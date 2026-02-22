import numpy as np
from typing import Dict, Any, List, Optional
import cv2
import time
import os
import json
import shutil

from real_robot.robot.single_arm_scene import SingleArmScene
from real_robot.robot.utils import get_grasp_rotation, snap_to_mask, process_depth
from real_robot.utils.mask_utils import get_mask_generator, get_mask_v2
from real_robot.utils.save_utils import *
from real_robot.utils.transform_utils import MOVE_ACC, MOVE_SPEED
from real_robot.primitives.single_arm_pick_and_place import SingleArmPickAndPlaceSkill
from real_robot.loggers.single_arm_pixel_logger import SingleArmPixelLogger
from actoris_harena import Arena

class SingleArmPickAndPlaceArena(Arena):
    """
    Real Single-Arm Arena implementation using only UR5e with only Pick and Place Primitive
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "single_arm_garment_pick_and_place_arena"
        self.config = config
        self.measure_time = config.get('measure_time', False)

        # Robot initialization
        dry_run = config.get("dry_run", False)
        
        self.single_arm = SingleArmScene(
            ur5e_robot_ip=config.get("ur5e_ip", "192.168.1.10"),
            dry_run=dry_run,
            radius=config.get('radius', [0.24, 0.54])
        )
        
        # Use the new Single Arm Skill
        self.pick_and_place_skill = SingleArmPickAndPlaceSkill(self.single_arm)
        
        self.logger = SingleArmPixelLogger()
        self.mask_generator = get_mask_generator()

        # Arena parameters
        self.num_train_trials = config.get("num_train_trials", 100)
        self.num_val_trials = config.get("num_val_trials", 10)
        self.num_eval_trials = config.get("num_eval_trials", 30)
        self.action_horizon = config.get("action_horizon", 20)
        self.snap_to_cloth_mask = config.get("snap_to_cloth_mask", False)
        self.init_from = config.get("init_from", "crumpled")
        self.maskout_background = config.get("maskout_background", False)
        self.use_sim_workspace = config.get("use_sim_workspace", False)
        self.asset_dir = f"{os.environ.get('MP_FOLD_PATH', '.')}/assets"
        self.track_trajectory = config.get("track_trajectory", False)
        
        self.current_episode = None
        self.frames = []
        self.all_infos = []
        self.goal = None
        self.debug = config.get("debug", False)

        self.resolution = (512, 512)
        self.action_step = 0
        self.evaluate_result = None
        self.last_flattened_step = -1
        self.id = 0
        self.init_coverage = None
        
        self.flattened_obs = None
        
        print('Finished init SingleArmPickAndPlaceArena')

    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.eid = episode_config.get("eid", np.random.randint(0, 9999)) if episode_config else 0
        print(f"[Arena] Resetting episode {self.eid}")

        self.flattened_obs = None
        self.get_flattened_obs()
        
        self.last_info = None
        self.action_step = 0
        
        self.init_coverage = None
        if hasattr(self, 'task'):
            self.task.reset(self)
        if self.init_from == 'crumpled':
            input("Press [Enter] to finish resetting cloth state to a crumpled state...")
        
        self.info = {}
        self.all_infos = [self.info]
        self.info = self._process_info(self.info)
        
        self.init_coverage = self.coverage
        self.clear_frames()
        self.primitive_time = []
        self.perception_time = []
        self.process_action_time = []
        self.all_infos = [self.info]
        return self.info

    def get_flattened_obs(self):
        if self.flattened_obs is None:
            self.garment_id = input("\n[Arena] Enter garment name (e.g. shirt_01): ").strip()
            base_asset_dir = f"{os.environ.get('MP_FOLD_PATH', '.')}/assets"
            save_dir = os.path.join(base_asset_dir, 'real_garments', self.garment_id)
            
            
            fn_info = os.path.join(save_dir, "info.json")

            if os.path.exists(save_dir) and os.path.exists(fn_info):
                print(f"[Arena] Found cached observation folder for '{self.garment_id}'. Loading images...")
                try:
                    rgb = load_colour("rgb", save_dir)
                    raw_rgb = load_colour("raw_rgb", save_dir)
                    depth = load_depth("depth", save_dir)
                    mask = load_mask("mask", save_dir)
                    r0_mask = load_mask("robot_0_mask", save_dir)

                    with open(fn_info, 'r') as f:
                        meta_info = json.load(f)

                    self.flattened_obs = {
                        'observation': {
                            "rgb": rgb,
                            "depth": depth,
                            "mask": mask,
                            "raw_rgb": raw_rgb,
                            "action_step": meta_info.get("action_step", 0),
                            "robot0_mask": r0_mask,
                        },
                        "eid": meta_info.get("eid", 0),
                        "arena_id": 0,
                        "arena": self
                    }
                    
                    self.flatten_coverage = np.sum(mask)
                    print(f"[Arena] Successfully loaded flattened state from PNGs/JSON, flatten coverage {self.flatten_coverage}.")

                except Exception as e:
                    print(f"[Arena] Error loading data: {e}. Will recapture manually.")
                    self.flattened_obs = None

            if self.flattened_obs is None:
                print("\n" + "=" * 60)
                print(f"No valid data found in '{save_dir}'.")
                print("Please prepare the flattened garment position manually.")
                print("=" * 60)
                input("Press [Enter] to capture the flattened cloth state...")
                self.flattened_obs = {}
                self.flattened_obs = self._process_info(self.flattened_obs, task_related=False, flattened_obs=False)
                self.flatten_coverage = self.coverage
                
                obs = self.flattened_obs['observation']
                print(f"[Arena] Saving human-readable observation to {save_dir}..., flatten coverage {self.flatten_coverage}")
                os.makedirs(save_dir, exist_ok=True)
                
                try:

                    save_colour(obs['rgb'], "rgb", save_dir)
                    save_colour(obs['raw_rgb'], "raw_rgb", save_dir)
                    save_depth(obs['depth'], "depth", save_dir)
                    save_mask(obs['mask'], "mask", save_dir)
                    save_mask(obs['robot0_mask'], "robot0_mask", save_dir)

                    
                    meta_info = {
                        "eid": int(self.flattened_obs.get("eid", 0)),
                        "action_step": int(obs.get("action_step", 0)),
                        "note": "Flattened state capture"
                    }
                    
                    with open(fn_info, 'w') as f:
                        json.dump(meta_info, f, indent=4)
                        
                    print("[Arena] Saved successfully.")
                except Exception as e:
                    print(f"[Arena] Warning: Could not save flattened obs: {e}")

                print("=" * 60 + "\n")
        
        return self.flattened_obs
    
    def _raw_to_crop(self, pt_raw):
        """Maps a point from raw robot coordinates back to the final square crop."""
        pt_rect = pt_raw - np.array([self.roi_x_min, self.roi_y_min])
        # Perfect 90 CW Rotation (with -1 offset for 0-index bounds)
        pt_rot = np.array([self.roi_h - 1 - pt_rect[1], pt_rect[0]])
        pt_crop = pt_rot - np.array([self.cx1, self.cy1])
        return pt_crop

    def _crop_to_raw(self, pt_crop):
        """Maps a point from the policy's crop coordinates out to raw robot coordinates."""
        pt_rot = pt_crop + np.array([self.cx1, self.cy1])
        # Undo Perfect 90 CW Rotation
        pt_rect = np.array([pt_rot[1], self.roi_h - 1 - pt_rot[0]])
        pt_raw = pt_rect + np.array([self.roi_x_min, self.roi_y_min])
        return pt_raw

    def step(self, action):
        if self.measure_time: start_time = time.time()

        norm_pixels = action.reshape(-1, 2)
        norm_pixels[:, [0, 1]] = norm_pixels[:, [1, 0]]
        #norm_pixels[:, 0], norm_pixels[:, 1] = norm_pixels[:, 1], norm_pixels[:, 0]

        
        # 1. Denormalize to crop space
        points_crop = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.float32)
        
        if self.debug:
            try:
                debug_raw = self.info['observation']['rgb'].copy()
                points_resize = ((norm_pixels + 1) / 2 * debug_raw.shape[0]).astype(np.float32)
        
                debug_raw = cv2.cvtColor(debug_raw, cv2.COLOR_RGB2BGR)
                
                
                px, py = int(points_resize[0, 0]), int(points_resize[0, 1])

                cv2.circle(debug_raw, (px, py), 6, (0, 0, 255), -1) 
                cv2.putText(debug_raw, "PICK", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                lx, ly = int(points_resize[1, 0]), int(points_resize[1, 1])
                cv2.circle(debug_raw, (lx, ly), 6, (255, 0, 0), -1) 
                cv2.putText(debug_raw, "PLACE", (lx+10, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                
                cv2.rectangle(debug_raw, (self.roi_x_min, self.roi_y_min), (self.roi_x_max, self.roi_y_max), (0, 255, 0), 2)

                os.makedirs('./tmp', exist_ok=True)
                cv2.imwrite('./tmp/debug_rgb_action.png', debug_raw)
            except Exception as e:
                print(f"[DEBUG] Failed to write debug image: {e}")

        # 2. Snap to Cloth Mask (in crop space)
        if self.snap_to_cloth_mask:
            mask = self.cloth_mask
            kernel = np.ones((3, 3), np.uint8) 
            eroded_mask = cv2.erode(mask, kernel, iterations=5)
            target_mask = mask if np.sum(eroded_mask) == 0 else eroded_mask
            
            # Snap both points
            points_crop[0] = snap_to_mask(points_crop[0], target_mask)
        
        # 3. Un-rotate and un-crop to raw robot coordinates
        points_orig = np.array([self._crop_to_raw(pt) for pt in points_crop])
        
        # 4. Snap to Robot Workspace in RAW space
        full_mask_0 = self.single_arm.get_workspace_mask()
        
        pre_snap_pick = points_orig[0].copy()
        final_pick = snap_to_mask(points_orig[0], full_mask_0)
        final_place = snap_to_mask(points_orig[1], full_mask_0)
        
        dist_moved = np.linalg.norm(final_pick - pre_snap_pick)
        if dist_moved > 10:
            print(f"[WARNING] Point was pushed {dist_moved:.1f} pixels by workspace mask snapping!")
        
        # 5. Calculate Rotation
        pt_crop_for_rot = self._raw_to_crop(final_pick)
        pick_angle_crop = get_grasp_rotation(self.cloth_mask, pt_crop_for_rot)

        # Subtract 90 degrees to undo the image rotation for the physical robot frame
        pick_angle = pick_angle_crop - (np.pi / 2)
        if pick_angle < -np.pi:
            pick_angle += np.pi
        
        # Construct Single Arm Action [px, py, lx, ly, rot]
        skill_action = np.concatenate([final_pick, final_place, [pick_angle]])

        # ---------------------------------------------------------------------
        # --- VISUAL DEBUGGING TOOL ---
        # ---------------------------------------------------------------------
        if self.debug:
            try:
                debug_raw = self.info['observation']['raw_rgb'].copy()
                debug_raw = cv2.cvtColor(debug_raw, cv2.COLOR_RGB2BGR)
                
                ox, oy = int(pre_snap_pick[0]), int(pre_snap_pick[1])
                cv2.circle(debug_raw, (ox, oy), 6, (0, 255, 255), -1) 
                
                px, py = int(final_pick[0]), int(final_pick[1])
                cv2.circle(debug_raw, (px, py), 6, (0, 0, 255), -1) 
                cv2.putText(debug_raw, "PICK", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                lx, ly = int(final_place[0]), int(final_place[1])
                cv2.circle(debug_raw, (lx, ly), 6, (255, 0, 0), -1) 
                cv2.putText(debug_raw, "PLACE", (lx+10, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                
                cv2.rectangle(debug_raw, (self.roi_x_min, self.roi_y_min), (self.roi_x_max, self.roi_y_max), (0, 255, 0), 2)

                os.makedirs('./tmp', exist_ok=True)
                cv2.imwrite('./tmp/debug_mapped_action.png', debug_raw)
            except Exception as e:
                print(f"[DEBUG] Failed to write debug image: {e}")
        # ---------------------------------------------------------------------

        if self.measure_time:
            self.process_action_time.append(time.time() - start_time)
            start_time = time.time()

        self.info = {}
        self.pick_and_place_skill.reset()
        
        self.pick_and_place_skill.step(skill_action)
        
        executed_points = np.concatenate([final_pick, final_place])

        if self.measure_time:
            self.primitive_time.append(time.time() - start_time)
            start_time = time.time()
        
        self.action_step += 1
        
        if self.action_step % 5 == 0:
            if hasattr(self.single_arm, 'restart_camera'):
                self.single_arm.restart_camera()
        
        self.all_infos.append(self.info)    
        self.info = self._process_info(self.info)
        
        # Normalize applied actions correctly using crop coordinates
        executed_crop_pts = np.array([self._raw_to_crop(pt) for pt in executed_points.reshape(-1, 2)])
        applied_action = (executed_crop_pts / self.crop_size) * 2.0 - 1.0
        applied_action[:, [0,1]] = applied_action[:,[1,0]]
        self.info['applied_action'] = applied_action.flatten()
            
        if self.measure_time:
            self.perception_time.append(time.time() - start_time)

        return self.info

    def _process_info(self, info, task_related=True, flattened_obs=True):
        self.single_arm.open_gripper()
        self.single_arm.home(MOVE_SPEED, MOVE_ACC)
        self.single_arm.out_scene(MOVE_SPEED, MOVE_ACC)
        time.sleep(1.0)

        raw_rgb, raw_depth = self.single_arm.take_rgbd()
        workspace_mask_0 = self.single_arm.get_workspace_mask()
        
        if workspace_mask_0 is None:
            workspace_mask_0 = np.zeros(raw_rgb.shape[:2], dtype=np.uint8)
        else:
            workspace_mask_0 = workspace_mask_0.astype(np.uint8)

        h, w = raw_rgb.shape[:2]
        
        # 1. Define Rectangular ROI
        self.roi_x_min = self.config.get("roi_x_min", w // 2)
        self.roi_x_max = self.config.get("roi_x_max", w)
        self.roi_y_min = self.config.get("roi_y_min", 0)
        self.roi_y_max = self.config.get("roi_y_max", h)
        
        rect_rgb = raw_rgb[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]
        rect_depth = raw_depth[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]
        rect_mask_0 = workspace_mask_0[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]
        
        self.roi_h, self.roi_w = rect_rgb.shape[:2]

        # 2. Rotate the ROI 90 degrees clockwise
        rot_rgb = cv2.rotate(rect_rgb, cv2.ROTATE_90_CLOCKWISE)
        rot_depth = cv2.rotate(rect_depth, cv2.ROTATE_90_CLOCKWISE)
        rot_mask_0 = cv2.rotate(rect_mask_0, cv2.ROTATE_90_CLOCKWISE)
        
        rot_h, rot_w = rot_rgb.shape[:2]

        # 3. Default Center square crop
        self.crop_size = min(rot_h, rot_w)
        self.cx1 = rot_w // 2 - self.crop_size // 2
        self.cy1 = rot_h // 2 - self.crop_size // 2

        crop_rgb = rot_rgb[self.cy1:self.cy1+self.crop_size, self.cx1:self.cx1+self.crop_size]
        crop_cloth_mask = get_mask_v2(
            self.mask_generator, crop_rgb, debug=self.debug, 
            mask_threshold_min=3000, mask_threshold_max=30000)

        # ---------------------------------------------------------------------
        # --- HEURISTIC WORKSPACE SAMPLING ---
        # ---------------------------------------------------------------------
        if np.sum(crop_cloth_mask) < 50:
            print("[Arena] Heuristic: Central window is empty. Searching for cloth in full ROI...")
            full_rot_mask = get_mask_v2(self.mask_generator, rot_rgb, debug=False, 
            mask_threshold_min=3000, mask_threshold_max=30000)
            
            if np.sum(full_rot_mask) > 50:
                # Find center of mass of the cloth
                ys, xs = np.where(full_rot_mask > 0)
                cy_cloth = int(np.mean(ys))
                cx_cloth = int(np.mean(xs))
                
                # Shift cx1, cy1 to perfectly center the cloth, clamped to edges
                self.cx1 = np.clip(cx_cloth - self.crop_size // 2, 0, rot_w - self.crop_size)
                self.cy1 = np.clip(cy_cloth - self.crop_size // 2, 0, rot_h - self.crop_size)
                
                print(f"[Arena] Heuristic: Shifted window to ({self.cx1}, {self.cy1}).")
                
                # Re-crop everything with the new centered coordinates
                crop_rgb = rot_rgb[self.cy1:self.cy1+self.crop_size, self.cx1:self.cx1+self.crop_size]
                crop_cloth_mask = full_rot_mask[self.cy1:self.cy1+self.crop_size, self.cx1:self.cx1+self.crop_size]
            else:
                print("[Arena] Warning: Cloth not found in the entire ROI.")
        # ---------------------------------------------------------------------

        # Proceed with depth and mask mapping using the (potentially shifted) cx1, cy1
        crop_depth = rot_depth[self.cy1:self.cy1+self.crop_size, self.cx1:self.cx1+self.crop_size]
        crop_mask_0 = rot_mask_0[self.cy1:self.cy1+self.crop_size, self.cx1:self.cx1+self.crop_size]
        
        self.cloth_mask = crop_cloth_mask
        
        

        # Debug Visualizations with Workspace Mask Shading
        if self.debug:
            os.makedirs('./tmp', exist_ok=True)
            
            def apply_shade_bgr(img_rgb, mask):
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                shaded = img_bgr.astype(np.float32)
                valid = mask > 0
                
                tint = np.array([0, 200, 0], dtype=np.float32)
                shaded[valid] = shaded[valid] * 0.8 + tint * 0.2
                shaded[~valid] = shaded[~valid] * 0.3
                
                return shaded.astype(np.uint8)

            cv2.imwrite('./tmp/debug_01_raw.png', apply_shade_bgr(raw_rgb, workspace_mask_0))
            cv2.imwrite('./tmp/debug_02_rect_roi.png', apply_shade_bgr(rect_rgb, rect_mask_0))
            cv2.imwrite('./tmp/debug_03_rotated.png', apply_shade_bgr(rot_rgb, rot_mask_0))
            cv2.imwrite('./tmp/debug_04_final_crop.png', apply_shade_bgr(crop_rgb, crop_mask_0))

        norm_depth = process_depth(crop_depth)

        resized_rgb = cv2.resize(crop_rgb, self.resolution)
        resized_depth = cv2.resize(norm_depth, self.resolution)
        resized_mask = cv2.resize(crop_cloth_mask.astype(np.uint8), self.resolution)
        resized_r0_mask = cv2.resize(crop_mask_0, self.resolution, interpolation=cv2.INTER_NEAREST)

        if self.maskout_background:
            is_background = resized_mask == 0
            resized_rgb[is_background] = 0
            resized_depth[is_background] = 0

        info.update({
            'observation': {
                "rgb": resized_rgb,
                "depth": resized_depth,
                "mask": resized_mask.astype(np.bool_),
                "raw_rgb": raw_rgb,
                "action_step": self.action_step,
                "robot0_mask": resized_r0_mask.astype(np.bool_),
                "cloth-workspace-mask": np.logical_and(resized_mask, resized_r0_mask)

            },
            "eid": getattr(self, 'eid', 0),
            "arena": self,
            "arena_id": getattr(self, 'id', 0),
        })

        self.coverage = np.sum(resized_mask)
        if self.init_coverage is None: self.init_coverage = self.coverage

        if flattened_obs:
            info['flattened_obs'] = self.get_flattened_obs()
            for k, v in info['flattened_obs'].items():
                info['observation'][f'flattened-{k}'] = v
        
        info['done'] = self.action_step >= self.action_horizon

        if task_related:
            info['evaluation'] = self.evaluate()
            if info['evaluation'].get('normalised_coverage', 0) > 0.9:
                self.last_flattened_step = self.action_step
        
            info['observation']['last_flattened_step'] = self.last_flattened_step
            info['success'] =  self.success()
            
            if info['evaluation'] != {}:
                print('evaluation', info['evaluation'])
                info['reward'] = self.task.reward(self.last_info, None, info)
            
            if hasattr(self, 'task'):
                goals = self.task.get_goals()
                if len(goals) > 0:
                    goal = goals[0]
                    info['goal'] = {}
                    for k, v in goal[-1]['observation'].items():
                        if k == 'rgb' and self.maskout_background and ('mask' in  goal[-1]['observation']):
                            is_background = goal[-1]['observation']['mask'] == 0
                            v[is_background] = 0

                        info['goal'][k] = v
                        info['observation'][f'goal_{k}'] = v
                        info['observation'][f'goal-{k}'] = v

        self.last_info = info
        self.single_arm.home(MOVE_SPEED, MOVE_ACC)
        return info

    def clear_frames(self): self.frames = []
    def evaluate(self): return self.task.evaluate(self) if hasattr(self, 'task') else {}
    def success(self): return self.task.success(self) if hasattr(self, 'task') else False

    def get_action_space(self):
        return self._action_space

    def sample_random_action(self):
        return self._action_space.sample()

    def get_no_op(self) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

if __name__ == "__main__":
    print("Testing Rotation, Cropping, Shading & Heuristic Pipeline...")
    
    test_config = {
        "debug": True,
        "dry_run": True,
        "action_horizon": 5,
        "roi_x_min": 771, 
        "roi_x_max": 1120,
        "roi_y_min": 50,
        "roi_y_max": 670
    }

    arena = SingleArmPickAndPlaceArena(test_config)
    
    # Create Mock RGB Data
    dummy_rgb = np.ones((720, 1280, 3), dtype=np.uint8) * 150
    
    # Add a visual reference (red square) EXTREMELY HIGH in the ROI
    # This ensures the standard center crop misses it, triggering the heuristic!
    cv2.rectangle(dummy_rgb, (800, 60), (900, 160), (255, 0, 0), -1) 
    
    dummy_depth = np.random.rand(720, 1280).astype(np.float32)
    
    # Create a dummy workspace mask
    dummy_mask = np.zeros((720, 1280), dtype=np.uint8)
    cv2.circle(dummy_mask, (950, 360), 300, 1, -1)
    
    # Inject Mock Handlers
    arena.single_arm.take_rgbd = lambda: (dummy_rgb, dummy_depth)
    arena.single_arm.get_workspace_mask = lambda: dummy_mask
    arena.get_flattened_obs = lambda: {"test_obs": 1}
    
    # Mock the get_mask_v2 to simply isolate the Red channel (simulating cloth detection)
    def mock_get_mask(generator, img, debug=False):
        # Return True where the Red channel is strong (finding our red square)
        return (img[:,:,2] > 200).astype(np.uint8) * 255
    
    arena.mask_generator = "dummy_generator"
    # Override global function locally for testing
    get_mask_v2 = mock_get_mask
    
    # Execute image pipeline
    test_info = {}
    arena._process_info(test_info, task_related=False, flattened_obs=False)
    
    print("\n[SUCCESS] Pipeline executed.")
    print("You should see '[Arena] Heuristic: Central window is empty.' in the logs above.")
    print("Check './tmp/debug_04_final_crop.png' - the red square should be perfectly centered!")