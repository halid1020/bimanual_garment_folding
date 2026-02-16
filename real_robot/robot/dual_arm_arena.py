import numpy as np
from typing import Dict, Any, List, Optional
import cv2
import time
import os
import json
import shutil

from real_robot.robot.dual_arm_scene import DualArmScene
from real_robot.utils.mask_utils import get_mask_generator, get_mask_v2
from real_robot.utils.transform_utils import MOVE_ACC, MOVE_SPEED
from real_robot.primitives.pick_and_place import PickAndPlaceSkill
from real_robot.primitives.pick_and_fling import PickAndFlingSkill
from real_robot.robot.pixel_based_primitive_env_logger import PixelBasedPrimitiveEnvLogger
from real_robot.robot.pixel_based_primitive_env_imp_logger import PixelBasedPrimitiveImpEnvLogger
from actoris_harena import Arena

class DualArmArena(Arena):
    """
    Real Dual-Arm Arena implementation using UR5e and UR16e robots.
    """

    def __init__(self, config):
        # super().__init__(config)
        self.name = "dual_arm_garment_arena"
        self.config = config

        self.measure_time = config.get('measure_time', False)

        # Robot initialization
        dry_run = config.get("dry_run", False)
        self.dual_arm = DualArmScene(
            ur5e_robot_ip=config.get("ur5e_ip", "192.168.1.10"),
            ur16e_robot_ip=config.get("ur16e_ip", "192.168.1.102"),
            dry_run=dry_run
        )

        self.pick_and_place_skill = PickAndPlaceSkill(self.dual_arm)
        self.pick_and_fling_skill = PickAndFlingSkill(self.dual_arm)
        self.logger = PixelBasedPrimitiveImpEnvLogger()

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
        self.asset_dir = f"{os.environ['MP_FOLD_PATH']}/assets"
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
        
        # Initialize flattened storage
        self.flattened_obs = None
        
        print('Finished init DualArmArena')

    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reset the arena (move both arms home, open grippers, capture initial observation).
        """
        self.eid = episode_config.get("eid", np.random.randint(0, 9999)) if episode_config else 0
        print(f"[Arena] Resetting episode {self.eid}")

        # Ensure we have the flattened observation (loaded or captured) before starting
        self.flattened_obs = None
        self.get_flattened_obs()
        
        self.last_info = None
        self.action_step = 0
        
        # Reset robot to safe state
        self.init_coverage = None
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

    def _process_depth(self, raw_depth):
        """
        Process raw depth image:
        1. Convert to meters
        2. Estimate table height (95th percentile)
        3. Clip to range [table - 0.1m, table + 0.1m]
        4. Normalize (0 = Deepest/Background, 1 = Shallowest/Closest)
        """
        # 1. Convert to Meters (RealSense is usually uint16 mm)
        if raw_depth.dtype == np.uint16:
            depth_m = raw_depth.astype(np.float32) / 1000.0
        else:
            depth_m = raw_depth.astype(np.float32)

        # 2. Estimate Camera Height (Distance to Table)
        # 95th percentile is robust against sensor noise (dropouts)
        camera_height = np.percentile(depth_m, 95)
        
        # 3. Define Clipping Range (+- 0.1m around table)
        # Objects on table will be closer (smaller value) than camera_height
        min_dist = camera_height - 0.1
        max_dist = camera_height + 0.1
        
        clipped_depth = np.clip(depth_m, min_dist, max_dist)
        
        # 4. Min-Max Normalization (Inverted)
        # 0 = Deepest point (max_dist), 1 = Shallowest point (min_dist)
        # Formula: (max - value) / (max - min)
        norm_depth = (max_dist - clipped_depth) / (max_dist - min_dist)
        
        return norm_depth
    
    def _process_info(self, info, task_related=True, flattened_obs=True):
        self.dual_arm.both_open_gripper()
        self.dual_arm.both_home(MOVE_SPEED, MOVE_ACC)
        self.dual_arm.both_out_scene(MOVE_SPEED, MOVE_ACC)

        # ADD THIS: Brief pause to let robot settle and camera clear
        time.sleep(1.0)

        raw_rgb, raw_depth = self.dual_arm.take_rgbd()
        
        workspace_mask_0, workspace_mask_1 = self.dual_arm.get_workspace_masks()
        workspace_mask_0 = workspace_mask_0.astype(np.uint8)
        workspace_mask_1 = workspace_mask_1.astype(np.uint8)

        h, w = raw_rgb.shape[:2]
        crop_size = min(h, w)  

        x1 = w // 2 - crop_size // 2
        y1 = h // 2 - crop_size // 2
        x2, y2 = x1 + crop_size, y1 + crop_size

        self.x1 = x1
        self.y1 = y1
        self.crop_size = crop_size

        crop_rgb = raw_rgb[y1:y2, x1:x2]
        crop_mask = get_mask_v2(self.mask_generator, crop_rgb, debug=self.debug)
        self.cloth_mask = crop_mask
        self.coverage = np.sum(self.cloth_mask)
        if self.init_coverage == None:
            self.init_coverage = self.coverage
        crop_depth = raw_depth[y1:y2, x1:x2]

        crop_depth = self._process_depth(crop_depth)
        
        crop_workspace_mask_0 = workspace_mask_0[y1:y2, x1:x2]
        crop_workspace_mask_1 = workspace_mask_1[y1:y2, x1:x2]

        ## Resize images
        self.cropped_resolution = crop_rgb.shape[:2]
        resized_rgb = cv2.resize(crop_rgb, self.resolution)
        resized_depth = cv2.resize(crop_depth, self.resolution)
        resized_mask = cv2.resize(crop_mask.astype(np.uint8), self.resolution)

        if self.maskout_background:
            # apply resized_mask on resized_rgb and resized_depeth.
            is_background = resized_mask == 0
            resized_rgb[is_background] = 0
            resized_depth[is_background] = 0
        
        self.resized_workspace_mask_0 = cv2.resize(crop_workspace_mask_0, self.resolution, interpolation=cv2.INTER_NEAREST)
        self.resized_workspace_mask_1 = cv2.resize(crop_workspace_mask_1, self.resolution, interpolation=cv2.INTER_NEAREST)
        
        input_mask_0 = self.resized_workspace_mask_1.astype(np.bool_)
        input_mask_1 = self.resized_workspace_mask_0.astype(np.bool_)
        

        if self.use_sim_workspace:
            mask_dir = f"{self.asset_dir}/sim_masks/"
            r0_path = os.path.join(mask_dir, "robot0_mask.png")
            r1_path = os.path.join(mask_dir, "robot1_mask.png")

            # 1. Load masks
            raw_mask_0 = cv2.imread(r0_path, cv2.IMREAD_GRAYSCALE)
            raw_mask_1 = cv2.imread(r1_path, cv2.IMREAD_GRAYSCALE)

            # 2. Resize with LINEAR interpolation to smooth edges (avoids blockiness)
            # Note: Do not use INTER_NEAREST here if you want smooth curves.
            resized_0 = cv2.resize(raw_mask_0, self.resolution, interpolation=cv2.INTER_LINEAR)
            resized_1 = cv2.resize(raw_mask_1, self.resolution, interpolation=cv2.INTER_LINEAR)

            # 3. Threshold to convert back to boolean (0 or 1)
            # Since Linear interpolation introduces gray values at edges, > 127 cleans it up.
            input_mask_0 = resized_0 > 127
            input_mask_1 = resized_1 > 127
        
        #print('input mask shape', input_mask_0.shape)

        info.update({
            'observation': {
                "rgb": resized_rgb,
                "depth": resized_depth,
                "mask": resized_mask.astype(np.bool_),
                "raw_rgb": raw_rgb,
                "action_step": self.action_step,
                "robot0_mask": input_mask_0,
                "robot1_mask": input_mask_1 
            },
            
            "eid": self.eid,
            "arena_id": 0,
            "arena": self,
        })

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
            
            goals = self.task.get_goals()
            if len(goals) > 0:
                goal = goals[0]
                info['goal'] = {}
                for k, v in goal[-1]['observation'].items():
                    if k == 'rgb' and self.maskout_background and ('mask' in  goal[-1]['observation']):
                        # apply resized_mask on resized_rgb and resized_depeth.
                        is_background = goal[-1]['observation']['mask'] == 0
                        v[is_background] = 0

                    info['goal'][k] = v
                    info['observation'][f'goal_{k}'] = v
        
        self.last_info = info
        self.dual_arm.both_home(MOVE_SPEED, MOVE_ACC)
        return info

    def get_trajectory_infos(self):
        return self.all_infos
    
    def get_flattened_obs(self):
        if self.flattened_obs is None:
            self.garment_id = input("\n[Arena] Enter garment name (e.g. shirt_01): ").strip()
            base_asset_dir = f"{os.environ.get('MP_FOLD_PATH', '.')}/assets"
            save_dir = os.path.join(base_asset_dir, 'real_garments', self.garment_id)
            
            fn_rgb = os.path.join(save_dir, "rgb.png")
            fn_raw_rgb = os.path.join(save_dir, "raw_rgb.png")
            fn_depth = os.path.join(save_dir, "depth.png")
            fn_mask = os.path.join(save_dir, "mask.png")
            fn_r0_mask = os.path.join(save_dir, "robot0_mask.png")
            fn_r1_mask = os.path.join(save_dir, "robot1_mask.png")
            fn_info = os.path.join(save_dir, "info.json")

            if os.path.exists(save_dir) and os.path.exists(fn_info):
                print(f"[Arena] Found cached observation folder for '{self.garment_id}'. Loading images...")
                try:
                    rgb = cv2.cvtColor(cv2.imread(fn_rgb), cv2.COLOR_BGR2RGB)
                    raw_rgb = cv2.cvtColor(cv2.imread(fn_raw_rgb), cv2.COLOR_BGR2RGB)
                    depth = cv2.imread(fn_depth, cv2.IMREAD_UNCHANGED)
                    mask_img = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)
                    mask = (mask_img > 127).astype(np.bool_)
                    r0_mask = (cv2.imread(fn_r0_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.bool_)
                    r1_mask = (cv2.imread(fn_r1_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.bool_)

                    with open(fn_info, 'r') as f:
                        meta_info = json.load(f)

                    self.flattened_obs = {
                        'observation': {
                            "rgb": rgb,
                            "depth": depth,
                            "mask": mask,
                            "raw_rgb": raw_rgb,
                            "action_step": meta_info.get("action_step", 0),
                            "robot0_mask": r1_mask,
                            "robot1_mask": r0_mask,
                        },
                        "eid": meta_info.get("eid", 0),
                        "arena_id": 0,
                        "arena": self
                    }
                    
                    self.flatten_coverage = np.sum(mask)
                    print("[Arena] Successfully loaded flattened state from PNGs/JSON.")

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
                print(f"[Arena] Saving human-readable observation to {save_dir}...")
                os.makedirs(save_dir, exist_ok=True)
                
                try:
                    cv2.imwrite(fn_rgb, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_raw_rgb, cv2.cvtColor(obs['raw_rgb'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_depth, obs['depth'])
                    cv2.imwrite(fn_mask, (obs['mask'] * 255).astype(np.uint8))
                    cv2.imwrite(fn_r0_mask, (obs['robot0_mask'] * 255).astype(np.uint8))
                    cv2.imwrite(fn_r1_mask, (obs['robot1_mask'] * 255).astype(np.uint8))
                    
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

    def _snap_to_mask(self, point, mask):
        point = np.array(point, dtype=int)
        h, w = mask.shape
        x, y = np.clip(point[0], 0, w - 1), np.clip(point[1], 0, h - 1)

        if mask[y, x] > 0:
            return np.array([x, y])

        valid_indices = np.argwhere(mask > 0)
        
        if len(valid_indices) == 0:
            print("[Warning] Workspace mask is empty! Cannot snap point.")
            return np.array([x, y])

        current_pos_yx = np.array([y, x])
        distances = np.sum((valid_indices - current_pos_yx) ** 2, axis=1)
        
        nearest_idx = np.argmin(distances)
        nearest_yx = valid_indices[nearest_idx]
        
        return np.array([nearest_yx[1], nearest_yx[0]])

    def _get_grasp_rotation(self, mask, point):
        """
        Calculates rotation angle by finding the strongest edge in the neighborhood.
        """
        x, y = int(point[0]), int(point[1])
        h, w = mask.shape

        r = 15
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r + 1), min(h, y + r + 1)
        
        roi = mask[y1:y2, x1:x2].astype(np.float32)
        
        if np.min(roi) == np.max(roi):
            return 0.0

        roi = cv2.GaussianBlur(roi, (7, 7), 1.0)

        gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        
        if magnitude[max_idx] < 1e-3:
            return 0.0
            
        best_gx = gx[max_idx]
        best_gy = gy[max_idx]
        
        angle = np.arctan2(best_gy, best_gx)
        angle += np.pi / 2 
        
        while angle > np.pi / 2:
            angle -= np.pi
        while angle < -np.pi / 2:
            angle += np.pi
            
        return angle

    def step(self, action):
        if self.measure_time:
            start_time = time.time()

        norm_pixels = np.array(list(action.values())[0]).reshape(-1, 2)
        action_type = list(action.keys())[0]

        if action_type in ['norm-pixel-pick-and-place', 'norm-pixel-pick-and-fling']:

            points_crop = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.int32)
            if self.snap_to_cloth_mask:
                mask = self.cloth_mask
                kernel = np.ones((3, 3), np.uint8) 
                eroded_mask = cv2.erode(mask, kernel, iterations=10)
                
                if np.sum(eroded_mask) == 0:
                    print("[Warning] Erosion removed entire mask. Using original mask.")
                    target_mask = mask
                else:
                    target_mask = eroded_mask

                snapped_points = []
                for i, pt in enumerate(points_crop):
                    if i < 2:
                        snapped_points.append(self._snap_to_mask(pt, target_mask))
                    else:
                        snapped_points.append(pt)
                points_crop = np.array(snapped_points)
            
            points_orig = points_crop + np.array([self.x1, self.y1])
            points_executed = points_orig.flatten()

            full_mask_0, full_mask_1 = self.dual_arm.get_workspace_masks()
            
            # --- Robot Assignment, Rotation, and Validity Logic ---
            pick_angles = [0.0, 0.0]
            valid_flags = [1.0, 1.0] # Default True

            # Helper to check if a crop-space point is on the cloth mask
            def check_validity(pt_crop):
                x, y = int(pt_crop[0]), int(pt_crop[1])
                h, w = self.cloth_mask.shape
                if 0 <= x < w and 0 <= y < h:
                    return 1.0 if self.cloth_mask[y, x] > 0 else 0.0
                return 0.0

            if len(points_orig) == 4:
                p0_orig, p1_orig = points_orig[0], points_orig[1]
                l0_orig, l1_orig = points_orig[2], points_orig[3]

                pair_a = (p0_orig, l0_orig)
                pair_b = (p1_orig, l1_orig)

                if pair_a[0][0] < pair_b[0][0]:
                    pair_a, pair_b = pair_b, pair_a
                
                final_pick_0 = self._snap_to_mask(pair_a[0], full_mask_0)
                final_place_0 = self._snap_to_mask(pair_a[1], full_mask_0)
                
                final_pick_1 = self._snap_to_mask(pair_b[0], full_mask_1)
                final_place_1 = self._snap_to_mask(pair_b[1], full_mask_1)

                points_executed = np.concatenate([
                    final_pick_0, final_pick_1, 
                    final_place_0, final_place_1
                ])
                
                # Rotation & Validity Calculation
                pt0_crop = final_pick_0 - np.array([self.x1, self.y1])
                pt1_crop = final_pick_1 - np.array([self.x1, self.y1])
                
                angle_0 = self._get_grasp_rotation(self.cloth_mask, pt0_crop)
                angle_1 = self._get_grasp_rotation(self.cloth_mask, pt1_crop)
                pick_angles = [angle_0, angle_1]

                # Validity: check mask at the crop coordinates
                valid_0 = check_validity(pt0_crop)
                valid_1 = check_validity(pt1_crop)
                valid_flags = [valid_0, valid_1]

            elif len(points_orig) == 2:
                p0_orig, p1_orig = points_orig[0], points_orig[1]

                if p0_orig[0] < p1_orig[0]:
                    p0_orig, p1_orig = p1_orig, p0_orig
                
                final_pick_0 = self._snap_to_mask(p0_orig, full_mask_0)
                final_pick_1 = self._snap_to_mask(p1_orig, full_mask_1)
                points_executed = np.concatenate([final_pick_0, final_pick_1])
                
                pt0_crop = final_pick_0 - np.array([self.x1, self.y1])
                pt1_crop = final_pick_1 - np.array([self.x1, self.y1])
                
                angle_0 = self._get_grasp_rotation(self.cloth_mask, pt0_crop)
                angle_1 = self._get_grasp_rotation(self.cloth_mask, pt1_crop)
                pick_angles = [angle_0, angle_1]

                valid_0 = check_validity(pt0_crop)
                valid_1 = check_validity(pt1_crop)
                valid_flags = [valid_0, valid_1]

            # Concatenate: [coords, angles, flags]
            # Size: 8 + 2 + 2 = 12 floats (for pick-place)
            full_action = np.concatenate([
                points_executed.copy(), 
                np.array(pick_angles),
                np.array(valid_flags)
            ])
           
        if self.measure_time:
            self.process_action_time.append(time.time() - start_time)
            start_time = time.time()

        self.info = {}
        print(f'action step {self.action_step}')
        if action_type == 'norm-pixel-pick-and-place':
            self.pick_and_place_skill.reset()
            self.pick_and_place_skill.step(full_action)
        elif action_type == 'norm-pixel-pick-and-fling':
            self.pick_and_fling_skill.reset()
            traj = self.pick_and_fling_skill.step(full_action, record_debug=self.track_trajectory)
            print('!!!len trj', len(traj))
            self.info['debug_trajectory'] = traj
        elif action_type == 'no-operation':
            print('no operation!!!')
            pass
        else:
            raise ValueError
        
        self.action_step += 1

        if self.action_step % 5 == 0:
            self.dual_arm.restart_camera()
        
        self.all_infos.append(self.info)

        if self.measure_time:
            self.primitive_time.append(time.time() - start_time)
            start_time = time.time()

        self.info = self._process_info(self.info)

        if action_type in ['norm-pixel-pick-and-place', 'norm-pixel-pick-and-fling']:
            applied_action = (1.0*points_executed.reshape(-1, 2) - np.array([self.x1, self.y1]))/self.crop_size * 2 - 1
            self.info['applied_action'] = {
                action_type: applied_action.flatten()
            }

        if self.measure_time:
            self.perception_time.append(time.time() - start_time)

        return self.info

    def get_frames(self) -> List[np.ndarray]:
        return self.frames

    def success(self):
        return self.task.success(self)
    
    def clear_frames(self):
        self.frames = []

    def get_goal(self) -> Dict[str, Any]:
        return self.goal or {}

    def get_action_space(self):
        return self._action_space

    def sample_random_action(self):
        return self._action_space.sample()

    def get_no_op(self) -> np.ndarray:
        return np.zeros(12, dtype=np.float32)

    def get_action_horizon(self) -> int:
        return self.action_horizon
    
    def evaluate(self):
        if (self.evaluate_result is None) or (self.action_step == 0):
            self.evaluate_result = self.task.evaluate(self)
        return self.evaluate_result

    def set_task(self, task):
        self.task = task

    def visualize_workspace(self):
        if self.dual_arm.dry_run:
            print("[Dry-run] Visualization skipped.")
            return None
        rgb, _ = self.dual_arm.camera.take_rgbd()
        shaded = self.dual_arm.apply_workspace_mask(rgb)
        return shaded