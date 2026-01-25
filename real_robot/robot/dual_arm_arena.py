import numpy as np
from typing import Dict, Any, List, Optional
import cv2
import time
import os
import json   # Added for human-readable metadata
import shutil # Added for directory management

from real_robot.robot.dual_arm_scene import DualArmScene
from real_robot.utils.mask_utils import get_mask_generator, get_mask_v2
from real_robot.utils.save_utils import save_colour
from real_robot.primitives.pick_and_place import PickAndPlaceSkill
from real_robot.primitives.pick_and_fling import PickAndFlingSkill
from real_robot.robot.pixel_based_primitive_env_logger import PixelBasedPrimitiveEnvLogger

from agent_arena import Arena

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
        self.logger = PixelBasedPrimitiveEnvLogger()

        self.mask_generator = get_mask_generator()

        # Arena parameters
        self.num_train_trials = config.get("num_train_trials", 100)
        self.num_val_trials = config.get("num_val_trials", 10)
        self.num_eval_trials = config.get("num_eval_trials", 30)
        self.action_horizon = config.get("action_horizon", 20)
        self.snap_to_cloth_mask = config.get("snap_to_cloth_mask", False)
        self.init_from = config.get("init_from", "crumpled")

        self.current_episode = None
        self.frames = []
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
        self.get_flattened_obs()
        
        self.last_info = None
        self.action_step = 0
        
        # Reset robot to safe state
        self.init_coverage = None
        self.task.reset(self)
        if self.init_from == 'crumpled':
            input("Press [Enter] to finish resetting cloth state to a crumpled state...")
        self.info = self._get_info()
        self.init_coverage = self.coverage
        self.clear_frames()
        self.primitive_time = []
        self.perception_time = []
        self.process_action_time = []
        return self.info
    
    def _get_info(self, task_related=True, flattened_obs=True):
        # Return to camera position after manipulation
        self.dual_arm.both_open_gripper()
        self.dual_arm.both_home()
        self.dual_arm.both_out_scene()

        # -----------------------------
        # Capture post-interaction scene
        # -----------------------------
        raw_rgb, raw_depth = self.dual_arm.take_rgbd()
        
        workspace_mask_0, workspace_mask_1 = self.dual_arm.get_workspace_masks()
        workspace_mask_0 = workspace_mask_0.astype(np.uint8)
        workspace_mask_1 = workspace_mask_1.astype(np.uint8)

        # -----------------------------
        # Center crop bird’s-eye view around cloth
        # -----------------------------
        h, w = raw_rgb.shape[:2]
        crop_size = min(h, w)  

        x1 = w // 2 - crop_size // 2
        y1 = h // 2 - crop_size // 2
        x2, y2 = x1 + crop_size, y1 + crop_size

        self.x1 = x1
        self.y1 = y1
        self.crop_size = crop_size

        # Perform crops
        crop_rgb = raw_rgb[y1:y2, x1:x2]
        crop_mask = get_mask_v2(self.mask_generator, crop_rgb, debug=self.config.debug)
        self.cloth_mask = crop_mask
        self.coverage = np.sum(self.cloth_mask)
        if self.init_coverage == None:
            self.init_coverage = self.coverage
        crop_depth = raw_depth[y1:y2, x1:x2]
        
        crop_workspace_mask_0 = workspace_mask_0[y1:y2, x1:x2]
        crop_workspace_mask_1 = workspace_mask_1[y1:y2, x1:x2]

        ## Resize images
        self.cropped_resolution = crop_rgb.shape[:2]
        resized_rgb = cv2.resize(crop_rgb, self.resolution)
        resized_depth = cv2.resize(crop_depth, self.resolution)
        resized_mask = cv2.resize(crop_mask.astype(np.uint8), self.resolution)
        
        # Store resized workspace masks for use in step()
        self.resized_workspace_mask_0 = cv2.resize(crop_workspace_mask_0, self.resolution, interpolation=cv2.INTER_NEAREST)
        self.resized_workspace_mask_1 = cv2.resize(crop_workspace_mask_1, self.resolution, interpolation=cv2.INTER_NEAREST)

        # -----------------------------
        # Store and return information
        # -----------------------------
        info = {
            'observation': {
                "rgb": resized_rgb,
                "depth": resized_depth,
                "mask": resized_mask.astype(np.bool_),
                "raw_rgb": raw_rgb,
                "action_step": self.action_step,
                "robot0_mask": self.resized_workspace_mask_0.astype(np.bool_),
                "robot1_mask": self.resized_workspace_mask_1.astype(np.bool_),
            },
            
            "eid": self.eid,
            "arena_id": 0,
            "arena": self,
        }

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
                    info['goal'][k] = v
        
        self.last_info = info
        self.dual_arm.both_home()
        return info

    def get_flattened_obs(self):
        """
        Loads flattened observation from a human-readable directory (PNGs + JSON).
        If not found, asks user to set it up, captures it, and saves it as images.
        """
        if self.flattened_obs is None:
            # 1. Ask for garment ID
            garment_id = input("\n[Arena] Enter garment name (e.g. shirt_01): ").strip()
            
            # 2. Setup asset directory path
            base_asset_dir = f"{os.environ.get('MP_FOLD_PATH', '.')}/assets"
            save_dir = os.path.join(base_asset_dir, 'real_garments', garment_id)
            
            # Define filenames
            fn_rgb = os.path.join(save_dir, "rgb.png")
            fn_raw_rgb = os.path.join(save_dir, "raw_rgb.png")
            fn_depth = os.path.join(save_dir, "depth.png")
            fn_mask = os.path.join(save_dir, "mask.png")
            fn_r0_mask = os.path.join(save_dir, "robot0_mask.png")
            fn_r1_mask = os.path.join(save_dir, "robot1_mask.png")
            fn_info = os.path.join(save_dir, "info.json")

            # 3. Check if directory and critical files exist
            if os.path.exists(save_dir) and os.path.exists(fn_info):
                print(f"[Arena] Found cached observation folder for '{garment_id}'. Loading images...")
                try:
                    # Load Images
                    # Note: cv2.imread loads as BGR, we usually work in RGB, so convert if needed. 
                    # Assuming the rest of your pipeline expects RGB.
                    rgb = cv2.cvtColor(cv2.imread(fn_rgb), cv2.COLOR_BGR2RGB)
                    raw_rgb = cv2.cvtColor(cv2.imread(fn_raw_rgb), cv2.COLOR_BGR2RGB)
                    
                    # Load Depth (IMREAD_UNCHANGED keeps 16-bit depth if saved as such)
                    depth = cv2.imread(fn_depth, cv2.IMREAD_UNCHANGED)
                    
                    # Load Masks (Read as grayscale/uint8)
                    mask_img = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)
                    # Convert back to boolean/binary mask
                    mask = (mask_img > 127).astype(np.bool_)
                    
                    r0_mask = (cv2.imread(fn_r0_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.bool_)
                    r1_mask = (cv2.imread(fn_r1_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.bool_)

                    # Load Metadata JSON
                    with open(fn_info, 'r') as f:
                        meta_info = json.load(f)

                    # Reconstruct Dictionary
                    self.flattened_obs = {
                        'observation': {
                            "rgb": rgb,
                            "depth": depth,
                            "mask": mask,
                            "raw_rgb": raw_rgb,
                            "action_step": meta_info.get("action_step", 0),
                            "robot0_mask": r0_mask,
                            "robot1_mask": r1_mask,
                        },
                        "eid": meta_info.get("eid", 0),
                        "arena_id": 0,
                        "arena": self
                    }
                    
                    # Set Coverage
                    self.flatten_coverage = np.sum(mask)
                    print("[Arena] Successfully loaded flattened state from PNGs/JSON.")

                except Exception as e:
                    print(f"[Arena] Error loading data: {e}. Will recapture manually.")
                    self.flattened_obs = None # Force recapture

            # 4. If not loaded (or failed load), capture manually
            if self.flattened_obs is None:
                print("\n" + "=" * 60)
                print(f"No valid data found in '{save_dir}'.")
                print("Please prepare the flattened garment position manually.")
                print("=" * 60)
                input("Press [Enter] to capture the flattened cloth state...")
                
                # Capture
                self.flattened_obs = self._get_info(task_related=False, flattened_obs=False)
                self.flatten_coverage = self.coverage
                
                obs = self.flattened_obs['observation']
                
                # 5. Save to disk as Human Readable Files
                print(f"[Arena] Saving human-readable observation to {save_dir}...")
                os.makedirs(save_dir, exist_ok=True)
                
                try:
                    # Save Images (Convert RGB -> BGR for OpenCV saving)
                    cv2.imwrite(fn_rgb, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_raw_rgb, cv2.cvtColor(obs['raw_rgb'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_depth, obs['depth']) # Depth usually saves fine as png (16bit or 8bit)
                    
                    # Save Masks (Convert Bool -> 0-255 Uint8)
                    cv2.imwrite(fn_mask, (obs['mask'] * 255).astype(np.uint8))
                    cv2.imwrite(fn_r0_mask, (obs['robot0_mask'] * 255).astype(np.uint8))
                    cv2.imwrite(fn_r1_mask, (obs['robot1_mask'] * 255).astype(np.uint8))
                    
                    # Save Metadata JSON
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

    # ... [The rest of the methods: _snap_to_mask, step, get_frames, etc. remain unchanged] ...
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

    def step(self, action):
        if self.measure_time:
            start_time = time.time()

        norm_pixels = np.array(list(action.values())[0]).reshape(-1, 2)
        action_type = list(action.keys())[0]

        points_crop = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.int32)
        if self.snap_to_cloth_mask:
            mask = self.cloth_mask
            kernel = np.ones((3, 3), np.uint8) 
            eroded_mask = cv2.erode(mask, kernel, iterations=2)
            
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

        elif len(points_orig) == 2:
            p0_orig, p1_orig = points_orig[0], points_orig[1]

            if p0_orig[0] < p1_orig[0]:
                p0_orig, p1_orig = p1_orig, p0_orig
            
            final_pick_0 = self._snap_to_mask(p0_orig, full_mask_0)
            final_pick_1 = self._snap_to_mask(p1_orig, full_mask_1)
            points_executed = np.concatenate([final_pick_0, final_pick_1])
           
        if self.measure_time:
            self.process_action_time.append(time.time() - start_time)
            start_time = time.time()

        if action_type == 'norm-pixel-pick-and-place':
            self.pick_and_place_skill.reset()
            self.pick_and_place_skill.step(points_executed.copy())
        elif action_type == 'norm-pixel-pick-and-fling':
            self.pick_and_fling_skill.reset()
            self.pick_and_fling_skill.step(points_executed.copy())
        elif action_type == 'no-operation':
            pass
        
        self.action_step += 1

        if self.measure_time:
            self.primitive_time.append(time.time() - start_time)
            start_time = time.time()

        self.info = self._get_info()

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