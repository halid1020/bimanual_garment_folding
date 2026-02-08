# dual_arm_real_arena.py
import numpy as np
from typing import Dict, Any, List, Optional
from gym import spaces
import cv2

# from actoris_harena import Arena
# from ..utilities.logger.dummy_logger import DummyLogger
from dual_arm_scene import DualArmScene  # <-- your robot class path
from mask_utils import get_mask_generator, get_mask_v2
from camera_utils import get_birdeye_rgb_and_pose, intrinsics_to_matrix
from save_utils import save_colour
from pick_and_place import PickAndPlaceSkill
from pick_and_fling import PickAndFlingSkill
from save_utils import save_mask, save_colour

class DualArmArena():
    """
    Real Dual-Arm Arena implementation using UR5e and UR16e robots.
    """

    def __init__(self, config):
        # super().__init__(config)
        self.name = "dual_arm_arena"
        self.config = config

        # Robot initialization
        dry_run = config.get("dry_run", False)
        self.dual_arm = DualArmScene(
            ur5e_robot_ip=config.get("ur5e_ip", "192.168.1.10"),
            ur16e_robot_ip=config.get("ur16e_ip", "192.168.1.102"),
            dry_run=dry_run
        )

        self.pick_and_place_skill = PickAndPlaceSkill(self.dual_arm)
        self.pick_and_fling_skill = PickAndFlingSkill(self.dual_arm)

        self.mask_generator = get_mask_generator()

        # Arena parameters
        self.num_train_trials = config.get("num_train_trials", 100)
        self.num_val_trials = config.get("num_val_trials", 10)
        self.num_eval_trials = config.get("num_eval_trials", 30)
        self.action_horizon = config.get("action_horizon", 20)

        self.current_episode = None
        self.frames = []
        self.goal = None
        self.debug=config.get("debug", False)

        self.resolution = (512, 512)
        self.action_step = 0
        self.horizon = self.config.horizon

        print('Finished init DualArmArena')


    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reset the arena (move both arms home, open grippers, capture initial observation).
        """
        self.eid = episode_config.get("eid", np.random.randint(0, 9999)) if episode_config else 0
        print(f"[Arena] Resetting episode {self.eid}")

        self.flattened_obs = None
        self.get_flattened_obs()

        self.action_step = 0
        
        # Reset robot to safe state
        info = self._get_info()
        self.clear_frames()
        return info

    
    def _get_info(self, task_related=True, flattened_obs=True):
        # Return to camera position after manipulation
        self.dual_arm.both_open_gripper()
        self.dual_arm.both_home()
        self.dual_arm.go_camera_pos()

        # -----------------------------
        # Capture post-interaction scene
        # -----------------------------
        raw_rgb, raw_depth = self.dual_arm.take_rgbd()
        print('raw_rgb shape', raw_rgb.shape)
        raw_cloth_mask = get_mask_v2(self.mask_generator, raw_rgb, debug=True)
        workspace_mask_0, workspace_mask_1 = self.dual_arm.get_workspace_masks()

        # Bird’s-eye transformation
        rgb_bird_eye, self.map_x, self.map_y, _, _ = get_birdeye_rgb_and_pose(
            raw_rgb,
            self.dual_arm.get_T_base_cam(),
            intrinsics_to_matrix(self.dual_arm.get_camera_intrinsic()),
            rotate_ccw=False,
        )
        print('rgb_bird_eye.shape', rgb_bird_eye.shape)

        if self.debug:
            save_colour(rgb_bird_eye, 'rgb_bird_eye', './tmp')
            save_colour(raw_rgb, 'raw_rgb', './tmp')

        depth_bird_eye = cv2.remap(raw_depth, self.map_x, self.map_y, interpolation=cv2.INTER_NEAREST)
        mask_bird_eye = cv2.remap(raw_cloth_mask.astype(np.uint8), self.map_x, self.map_y, interpolation=cv2.INTER_NEAREST)
        workspace_mask_0_be = cv2.remap(workspace_mask_0.astype(np.uint8), self.map_x, self.map_y, interpolation=cv2.INTER_NEAREST)
        workspace_mask_1_be = cv2.remap(workspace_mask_1.astype(np.uint8), self.map_x, self.map_y, interpolation=cv2.INTER_NEAREST)

        # -----------------------------
        # Center crop bird’s-eye view around cloth
        # -----------------------------
        ys, xs = np.where(mask_bird_eye > 0)
        h, w = rgb_bird_eye.shape[:2]
        crop_size = min(h, w)  # ✅ Use minimum of width and height

        if len(xs) > 0:
            # Center around the cloth mask
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            x1 = np.clip(cx - crop_size // 2, 0, w - crop_size)
            y1 = np.clip(cy - crop_size // 2, 0, h - crop_size)
        else:
            # Fallback: use image center
            x1 = w // 2 - crop_size // 2
            y1 = h // 2 - crop_size // 2

        x2, y2 = x1 + crop_size, y1 + crop_size

        self.x1 = x1
        self.y1 = y1
        self.crop_size = crop_size

        # Perform crops
        crop_rgb = rgb_bird_eye[y1:y2, x1:x2]
        crop_depth = depth_bird_eye[y1:y2, x1:x2]
        crop_mask = mask_bird_eye[y1:y2, x1:x2]
        crop_workspace_mask_0 = workspace_mask_0_be[y1:y2, x1:x2]
        crop_workspace_mask_1 = workspace_mask_1_be[y1:y2, x1:x2]

        ## Resize images
        self.cropped_resolution = crop_rgb.shape[:2]
        resized_rgb = cv2.resize(crop_rgb, self.resolution)
        resized_depth = cv2.resize(crop_depth, self.resolution)
        resized_mask = cv2.resize(crop_mask, self.resolution)
        resized_workspace_mask_0 = cv2.resize(crop_workspace_mask_0, self.resolution)
        resized_workspace_mask_1 = cv2.resize(crop_workspace_mask_1, self.resolution)
        
        print('max resized mask', np.max(resized_mask), resized_mask.dtype)
    

        # -----------------------------
        # Store and return information
        # -----------------------------
        info = {
            'observation': {
                "rgb": resized_rgb,
                "depth": resized_depth,
                "mask": resized_mask.astype(np.bool),
            },
            "workspace_mask_0": resized_workspace_mask_0.astype(np.bool),
            "workspace_mask_1": resized_workspace_mask_1.astype(np.bool),
            "eid": self.eid,
            
        }

        if flattened_obs:
            info['flattened_obs'] = self.get_flattened_obs()

            for k, v in info['flattened_obs'].items():
                info['observation'][f'flattened-{k}'] = v

        info['done'] = self.action_step >= self.horizon

        if task_related:
            info['evaluation'] = self.evaluate()
            if info['evaluation'].get('normalised_coverage', 0) > 0.9:
                self.last_flattened_step = self.action_step

            
           
            info['observation']['last_flattened_step'] = self.last_flattened_step
            
            info['success'] =  self.success()
            if info['success']:
                info['done'] = True
            
            #print('ev', info['evaluation'])
            if info['evaluation'] != {}:
                #print('self.last_info', self.last_info)
                #print(info['evaluation'])
                info['reward'] = self.task.reward(self.last_info, None, info)
            

            goals = self.task.get_goals()
            if len(goals) > 0:
                goal = goals[0]
                info['goal'] = {}
                for k, v in goal[-1]['observation'].items():
                    info['goal'][k] = v

        
        return info
    
    def get_flattened_obs(self):
        """
        Ask the user to manually set the garment to the flattened position,
        then capture and store that observation as the reference (goal) state.
        """
        if self.flattened_obs is None:
            print("\n" + "=" * 60)
            print("🧺  Please prepare the flattened garment position manually.")
            print("    - Adjust the cloth so it is as flat and spread as possible.")
            print("    - When you are ready, press [Enter] to capture the flattened observation.")
            print("=" * 60)
            input("👉 Press [Enter] to capture the flattened cloth state...")

            # Capture flattened observation (without triggering evaluation)
            self.flattened_obs = self._get_info(task_related=False, flattened_obs=False)

            # Compute flatten coverage ratio (cloth area / total area)
            mask = self.flattened_obs['observation']["mask"]
            save_colour(self.flattened_obs['observation']['rgb'], 'flattended_rgb', 'tmp')
            save_mask(mask, 'flattened_mask', 'tmp')
            cloth_pixels = np.sum(mask)
            total_pixels = mask.size
            self.flatten_coverage = cloth_pixels / total_pixels

            print(f"\n✅ Flattened observation captured successfully.")
            print(f"   Cloth coverage ratio: {self.flatten_coverage:.3f}")
            print("=" * 60 + "\n")

        else:
            print("[Arena] Using cached flattened observation.")

        return self.flattened_obs



    def step(self, action):
        """
        Convert normalized bird-eye pixels to original RGB image coordinates and perform skill action.
        """
        norm_pixels = np.array(list(action.values())[0]).reshape(-1, 2)
        action_type = list(action.keys())[0]

        # --- Step 1: Convert normalized → crop → full bird-eye pixels ---
        standard_pixel_bird_eye = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.int32)

        # add crop offset (x1, y1)
        standard_pixel_bird_eye += np.array([self.x1, self.y1])

        # --- Step 2: Map back to original RGB image coordinates ---
        points_orig = []
        for pixel in standard_pixel_bird_eye:
            x, y = pixel  # pixel = [x, y]
            u = self.map_x[y, x]
            v = self.map_y[y, x]
            points_orig.append((u, v))
        points_orig = np.array(points_orig).flatten()

        # --- Step 3: Execute robot skill ---
        if action_type == 'norm-pixel-pick-and-place':
            self.pick_and_place_skill.reset()
            self.pick_and_place_skill.step(points_orig)
        elif action_type == 'norm-pixel-pick-and-fling':
            self.pick_and_fling_skill.reset()
            self.pick_and_fling_skill.step(points_orig)
        
        self.action_step += 1

        # --- Step 4: Capture new state ---
        return self._get_info()


    def get_frames(self) -> List[np.ndarray]:
        return self.frames

    def success(self):
        return self.task.success(self)
    
    def clear_frames(self):
        self.frames = []

    def get_goal(self) -> Dict[str, Any]:
        """
        Return current episode goal (if defined by task).
        """
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

    # ------------------------------------------------------------------
    # Optional convenience methods
    # ------------------------------------------------------------------

    def visualize_workspace(self):
        """
        Capture and visualize the masked workspace (for debugging or manual control).
        """
        if self.dual_arm.dry_run:
            print("[Dry-run] Visualization skipped.")
            return None

        rgb, _ = self.dual_arm.camera.take_rgbd()
        shaded = self.dual_arm.apply_workspace_mask(rgb)
        return shaded

