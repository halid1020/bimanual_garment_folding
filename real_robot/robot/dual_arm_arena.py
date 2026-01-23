import numpy as np
from typing import Dict, Any, List, Optional
import cv2
import time

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

        self.current_episode = None
        self.frames = []
        self.goal = None
        self.debug = config.get("debug", False)

        self.resolution = (512, 512)
        self.action_step = 0
        #self.horizon = self.config.horizon
        self.evaluate_result = None
        self.last_flattened_step = -1
        self.id = 0
        self.init_coverage = None
        print('Finished init DualArmArena')

    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reset the arena (move both arms home, open grippers, capture initial observation).
        """
        self.eid = episode_config.get("eid", np.random.randint(0, 9999)) if episode_config else 0
        print(f"[Arena] Resetting episode {self.eid}")

        self.flattened_obs = None
        self.get_flattened_obs()
        self.last_info = None

        self.action_step = 0
        
        # Reset robot to safe state
        self.init_coverage = None
        self.task.reset(self)
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
        #crop_mask = raw_cloth_mask[y1:y2, x1:x2]
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
                "action_step": self.action_step
            },
            "robot0_mask": self.resized_workspace_mask_0.astype(np.bool_),
            "robot1_mask": self.resized_workspace_mask_1.astype(np.bool_),
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
            # if info['success']:
            #     info['done'] = True
            
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
        if self.flattened_obs is None:
            print("\n" + "=" * 60)
            print("Please prepare the flattened garment position manually.")
            print("=" * 60)
            input("Press [Enter] to capture the flattened cloth state...")
            self.flattened_obs = self._get_info(task_related=False, flattened_obs=False)
            self.flatten_coverage = self.coverage
            if self.config.debug:
                save_colour(self.flattened_obs['observation']['rgb'], 'flattended_rgb', 'tmp')
            print(f"\n Flattened observation captured successfully.")
            print("=" * 60 + "\n")
        else:
            print("[Arena] Using cached flattened observation.")
        return self.flattened_obs

    def _snap_to_mask(self, point, mask):
        """
        Snaps a (x, y) point to the nearest valid pixel within the provided binary mask.
        Args:
            point: np.array([x, y])
            mask: 2D binary numpy array (h, w)
        Returns:
            np.array([x_snapped, y_snapped])
        """
        point = np.array(point, dtype=int)
        h, w = mask.shape
        x, y = np.clip(point[0], 0, w - 1), np.clip(point[1], 0, h - 1)

        # If point is already in mask, return it
        if mask[y, x] > 0:
            return np.array([x, y])

        # If not, find the nearest non-zero pixel
        # Note: argwhere returns [y, x]
        valid_indices = np.argwhere(mask > 0)
        
        # If mask is empty, we can't snap. Return original (or safe default)
        if len(valid_indices) == 0:
            print("[Warning] Workspace mask is empty! Cannot snap point.")
            return np.array([x, y])

        # Calculate Euclidean distances to all valid points
        # valid_indices is (N, 2) -> [y, x]
        # point is [x, y]
        current_pos_yx = np.array([y, x])
        distances = np.sum((valid_indices - current_pos_yx) ** 2, axis=1)
        
        nearest_idx = np.argmin(distances)
        nearest_yx = valid_indices[nearest_idx]
        
        # Return as [x, y]
        return np.array([nearest_yx[1], nearest_yx[0]])

    def step(self, action):
        """
        Convert normalized bird-eye pixels to original RGB image coordinates,
        snap them to the robot workspace masks, and execute the skill.
        """

        if self.measure_time:
            start_time = time.time()

        norm_pixels = np.array(list(action.values())[0]).reshape(-1, 2)
        action_type = list(action.keys())[0]

        # --- Step 1: Convert normalized → crop → full bird-eye pixels ---
        # Scale from [-1, 1] to [0, crop_size]
        points_crop = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.int32)
        print('points crop before snapping', points_crop)
        if self.snap_to_cloth_mask:
            # Get the current mask
            mask = self.cloth_mask
            
            # --- EROSION STEP START ---
            # Erode the mask by 2 pixels to force points slightly inward
            kernel = np.ones((3, 3), np.uint8) # 3x3 kernel is standard for small erosions
            eroded_mask = cv2.erode(mask, kernel, iterations=2)
            
            # Safety check: If erosion wiped out the whole mask (tiny cloth), revert to original
            if np.sum(eroded_mask) == 0:
                print("[Warning] Erosion removed entire mask. Using original mask.")
                target_mask = mask
            else:
                target_mask = eroded_mask
            # --- EROSION STEP END ---

            snapped_points = []
            for i, pt in enumerate(points_crop):
                # Use the eroded 'target_mask' for snapping
                if i < 2:
                    snapped_points.append(self._snap_to_mask(pt, target_mask))
                else:
                    snapped_points.append(pt)
            points_crop = np.array(snapped_points)
        print('point scrop after snapping', points_crop)
        # Add crop offset (x1, y1) to get,  coordinates in the Full Raw Image
        # Shape is (N, 2) where N is usually 4 (Pick0, Pick1, Place0, Place1)
        points_orig = points_crop + np.array([self.x1, self.y1])
        
        # Default fallback if logic below is skipped
        points_executed = points_orig.flatten()

        # --- Step 2: Workspace Constraint Logic (Snapping) ---
        full_mask_0, full_mask_1 = self.dual_arm.get_workspace_masks()
        if len(points_orig) == 4:
            # We need to grab the Full Size masks
            
            
            # Extract pairs from the unsorted original points
            # points_orig is [pick_0, pick_1, place_0, place_1] (from network perspective)
            p0_orig, p1_orig = points_orig[0], points_orig[1]
            l0_orig, l1_orig = points_orig[2], points_orig[3]

            # -------------------------------------------------------------
            # MIMIC SKILL SORTING LOGIC TO IDENTIFY ROBOT ASSIGNMENT
            # -------------------------------------------------------------
            # We must sort these pairs exactly how the skill will sort them 
            # to know which mask (Left/UR16e or Right/UR5e) applies to which point.
            
            # Create pairs: (pick, place)
            pair_a = (p0_orig, l0_orig)
            pair_b = (p1_orig, l1_orig)

            # Sort 1: By Pick X-coordinate (Primary Sort)
            # If pair_a is to the LEFT of pair_b, swap.
            # In this setup: Right side (Higher X) -> UR5e (Mask 0), Left side (Lower X) -> UR16e (Mask 1)
            # (Note: Verify your specific robot coordinate system. Usually UR5e is base, UR16e is other side)
            if pair_a[0][0] < pair_b[0][0]:
                pair_a, pair_b = pair_b, pair_a
            
            # # Sort 2: By Place X-coordinate (Secondary Sort, used in skill)
            # if pair_a[1][0] < pair_b[1][0]:
            #     pair_a, pair_b = pair_b, pair_a

            # Now:
            # pair_a is destined for Robot 0 (UR5e / Mask 0)
            # pair_b is destined for Robot 1 (UR16e / Mask 1)
            
            # --- Snap Coordinates ---
            final_pick_0 = self._snap_to_mask(pair_a[0], full_mask_0)
            final_place_0 = self._snap_to_mask(pair_a[1], full_mask_0)
            
            final_pick_1 = self._snap_to_mask(pair_b[0], full_mask_1)
            final_place_1 = self._snap_to_mask(pair_b[1], full_mask_1)

            # Reconstruct the flat array for execution [p0, p1, l0, l1]
            points_executed = np.concatenate([
                final_pick_0, final_pick_1, 
                final_place_0, final_place_1
            ])

            # -------------------------------------------------------------
            # DEBUG VISUALIZATION
            # -------------------------------------------------------------
            if True:
                # 1. Get a fresh image for drawing (or use cached if available)
                debug_img = self.info['observation']['raw_rgb']
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

                # 2. Draw Workspace Boundaries (Contours)
                # Mask 0 (UR5e) in Red
                contours_0, _ = cv2.findContours(full_mask_0.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_img, contours_0, -1, (0, 0, 255), 2) # Blue

                # Mask 1 (UR16e) in Blue
                contours_1, _ = cv2.findContours(full_mask_1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_img, contours_1, -1, (255, 0, 0), 2) # Yellow

                # 3. Draw Points and Arrows
                # Helper to draw
                def draw_correction(img, start, end, label):
                    # Original (Red)
                    cv2.circle(img, tuple(start), 5, (0, 0, 255), -1)
                    # Snapped (Green)
                    cv2.circle(img, tuple(end), 5, (0, 255, 0), -1)
                    # Arrow (White)
                    if np.linalg.norm(start - end) > 1.0: # Only draw arrow if moved
                        cv2.arrowedLine(img, tuple(start), tuple(end), (255, 255, 255), 2, tipLength=0.3)
                    # Label
                    cv2.putText(img, label, tuple(end + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                draw_correction(debug_img, pair_a[0], final_pick_0, "Pick0")
                draw_correction(debug_img, pair_a[1], final_place_0, "Place0")
                draw_correction(debug_img, pair_b[0], final_pick_1, "Pick1")
                draw_correction(debug_img, pair_b[1], final_place_1, "Place1")

                # 4. Save Image
                import os
                save_dir = "debug_snaps"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{save_dir}/step_{self.action_step:03d}_snap.png"
                cv2.imwrite(filename, debug_img)
                print(f"[Debug] Saved snap visualization to {filename}")
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

        # --- Step 3: Execute robot skill ---
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

        # --- Step 4: Capture new state ---
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