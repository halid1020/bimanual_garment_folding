# dual_arm_real_arena.py
import numpy as np
from typing import Dict, Any, List, Optional
from gym import spaces
import cv2

# from agent_arena import Arena
# from ..utilities.logger.dummy_logger import DummyLogger
from dual_arm_scene import DualArmScene  # <-- your robot class path
from mask_utils import get_mask_generator, get_mask_v2
from camera_utils import get_birdeye_rgb_and_pose, intrinsics_to_matrix
from save_utils import save_colour
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

        # Reset robot to safe state
        self.dual_arm.both_open_gripper()
        self.dual_arm.go_camera_pos()

        # Capture initial scene
        raw_rgb, raw_depth = self.dual_arm.take_rgbd()
        raw_cloth_mask = get_mask_v2(self.mask_generator, raw_rgb)

        # # -----------------------------
        # # Randomly sample pick position on the cloth
        # # -----------------------------
        # ys, xs = np.where(raw_cloth_mask > 0)
        # if len(xs) > 0:
        #     idx = np.random.randint(len(xs))
        #     pick_pos_px = np.array([xs[idx], ys[idx]])
        #     pick_depth = raw_depth[pick_pos_px[1], pick_pos_px[0]]

        #     # Convert pick pixel to 3D world coordinates
        #     pick_pos_3d = self.dual_arm.pixel_to_3d(
        #         pick_pos_px, pick_depth, self.dual_arm.get_camera_intrinsic(), self.dual_arm.get_T_base_cam()
        #     )

        #     # Move to pick position, grasp, lift randomly, and drop
        #     self.dual_arm.move_to(pick_pos_3d + np.array([0, 0, 0.05]))  # approach
        #     self.dual_arm.move_to(pick_pos_3d)
        #     self.dual_arm.close_gripper()

        #     # Random release position and height
        #     release_offset = np.random.uniform([-0.1, -0.1, 0.05], [0.1, 0.1, 0.2])
        #     release_pos_3d = pick_pos_3d + release_offset
        #     self.dual_arm.move_to(release_pos_3d)
        #     self.dual_arm.open_gripper()

        # Return to camera position after manipulation
        self.dual_arm.both_open_gripper()
        self.dual_arm.both_home()
        self.dual_arm.go_camera_pos()

        # -----------------------------
        # Capture post-interaction scene
        # -----------------------------
        raw_rgb, raw_depth = self.dual_arm.take_rgbd()
        raw_cloth_mask = get_mask_v2(self.mask_generator, raw_rgb)
        workspace_mask_0, workspace_mask_1 = self.dual_arm.get_workspace_masks()

        # Bird’s-eye transformation
        rgb_bird_eye, map_x, map_y, _, _ = get_birdeye_rgb_and_pose(
            raw_rgb,
            self.dual_arm.get_T_base_cam(),
            intrinsics_to_matrix(self.dual_arm.get_camera_intrinsic()),
            rotate_ccw=False,
        )

        if self.debug:
            save_colour(rgb_bird_eye, 'rgb_bird_eye', './tmp')
            save_colour(raw_rgb, 'raw_rgb', './tmp')

        depth_bird_eye = cv2.remap(raw_depth, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        mask_bird_eye = cv2.remap(raw_cloth_mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST)
        workspace_mask_0_be = cv2.remap(workspace_mask_0.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST)
        workspace_mask_1_be = cv2.remap(workspace_mask_1.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST)

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

        # Perform crops
        crop_rgb = rgb_bird_eye[y1:y2, x1:x2]
        crop_depth = depth_bird_eye[y1:y2, x1:x2]
        crop_mask = mask_bird_eye[y1:y2, x1:x2]
        crop_workspace_mask_0 = workspace_mask_0_be[y1:y2, x1:x2]
        crop_workspace_mask_1 = workspace_mask_1_be[y1:y2, x1:x2]

        # -----------------------------
        # Store and return information
        # -----------------------------
        self.info = {
            "rgb": crop_rgb,
            "depth": crop_depth,
            "workspace_mask_0": crop_workspace_mask_0,
            "workspace_mask_1": crop_workspace_mask_1,
            "mask": crop_mask,
            "done": False,
            "eid": self.eid,
        }

        self.clear_frames()
        return self.info

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Execute a low-level robot command (or abstract high-level action).
        """
        assert action.shape == (12,), f"Expected action of shape (12,), got {action.shape}"

        ur5e_pose_delta = action[:6]
        ur16e_pose_delta = action[6:]

        # Apply relative move (real robot)
        print(f"[Arena] Executing action for episode {self.eid}")
        current_step = self.current_episode["step"]
        self.dual_arm.both_movel(
            ur5e_pose_delta,
            ur16e_pose_delta,
            speed=0.2,
            acceleration=0.1,
            blocking=True
        )

        # Observe new state
        rgb, depth = None, None
        if not self.dual_arm.dry_run:
            try:
                rgb, depth = self.dual_arm.camera.take_rgbd()
            except Exception as e:
                print(f"[Error] Camera capture failed during step: {e}")

        self.frames.append(rgb)
        self.current_episode["step"] = current_step + 1

        info = {
            "rgb": rgb,
            "depth": depth,
            "step": self.current_episode["step"],
            "done": False
        }
        return info

    def get_frames(self) -> List[np.ndarray]:
        return self.frames

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

