import numpy as np
import time
import os
import sys

from real_robot.utils.motion_utils import safe_movel, safe_gripper, safe_home, safe_out_scene
from real_robot.utils.scene_utils import load_camera_to_base
from real_robot.utils.transform_utils import pixels2base_on_table, SURFACE_HEIGHT
from real_robot.robot.ur import UR_RTDE
from real_robot.robot.realsense_camera import RealsenseCamera

class SingleArmScene:
    """
    Control scene for Single-Arm (UR5e only) setup.
    Implements 'both_' methods as wrappers for single arm to maintain 
    compatibility with existing Skills.
    """

    def __init__(self, ur5e_robot_ip="192.168.1.10", dry_run=False):
        self.dry_run = dry_run
        self.gripper_type = 'rg2'   
        
        # UR5e uses Eye-to-Hand (Static Camera) calibration
        self.ur5e_eye2hand_calib_file = f"{os.environ['MP_FOLD_PATH']}/real_robot/calibration/ur5e-calib.yaml" 
        self.ur5e_radius = (0.1, 0.85)

        if not dry_run:
            self.ur5e = UR_RTDE(ur5e_robot_ip, self.gripper_type)
            self.camera = RealsenseCamera(debug=True)
            self.intr = self.camera.get_intrinsic()
            self.ur5e.open_gripper()
            self.ur5e.home()
        else:
            print("[Dry-run] Skipping robot and camera init.")
            self.ur5e = None
            self.camera = None
            self.intr = np.eye(3)

        # --------------------------------------------------------
        # Calibration & World Frame Calculation
        # --------------------------------------------------------
        if not self.dry_run:
            # Load Calibration
            self.T_ur5e_cam = load_camera_to_base(self.ur5e_eye2hand_calib_file)
            
            # For Single Arm, World Frame is usually the UR5e Base Frame
            self.T_ur5e_world = np.eye(4)
            
            # Calculate Masks
            rgb, _ = self.take_rgbd()
            self._calculate_wrkspace_masks(rgb)
        else:
            self.T_ur5e_cam = np.eye(4)
            self.T_ur5e_world = np.eye(4)

        print('Finished init SingleArm Scene')

    # ------------------------------------------------------------------
    # Compatibility Methods (Mimic DualArmScene)
    # ------------------------------------------------------------------

    def movel(self, ur5e_pose, speed, acc, blocking=True, record=False):
        """Ignores ur16e_pose and moves only UR5e."""
        return safe_movel(self.ur5e, ur5e_pose, speed, acc, blocking, self.dry_run)

    def home(self, speed=1.5, acc=1.0, blocking=True):
        return safe_home(self.ur5e, speed, acc, blocking, self.dry_run)
    
    def out_scene(self, speed=1.5, acc=1.0, blocking=True):
        return safe_out_scene(self.ur5e, speed, acc, blocking, self.dry_run)

    def open_gripper(self):
        return safe_gripper(self.ur5e, "open", self.dry_run)

    def close_gripper(self):
        return safe_gripper(self.ur5e, "close", self.dry_run)
    
    # ------------------------------------------------------------------
    # Standard Methods
    # ------------------------------------------------------------------

    def take_rgbd(self):
        return self.camera.take_rgbd()
    
    def restart_camera(self):
        self.camera.restart()
    
    def get_workspace_mask(self):
        # Return a zero-mask for the non-existent robot
        return self.ur5e_mask

    def _calculate_wrkspace_masks(self, rgb):
        H, W, _ = rgb.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        pixel_points = np.stack([uu.ravel(), vv.ravel()], axis=1)

        ur5e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur5e_cam, SURFACE_HEIGHT)
        ur5e_dist = np.linalg.norm(ur5e_base_points[:, :2], axis=1)
        
        ur5e_mask = (ur5e_dist >= self.ur5e_radius[0]) & (ur5e_dist <= self.ur5e_radius[1])
        self.ur5e_mask = ur5e_mask.reshape(H, W)