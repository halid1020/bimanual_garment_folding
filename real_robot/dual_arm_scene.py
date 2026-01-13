
import numpy as np

from thread_utils import ThreadWithResult
from constants import *
from motion_utils import safe_movel, safe_gripper, safe_home
from scene_utils import load_camera_to_base, load_camera_to_gripper
from transform_utils import tcp_pose_to_transform, pixels2base_on_table, transform_pose
from ur import UR_RTDE
from realsense_camera import RealsenseCamera


class DualArmScene:
    """
    Control scene for dual-arm setup with UR5e and UR16e.
    Handles initialization, motion, perception, and user-guided tasks.
    """

    def __init__(self, ur5e_robot_ip="192.168.1.10", ur16e_robot_ip="192.168.1.102", dry_run=False):
        self.dry_run = dry_run
        self.gripper_type = 'rg2'   
        self.ur16e_eye2hand_calib_file = "ur16e_eye_to_hand_calib.yaml"
        self.ur5e_eyeinhand_calib_file = "ur5e_eye_in_hand_calib.yaml"
        
        self.ur5e_radius = (0.1, 0.85)
        self.ur16e_radius = (0.1, 0.9)

        if not dry_run:
            self.ur5e = UR_RTDE(ur5e_robot_ip, self.gripper_type)
            self.ur16e = UR_RTDE(ur16e_robot_ip, self.gripper_type)
            self.camera = RealsenseCamera()
            self.intr = self.camera.get_intrinsic()
            self.both_home()
        else:
            print("[Dry-run] Skipping robot and camera init.")
            self.ur5e = self.ur16e = self.camera = None
            self.intr = np.eye(3)

        # Calibration transforms
        
        if not self.dry_run:
            self.T_ur16e_cam = load_camera_to_base(self.ur16e_eye2hand_calib_file)
            self.go_camera_pos()
            rgb, _ = self.take_rgbd()
            self.T_gripper_cam = load_camera_to_gripper(self.ur5e_eyeinhand_calib_file)
            tcp_pose = self.ur5e.get_tcp_pose()
            T_base_gripper = tcp_pose_to_transform(tcp_pose)
            self.T_ur5e_cam =   T_base_gripper @ self.T_gripper_cam
            self._calculate_wrkspace_masks(rgb)
        else:
            self.T_ur16e_cam = np.eye(4)
            self.T_ur5e_cam = np.eye(4)

        self.T_ur5e_ur16e = self.T_ur5e_cam @ np.linalg.inv(self.T_ur16e_cam)

        print('Finished init RobotArm Scene')

    # ------------------------------------------------------------------
    # Multi-threaded motion commands
    # ------------------------------------------------------------------

    def both_movel(self, ur5e_pose, ur16e_pose, speed, acc, blocking=True):
        t1 = ThreadWithResult(target=safe_movel, args=(self.ur5e, ur5e_pose, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_movel, args=(self.ur16e, ur16e_pose, speed, acc, blocking, self.dry_run))
        t1.start(); t2.start()
        if blocking:
            t1.join(); t2.join()
            return t1.result and t2.result
        return True

    def both_home(self, speed=1.5, acc=1.0, blocking=True):
        t1 = ThreadWithResult(target=safe_home, args=(self.ur5e, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_home, args=(self.ur16e, speed, acc, blocking, self.dry_run))
        t1.start(); t2.start()
        if blocking:
            t1.join(); t2.join()
            return t1.result and t2.result
        return True

    def both_open_gripper(self):
        t1 = ThreadWithResult(target=safe_gripper, args=(self.ur5e, "open", self.dry_run))
        t2 = ThreadWithResult(target=safe_gripper, args=(self.ur16e, "open", self.dry_run))
        t1.start(); t2.start()
        t1.join(); t2.join()
        return True

    def both_close_gripper(self):
        t1 = ThreadWithResult(target=safe_gripper, args=(self.ur5e, "close", self.dry_run))
        t2 = ThreadWithResult(target=safe_gripper, args=(self.ur16e, "close", self.dry_run))
        t1.start(); t2.start()
        t1.join(); t2.join()
        return True
    
    def both_fling(self, ur5e_path, ur16e_path, speed, acc):
        r = self.both_movel(ur5e_path[0], ur16e_path[0], speed=speed, acc=acc)
        if not r: return False
        r = self.both_movel(ur5e_path[1:], ur16e_path[1:], speed=speed, acc=acc)
        return r
    
    def get_tcp_distance(self):
        ur5e_tcp_pose = self.ur5e.get_tcp_pose()
        ur16e_tcp_pose = transform_pose(self.T_ur5e_ur16e,
            self.ur16e.get_tcp_pose())
        tcp_distance = np.linalg.norm((ur16e_tcp_pose - ur5e_tcp_pose)[:3])
        return tcp_distance

    def go_camera_pos(self):
        self.both_home()
        self.ur5e.camera_state()
    
    def take_rgbd(self):
        return self.camera.take_rgbd()
    
    def get_workspace_masks(self):
        return self.ur5e_mask, self.ur16e_mask
    
    def get_T_base_cam(self):
        return self.T_ur5e_cam
    
    def get_camera_intrinsic(self):
        return self.intr
    
    def _calculate_wrkspace_masks(self, rgb):
        """
        Apply workspace mask to RGB image.
        Pixels outside UR5e and UR16e workspaces are shaded differently (more visible tint).
        """
        H, W, _ = rgb.shape

        # 1. Generate all pixel coordinates (u, v)
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        pixel_points = np.stack([uu.ravel(), vv.ravel()], axis=1)  # (N, 2)

        # 2. Project pixels to base frame for each robot
        ur5e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur5e_cam, TABLE_HEIGHT)
        ur16e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur16e_cam, TABLE_HEIGHT)

        # 3. Compute planar distances
        ur5e_dist = np.linalg.norm(ur5e_base_points[:, :2], axis=1)
        ur16e_dist = np.linalg.norm(ur16e_base_points[:, :2], axis=1)

        # 4. Create boolean masks
        ur5e_mask = (ur5e_dist >= self.ur5e_radius[0]) & (ur5e_dist <= self.ur5e_radius[1])
        ur16e_mask = (ur16e_dist >= self.ur16e_radius[0]) & (ur16e_dist <= self.ur16e_radius[1])

        # 5. Reshape masks
        self.ur5e_mask = ur5e_mask.reshape(H, W)
        self.ur16e_mask = ur16e_mask.reshape(H, W)
