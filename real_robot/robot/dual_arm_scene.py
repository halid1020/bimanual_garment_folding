import numpy as np
import argparse
import time
import sys
import os

# Helper to ensure we can import from local folders if needed
sys.path.append(os.getcwd())

from real_robot.utils.thread_utils import ThreadWithResult
# from constants import *
from real_robot.utils.motion_utils import safe_movel, safe_gripper, safe_home, safe_out_scene
from real_robot.utils.scene_utils import load_camera_to_base

# Ensure matrix_to_pose is in this file!
from real_robot.utils.transform_utils import tcp_pose_to_transform, pixels2base_on_table, \
    transform_pose, matrix_to_pose, SURFACE_HEIGHT
from real_robot.robot.ur import UR_RTDE
from real_robot.robot.realsense_camera import RealsenseCamera

# World frame is at the middle of two arms and SURFACE_HEIGHT above the base frames
# World frame axis follows the ones of the ur5e.

class DualArmScene:
    """
    Control scene for dual-arm setup with UR5e and UR16e.
    Handles initialization, motion, perception, and user-guided tasks.
    """

    def __init__(self, ur5e_robot_ip="192.168.1.10", ur16e_robot_ip="192.168.1.102", dry_run=False):
        self.dry_run = dry_run
        self.gripper_type = 'rg2'   
        
        # Initialize storage for trajectory recording
        self.last_trajectory = None
        
        # Both robots use Eye-to-Hand (Static Camera) calibration
        self.ur16e_eye2hand_calib_file = f"{os.environ['MP_FOLD_PATH']}/real_robot/calibration/ur16e-calib.yaml"
        self.ur5e_eye2hand_calib_file = f"{os.environ['MP_FOLD_PATH']}/real_robot/calibration/ur5e-calib.yaml" 

        self.ur5e_radius = (0.1, 0.85)
        self.ur16e_radius = (0.25, 0.9)

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

        # --------------------------------------------------------
        # Calibration & World Frame Calculation
        # --------------------------------------------------------
        if not self.dry_run:
            # 1. Load Calibration Matrices (Points in Cam -> Points in Base)
            self.T_ur16e_cam = load_camera_to_base(self.ur16e_eye2hand_calib_file)
            self.T_ur5e_cam = load_camera_to_base(self.ur5e_eye2hand_calib_file)
            
            # 2. Calculate T_ur5e_ur16e (UR16e Base expressed in UR5e Base Frame)
            # Logic: Base5 -> Cam -> Base16
            self.T_ur5e_ur16e = self.T_ur5e_cam @ np.linalg.inv(self.T_ur16e_cam)

            # 3. Define World Frame relative to UR5e
            self.T_ur5e_world = np.eye(4)
            self.T_ur5e_world[:3, 3] = self.T_ur5e_ur16e[:3, 3] / 2.0
            self.T_ur5e_world[2][3] = SURFACE_HEIGHT
            
            # 4. Define World Frame relative to UR16e
            self.T_ur16e_world = np.linalg.inv(self.T_ur5e_ur16e) @ self.T_ur5e_world
            
            # 5. Define Camera in World Frame (for checking height)
            self.T_world_cam = np.linalg.inv(self.T_ur5e_world) @ self.T_ur5e_cam
            self.T_cam_world = np.linalg.inv(self.T_world_cam)
            # Capture for masks
            rgb, _ = self.take_rgbd()
            self._calculate_wrkspace_masks(rgb)
        else:
            # Dry run dummy values
            self.T_ur16e_cam = np.eye(4)
            self.T_ur5e_cam = np.eye(4)
            self.T_ur5e_cam[0, 3] = 0.5 # Fake separation
            self.T_ur16e_cam[0, 3] = -0.5
            self.T_cam_world = np.eye(4)
            self.T_cam_world[2, 3] = 1.32
            self.T_ur5e_world = np.eye(4)
            self.T_ur16e_world = np.eye(4)

        self.T_ur5e_ur16e = self.T_ur5e_cam @ np.linalg.inv(self.T_ur16e_cam)

        print('Finished init RobotArm Scene')

    # ------------------------------------------------------------------
    # Helper: Coordinate Transformations
    # ------------------------------------------------------------------

    def world_to_robot_base(self, world_pose, robot_name='ur5e'):
        """
        Convert a pose [x,y,z, rx,ry,rz] from World Frame to Robot Base Frame.
        """
        T_world = tcp_pose_to_transform(world_pose)
        
        if robot_name == 'ur5e':
            T_base_world = self.T_ur5e_world
        elif robot_name == 'ur16e':
            T_base_world = self.T_ur16e_world
        else:
            raise ValueError("Unknown robot name")
            
        T_base = T_base_world @ T_world
        return matrix_to_pose(T_base)

    def robot_base_to_world(self, robot_pose, robot_name='ur5e'):
        """
        Convert a pose from Robot Base Frame to World Frame.
        """
        T_base = tcp_pose_to_transform(robot_pose)
        
        if robot_name == 'ur5e':
            T_world_base = np.linalg.inv(self.T_ur5e_world)
            T_world = T_world_base @ T_base
        elif robot_name == 'ur16e':
            T_world_base = np.linalg.inv(self.T_ur16e_world)
            T_world = T_world_base @ T_base
            
        return matrix_to_pose(T_world)

    # ------------------------------------------------------------------
    # Multi-threaded motion commands
    # ------------------------------------------------------------------

    def both_movel(self, ur5e_pose, ur16e_pose, speed, acc, blocking=True, record=False):
        """
        Executes moveL on both robots.
        Args:
            record (bool): If True, records TCP positions while threads are alive.
        """
        t1 = ThreadWithResult(target=safe_movel, args=(self.ur5e, ur5e_pose, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_movel, args=(self.ur16e, ur16e_pose, speed, acc, blocking, self.dry_run))
        t1.start(); t2.start()
        
        # Trajectory recording logic
        self.last_trajectory = None
        if blocking and record and not self.dry_run:
            traj = {'ur5e': [], 'ur16e': []}
            # Poll while threads are running
            while t1.is_alive() or t2.is_alive():
                try:
                    p5 = self.ur5e.get_tcp_pose()[:3]
                    p16 = self.ur16e.get_tcp_pose()[:3]
                    traj['ur5e'].append(p5)
                    traj['ur16e'].append(p16)
                except Exception:
                    pass # Ignore read errors during loop
                time.sleep(0.01) # ~100Hz max polling
            self.last_trajectory = traj

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
    
    def both_out_scene(self, speed=1.5, acc=1.0, blocking=True):
        t1 = ThreadWithResult(target=safe_out_scene, args=(self.ur5e, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_out_scene, args=(self.ur16e, speed, acc, blocking, self.dry_run))
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
    
    def both_fling(self, ur5e_path, ur16e_path, speed, acc, record=False):
        """
        Execute fling motion (trajectory).
        Args:
            record (bool): If True, records trajectory during the curve execution.
        """
        # 1. Move to start point (usually linear, short move)
        r = self.both_movel(ur5e_path[0], ur16e_path[0], speed=speed, acc=acc)
        if not r: return False
        
        # 2. Execute the fling curve
        # We pass record=True here to capture the actual fling arc
        r = self.both_movel(ur5e_path[1:], ur16e_path[1:], speed=speed, acc=acc, record=record)
        return r
    
    def get_tcp_distance(self):
        if self.dry_run: return 0.5
        ur5e_tcp_pose = self.ur5e.get_tcp_pose()
        ur16e_tcp_pose = transform_pose(self.T_ur5e_ur16e,
            self.ur16e.get_tcp_pose())
        tcp_distance = np.linalg.norm((ur16e_tcp_pose - ur5e_tcp_pose)[:3])
        return tcp_distance

    def go_camera_pos(self):
        self.both_home()
        self.both_out_scene()
    
    def take_rgbd(self):
        return self.camera.take_rgbd()
    
    def get_workspace_masks(self):
        return self.ur5e_mask, self.ur16e_mask
    
    def get_T_base_cam(self):
        return self.T_ur5e_cam
    
    def get_camera_intrinsic(self):
        return self.intr
    
    def _calculate_wrkspace_masks(self, rgb):
        H, W, _ = rgb.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        pixel_points = np.stack([uu.ravel(), vv.ravel()], axis=1)

        ur5e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur5e_cam, SURFACE_HEIGHT)
        ur16e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur16e_cam, SURFACE_HEIGHT)

        ur5e_dist = np.linalg.norm(ur5e_base_points[:, :2], axis=1)
        ur16e_dist = np.linalg.norm(ur16e_base_points[:, :2], axis=1)

        ur5e_mask = (ur5e_dist >= self.ur5e_radius[0]) & (ur5e_dist <= self.ur5e_radius[1])
        ur16e_mask = (ur16e_dist >= self.ur16e_radius[0]) & (ur16e_dist <= self.ur16e_radius[1])

        self.ur5e_mask = ur5e_mask.reshape(H, W)
        self.ur16e_mask = ur16e_mask.reshape(H, W)

if __name__ == "__main__":
    # Test block remains same as original
    pass