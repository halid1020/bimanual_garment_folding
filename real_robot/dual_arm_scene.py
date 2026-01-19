import numpy as np
import argparse
import time
import sys
import os

# Helper to ensure we can import from local folders if needed
sys.path.append(os.getcwd())

from thread_utils import ThreadWithResult
from constants import *
from motion_utils import safe_movel, safe_gripper, safe_home
from scene_utils import load_camera_to_base, load_camera_to_gripper

# Ensure matrix_to_pose is in this file!
from real_robot.transform_utils import tcp_pose_to_transform, pixels2base_on_table, transform_pose, matrix_to_pose
from ur import UR_RTDE
from realsense_camera import RealsenseCamera

SURFACE_HEIGHT = 0.03 # relavent to base
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
        
        # Both robots use Eye-to-Hand (Static Camera) calibration
        self.ur16e_eye2hand_calib_file = "calibration/ur16e-calib.yaml"
        self.ur5e_eye2hand_calib_file = "calibration/ur5e-calib.yaml" 

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

        # --------------------------------------------------------
        # Calibration & World Frame Calculation
        # --------------------------------------------------------
        if not self.dry_run:
            # 1. Load Calibration Matrices (Points in Cam -> Points in Base)
            self.T_ur16e_cam = load_camera_to_base(self.ur16e_eye2hand_calib_file)
            self.T_ur5e_cam = load_camera_to_base(self.ur5e_eye2hand_calib_file)
            
            # 2. Calculate T_ur5e_ur16e (UR16e Base expressed in UR5e Base Frame)
            # Logic: Base5 -> Cam -> Base16
            # T_base5_base16 = T_base5_cam * T_cam_base16
            #                = T_base5_cam * inv(T_base16_cam)
            self.T_ur5e_ur16e = self.T_ur5e_cam @ np.linalg.inv(self.T_ur16e_cam)

            # 3. Define World Frame relative to UR5e
            # Origin: Exactly halfway between the two robots
            # Orientation: Aligned with UR5e Base (So Z is UP, same as robots)
            self.T_ur5e_world = np.eye(4)
            self.T_ur5e_world[:3, 3] = self.T_ur5e_ur16e[:3, 3] / 2.0
            self.T_ur5e_world[2][3] = SURFACE_HEIGHT
            
            # 4. Define World Frame relative to UR16e
            # T_ur16e_world = inv(T_ur5e_ur16e) * T_ur5e_world
            self.T_ur16e_world = np.linalg.inv(self.T_ur5e_ur16e) @ self.T_ur5e_world
            
            # 5. Define Camera in World Frame (for checking height)
            # T_world_cam = inv(T_ur5e_world) * T_ur5e_cam
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

        # Use TABLE_HEIGHT 0.0 if using World Frame logic, but keep existing for now
        ur5e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur5e_cam, TABLE_HEIGHT)
        ur16e_base_points = pixels2base_on_table(pixel_points, self.intr, self.T_ur16e_cam, TABLE_HEIGHT)

        ur5e_dist = np.linalg.norm(ur5e_base_points[:, :2], axis=1)
        ur16e_dist = np.linalg.norm(ur16e_base_points[:, :2], axis=1)

        ur5e_mask = (ur5e_dist >= self.ur5e_radius[0]) & (ur5e_dist <= self.ur5e_radius[1])
        ur16e_mask = (ur16e_dist >= self.ur16e_radius[0]) & (ur16e_dist <= self.ur16e_radius[1])

        self.ur5e_mask = ur5e_mask.reshape(H, W)
        self.ur16e_mask = ur16e_mask.reshape(H, W)


# ------------------------------------------------------------------
# MAIN TEST BLOCK
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# MAIN TEST BLOCK
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    
    # CLI for safe testing
    parser = argparse.ArgumentParser(description="Test DualArmScene functionalities")
    parser.add_argument('--dry-run', action='store_true', help="Run without connecting to real robots")
    parser.add_argument('--test-calib', action='store_true', help="Print calibration matrices and world frame info")
    parser.add_argument('--test-transform', action='store_true', help="Test World <-> Base coordinate conversion")
    parser.add_argument('--test-motion', action='store_true', help="Perform a small, safe test motion")
    args = parser.parse_args()

    print(f"--- Initializing DualArmScene (Dry Run: {args.dry_run}) ---")
    
    # Initialize the scene
    scene = DualArmScene(dry_run=args.dry_run)

    # ---------------------------------------------------------
    # Test 1: Calibration & World Frame Logic
    # ---------------------------------------------------------
    if args.test_calib:
        print("\n--- Testing Calibration & Frames ---")
        np.set_printoptions(precision=3, suppress=True)
        
        # Verify the matrices loaded correctly
        print("T_ur5e_cam (Camera to UR5e Base):\n", scene.T_ur5e_cam)
        print("\nT_ur16e_cam (Camera to UR16e Base):\n", scene.T_ur16e_cam)
        print("\nT_ur5e_ur16e (UR16e to UR5e Base):\n", scene.T_ur5e_ur16e)

        
        # Verify the new World Frame
        # CORRECTION: Variable is named T_cam_world in the class
        print("\nT_ur5e_world (World to UR5e Base):\n", scene.T_ur5e_world)
        print("\nT_cam_world (World Pose in Camera Frame):\n", scene.T_cam_world)
        print("\nT_ur16e_world (World to UR16e Base):\n", scene.T_ur16e_world)
        
        if not args.dry_run:
            print("\n>> Midpoint Logic Check:")
            print(f"   Camera Z (Depth): {scene.T_cam_world[2,3]:.3f} meters (Should be 1.32)")

    # ---------------------------------------------------------
    # Test 2: Coordinate Transformations
    # ---------------------------------------------------------
    if args.test_transform:
        print("\n--- Testing Coordinate Transforms ---")
        # Test Point: World Origin (0,0,0)
        test_world_pose = [0.0, 0.0, 0.0, np.pi, 0, 0] 
        
        print(f"Test World Pose: {test_world_pose}")
        
        # Convert World -> Robot Base
        ur5e_pose = scene.world_to_robot_base(test_world_pose, robot_name='ur5e')
        ur16e_pose = scene.world_to_robot_base(test_world_pose, robot_name='ur16e')
        
        print(f" -> UR5e Base Frame Command: {ur5e_pose}")
        print(f" -> UR16e Base Frame Command: {ur16e_pose}")
        
        # Convert back Robot Base -> World (Sanity Check)
        world_from_ur5e = scene.robot_base_to_world(ur5e_pose, robot_name='ur5e')
        print(f" -> Re-calculated World from UR5e: {world_from_ur5e}")
        
        err = np.linalg.norm(np.array(test_world_pose[:3]) - np.array(world_from_ur5e[:3]))
        if err < 1e-4:
            print("✅ Transform sanity check PASSED.")
        else:
            print(f"❌ Transform sanity check FAILED (Error: {err:.6f} m)")

    # ---------------------------------------------------------
    # Test 3: Physical Motion
    # ---------------------------------------------------------
    if args.test_motion:
        print("\n--- Testing Dual Arm Motion ---")
        print("WARNING: This will move the robots. Ensure area is clear.")
        
        if not args.dry_run:
            x = input("Type 'YES' to proceed with real motion: ")
            if x != 'YES':
                print("Aborting motion test.")
                sys.exit()
        
        # Get current poses
        if not args.dry_run:
            current_ur5 = scene.ur5e.get_tcp_pose()
            current_ur16 = scene.ur16e.get_tcp_pose()
        else:
            current_ur5 = [0.5, -0.2, 0.3, 3.14, 0, 0]
            current_ur16 = [-0.5, -0.2, 0.3, 3.14, 0, 0]

        print("Testing Grippers...")
        scene.both_close_gripper()
        time.sleep(0.5)
        scene.both_open_gripper()
        time.sleep(0.5)
        
        # Simple vertical move test (Z-axis is base Z, usually up)
        print(f"Testing Linear Move (Up 2cm)...")
        
        target_ur5 = list(current_ur5)
        target_ur5[2] += 0.02
        
        target_ur16 = list(current_ur16)
        target_ur16[2] += 0.02
        
        # Move UP
        scene.both_movel(target_ur5, target_ur16, speed=0.1, acc=0.1)
        
        # Move DOWN (Back to start)
        print("Returning to start...")
        scene.both_movel(current_ur5, current_ur16, speed=0.1, acc=0.1)
        
        print("Motion test done.")

    if not (args.test_calib or args.test_transform or args.test_motion):
        print("No test flag provided.")
        print("Usage examples:")
        print("  python dual_arm_scene.py --dry-run --test-transform")
        print("  python dual_arm_scene.py --test-calib")