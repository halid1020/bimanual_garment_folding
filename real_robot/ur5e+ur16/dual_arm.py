import argparse
import numpy as np
import time
import math
import sys
import threading

from ur import UR_RTDE
from realsense_camera import RealsenseCamera
from utils import (
    load_camera_to_base,
    click_points_pick_and_place,
    safe_depth_at,
    pixel_to_camera_point,
    transform_point,
    load_camera_to_gripper,
    tcp_pose_to_transform,
)

MIN_Z = 0.015
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2
MOVE_ACC = 0.2
HOME_AFTER = True
TABLE_OFFSET = 0.03        # Gripper length offset


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


class DualArm:

    def __init__(self, ur5e_robot_ip="192.168.1.10", ur16e_robot_ip="192.168.1.102", dry_run=False):
        self.gripper_type = 'rg2'
        self.dry_run = dry_run

        self.ur16e_eye2hand_calib_file = "ur16e_eye_to_hand_calib.yaml"
        self.ur5e_eyeinhand_calib_file = "ur5e_eye_in_hand_calib.yaml"

        if not self.dry_run:
            self.ur5e = UR_RTDE(ur5e_robot_ip, self.gripper_type)
            self.ur16e = UR_RTDE(ur16e_robot_ip, self.gripper_type)
            self.camera = RealsenseCamera()
            self.intr = self.camera.get_intrinsic()
            self.both_home()
        else:
            print("[Dry-run] Skipping robot and camera initialization.")
            self.ur5e = None
            self.ur16e = None
            self.camera = None
            self.intr = np.eye(3)

        self.action_step = 20
        self.step = 0

        if not self.dry_run:
            self.ur16e_T_cam2base = load_camera_to_base(self.ur16e_eye2hand_calib_file)
            self.ur5e.camera_state()
            self.ur5e_T_cam2gripper = load_camera_to_gripper(self.ur5e_eyeinhand_calib_file)
            tcp_pose = self.ur5e.get_tcp_pose()
            T_gripper2base = tcp_pose_to_transform(tcp_pose)
            self.ur5e_T_cam2base = self.ur5e_T_cam2gripper @ T_gripper2base
        else:
            self.ur16e_T_cam2base = np.eye(4)
            self.ur5e_T_cam2base = np.eye(4)

    # --------------------- DRY-RUN SAFE MOTION HELPERS ---------------------

    def safe_movel(self, robot_name, pose, speed, acc, blocking, avoid_singularity=False):
        """Move robot safely or simulate if dry-run."""
        if self.dry_run:
            print(f"[Dry-run] {robot_name}: movel to {pose}, speed={speed}, acc={acc}")
            time.sleep(0.2)
            return True
        robot = self.ur16e if robot_name == "ur16e" else self.ur5e
        return robot.movel(pose, speed, acc, blocking, avoid_singularity=avoid_singularity)

    def safe_open_gripper(self, robot_name):
        if self.dry_run:
            print(f"[Dry-run] {robot_name}: open gripper")
            time.sleep(0.1)
            return True
        robot = self.ur16e if robot_name == "ur16e" else self.ur5e
        return robot.open_gripper()

    def safe_close_gripper(self, robot_name):
        if self.dry_run:
            print(f"[Dry-run] {robot_name}: close gripper")
            time.sleep(0.1)
            return True
        robot = self.ur16e if robot_name == "ur16e" else self.ur5e
        return robot.close_gripper()

    def safe_home(self, robot_name, speed=1.5, acc=1.0, blocking=True):
        if self.dry_run:
            print(f"[Dry-run] {robot_name}: moving home")
            time.sleep(0.2)
            return True
        robot = self.ur16e if robot_name == "ur16e" else self.ur5e
        return robot.home(speed, acc, blocking)

    # --------------------- THREADING WRAPPERS ---------------------

    def both_movel(self, p_left, p_right, speed=0.25, acceleration=1.2, blocking=True, avoid_singularity=False):
        t1 = ThreadWithResult(target=self.safe_movel, args=("ur16e", p_left, speed, acceleration, blocking, avoid_singularity))
        t2 = ThreadWithResult(target=self.safe_movel, args=("ur5e", p_right, speed, acceleration, blocking, avoid_singularity))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            return t1.result and t2.result
        return True

    def both_open_gripper(self, sleep_time=1, blocking=True):
        t1 = ThreadWithResult(target=self.safe_open_gripper, args=("ur16e",))
        t2 = ThreadWithResult(target=self.safe_open_gripper, args=("ur5e",))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            time.sleep(sleep_time)
            return t1.result and t2.result
        return True

    def both_close_gripper(self, sleep_time=1, blocking=True):
        t1 = ThreadWithResult(target=self.safe_close_gripper, args=("ur16e",))
        t2 = ThreadWithResult(target=self.safe_close_gripper, args=("ur5e",))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            time.sleep(sleep_time)
            return t1.result and t2.result
        return True

    def both_home(self, speed=1.5, acceleration=1, blocking=True):
        t1 = ThreadWithResult(target=self.safe_home, args=("ur5e", speed, acceleration, blocking))
        t2 = ThreadWithResult(target=self.safe_home, args=("ur16e", speed, acceleration, blocking))
        t1.start()
        t2.start()
        if blocking:
            t1.join()
            t2.join()
            return t1.result and t2.result
        return True

    # --------------------- PICK & PLACE SEQUENCE ---------------------

    # def execute_pick_and_place(self):
    #     if self.dry_run:
    #         print("[Dry-run] Simulating pick and place cycle...")
    #         time.sleep(1)
    #         return

    #     # ... (your existing pick-and-place logic here, unchanged) ...
    #     print("[INFO] Real pick-and-place executed (not dry-run).")

    
    def execute_pick_and_place(self):
        # update camera state if available (eye-in-hand)
        try:
            self.ur5e.camera_state()
        except Exception:
            pass

        # Acquire images
        try:
            rgb, depth = self.camera.take_rgbd()
        except Exception as e:
            print(f"[Error] Camera capture failed: {e}")
            return

        # Let user click four points (pick0, place0, pick1, place1)
        picked = click_points_pick_and_place("Pick & Place", rgb)
        if not picked or len(picked) != 4:
            print("[Error] click_points_pick_and_place did not return 4 points.")
            return

        pick_0, place_0, pick_1, place_1 = picked
        print("Picked pixels:", pick_0, place_0, pick_1, place_1)

        # Safe depths (handle missing/NaN depth values)
        dz_pick_0 = safe_depth_at(depth, pick_0)
        dz_place_0 = safe_depth_at(depth, place_0)
        dz_pick_1 = safe_depth_at(depth, pick_1)
        dz_place_1 = safe_depth_at(depth, place_1)

        # If safe_depth_at returns None/NaN/0, fall back to a conservative depth
        def validate_depth(d):
            try:
                if d is None or np.isnan(d) or d <= 0:
                    return 0.10  # fallback depth 10 cm
                return float(d)
            except Exception:
                return 0.10

        dz_pick_0 = validate_depth(dz_pick_0)
        dz_place_0 = validate_depth(dz_place_0)
        dz_pick_1 = validate_depth(dz_pick_1)
        dz_place_1 = validate_depth(dz_place_1)

        print("Depths (m):", dz_pick_0, dz_place_0, dz_pick_1, dz_place_1)

        # Convert pixel -> camera points
        p_cam_pick_0 = pixel_to_camera_point(pick_0, dz_pick_0, self.intr)
        p_cam_place_0 = pixel_to_camera_point(place_0, dz_place_0, self.intr)
        p_cam_pick_1 = pixel_to_camera_point(pick_1, dz_pick_1, self.intr)
        # ---------- FIXED: use place_1 pixel for place point ----------
        p_cam_place_1 = pixel_to_camera_point(place_1, dz_place_1, self.intr)

        print("p_cam_pick_0:", p_cam_pick_0, "p_cam_place_0:", p_cam_place_0)
        print("p_cam_pick_1:", p_cam_pick_1, "p_cam_place_1:", p_cam_place_1)

        # Transform to corresponding robot base frames
        try:
            p_base_pick_0 = transform_point(self.ur16e_T_cam2base, p_cam_pick_0)
            p_base_place_0 = transform_point(self.ur16e_T_cam2base, p_cam_place_0)
        except Exception as e:
            print(f"[Warning] transform for ur16e failed: {e}")
            p_base_pick_0 = np.array([0.0, 0.0, MIN_Z])
            p_base_place_0 = np.array([0.0, 0.0, MIN_Z])

        try:
            p_base_pick_1 = transform_point(self.ur5e_T_cam2base, p_cam_pick_1)
            p_base_place_1 = transform_point(self.ur5e_T_cam2base, p_cam_place_1)
        except Exception as e:
            print(f"[Warning] transform for ur5e failed: {e}")
            p_base_pick_1 = np.array([0.0, 0.0, MIN_Z])
            p_base_place_1 = np.array([0.0, 0.0, MIN_Z])

        print(
            "p_base_pick_0 (pre-offset):", p_base_pick_0,
            "p_base_place_0 (pre-offset):", p_base_place_0
        )
        print(
            "p_base_pick_1 (pre-offset):", p_base_pick_1,
            "p_base_place_1 (pre-offset):", p_base_place_1
        )

        # Ensure z is sensible
        def clamp_z(arr):
            a = np.array(arr, dtype=float).copy()
            if a.shape[0] < 3:
                a = np.pad(a, (0, 3 - a.shape[0]), 'constant', constant_values=0.0)
            a[2] = max(MIN_Z, float(a[2]))
            return a

        p_base_pick_0 = clamp_z(p_base_pick_0)
        p_base_place_0 = clamp_z(p_base_place_0)
        p_base_pick_1 = clamp_z(p_base_pick_1)
        p_base_place_1 = clamp_z(p_base_place_1)

        # Apply table offset (gripper length)
        p_base_pick_0 += np.array([0.0, 0.0, TABLE_OFFSET])
        p_base_pick_1 += np.array([0.0, 0.0, TABLE_OFFSET])
        p_base_place_0 += np.array([0.0, 0.0, TABLE_OFFSET])
        p_base_place_1 += np.array([0.0, 0.0, TABLE_OFFSET])

        # Use current orientation for the TCP during motion (keep orientation same)
        # The code expects rotation-vector-like [rx,ry,rz] for the TCP orientation
        vertical_rotvec = [math.pi, 0.0, 0.0]

        # Compose approach/grasp/lift poses
        approach_pick_0 = p_base_pick_0 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_0 = p_base_pick_0
        lift_after_0 = grasp_pick_0 + np.array([0.0, 0.0, LIFT_DIST])
        approach_place_0 = p_base_place_0 + np.array([0.0, 0.0, APPROACH_DIST])
        place_pose_0 = p_base_place_0

        approach_pick_1 = p_base_pick_1 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_1 = p_base_pick_1
        lift_after_1 = grasp_pick_1 + np.array([0.0, 0.0, LIFT_DIST])
        approach_place_1 = p_base_place_1 + np.array([0.0, 0.0, APPROACH_DIST])
        place_pose_1 = p_base_place_1

        # Motion sequence
        print("Moving to home (safe start)")
        self.both_home(speed=1.0, acceleration=0.8, blocking=True)

        # move to approach above picks (both arms)
        self.both_movel(
            np.concatenate([approach_pick_0, vertical_rotvec]),
            np.concatenate([approach_pick_1, vertical_rotvec]),
            speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True
        )

        # descend to grasp poses
        self.both_movel(
            np.concatenate([grasp_pick_0, vertical_rotvec]),
            np.concatenate([grasp_pick_1, vertical_rotvec]),
            speed=0.08, acceleration=0.05, blocking=True
        )

        # Close gripper
        print("Closing gripper...")
        self.both_close_gripper()
        time.sleep(1.0)

        # Lift
        print("Lifting object")
        self.both_movel(
            np.concatenate([lift_after_0, vertical_rotvec]),
            np.concatenate([lift_after_1, vertical_rotvec]),
            speed=0.2, acceleration=0.1, blocking=True
        )

        # Move to approach above place points
        print("Move to approach above place point")
        self.both_movel(
            np.concatenate([approach_place_0, vertical_rotvec]),
            np.concatenate([approach_place_1, vertical_rotvec]),
            speed=0.2, acceleration=0.1, blocking=True
        )

        # Descend to place
        print("Descending to place point")
        self.both_movel(
            np.concatenate([place_pose_0, vertical_rotvec]),
            np.concatenate([place_pose_1, vertical_rotvec]),
            speed=0.08, acceleration=0.05, blocking=True
        )

        # Open gripper
        print("Opening gripper...")
        self.both_open_gripper()
        time.sleep(0.5)

        # Lift after release
        print("Lifting after releasing")
        self.both_movel(
            np.concatenate([approach_place_0, vertical_rotvec]),
            np.concatenate([approach_place_1, vertical_rotvec]),
            speed=0.2, acceleration=0.1, blocking=True
        )

        if HOME_AFTER:
            print("Returning home")
            self.both_home(speed=1.0, acceleration=0.8, blocking=True)

        print("Pick-and-place sequence finished.")

    def run(self, iterations=1):
        self.both_home()

        for i in range(iterations):
            print(f"--- Cycle {i + 1}/{iterations} ---")
            self.execute_pick_and_place()
            self.step += 1

        print("All cycles completed.")


def main():
    parser = argparse.ArgumentParser(description="Dual-arm pick-and-place test harness")
    parser.add_argument("--ur5e-ip", type=str, default="192.168.1.10", help="IP of UR5e robot")
    parser.add_argument("--ur16e-ip", type=str, default="192.168.1.102", help="IP of UR16e robot")
    parser.add_argument("--iterations", type=int, default=10, help="Number of pick-and-place cycles to run")
    parser.add_argument("--dry-run", action="store_true", help="Do not send commands to robots; just print actions")
    args = parser.parse_args()

    print(f"Starting DualArm with dry_run={args.dry_run}")
    dual = DualArm(
            ur5e_robot_ip=args.ur5e_ip,
            ur16e_robot_ip=args.ur16e_ip,
            dry_run=args.dry_run,
        )
    
    # try:
        
    # except Exception as e:
    #     print(f"[Fatal] Failed to create DualArm instance: {e}")
    #     sys.exit(1)

    dual.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
