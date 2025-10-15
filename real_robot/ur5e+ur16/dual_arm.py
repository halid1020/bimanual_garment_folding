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
    point_on_table_base,
    points_to_gripper_pose,
    points_to_fling_path,
    transform_pose,
    click_points_pick_and_fling
)

MIN_Z = 0.015
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2
MOVE_ACC = 0.2
HOME_AFTER = True
GRIPPER_OFFSET_UR5e = 0.012       # Gripper length offset
GRIPPER_OFFSET_UR16e = 0
TABLE_HEIGHT = 0.074
FLING_LIFT_DIST = 0.4

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


class DualArm:
    """
        World frame is the base of robot 0.
        robot0 is ur5e and robot1 is ur16e in our case
    """
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
            self.T_ur16e_cam = load_camera_to_base(self.ur16e_eye2hand_calib_file)
            self.ur5e.camera_state()
            self.T_gripper_cam = load_camera_to_gripper(self.ur5e_eyeinhand_calib_file)
            tcp_pose = self.ur5e.get_tcp_pose()
            T_base_gripper = tcp_pose_to_transform(tcp_pose)
            self.T_ur5e_cam =   T_base_gripper @ self.T_gripper_cam
        else:
            self.T_ur16e_cam = np.eye(4)
            self.T_ur5e_cam = np.eye(4)
        
        self.T_ur5e_ur16e = self.T_ur5e_cam @ np.linalg.inv(self.T_ur16e_cam )

    # --------------------- DRY-RUN SAFE MOTION HELPERS ---------------------

    def safe_movel(self, robot_name, pose, speed, acc=1.2, blocking=True, avoid_singularity=False):
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

    def both_movel(self, p_ur5e, p_ur16e, speed=0.25, acceleration=1.2, blocking=True, avoid_singularity=False):
        t1 = ThreadWithResult(target=self.safe_movel, args=("ur5e", p_ur5e, speed, acceleration, blocking, avoid_singularity))
        t2 = ThreadWithResult(target=self.safe_movel, args=("ur16e", p_ur16e, speed, acceleration, blocking, avoid_singularity))
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

    def both_fling(self, ur5e_path, ur16e_path, speed, acceleration):
        r = self.both_movel(ur5e_path[0], ur5e_path[0], speed=speed, acceleration=acceleration)
        if not r: return False
        r = self.both_movel(ur16e_path[1:], ur16e_path[1:], speed=speed, acceleration=acceleration)
        return r

    
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
        if pick_0[0] < pick_1[0]:
            pick_0, place_0, pick_1, place_1 = pick_1, place_1, pick_0, place_0

        p_base_pick_0 = point_on_table_base(pick_0[0], pick_0[1], self.intr, self.T_ur5e_cam, TABLE_HEIGHT)
        p_base_place_0 = point_on_table_base(place_0[0], place_0[1], self.intr, self.T_ur5e_cam, TABLE_HEIGHT)
        p_base_pick_1 = point_on_table_base(pick_1[0], pick_1[1], self.intr, self.T_ur16e_cam, TABLE_HEIGHT)
        p_base_place_1 = point_on_table_base(place_1[0], place_1[1], self.intr, self.T_ur16e_cam, TABLE_HEIGHT)

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
        p_base_pick_0 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR5e])
        p_base_place_0 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR5e])
        p_base_pick_1 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR16e])
        
        p_base_place_1 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR16e])

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

    def execute_pick_and_fling(self):


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
        picked = click_points_pick_and_fling("Pick & fling", rgb)
        if not picked or len(picked) != 2:
            print("[Error] click_points_pick_and_fling did not return 2 points.")
            return

        pick_0, pick_1 = picked
        print("Picked pixels:", pick_0, pick_1)
        if pick_0[0] < pick_1[0]:
            pick_0,  pick_1 = pick_1, pick_0

        
        p_base_pick_0 = point_on_table_base(pick_0[0], pick_0[1], self.intr, self.T_ur5e_cam, TABLE_HEIGHT)
        p_base_pick_1 = point_on_table_base(pick_1[0], pick_1[1], self.intr, self.T_ur16e_cam, TABLE_HEIGHT)
    

        print(
            "p_base_pick_0 (pre-offset):", p_base_pick_0,
        )
        print(
            "p_base_pick_1 (pre-offset):", p_base_pick_1,
        )

        # Ensure z is sensible
        def clamp_z(arr):
            a = np.array(arr, dtype=float).copy()
            if a.shape[0] < 3:
                a = np.pad(a, (0, 3 - a.shape[0]), 'constant', constant_values=0.0)
            a[2] = max(MIN_Z, float(a[2]))
            return a

        p_base_pick_0 = clamp_z(p_base_pick_0)
        p_base_pick_1 = clamp_z(p_base_pick_1)

        # Apply table offset (gripper length)
        p_base_pick_0 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR5e])
        p_base_pick_1 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR16e])

        # Use current orientation for the TCP during motion (keep orientation same)
        # The code expects rotation-vector-like [rx,ry,rz] for the TCP orientation
        vertical_rotvec = [math.pi, 0.0, 0.0]

        # Compose approach/grasp/lift poses
        approach_pick_0 = p_base_pick_0 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_0 = p_base_pick_0
        lift_after_0 = grasp_pick_0 + np.array([0.0, 0.0, FLING_LIFT_DIST])
        

        approach_pick_1 = p_base_pick_1 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_1 = p_base_pick_1
        lift_after_1 = grasp_pick_1 + np.array([0.0, 0.0, FLING_LIFT_DIST])
        

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

        self.dual_arm_stretch_and_fling(p_base_pick_0, self.T_ur5e_ur16e@p_base_pick_1)


    def dual_arm_stretch_and_fling(self, 
            ur5e_pick_point_world, 
            ur16e_pick_point_world,
            stretch_force=20,
            stretch_max_speed=0.15,
            stretch_max_width=0.7, 
            stretch_max_time=5,
            swing_stroke=0.6,
            swing_height=0.45,
            swing_angle=np.pi/4,
            lift_height=0.4,
            place_height=0.05,
            fling_speed=1.3,
            fling_acceleration=5
            ):
        ur5e_pose_world, ur16e_pose_world = points_to_gripper_pose(
            ur5e_pick_point_world, ur16e_pick_point_world, max_width=stretch_max_width)

        # stretch
        r = self.dual_arm_stretch(ur5e_pose_world, ur16e_pose_world, 
            force=stretch_force, 
            max_speed=stretch_max_speed, 
            max_width=stretch_max_width,
            max_time=stretch_max_time)
        if not r: return False
        width = self.get_tcp_distance()
        print('Width: {}'.format(width))
        
        # fling
        ur5e_path, ur16e_path = points_to_fling_path(
            ur5e_point=ur5e_pick_point_world,
            ur16e_point=ur16e_pick_point_world,
            width=width,
            swing_stroke=swing_stroke,
            swing_height=swing_height,
            swing_angle=swing_angle,
            lift_height=lift_height,
            place_height=place_height
        )
        return self.both_fling(ur5e_path, ur16e_path, 
            fling_speed, fling_acceleration)

    def dual_arm_stretch(self, 
        ur5e_pose_world, ur16e_pose_world,
        force=12, init_force=30,
        max_speed=0.15, 
        max_width=0.7, max_time=5,
        speed_threshold=0.001):
        """
        Assuming specific gripper and tcp orientation.
        """
        r = self.both_movel(ur5e_pose_world, ur16e_pose_world, speed=max_speed)
        if not r: return False

        ur5e_tcp_pose = self.ur5e.get_tcp_pose()
        ur16e_tcp_pose = self.ur16e.get_tcp_pose()

        # task_frame = [0, 0, 0, 0, 0, 0]
        selection_vector = [1, 0, 0, 0, 0, 0]
        force_type = 2
        # speed for compliant axis, deviation for non-compliant axis
        limits = [max_speed, 2, 2, 1, 1, 1]
        dt = 1.0/125

        # enable force mode on both robots
        tcp_distance = self.get_tcp_distance()
        with self.ur5e.start_force_mode() as left_force_guard:
            with self.ur16e.start_force_mode() as right_force_guard:
                start_time = time.time()
                prev_time = start_time
                max_acutal_speed = 0
                while (time.time() - start_time) < max_time:
                    f = force
                    if max_acutal_speed < max_speed/20:
                        f = init_force
                    left_wrench = [f, 0, 0, 0, 0, 0]
                    right_wrench = [-f, 0, 0, 0, 0, 0]

                    # apply force
                    r = left_force_guard.apply_force(ur5e_tcp_pose, selection_vector, 
                        left_wrench, force_type, limits)
                    if not r: return False
                    r = right_force_guard.apply_force(ur16e_tcp_pose, selection_vector, 
                        right_wrench, force_type, limits)
                    if not r: return False

                    # check for distance
                    tcp_distance = self.get_tcp_distance()
                    if tcp_distance >= max_width:
                        print('Max distance reached: {}'.format(tcp_distance))
                        break

                    # check for speed
                    l_speed = np.linalg.norm(self.ur5e.get_tcp_speed()[:3])
                    r_speed = np.linalg.norm(self.ur16e.get_tcp_speed()[:3])
                    actual_speed = max(l_speed, r_speed)
                    max_acutal_speed = max(max_acutal_speed, actual_speed)
                    if max_acutal_speed > (max_speed * 0.4):
                        if actual_speed < speed_threshold:
                            print('Action stopped at acutal_speed: {} with  max_acutal_speed: {}'.format(
                                actual_speed, max_acutal_speed))
                            break

                    curr_time = time.time()
                    duration = curr_time - prev_time
                    if duration < dt:
                        time.sleep(dt - duration)
        return r


    def get_tcp_distance(self):
        ur5e_tcp_pose = self.ur5e.get_tcp_pose()
        ur16e_tcp_pose = transform_pose(self.T_ur5e_ur16e,
            self.ur16e.get_tcp_pose())
        tcp_distance = np.linalg.norm((ur16e_tcp_pose - ur5e_tcp_pose)[:3])
        return tcp_distance

    def human_run(self, iterations=1):
    
        """
        Interactive control loop for dual-arm robot.
        Allows user to choose between pick-and-place, pick-and-fling, or exit.
        """
        print("\n=== DualArm Interactive Control ===")
        print("Type one of the following commands each cycle:")
        print("  [1] pick-and-place")
        print("  [2] pick-and-fling")
        print("  [q] quit\n")

        self.both_home()

        while True:
            cmd = input("Enter command [1=pick-place, 2=pick-fling, q=quit]: ").strip().lower()

            if cmd in ("q", "quit", "exit"):
                print("Exiting interactive mode...")
                if HOME_AFTER:
                    print("Returning both arms home...")
                    self.both_home()
                break

            elif cmd in ("1", "pick", "pick-and-place", "p"):
                print("\n--- Executing Pick and Place ---")
                try:
                    self.execute_pick_and_place()
                except Exception as e:
                    print(f"[Error] Pick-and-place failed: {e}")
                print("\n--- Pick and Place Completed ---\n")

            elif cmd in ("2", "fling", "pick-and-fling", "f"):
                print("\n--- Executing Pick and Fling ---")
                try:
                    self.execute_pick_and_fling()
                except Exception as e:
                    print(f"[Error] Pick-and-fling failed: {e}")
                print("\n--- Pick and Fling Completed ---\n")

            else:
                print("Invalid command. Please enter 1, 2, or q.\n")


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

    dual.human_run(iterations=args.iterations)


if __name__ == "__main__":
    main()
