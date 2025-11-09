# skills/pick_and_fling_skill.py
import math
import numpy as np
import time
from transform_utils import point_on_table_base, transform_point, points_to_gripper_pose, \
    points_to_action_frame, get_base_fling_poses, transform_pose

MIN_Z = 0.015
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2
MOVE_ACC = 0.2
HOME_AFTER = TrueMIN_Z = 0.015
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2
MOVE_ACC = 0.2
HOME_AFTER = True
GRIPPER_OFFSET_UR5e = 0.012       # Gripper length offset
GRIPPER_OFFSET_UR16e = 0
TABLE_HEIGHT = 0.074
FLING_LIFT_DIST = 0.1
GRIPPER_OFFSET_UR5e = 0.012       # Gripper length offset
GRIPPER_OFFSET_UR16e = 0
TABLE_HEIGHT = 0.074
FLING_LIFT_DIST = 0.1


def points_to_fling_path(
        left_point, right_point,
        width=None,   
        swing_stroke=0.6, 
        #swing_height=0.45, 
        swing_angle=np.pi/4,
        lift_height=0.4,
        place_height=0.05):
    tx_world_action = points_to_action_frame(left_point, right_point)
    tx_world_fling_base = tx_world_action.copy()
    # height is managed by get_base_fling_poses
    tx_world_fling_base[2,3] = 0
    base_fling = get_base_fling_poses(
        stroke=swing_stroke,
        #lift_height=swing_height,
        swing_angle=swing_angle,
        lift_height=lift_height,
        place_height=place_height)
    if width is None:
        width = np.linalg.norm((left_point - right_point)[:2])
    left_path = base_fling.copy()
    left_path[:,0] = -width/2
    right_path = base_fling.copy()
    right_path[:,0] = width/2
    left_path_w = transform_pose(tx_world_fling_base, left_path)
    right_path_w = transform_pose(tx_world_fling_base, right_path)
    return left_path_w, right_path_w


class PickAndFlingSkill:
    """from transform_utils import point_on_table_base
    Pick-and-Fling skill primitive.
    Step executes a fling motion with pick coordinates.
    """

    def __init__(self, scene):
        self.scene = scene
        self.vertical_rotvec = [math.pi, 0.0, 0.0]
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_dist = 0.1
        self.move_speed = 0.2
        self.move_acc = 0.2

    def reset(self):
        print("[PickAndFlingSkill] Resetting...")
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def step(self, action):
        pick_0, pick_1 = action[:2], action[2:4]
        print("Picked pixels:", pick_0, pick_1)
        if pick_0[0] < pick_1[0]:
            pick_0,  pick_1 = pick_1, pick_0

        
        p_base_pick_0 = point_on_table_base(pick_0[0], pick_0[1], self.scene.intr, self.scene.T_ur5e_cam, TABLE_HEIGHT)
        p_base_pick_1 = point_on_table_base(pick_1[0], pick_1[1], self.scene.intr, self.scene.T_ur16e_cam, TABLE_HEIGHT)
    

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
        lift_after_0 = grasp_pick_0.copy()
        lift_after_0 [2] += FLING_LIFT_DIST
        

        approach_pick_1 = p_base_pick_1 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_1 = p_base_pick_1
        lift_after_1 = grasp_pick_1.copy()
        lift_after_1 [2] += FLING_LIFT_DIST
        

        # Motion sequence
        print("Moving to home (safe start)")
        self.scene.both_open_gripper()
        self.scene.both_home(speed=1.0, acc=0.8, blocking=True)

        # move to approach above picks (both arms)
        self.scene.both_movel(
            np.concatenate([approach_pick_0, vertical_rotvec]),
            np.concatenate([approach_pick_1, vertical_rotvec]),
            speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True
        )

        # descend to grasp poses
        print('grasp_pick_0', grasp_pick_0)
        print('grasp_pick_1', grasp_pick_1)
        self.scene.both_movel(
            np.concatenate([grasp_pick_0, vertical_rotvec]),
            np.concatenate([grasp_pick_1, vertical_rotvec]),
            speed=0.08, acc=0.05, blocking=True
        )

        # Close gripper
        print("Closing gripper...")
        self.scene.both_close_gripper()
        time.sleep(1.0)

        # Lift
        print("Lifting object")
        self.scene.both_movel(
            np.concatenate([lift_after_0, vertical_rotvec]),
            np.concatenate([lift_after_1, vertical_rotvec]),
            speed=0.2, acc=0.1, blocking=True
        )
        print('lift_pick_0', lift_after_0, 'lift_pick_1', \
              lift_after_1, 'lift_pick_1_world', transform_point(self.scene.T_ur5e_ur16e, lift_after_1))
        self.dual_arm_stretch_and_fling(lift_after_0, transform_point(self.scene.T_ur5e_ur16e, lift_after_1))

        self.scene.both_home()


    def dual_arm_stretch_and_fling(self, 
            ur5e_pick_point_world, 
            ur16e_pick_point_world,
            stretch_force=20,
            stretch_max_speed=0.15,
            stretch_max_width=0.7, 
            stretch_max_time=5,
            swing_stroke=0.4,
            swing_height=0.45,
            swing_angle=np.pi/4,
            lift_height=0.3,
            place_height= 0.15, #0.05,
            fling_speed=1.0, #1.3,
            fling_acc=3 #5
            ):
        
        width = self.scene.get_tcp_distance()
        print('Width before stretch: {}'.format(width))
    
        ur5e_pose_world, ur16e_pose_world = points_to_gripper_pose(
            ur5e_pick_point_world, ur16e_pick_point_world, max_width=stretch_max_width)
        
        print('ur5e_pose_world', ur5e_pose_world, 'ur16e_pose_world', ur16e_pose_world)

        # stretch
        r = self.dual_arm_stretch(ur5e_pose_world, ur16e_pose_world, 
            force=stretch_force, 
            max_speed=stretch_max_speed, 
            max_width=stretch_max_width,
            max_time=stretch_max_time)
        if not r: return False
        width = self.scene.get_tcp_distance()
        print('Width: {}'.format(width))
        
        # fling
        ur5e_path_world, ur16e_path_world = points_to_fling_path(
            left_point=ur5e_pick_point_world,
            right_point=ur16e_pick_point_world,
            width=width,
            swing_stroke=swing_stroke,
            #swing_height=swing_height,
            swing_angle=swing_angle,
            lift_height=lift_height,
            place_height=place_height
        )

        print('ur5e_path_world', ur5e_path_world)
        print('ur16e_path_world', ur16e_path_world)

        self.scene.both_fling(ur5e_path_world, transform_pose(np.linalg.inv(self.scene.T_ur5e_ur16e), ur16e_path_world), 
            fling_speed, fling_acc)
        
        self.scene.both_open_gripper()

    def dual_arm_stretch(self, 
        ur5e_pose_world, ur16e_pose_world,
        force=12, init_force=30,
        max_speed=0.15, 
        max_width=0.7, max_time=5,
        speed_threshold=0.001):
        """
        Assuming specific gripper and tcp orientation.
        """
        r = self.scene.both_movel(ur5e_pose_world, \
            transform_pose(np.linalg.inv(self.scene.T_ur5e_ur16e), ur16e_pose_world), \
            speed=max_speed,
            acc=1.2)
        if not r: return False

        ur5e_tcp_pose = self.scene.ur5e.get_tcp_pose()
        ur16e_tcp_pose = self.scene.ur16e.get_tcp_pose()

        # task_frame = [0, 0, 0, 0, 0, 0]
        selection_vector = [1, 0, 0, 0, 0, 0]
        force_type = 2
        # speed for compliant axis, deviation for non-compliant axis
        limits = [max_speed, 2, 2, 1, 1, 1]
        dt = 1.0/125

        # enable force mode on both robots
        tcp_distance = self.scene.get_tcp_distance()
        print('Force Mode')
        with self.scene.ur5e.start_force_mode() as left_force_guard:
            with self.scene.ur16e.start_force_mode() as right_force_guard:
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
                    tcp_distance = self.scene.get_tcp_distance()
                    if tcp_distance >= max_width:
                        print('Max distance reached: {}'.format(tcp_distance))
                        break

                    # check for speed
                    l_speed = np.linalg.norm(self.scene.ur5e.get_tcp_speed()[:3])
                    r_speed = np.linalg.norm(self.scene.ur16e.get_tcp_speed()[:3])
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


