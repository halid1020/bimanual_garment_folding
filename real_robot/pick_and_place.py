# skills/pick_and_place_skill.py
import math
import numpy as np
import time
from transform_utils import point_on_table_base

MIN_Z = 0.015
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2
MOVE_ACC = 0.2
HOME_AFTER = True
GRIPPER_OFFSET_UR5e = 0.012       # Gripper length offset
GRIPPER_OFFSET_UR16e = 0
TABLE_HEIGHT = 0.074
FLING_LIFT_DIST = 0.1

class PickAndPlaceSkill:
    """
    Pick-and-Place skill primitive.
    Each step executes a pick-place with given pick/place coordinates.
    """

    def __init__(self, scene):
        self.scene = scene
        self.vertical_rotvec = [math.pi, 0.0, 0.0]
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_dist = 0.08
        self.move_speed = 0.2
        self.move_acc = 0.2
        self.home_after = True

    def reset(self):
        """Reset robot state for the skill."""
        print("[PickAndPlaceSkill] Resetting...")
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def step(self, action):
        """
        Args:
            action: dict with keys:
                pick_0, pick_1, place_0, place_1 -> each np.array([x, y, z])
        """
        pick_0, pick_1, place_0, place_1 = action[:2], action[2:4], action[4:6], action[6:8]
        print("Picked pixels:", pick_0, place_0, pick_1, place_1)
        if pick_0[0] < pick_1[0]:
            pick_0, place_0, pick_1, place_1 = pick_1, place_1, pick_0, place_0
        
        if place_0[0] < place_1[0]:
            pick_0, place_0, pick_1, place_1 = pick_1, place_1, pick_0, place_0

        p_base_pick_0 = point_on_table_base(pick_0[0], pick_0[1], self.scene.intr, self.scene.T_ur5e_cam, TABLE_HEIGHT)
        p_base_place_0 = point_on_table_base(place_0[0], place_0[1], self.scene.intr, self.scene.T_ur5e_cam, TABLE_HEIGHT)
        p_base_pick_1 = point_on_table_base(pick_1[0], pick_1[1], self.scene.intr, self.scene.T_ur16e_cam, TABLE_HEIGHT)
        p_base_place_1 = point_on_table_base(place_1[0], place_1[1], self.scene.intr, self.scene.T_ur16e_cam, TABLE_HEIGHT)

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
        self.scene.both_home(speed=1.0, acc=0.8, blocking=True)

        # move to approach above picks (both arms)
        self.scene.both_movel(
            np.concatenate([approach_pick_0, vertical_rotvec]),
            np.concatenate([approach_pick_1, vertical_rotvec]),
            speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True
        )

        # descend to grasp poses
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

        # Move to approach above place points
        print("Move to approach above place point")
        self.scene.both_movel(
            np.concatenate([approach_place_0, vertical_rotvec]),
            np.concatenate([approach_place_1, vertical_rotvec]),
            speed=0.2, acc=0.1, blocking=True
        )

        # Descend to place
        print("Descending to place point")
        self.scene.both_movel(
            np.concatenate([place_pose_0, vertical_rotvec]),
            np.concatenate([place_pose_1, vertical_rotvec]),
            speed=0.08, acc=0.05, blocking=True
        )

        # Open gripper
        print("Opening gripper...")
        self.scene.both_open_gripper()
        time.sleep(0.5)

        # Lift after release
        print("Lifting after releasing")
        self.scene.both_movel(
            np.concatenate([approach_place_0, vertical_rotvec]),
            np.concatenate([approach_place_1, vertical_rotvec]),
            speed=0.2, acc=0.1, blocking=True
        )

        if HOME_AFTER:
            print("Returning home")
            self.scene.both_home(speed=1.0, acc=0.8, blocking=True)

        print("Pick-and-place sequence finished.")
