import math
import threading
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from real_robot.utils.transform_utils \
    import point_on_table_base, GRIPPER_OFFSET_UR5e, \
        GRIPPER_OFFSET_UR16e, SURFACE_HEIGHT, MOVE_ACC, MOVE_SPEED
from .utils import *

from real_robot.utils.thread_utils import ThreadWithResult


class SingleArmPickAndPlaceSkill:
    def __init__(self, scene, config=None):
        self.scene = scene
        if config is None: config = {}
        
        # Motion Parameters
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_height = 0.10
        self.move_speed = config.get('speed', 1.0)
        self.move_acc = config.get('acc', 0.5)
        self.trajectory_mode = config.get('trajectory_mode', 'rectangular') # 'rectangular' or 'arc'
        self.home_after = True

    def reset(self):
        self.scene.open_gripper()
        self.scene.home(speed=self.move_speed, acc=self.move_acc, blocking=True)
        time.sleep(0.5)

    def step(self, action):
        """
        Executes a single-arm pick and place.
        Action format: [pick_x, pick_y, place_x, place_y, rotation_radians]
        Coordinates are in pixels.
        """
        if len(action) != 5:
            raise ValueError(f"Expected 5 action values (px, py, lx, ly, rot), got {len(action)}")

        pick_pixel = action[0:2]
        place_pixel = action[2:4]
        rotation_angle = action[4]

        # 1. Convert Pixels to Robot Base Coordinates
        # ----------------------------------------------------------------
        # Get current home rotation to apply the offset angle to
        current_tcp = self.scene.ur5e.get_tcp_pose()
        base_rot_vec = current_tcp[3:6]
        
        # Calculate target rotation vector (Base Rotation + Z-axis offset)
        target_rot = apply_local_z_rotation(base_rot_vec, rotation_angle)

        # Convert pixels to 3D points
        p_pick = point_on_table_base(
            pick_pixel[0], pick_pixel[1], 
            self.scene.intr, 
            self.scene.T_ur5e_cam, 
            SURFACE_HEIGHT
        )
        
        p_place = point_on_table_base(
            place_pixel[0], place_pixel[1], 
            self.scene.intr, 
            self.scene.T_ur5e_cam, 
            SURFACE_HEIGHT
        )

        # Add Gripper Offset to Z (so we grasp the cloth, not the table)
        p_pick[2] = max(self.min_z, float(p_pick[2])) + GRIPPER_OFFSET_UR5e
        p_place[2] = max(self.min_z, float(p_place[2])) + GRIPPER_OFFSET_UR5e

        # 2. Execute Motion Sequence
        # ----------------------------------------------------------------
        print(f"[SingleArmPnP] Pick: {pick_pixel} -> Place: {place_pixel} | Rot: {rotation_angle:.2f}")

        # A. Approach Pick
        approach_pick = np.concatenate([p_pick + [0, 0, self.approach_dist], target_rot])
        self.scene.movel(approach_pick, speed=MOVE_SPEED, acc=MOVE_ACC)
        self.scene.open_gripper()

        # B. Move to Contact (Grasp)
        # Try to move down to p_pick, but stop if force threshold met
        contact_pose = move_until_contact(
            self.scene.ur5e, 
            approach_pick, 
            self.approach_dist + 0.02, 
            force_threshold=CONTACT_FORCE_THRESH_UR5e
        )
        
        self.scene.close_gripper()
        time.sleep(0.5) # Wait for grip

        # C. Lift and Move to Place
        # Use the actual contact Z as the starting height
        start_pt = contact_pose[:3]
        
        lift_pt = np.concatenate([start_pt + [0, 0, self.lift_height], target_rot])
        approach_place = np.concatenate([p_place + [0, 0, self.approach_dist], target_rot])
        final_place = np.concatenate([p_place + [0, 0, 0.02], target_rot]) # Place slightly above table

        # Chain moves for smoothness
        # Lift -> Approach Place -> Place Down
        self.scene.movel([lift_pt, approach_place, final_place], speed=self.move_speed, acc=self.move_acc)

        # D. Release and Retract
        self.scene.open_gripper()
        time.sleep(0.2)
        
        retract_pt = np.concatenate([p_place + [0, 0, self.approach_dist], target_rot])
        self.scene.movel(retract_pt, speed=self.move_speed, acc=self.move_acc)

        if self.home_after:
            self.scene.home(speed=MOVE_SPEED, acc=MOVE_ACC)