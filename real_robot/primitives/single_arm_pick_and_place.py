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
        self.trajectory_mode = config.get('trajectory_mode', 'rectangular') 
        self.home_after = True
        
        # Collision Avoidance Parameters
        # Lower radius is usually ~0.24m. We add a 4cm buffer for the gripper/wrist
        self.base_safe_radius = config.get('base_safe_radius', 0.25) 

    def reset(self):
        self.scene.open_gripper()
        self.scene.home(speed=self.move_speed, acc=self.move_acc, blocking=True)
        time.sleep(0.5)

    def _get_collision_free_path(self, start_pose, end_pose):
        """
        Checks if the 2D line segment between start and end intersects the base keepout circle.
        Returns a list of intermediate poses to route around it safely.
        """
        A = start_pose[:2]
        B = end_pose[:2]
        D = B - A
        len_sq = np.dot(D, D)

        # If start and end are exactly the same X, Y, no collision route needed
        if len_sq == 0:
            return [end_pose]

        # Calculate 't' (parameter from 0 to 1) for the closest point on the line to the origin (0,0)
        t = -np.dot(A, D) / len_sq

        # If t is outside [0, 1], the closest point to the base is NOT on our line segment
        if t <= 0 or t >= 1:
            return [end_pose]

        # Find the actual closest XY point to the robot base
        closest_pt = A + t * D
        dist_to_base = np.linalg.norm(closest_pt)

        # If the closest point is outside our safe radius, we are safe
        if dist_to_base >= self.base_safe_radius:
            return [end_pose]

        # --- COLLISION DETECTED ---
        print(f"[Collision Avoidance] Path intersects base at dist {dist_to_base:.2f}m. Routing around...")
        
        # Determine the direction to push the waypoint outward
        if dist_to_base > 0.001:
            dir_to_safe = closest_pt / dist_to_base
        else:
            # Edge case: The path goes *exactly* through the center (0,0)
            # Pick a perpendicular direction to route around
            dir_to_safe = np.array([-D[1], D[0]])
            dir_to_safe = dir_to_safe / np.linalg.norm(dir_to_safe)

        # Create the intermediate waypoint on the edge of the safe circle
        waypoint_xy = dir_to_safe * self.base_safe_radius
        
        # Construct the full 6D waypoint pose
        waypoint_pose = np.copy(start_pose)
        waypoint_pose[:2] = waypoint_xy
        waypoint_pose[2] = max(start_pose[2], end_pose[2]) # Keep the height safe
        waypoint_pose[3:6] = end_pose[3:6] # Match target rotation

        # Return the waypoint, followed by the actual destination
        return [waypoint_pose, end_pose]

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
        current_tcp = self.scene.ur5e.get_tcp_pose()
        base_rot_vec = current_tcp[3:6]
        
        target_rot = apply_local_z_rotation(base_rot_vec, rotation_angle)

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
        contact_pose = move_until_contact(
            self.scene.ur5e, 
            approach_pick, 
            self.approach_dist + 0.02, 
            force_threshold=CONTACT_FORCE_THRESH_UR5e
        )
        
        self.scene.close_gripper()
        time.sleep(0.5) 

        # C. Lift and Move to Place
        start_pt = contact_pose[:3]
        
        lift_pt = np.concatenate([start_pt + [0, 0, self.lift_height], target_rot])
        approach_place = np.concatenate([p_place + [0, 0, self.approach_dist], target_rot])
        final_place = np.concatenate([p_place + [0, 0, 0.02], target_rot]) 

        # --- COLLISION CHECK INJECTED HERE ---
        # Get transit path (will be 1 point if safe, 2 points if routing around base)
        transit_path = self._get_collision_free_path(lift_pt, approach_place)

        # Chain moves for smoothness: Lift -> (Optional Waypoint) -> Approach Place -> Place Down
        full_path = [lift_pt] + transit_path + [final_place]
        self.scene.movel(full_path, speed=self.move_speed, acc=self.move_acc)
        # -------------------------------------

        # D. Release and Retract
        self.scene.open_gripper()
        time.sleep(0.2)
        
        retract_pt = np.concatenate([p_place + [0, 0, self.approach_dist], target_rot])
        self.scene.movel(retract_pt, speed=self.move_speed, acc=self.move_acc)

        if self.home_after:
            self.scene.home(speed=MOVE_SPEED, acc=MOVE_ACC)