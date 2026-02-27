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


MIN_Z = 0.015
APPROACH_DIST = 0.08
LIFT_DIST = 0.08
HOME_AFTER = True

class PickAndPlaceSkill:
    def __init__(self, scene, config=None):
        self.scene = scene
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.move_speed = 1.0  # Increased slightly for smoother momentum
        self.move_acc = 0.5
        self.home_after = True
        self.collision_threshold = 0.15
        
        if config is None: config = {}
        self.trajectory_mode = config.get('trajectory_mode', 'arc') 
        
        # FIX 1: Lower the max height (15cm is usually plenty for cloth)
        self.target_arc_height = config.get('target_arc_height', 0.2) 
        
        # FIX 2: Reduce resolution. 
        # Fewer points = fewer stops. 6 is enough for a simple arc.
        self.arc_resolution = config.get('arc_resolution', 6) 

    def reset(self):
        self.scene.both_open_gripper()
        self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
        time.sleep(0.5)
        
    def _generate_arc_path(self, start_pos, end_pos, rot, num_points=6):
        """
        Generates a parabolic arc. 
        """
        dist = np.linalg.norm(start_pos[:2] - end_pos[:2])
        
        # FIX 3: Dynamic Height Adjustment (Shallower Arc)
        # Use 30% of distance as height, instead of 80%.
        # Example: 30cm move -> 9cm height. 10cm move -> 3cm height.
        adjusted_height = min(self.target_arc_height, dist * 0.4)
        
        
        # Ensure minimum clearance of 2cm so we don't drag
        adjusted_height = max(0.02, adjusted_height)

        print('adjusted height!', adjusted_height)

        path = []
        for i in range(num_points):
            t = (i + 1) / num_points # Start from t > 0 because start_pos is already added separately
            
            # Linear Interpolation
            linear_p = (1 - t) * start_pos + t * end_pos
            
            # Parabolic Z offset: 4 * H * t * (1-t)
            z_offset = 4 * adjusted_height * t * (1 - t)
            
            point = linear_p.copy()
            point[2] += z_offset
            
            pose = np.concatenate([point, rot])
            path.append(pose)
            
        return path

    def step(self, action):
        from real_robot.utils.transform_utils import transform_point

        # 1. Unpack Action
        # Format: [p0, p1, pl0, pl1, ang0, ang1, active0, active1]
        # Indices: 0-1, 2-3, 4-5, 6-7, 8, 9, 10, 11
        if len(action) >= 12:
            pick_0_xy, pick_1_xy = action[0:2], action[2:4]
            place_0_xy, place_1_xy = action[4:6], action[6:8]
            angle_0, angle_1 = action[8], action[9]
            active_0, active_1 = bool(action[10]), bool(action[11])
        else:
            pick_0_xy, pick_1_xy = action[:2], action[2:4]
            place_0_xy, place_1_xy = action[4:6], action[6:8]
            angle_0, angle_1 = 0.0, 0.0
            active_0, active_1 = True, True

        # Group data for sorting
        pair_data_0 = {'pick': pick_0_xy, 'place': place_0_xy, 'angle': angle_0, 'active': active_0}
        pair_data_1 = {'pick': pick_1_xy, 'place': place_1_xy, 'angle': angle_1, 'active': active_1}

        # Sort based on pick X coordinate (Left to Right)
        if pair_data_0['pick'][0] < pair_data_1['pick'][0]:
            pair_data_0, pair_data_1 = pair_data_1, pair_data_0
        
        pick_0, place_0, rot_angle_0, active_0 = pair_data_0.values()
        pick_1, place_1, rot_angle_1, active_1 = pair_data_1.values()

        # Coordinate processing
        p_pick_0, p_place_0, rot_0 = None, None, None
        if active_0:
            p_pick_0, p_place_0, rot_0 = self._process_coords(pick_0, place_0, rot_angle_0, self.scene.T_ur5e_cam, GRIPPER_OFFSET_UR5e, self.scene.ur5e.get_tcp_pose())

        p_pick_1, p_place_1, rot_1 = None, None, None
        if active_1:
            p_pick_1, p_place_1, rot_1 = self._process_coords(pick_1, place_1, rot_angle_1, self.scene.T_ur16e_cam, GRIPPER_OFFSET_UR16e, self.scene.ur16e.get_tcp_pose())

        # Execution Logic with Collision Check
        print(f"[PickAndPlace] Execution Flags -> Robot 0: {active_0}, Robot 1: {active_1}")

        if active_0 and active_1:
            # Transform UR16e points to UR5e frame for check
            p_pick_1_in_0 = transform_point(self.scene.T_ur5e_ur16e, p_pick_1)
            p_place_1_in_0 = transform_point(self.scene.T_ur5e_ur16e, p_place_1)
            
            # Simple linear trajectory approximation for collision check
            # (Start -> High Pick -> Pick -> High Pick -> High Place -> Place)
            traj0 = [p_pick_0 + [0,0,0.08], p_pick_0, p_pick_0 + [0,0,0.08], p_place_0 + [0,0,0.08], p_place_0]
            traj1 = [p_pick_1_in_0 + [0,0,0.08], p_pick_1_in_0, p_pick_1_in_0 + [0,0,0.08], p_place_1_in_0 + [0,0,0.08], p_place_1_in_0]

            conflict, _ = check_trajectories_close(traj0, traj1, threshold=self.collision_threshold)
            
            # Always reset before moving
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
            self.scene.both_open_gripper()

            if conflict:
                print("[PickAndPlace] Collision detected! Executing sequentially.")
                self._execute_single_arm(self.scene.ur5e, p_pick_0, p_place_0, rot_0, CONTACT_FORCE_THRESH_UR5e)
                self.scene.ur5e.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)
                self._execute_single_arm(self.scene.ur16e, p_pick_1, p_place_1, rot_1, CONTACT_FORCE_THRESH_UR16e)
                self.scene.ur16e.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)
            else:
                self._execute_dual_arm(p_pick_0, p_place_0, rot_0, p_pick_1, p_place_1, rot_1)

        elif active_0:
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
            self.scene.ur5e.open_gripper()
            self._execute_single_arm(self.scene.ur5e, p_pick_0, p_place_0, rot_0, CONTACT_FORCE_THRESH_UR5e)
            if self.home_after:
                self.scene.ur5e.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)

        elif active_1:
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
            self.scene.ur16e.open_gripper()
            self._execute_single_arm(self.scene.ur16e, p_pick_1, p_place_1, rot_1, CONTACT_FORCE_THRESH_UR16e)
            if self.home_after:
                self.scene.ur16e.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)
        else:
            print("[PickAndPlace] No active arms. Skipping execution.")

    def _process_coords(self, pick_xy, place_xy, rot_angle, cam_T, gripper_offset, home_pose):
        p_pick = point_on_table_base(pick_xy[0], pick_xy[1], self.scene.intr, cam_T, SURFACE_HEIGHT)
        p_place = point_on_table_base(place_xy[0], place_xy[1], self.scene.intr, cam_T, SURFACE_HEIGHT)
        p_pick[2] = max(self.min_z, float(p_pick[2])) + gripper_offset
        p_place[2] = max(self.min_z, float(p_place[2])) + gripper_offset
        rot_base = home_pose[3:6]
        rot = apply_local_z_rotation(rot_base, rot_angle)
        return p_pick, p_place, rot

    def _execute_single_arm(self, robot, pick_pt, place_pt, rot, force_threshold):
        # 1. Approach & Grasp
        approach_pick = np.concatenate([pick_pt + [0,0,APPROACH_DIST], rot])
        robot.movel(approach_pick, speed=self.move_speed, acceleration=self.move_acc, blocking=True)
        
        grasp_pose = move_until_contact(robot, approach_pick, APPROACH_DIST+0.1, force_threshold=force_threshold)
        robot.close_gripper()
        time.sleep(0.5) # Reduced sleep for flow

        start_pt = grasp_pose[:3]
        
        if self.trajectory_mode == 'arc':
            # Create the full path: [Lift Clearance -> Arc Waypoints -> Destination]
            clearance_pt = start_pt + [0, 0, 0.05] 
            dest_pt = place_pt + [0, 0, 0.02] 
            
            arc_traj = self._generate_arc_path(clearance_pt, dest_pt, rot, num_points=self.arc_resolution)
            
            # Combine into one list. 
            # With the new UR_RTDE fix, the robot will blend through the lift 
            # and all arc points, only stopping at dest_pt.
            full_path = [np.concatenate([clearance_pt, rot])] + arc_traj
            
            # You can tune blend_radius here if 0.02 is too small/large
            robot.movel(full_path, speed=self.move_speed, acceleration=self.move_acc, blocking=True, blend_radius=0.02)
            
        else:
            # Rectangular
            lift_pick = np.concatenate([start_pt + [0, 0, LIFT_DIST], rot])
            approach_place = np.concatenate([place_pt + [0,0,APPROACH_DIST], rot])
            place_pose = np.concatenate([place_pt + [0,0,0.02], rot]) 
            robot.movel([lift_pick, approach_place, place_pose], speed=self.move_speed, acceleration=self.move_acc, blocking=True)

        robot.open_gripper()
        time.sleep(0.3)

        # Retract
        approach_place = np.concatenate([place_pt + [0,0,APPROACH_DIST], rot])
        robot.movel(approach_place, speed=self.move_speed, acceleration=self.move_acc, blocking=True)
        if self.home_after:
             robot.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)

    
    def _execute_dual_arm(self, p0_pick, p0_place, rot0, p1_pick, p1_place, rot1):
        # 1. Approach & Grasp
        # -------------------------------------------------------------------------
        app_pick_0 = np.concatenate([p0_pick + [0,0,APPROACH_DIST], rot0])
        app_pick_1 = np.concatenate([p1_pick + [0,0,APPROACH_DIST], rot1])
        
        self.scene.both_movel(app_pick_0, app_pick_1, speed=self.move_speed, acc=self.move_acc, blocking=True)

        # 2. Parallel Force Detection (Threaded Contact)
        # -------------------------------------------------------------------------
        results = [None, None]
        def run_contact(robot, start_pose, index, force_threshold):
            # Move down until contact
            results[index] = move_until_contact(robot, start_pose, APPROACH_DIST+0.01, force_threshold)

        t0 = ThreadWithResult(target=run_contact, args=(self.scene.ur5e, app_pick_0, 0, CONTACT_FORCE_THRESH_UR5e))
        t1 = ThreadWithResult(target=run_contact, args=(self.scene.ur16e, app_pick_1, 1, CONTACT_FORCE_THRESH_UR16e))
        t0.start(); t1.start()
        t0.join(); t1.join()
        
        # Get the actual contact points (where the gripper is now)
        start_0 = results[0][:3]
        start_1 = results[1][:3]

        self.scene.both_close_gripper()
        time.sleep(0.5)

        # 3. Trajectory Generation & Execution
        # -------------------------------------------------------------------------
        if self.trajectory_mode == 'arc':
            # --- ARC MODE ---
            
            # A. Create clearance points (Small lift relative to current height)
            clear_0 = start_0 + [0, 0, 0.05]
            clear_1 = start_1 + [0, 0, 0.05]
            
            # B. Create destination points
            dest_0 = p0_place + [0, 0, 0.02]
            dest_1 = p1_place + [0, 0, 0.02]
            
            # C. Generate the arc waypoints
            traj_0 = self._generate_arc_path(clear_0, dest_0, rot0, num_points=self.arc_resolution)
            traj_1 = self._generate_arc_path(clear_1, dest_1, rot1, num_points=self.arc_resolution)
            
            # D. Combine into continuous path: [Clearance] + [Arc Points...]
            # The robot will blend through these points without stopping.
            full_path_0 = [np.concatenate([clear_0, rot0])] + traj_0
            full_path_1 = [np.concatenate([clear_1, rot1])] + traj_1
            
            # Execute
            self.scene.both_movel(full_path_0, full_path_1, speed=self.move_speed, acc=self.move_acc, blocking=True)
            
        else:
            # --- RECTANGULAR MODE ---
            
            # Even for rectangular, we bundle points into a list so the robot blends 
            # the corners (Lift -> Move -> Place) instead of stopping 3 times.
            
            lift_0 = np.concatenate([start_0 + [0,0,LIFT_DIST], rot0])
            lift_1 = np.concatenate([start_1 + [0,0,LIFT_DIST], rot1])
            
            app_place_0 = np.concatenate([p0_place + [0,0,APPROACH_DIST], rot0])
            app_place_1 = np.concatenate([p1_place + [0,0,APPROACH_DIST], rot1])
            
            place_0 = np.concatenate([p0_place + [0,0,0.02], rot0])
            place_1 = np.concatenate([p1_place + [0,0,0.02], rot1])

            # Create path lists
            path_0 = [lift_0, app_place_0, place_0]
            path_1 = [lift_1, app_place_1, place_1]

            self.scene.both_movel(path_0, path_1, speed=self.move_speed, acc=self.move_acc, blocking=True)

        # 4. Release & Retract
        # -------------------------------------------------------------------------
        self.scene.both_open_gripper()
        time.sleep(0.3)

        app_place_0 = np.concatenate([p0_place + [0,0,APPROACH_DIST], rot0])
        app_place_1 = np.concatenate([p1_place + [0,0,APPROACH_DIST], rot1])
        self.scene.both_movel(app_place_0, app_place_1, speed=self.move_speed, acc=self.move_acc, blocking=True)
        
        if self.home_after:
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)