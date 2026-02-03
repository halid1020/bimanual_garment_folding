import math
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from real_robot.utils.transform_utils import point_on_table_base, GRIPPER_OFFSET_UR5e, GRIPPER_OFFSET_UR16e, SURFACE_HEIGHT

# --- HELPER FUNCTIONS FOR COLLISION CHECKING ---
def segment_distance(p1, p2, p3, p4):
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D

    if D < 1e-6: 
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < 1e-6 else sN / sD
    tc = 0.0 if abs(tN) < 1e-6 else tN / tD

    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

def check_trajectories_close(traj0_points, traj1_points, threshold=0.3):
    min_dist = float("inf")
    for i in range(len(traj0_points)-1):
        for j in range(len(traj1_points)-1):
            dist = segment_distance(
                traj0_points[i], traj0_points[i+1], 
                traj1_points[j], traj1_points[j+1]
            )
            min_dist = min(min_dist, dist)
            if min_dist < threshold:
                return True, min_dist
    return False, min_dist

def apply_local_z_rotation(axis_angle, angle_rad):
    if abs(angle_rad) < 1e-4:
        return axis_angle
    r_current = R.from_rotvec(axis_angle)
    r_diff = R.from_euler('z', angle_rad, degrees=False)
    r_new = r_current * r_diff
    return r_new.as_rotvec()

# -----------------------------------------------

MIN_Z = 0.015
APPROACH_DIST = 0.08
LIFT_DIST = 0.08
MOVE_SPEED = 1.0
MOVE_ACC = 0.5
HOME_AFTER = True

class PickAndPlaceSkill:
    def __init__(self, scene):
        self.scene = scene
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_dist = 0.08
        self.move_speed = 0.2
        self.move_acc = 0.2
        self.home_after = True
        self.collision_threshold = 0.35 

    def reset(self):
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def step(self, action):
        from real_robot.utils.transform_utils import transform_point

        # 1. Unpack Action
        # Format: [p0, p1, pl0, pl1, ang0, ang1, active0, active1]
        # Indices: 0-1, 2-3, 4-5, 6-7, 8, 9, 10, 11
        if len(action) >= 12:
            pick_0_xy = action[0:2]
            pick_1_xy = action[2:4]
            place_0_xy = action[4:6]
            place_1_xy = action[6:8]
            angle_0 = action[8]
            angle_1 = action[9]
            active_0 = bool(action[10])
            active_1 = bool(action[11])
        else:
            # Fallback
            pick_0_xy = action[:2]
            pick_1_xy = action[2:4]
            place_0_xy = action[4:6]
            place_1_xy = action[6:8]
            angle_0, angle_1 = 0.0, 0.0
            active_0, active_1 = True, True

        # Group data for sorting
        pair_data_0 = {'pick': pick_0_xy, 'place': place_0_xy, 'angle': angle_0, 'active': active_0}
        pair_data_1 = {'pick': pick_1_xy, 'place': place_1_xy, 'angle': angle_1, 'active': active_1}

        # Sort based on pick X coordinate (Left to Right)
        if pair_data_0['pick'][0] < pair_data_1['pick'][0]:
            pair_data_0, pair_data_1 = pair_data_1, pair_data_0
        
        # Unpack sorted
        pick_0 = pair_data_0['pick']
        place_0 = pair_data_0['place']
        rot_angle_0 = pair_data_0['angle']
        active_0 = pair_data_0['active']
        
        pick_1 = pair_data_1['pick']
        place_1 = pair_data_1['place']
        rot_angle_1 = pair_data_1['angle']
        active_1 = pair_data_1['active']

        # 2. Convert to Robot Base Points (Only needed if active)
        def process_coords(pick_xy, place_xy, rot_angle, cam_T, gripper_offset, home_pose):
            if not np.any(pick_xy): return None, None, None # Safety
            
            p_pick = point_on_table_base(pick_xy[0], pick_xy[1], self.scene.intr, cam_T, SURFACE_HEIGHT)
            p_place = point_on_table_base(place_xy[0], place_xy[1], self.scene.intr, cam_T, SURFACE_HEIGHT)
            
            # Clamp Z
            p_pick[2] = max(self.min_z, float(p_pick[2]))
            p_place[2] = max(self.min_z, float(p_place[2]))
            
            # Offsets
            p_pick += np.array([0.0, 0.0, gripper_offset])
            p_place += np.array([0.0, 0.0, gripper_offset])
            
            # Rotation
            rot_base = home_pose[3:6]
            rot = apply_local_z_rotation(rot_base, rot_angle)
            
            return p_pick, p_place, rot

        # Process Robot 0 (UR5e)
        p_pick_0, p_place_0, rot_0 = None, None, None
        if active_0:
            p_pick_0, p_place_0, rot_0 = process_coords(
                pick_0, place_0, rot_angle_0, 
                self.scene.T_ur5e_cam, GRIPPER_OFFSET_UR5e, 
                self.scene.ur5e.get_tcp_pose()
            )

        # Process Robot 1 (UR16e)
        p_pick_1, p_place_1, rot_1 = None, None, None
        if active_1:
            p_pick_1, p_place_1, rot_1 = process_coords(
                pick_1, place_1, rot_angle_1, 
                self.scene.T_ur16e_cam, GRIPPER_OFFSET_UR16e, 
                self.scene.ur16e.get_tcp_pose()
            )

        # 3. Execution Logic
        print(f"[PickAndPlace] Execution Flags -> Robot 0: {active_0}, Robot 1: {active_1}")

        if active_0 and active_1:
            # Dual Execution (Collision Checked)
            # Transform UR16e points to UR5e frame for check
            p_pick_1_in_0 = transform_point(self.scene.T_ur5e_ur16e, p_pick_1)
            p_place_1_in_0 = transform_point(self.scene.T_ur5e_ur16e, p_place_1)
            
            traj0 = [p_pick_0 + [0,0,0.08], p_pick_0, p_pick_0 + [0,0,0.08], p_place_0 + [0,0,0.08], p_place_0]
            traj1 = [p_pick_1_in_0 + [0,0,0.08], p_pick_1_in_0, p_pick_1_in_0 + [0,0,0.08], p_place_1_in_0 + [0,0,0.08], p_place_1_in_0]

            conflict, _ = check_trajectories_close(traj0, traj1, threshold=self.collision_threshold)
            
            self.scene.both_home(speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
            self.scene.both_open_gripper()

            if conflict:
                self._execute_single_arm(self.scene.ur5e, p_pick_0, p_place_0, rot_0)
                self.scene.ur5e.home(speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
                self._execute_single_arm(self.scene.ur16e, p_pick_1, p_place_1, rot_1)
                self.scene.ur16e.home(speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
            else:
                self._execute_dual_arm(p_pick_0, p_place_0, rot_0, p_pick_1, p_place_1, rot_1)

        elif active_0:
            # Single Arm 0 Only
            self.scene.both_home(speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
            self.scene.ur5e.open_gripper()
            self._execute_single_arm(self.scene.ur5e, p_pick_0, p_place_0, rot_0)
            if HOME_AFTER:
                self.scene.ur5e.home(speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

        elif active_1:
            # Single Arm 1 Only
            self.scene.both_home(speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
            self.scene.ur16e.open_gripper()
            self._execute_single_arm(self.scene.ur16e, p_pick_1, p_place_1, rot_1)
            if HOME_AFTER:
                self.scene.ur16e.home(speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        
        else:
            print("[PickAndPlace] No active arms. Skipping execution.")

    def _execute_single_arm(self, robot, pick_pt, place_pt, rot):
        approach_pick = np.concatenate([pick_pt + [0,0,APPROACH_DIST], rot])
        grasp_pick    = np.concatenate([pick_pt, rot])
        lift_pick     = np.concatenate([pick_pt + [0,0,LIFT_DIST], rot])
        approach_place= np.concatenate([place_pt + [0,0,APPROACH_DIST], rot])
        place_pose    = np.concatenate([place_pt + [0,0,0.02], rot]) 

        robot.movel(approach_pick, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        robot.movel(grasp_pick, speed=0.1, acceleration=0.1, blocking=True) 
        
        robot.close_gripper()
        time.sleep(0.8)

        robot.movel(lift_pick, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        robot.movel(approach_place, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        robot.movel(place_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

        robot.open_gripper()
        time.sleep(0.5)

        robot.movel(approach_place, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

    def _execute_dual_arm(self, p0_pick, p0_place, rot0, p1_pick, p1_place, rot1):
        app_pick_0 = np.concatenate([p0_pick + [0,0,APPROACH_DIST], rot0])
        pick_0     = np.concatenate([p0_pick, rot0])
        lift_0     = np.concatenate([p0_pick + [0,0,LIFT_DIST], rot0])
        app_place_0= np.concatenate([p0_place + [0,0,APPROACH_DIST], rot0])
        place_0    = np.concatenate([p0_place + [0,0,0.02], rot0])

        app_pick_1 = np.concatenate([p1_pick + [0,0,APPROACH_DIST], rot1])
        pick_1     = np.concatenate([p1_pick, rot1])
        lift_1     = np.concatenate([p1_pick + [0,0,LIFT_DIST], rot1])
        app_place_1= np.concatenate([p1_place + [0,0,APPROACH_DIST], rot1])
        place_1    = np.concatenate([p1_place + [0,0,0.02], rot1])

        self.scene.both_movel(app_pick_0, app_pick_1, speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
        self.scene.both_movel(pick_0, pick_1, speed=0.1, acc=0.1, blocking=True)
        
        self.scene.both_close_gripper()
        time.sleep(0.8)

        self.scene.both_movel(lift_0, lift_1, speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
        self.scene.both_movel(app_place_0, app_place_1, speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
        self.scene.both_movel(place_0, place_1, speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)

        self.scene.both_open_gripper()
        time.sleep(0.5)

        self.scene.both_movel(app_place_0, app_place_1, speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
        
        if HOME_AFTER:
            self.scene.both_home(speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)