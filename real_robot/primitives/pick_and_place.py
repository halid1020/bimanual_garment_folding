import math
import numpy as np
import time
from real_robot.utils.transform_utils import point_on_table_base, GRIPPER_OFFSET_UR5e, GRIPPER_OFFSET_UR16e, SURFACE_HEIGHT

# --- HELPER FUNCTIONS FOR COLLISION CHECKING ---
def segment_distance(p1, p2, p3, p4):
    """
    Calculate the minimum distance between two line segments (p1-p2) and (p3-p4).
    Points are 3D numpy arrays.
    """
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

    if D < 1e-6: # Parallel lines
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
    """
    Check if two trajectories (lists of points) come closer than threshold.
    traj_points: list of [x, y, z] arrays
    threshold: safe distance in meters (0.3m is a safe conservative starting point)
    """
    min_dist = float("inf")
    # Iterate through segments
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
        # Safety threshold for real world (larger than sim usually)
        self.collision_threshold = 0.35 

    def reset(self):
        print("[PickAndPlaceSkill] Resetting...")
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def step(self, action):
        from real_robot.utils.transform_utils import transform_point # Ensure this is imported

        pick_0, pick_1, place_0, place_1 = action[:2], action[2:4], action[4:6], action[6:8]
        
        # Sort logic
        if pick_0[0] < pick_1[0]:
            pick_0, place_0, pick_1, place_1 = pick_1, place_1, pick_0, place_0
        
        # 1. Calculate World Coordinates (In Local Robot Frames)
        # Robot 0 (UR5e) Points -> In UR5e Frame
        p_base_pick_0 = point_on_table_base(pick_0[0], pick_0[1], self.scene.intr, self.scene.T_ur5e_cam, SURFACE_HEIGHT)
        p_base_place_0 = point_on_table_base(place_0[0], place_0[1], self.scene.intr, self.scene.T_ur5e_cam, SURFACE_HEIGHT)
        
        # Robot 1 (UR16e) Points -> In UR16e Frame
        p_base_pick_1 = point_on_table_base(pick_1[0], pick_1[1], self.scene.intr, self.scene.T_ur16e_cam, SURFACE_HEIGHT)
        p_base_place_1 = point_on_table_base(place_1[0], place_1[1], self.scene.intr, self.scene.T_ur16e_cam, SURFACE_HEIGHT)

        # Z-Clamping
        def clamp_z(arr):
            a = np.array(arr, dtype=float).copy()
            if a.shape[0] < 3:
                a = np.pad(a, (0, 3 - a.shape[0]), 'constant', constant_values=0.0)
            a[2] = max(self.min_z, float(a[2]))
            return a

        p_base_pick_0 = clamp_z(p_base_pick_0)
        p_base_place_0 = clamp_z(p_base_place_0)
        p_base_pick_1 = clamp_z(p_base_pick_1)
        p_base_place_1 = clamp_z(p_base_place_1)

        # Apply offsets (Gripper length)
        p_base_pick_0 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR5e])
        p_base_place_0 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR5e])
        p_base_pick_1 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR16e])
        p_base_place_1 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR16e])

        # Get orientation
        pose_0_home = self.scene.ur5e.get_tcp_pose()
        pose_1_home = self.scene.ur16e.get_tcp_pose()
        rot_0 = pose_0_home[3:6]
        rot_1 = pose_1_home[3:6]

        # -------------------------------------------------------------------
        # 2. PREPARE TRAJECTORIES FOR COLLISION CHECKING
        # -------------------------------------------------------------------
        # We must transform UR16e points (Frame 1) into UR5e Frame (Frame 0)
        # T_ur5e_ur16e: Pose of UR16e base in UR5e base frame
        
        # Transform the UR16e POINTS to UR5e frame just for the math check
        p_pick_1_in_0 = transform_point(self.scene.T_ur5e_ur16e, p_base_pick_1)
        p_place_1_in_0 = transform_point(self.scene.T_ur5e_ur16e, p_base_place_1)
        
        # Trajectory 0 (Already in UR5e Frame)
        traj0_check = [
            p_base_pick_0 + np.array([0.0, 0.0, self.approach_dist]),
            p_base_pick_0,
            p_base_pick_0 + np.array([0.0, 0.0, self.lift_dist]),
            p_base_place_0 + np.array([0.0, 0.0, self.approach_dist]),
            p_base_place_0
        ]
        
        # Trajectory 1 (Transformed to UR5e Frame)
        traj1_check = [
            p_pick_1_in_0 + np.array([0.0, 0.0, self.approach_dist]),
            p_pick_1_in_0,
            p_pick_1_in_0 + np.array([0.0, 0.0, self.lift_dist]),
            p_place_1_in_0 + np.array([0.0, 0.0, self.approach_dist]),
            p_place_1_in_0
        ]

        # -------------------------------------------------------------------
        # 3. Check Collision 
        # -------------------------------------------------------------------
        conflict, min_dist = check_trajectories_close(traj0_check, traj1_check, threshold=self.collision_threshold)
        
        print(f"[PickAndPlace] Trajectory Min Dist: {min_dist:.4f}m | Conflict: {conflict}")
        
        # Reset to home before starting
        self.scene.both_home(speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)
        self.scene.both_open_gripper()

        # 4. Execute
        # Note: We pass the ORIGINAL (local frame) points to the execute functions
        # because the robots expect commands in their own base frames.
        if conflict:
            print(">>> COLLISION DETECTED. Executing SEQUENTIAL Pick-and-Place. <<<")
            
            # Robot 0
            self._execute_single_arm(
                self.scene.ur5e, 
                p_base_pick_0, p_base_place_0, rot_0
            )
            self.scene.ur5e.home(speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

            # Robot 1
            self._execute_single_arm(
                self.scene.ur16e, 
                p_base_pick_1, p_base_place_1, rot_1
            )
            self.scene.ur16e.home(speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

        else:
            print(">>> Path Clear. Executing SIMULTANEOUS Pick-and-Place. <<<")
            self._execute_dual_arm(
                p_base_pick_0, p_base_place_0, rot_0,
                p_base_pick_1, p_base_place_1, rot_1
            )

        print("Pick-and-place sequence finished.")

    def _execute_single_arm(self, robot, pick_pt, place_pt, rot):
        """Helper to run PnP on a single robot instance."""
        approach_pick = np.concatenate([pick_pt + [0,0,APPROACH_DIST], rot])
        grasp_pick    = np.concatenate([pick_pt, rot])
        lift_pick     = np.concatenate([pick_pt + [0,0,LIFT_DIST], rot])
        approach_place= np.concatenate([place_pt + [0,0,APPROACH_DIST], rot])
        place_pose    = np.concatenate([place_pt + [0,0,0.02], rot]) # Drop slightly above

        # Move
        robot.movel(approach_pick, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        robot.movel(grasp_pick, speed=0.1, acceleration=0.1, blocking=True) # Slow for grasp
        
        robot.close_gripper()
        time.sleep(0.8)

        robot.movel(lift_pick, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        robot.movel(approach_place, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)
        robot.movel(place_pose, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

        robot.open_gripper()
        time.sleep(0.5)

        robot.movel(approach_place, speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

    def _execute_dual_arm(self, p0_pick, p0_place, rot0, p1_pick, p1_place, rot1):
        """Helper to run existing dual PnP logic."""
        # Compose poses
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

        # Execute Dual
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