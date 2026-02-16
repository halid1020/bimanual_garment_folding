
import numpy as np
from scipy.spatial.transform import Rotation as R
from real_robot.utils.transform_utils import (
    points_to_action_frame, get_base_fling_poses, transform_pose,
)
import time

RETRACT_OFFSET = 0.00
DESCEND_STEP = 0.002
DESCEND_SPEED = 0.5
MAX_DESCEND_DIST = 0.1
CONTACT_FORCE_THRESH_UR16e = 10
CONTACT_FORCE_THRESH_UR5e = 5

# --- HELPER FUNCTIONS FOR COLLISION CHECKING ---
def segment_distance(p1, p2, p3, p4):
    """Calculates the closest distance between two line segments (p1-p2) and (p3-p4)."""
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

def check_trajectories_close(traj0_points, traj1_points, threshold=0.1):
    """Checks if two point-sequences (polylines) ever get closer than threshold."""
    min_dist = float("inf")
    for i in range(len(traj0_points)-1):
        for j in range(len(traj1_points)-1):
            dist = segment_distance(
                np.array(traj0_points[i]), np.array(traj0_points[i+1]), 
                np.array(traj1_points[j]), np.array(traj1_points[j+1])
            )
            min_dist = min(min_dist, dist)
            if min_dist < threshold:
                return True, min_dist
    return False, min_dist

# --- HELPER: Apply Rotation ---
def apply_local_z_rotation(axis_angle, angle_rad):
    if abs(angle_rad) < 1e-4:
        return axis_angle
    r_current = R.from_rotvec(axis_angle)
    r_diff = R.from_euler('z', angle_rad, degrees=False)
    r_new = r_current * r_diff
    return r_new.as_rotvec()

# --- HELPER: Fling Path ---
def points_to_fling_path(
        right_point, left_point,
        width=None,   
        swing_stroke=0.6, 
        swing_angle=np.pi/4,
        lift_height=0.35,
        place_height=0.05):
    tx_world_action = points_to_action_frame(right_point, left_point)
    tx_world_fling_base = tx_world_action.copy()
    tx_world_fling_base[2,3] = 0
    base_fling = get_base_fling_poses(
        stroke=swing_stroke,
        swing_angle=swing_angle,
        lift_height=lift_height,
        place_height=place_height)
    if width is None:
        width = np.linalg.norm((right_point - left_point)[:2])
    right_path = base_fling.copy()
    right_path[:,0] = -width/2
    left_path = base_fling.copy()
    left_path[:,0] = width/2
    right_path_w = transform_pose(tx_world_fling_base, right_path)
    left_path_w = transform_pose(tx_world_fling_base, left_path)
    return right_path_w, left_path_w

def move_until_contact(robot, start_pose, max_dist=0.10, force_threshold=CONTACT_FORCE_THRESH_UR16e):
    """
    Moves downwards continuously until contact is detected, then stops immediately.
    """
    # 1. Zero the sensor
    robot.rtde_c.zeroFtSensor()
    time.sleep(0.1)
    
    # 2. Get Baseline Z force
    baseline_z = robot.get_tcp_force()[2]
    
    # 3. Define the full target pose (bottom of the search)
    target_pose = np.array(start_pose)
    target_pose[2] -= max_dist
    
    # 4. Start a NON-BLOCKING move
    # Note: We use a moderate speed for safety
    search_speed = 0.05 
    search_acc = 0.5
    
    # Send the async move command
    # In standard RTDE, passing async=True is usually done by simply NOT waiting.
    # If your wrapper's movel doesn't support async, we use the raw rtde_c.moveL with async=True
    robot.rtde_c.moveL(target_pose.tolist(), search_speed, search_acc, True) 
    
    contact_detected = False
    t_start = time.time()
    
    # 5. Monitor force while moving
    # We loop until we either hit force, or we estimate the move should be done
    # (max_dist / speed) + buffer gives us a timeout
    timeout = (max_dist / search_speed) * 1.5 
    
    while (time.time() - t_start) < timeout:
        current_z = robot.get_tcp_force()[2]
        delta = abs(current_z - baseline_z)
        
        # Check Force
        if delta > force_threshold:
            robot.rtde_c.stopL(10.0) # Stop immediately
            contact_detected = True
            print(f"Contact! Delta: {delta:.2f}N")
            break
            
        # Check if robot has actually reached the target (meaning no contact found)
        # We check simply if we are close to the target Z
        curr_pose = robot.get_tcp_pose()
        if abs(curr_pose[2] - target_pose[2]) < 0.005:
            print("Reached target depth without contact.")
            break
        
        time.sleep(0.002) # 500Hz check
        
    # 6. Handle Post-Contact
    # Wait briefly for the stop to settle
    time.sleep(0.1)
    final_pose = robot.get_tcp_pose()
    
    if contact_detected:
        # Apply the retract offset to not crush the object
        final_pose[2] += RETRACT_OFFSET
        robot.movel(final_pose, speed=0.1, acceleration=0.5, blocking=True)
    else:
        # If we didn't hit contact, we are likely at the bottom. 
        # You might want to just return this pose or retract slightly.
        pass
        
    return final_pose