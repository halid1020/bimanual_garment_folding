
import numpy as np
from scipy.spatial.transform import Rotation as R
from real_robot.utils.transform_utils import (
    points_to_action_frame, get_base_fling_poses, transform_pose,
)

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
