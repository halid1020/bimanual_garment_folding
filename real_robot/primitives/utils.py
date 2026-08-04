
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
CONTACT_FORCE_THRESH_UR16e = 5
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

# --- HELPER: which arm gets which pick ---
def sort_pairs_by_table_x(pair_0, pair_1, intr, T_left_cam, table_z, key='pick'):
    """Split two action pairs between the arms -> ``(left_pair, right_pair)``.

    The pick nearer the LEFT base along the base-to-base axis goes to the left arm.

    Assigns by position on the TABLE, not by pixel column. The xArm skills used to
    sort on pixel x ("larger pixel-x -> left arm"), which was only ever true for one
    particular camera roll: with the frame laid out so the left arm appears on the
    left, the same line hands each arm the OTHER one's target, and nothing downstream
    notices -- both grasps are individually reachable and the fling just stretches
    the cloth the wrong way. Which pixel column is "left" is decided by how the
    camera is bolted on and reported by hand-eye calibration, so it is not something
    a primitive can assume. Base-frame x is the same answer under any roll.

    ``pair_*`` are dicts carrying whatever else belongs with that pick (place, angle,
    active flags); whole dicts are swapped, so those stay attached to their pick.
    ``T_left_cam`` must be the LEFT arm's camera transform, so both picks are
    compared in ONE frame -- projecting each through its own arm's transform would
    compare two different x axes, which point opposite ways.
    """
    from real_robot.utils.transform_utils import point_on_table_base

    def base_x(pair):
        px = pair[key]
        return float(point_on_table_base(px[0], px[1], intr, T_left_cam, table_z)[0])

    return (pair_0, pair_1) if base_x(pair_0) <= base_x(pair_1) else (pair_1, pair_0)


# --- HELPER: which of the two equivalent wrist orientations to command ---
def nearest_wrist_branch(target_rotvec, current_rotvec):
    """The equivalent grasp orientation nearest the wrist's CURRENT one.

    A parallel jaw is symmetric under 180 deg about the tool z, so ``R`` and
    ``R * Rz(pi)`` are the SAME physical grasp: same gripper axis, same finger
    line, same contact. They are not the same joint configuration, and the arm
    does not get to pick -- it goes wherever the commanded rotation says.

    ⚠️ This is not a micro-optimisation, it is a joint-limit fix. get_base_fling_poses
    builds its orientations from ``init_rot = Rz(pi)``, which is 180 deg about the
    tool z away from XARM_DOWN_ROTVEC. Transforming the right arm's copy of the
    path by inv(T_left_right) (a 180 deg yaw) cancels that for the right arm and
    leaves it standing for the left, so the left arm was being told to spin its
    wrist half a turn -- under load, mid-swing, at fling speed -- to enter the
    fling. On 2026-08-04 that drove J4 to -363 deg against a -360 deg limit and
    both arms e-stopped with ``servo_id=4, code=23`` (joint angle exceeds limit).
    Both taught homes are ~178 deg about tool z from XARM_DOWN_ROTVEC as well, so
    the approach was already flipping the wrist before the fling flipped it back.

    Snapping to the nearer branch removes every such flip while leaving the tool z
    axis -- where the gripper actually points -- bit-identical. Only the roll about
    it changes, and that is the axis the jaw is symmetric in.
    """
    r_target = R.from_rotvec(np.asarray(target_rotvec, dtype=float))
    r_current = R.from_rotvec(np.asarray(current_rotvec, dtype=float))
    r_flipped = r_target * R.from_euler('z', np.pi)
    if (r_flipped * r_current.inv()).magnitude() < (r_target * r_current.inv()).magnitude():
        return r_flipped.as_rotvec()
    return r_target.as_rotvec()


def snap_path_wrist(poses, current_rotvec):
    """``nearest_wrist_branch`` along a whole trajectory, in order.

    Each waypoint is snapped against the PREVIOUS SNAPPED one, not against the
    pose the arm started in, so a multi-waypoint path stays in one branch instead
    of flipping somewhere in the middle -- which is the failure this exists to
    prevent. ``poses`` is (N, 6) UR-convention; a copy is returned.
    """
    poses = np.array(poses, dtype=float)
    single = poses.ndim == 1
    if single:
        poses = poses.reshape(1, -1)
    ref = np.asarray(current_rotvec, dtype=float)
    for i in range(poses.shape[0]):
        poses[i, 3:6] = nearest_wrist_branch(poses[i, 3:6], ref)
        ref = poses[i, 3:6]
    return poses[0] if single else poses


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
        swing_stroke=0.7, 
        swing_angle=np.pi/4,
        lift_height=0.35,
        place_height=0.05):
    tx_world_action = points_to_action_frame(right_point, left_point)
    tx_world_fling_base = tx_world_action.copy()
    tx_world_fling_base[2,3] = 0
    base_fling = get_base_fling_poses(
        place_y=0,
        stroke=swing_stroke, #swing_stroke,
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
            #print(f"Contact! Delta: {delta:.2f}N")
            break
            
        # Check if robot has actually reached the target (meaning no contact found)
        # We check simply if we are close to the target Z
        curr_pose = robot.get_tcp_pose()
        if abs(curr_pose[2] - target_pose[2]) < 0.005:
            #print("Reached target depth without contact.")
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