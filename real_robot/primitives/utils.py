
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


# --- HELPER: keep a fling path continuous with the grasp being held ---
def retarget_path_to_grasp(path, grasp_rot):
    """Re-express a fling path's orientations as tilts ON TOP OF the grasp held.

    ``get_base_fling_poses`` writes absolute orientations built from a fixed
    ``init_rot`` in the fling frame, which owes nothing to the pose the arm is
    actually holding the cloth in. Because the two bases face each other, that
    constant is 180 deg about the tool z away from the grasp for exactly one of
    the arms -- so one arm was told to spin its wrist half a turn, under load and
    at fling speed, to enter the swing. On 2026-08-04 that drove J4 to -363 deg
    and e-stopped both arms with ``servo_id=4, code=23``.

    Only the DIFFERENCE between waypoints is meaningful in that path: it is the
    wrist pitch that makes the swing a swing (``swing_angle`` about the
    base-to-base axis). So keep the differences and drop the absolute reference:
    each waypoint becomes ``(R_i * R_0^-1) * grasp_rot``, a base-frame tilt
    applied to the orientation the arm is already in. Waypoint 0 then reproduces
    the grasp exactly, so entering the fling costs no wrist motion at all, and
    every later waypoint carries only the tilt that was intended.

    ``path`` is (N, 6) UR-convention in ONE frame, and ``grasp_rot`` (a scipy
    Rotation) must be in that same frame. A copy is returned.

    ⚠️ Do NOT "improve" this by snapping each commanded orientation to whichever
    of the two equivalent wrist branches is nearer the arm's current one. That was
    tried and measured: because the criterion compares whole orientations, a tilt
    past 90 deg makes the flipped branch look nearer and the wrist flips in the
    MIDDLE of the swing. In the cell it took reconfigurations from 2 back up to 35
    and left the right arm's J4 with 9 deg of margin. Fixing the reference here,
    once, is what makes the branch question go away.
    """
    path = np.array(path, dtype=float)
    r0_inv = R.from_rotvec(path[0, 3:6]).inv()
    for i in range(path.shape[0]):
        tilt = R.from_rotvec(path[i, 3:6]) * r0_inv
        path[i, 3:6] = (tilt * grasp_rot).as_rotvec()
    return path


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

# --- HELPER: the xArm fling swing ---------------------------------------------
def xarm_base_fling_poses(stroke=0.25, swing_angle=np.pi / 4, lift_height=0.25,
                          place_height=0.10, windup=0.06, place_y=-0.05):
    """The Lite 6 fling swing, in the action frame (+y = forward = the FRONT).

    ⚠️ SEPARATE FROM ``get_base_fling_poses`` ON PURPOSE. That one is in
    ``transform_utils`` and belongs to the UR cell, which works and must not be
    disturbed; this is the xArm shape, and the two are different motions.

    The order matters and is what the operator asked for after watching it run::

        0  centre, high                     -- where the stretch/shake left it
        1  back a little, high (y=-windup)  -- wind up, JUST enough for momentum
        2  FRONT, high         (y=+stroke)  -- STROKE: the fling itself
        3  FRONT, low          (y=+stroke)  -- touch down at the far end
        4  behind the line, low (y=place_y) -- DRAG BACK to lay the cloth flat

                            z
        --------------------^-------------------------> y (front)
        |                                             |
        |            1      0                    2    |  lift_height
        |                                             |
        |         4                             <-3   |  place_height
        -----------------------------------------------
                    ^ place_y (just behind y=0, the base-to-base line)

    ⚠️ THE SHAPE IS ASYMMETRIC AND THAT IS THE POINT. ``windup`` is a small
    fraction of ``stroke``, not its mirror. The first version used ``-stroke``
    for waypoint 1, which made half the motion backwards -- watching it run, the
    fling reads as "backwards, then forwards" rather than as a fling. The
    backward leg exists only to load the cloth, so it gets the minimum that does
    that and no more. If someone widens it back toward ``stroke``, this is undone.

    The cloth is thrown FORWARD, lands ahead of the grippers, and is then pulled
    back THROUGH the base-to-base line to ``place_y`` just behind it, so it
    settles flat and stretched under the grippers rather than in a heap at the
    far end -- and the release stays clear of the front table edge.

    Returns (5, 6) poses -- xyz + rotvec -- in the action frame. The absolute
    wrist reference here is meaningless; ``retarget_path_to_grasp`` replaces it
    with the grasp the arm is actually holding, and only the DIFFERENCES between
    these orientations survive.
    """
    windup = abs(float(windup))
    pos = np.array([
        [0, 0.0,        lift_height],    # 0: centre, high
        [0, -windup,    lift_height],    # 1: small wind-up, toward the back
        [0, +stroke,    lift_height],    # 2: stroke, toward the front
        [0, +stroke,    place_height],   # 3: touch down at the far end
        [0, place_y,    place_height],   # 4: drag back past the line, laying it down
    ], dtype=float)

    # Pitch about the action frame's x axis -- the base-to-base axis -- so the
    # wrist leads the swing. Negative at the wind-up, positive at the stroke, and
    # held constant through the drag so the gripper does not twist on the cloth.
    # The wind-up pitch is scaled by how short the wind-up now is: a full -45 deg
    # cock over a 6 cm move is a wrist flick, not a swing, and it throws the cloth.
    cock = swing_angle * min(1.0, windup / max(stroke, 1e-6))
    rot = R.from_euler('xyz', [
        [0, 0, 0],
        [-cock, 0, 0],
        [+swing_angle, 0, 0],
        [swing_angle / 8, 0, 0],
        [swing_angle / 8, 0, 0],
    ])
    init_rot = R.from_rotvec([0, np.pi, 0])
    out = np.zeros((5, 6))
    out[:, :3] = pos
    out[:, 3:] = (rot * init_rot).as_rotvec()
    return out


def xarm_points_to_fling_path(right_point, left_point, width=None,
                              swing_stroke=0.25, swing_angle=np.pi / 4,
                              lift_height=0.25, place_height=0.10,
                              windup=0.06, place_y=-0.05):
    """``xarm_base_fling_poses`` placed in the world -> ``(right_path, left_path)``.

    Mirrors ``points_to_fling_path`` exactly, including that ``right_point`` /
    ``left_point`` name the FLING frame rather than our arms, and that the
    forward direction is DERIVED as ``z x (left_point - right_point)``.
    """
    right_point = np.asarray(right_point, dtype=float)
    left_point = np.asarray(left_point, dtype=float)
    tx_world_action = points_to_action_frame(right_point, left_point)
    tx_world_fling_base = tx_world_action.copy()
    tx_world_fling_base[2, 3] = 0
    base_fling = xarm_base_fling_poses(
        stroke=swing_stroke, swing_angle=swing_angle, lift_height=lift_height,
        place_height=place_height, windup=windup, place_y=place_y)
    if width is None:
        width = np.linalg.norm((right_point - left_point)[:2])
    right_path = base_fling.copy()
    right_path[:, 0] = -width / 2
    left_path = base_fling.copy()
    left_path[:, 0] = width / 2
    return (transform_pose(tx_world_fling_base, right_path),
            transform_pose(tx_world_fling_base, left_path))


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