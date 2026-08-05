"""Constants for the UFACTORY xArm Lite 6 real-world setup.

Kept separate from ``transform_utils.py`` (which holds the UR constants) so the UR
code stays byte-for-byte untouched. All lengths are in METRES and all speeds in
METRES/SECOND to match the UR convention used throughout the geometry/primitive
code; the ``XArmLite6`` driver converts to the xArm SDK's mm / mm-s units.

⚠️ Several of these are HARDWARE-SPECIFIC and must be validated/tuned on the real
cell during bring-up (marked below). Defaults are conservative.
"""
import numpy as np

# --- Grasp geometry (no F/T sensor: we descend to a fixed, calibrated height) ---
# Vertical offset added to the table plane so the closed Lite6 gripper fingertips
# sit at the fabric. CALIBRATE on hardware.
XARM_GRIPPER_OFFSET = 0.0869         # m
# Table surface height in each arm's base frame (z). CALIBRATE on hardware.
XARM_TABLE_Z = 0.0                  # m
XARM_MIN_Z = 0.005                  # m, never command below this (safety floor)

# --- Motion (UR-convention units; converted inside the driver) ---
XARM_MOVE_SPEED = 0.20              # m/s, Cartesian
XARM_MOVE_ACC = 0.50               # m/s^2, Cartesian
# Joint moves (home/out_scene). RAD/S and RAD/S^2 -- a different unit from the
# Cartesian speeds above, which is why the scenes' home()/out_scene() default to
# None and let the driver apply these rather than forwarding a m/s value.
# The controller reports a max joint speed of pi rad/s, so 1.0 is about a third of
# it: brisk for an unloaded homing move, still well short of the limit.
XARM_JOINT_SPEED = 1.0             # rad/s, joint moves (home/out_scene)
XARM_JOINT_ACC = 5.0               # rad/s^2
XARM_APPROACH_DIST = 0.08          # m, hover above pick/place before descending
XARM_LIFT_DIST = 0.10              # m, lift after grasp
XARM_BLEND_RADIUS = 0.02           # m, path-blend radius for multi-waypoint moves

# --- Workspace (Lite 6 reach ~0.44 m; bases ~0.80 m apart, top-down camera) ---
# Annular reach test in the base XY plane: keep out of a small base cylinder and
# inside the usable reach. TUNE on hardware (test_xarm_teach.py --reach).
XARM_WORKSPACE_RADIUS = (0.12, 0.40)   # (min, max) m from the arm base Z-axis
# Cell: the two arms are mounted on the two LONG (120 cm) edges of the 80 x 120 cm
# table, facing each other across the short axis, so the right base is rotated
# 180 deg about z w.r.t. the left. Measured base-to-base centre distance is 0.66 m
# (i.e. each base is inset ~0.07 m from its long edge), and the arm line sits
# 0.52 m from the FRONT 80 cm edge -- so the table is asymmetric about the arms:
# 0.52 m of table in front of them (+y) and 0.68 m behind (-y).
# At the conservative 0.40 m usable radius the two workspaces now genuinely
# overlap, by 2*0.40 - 0.66 = 0.14 m about the midline, which is what dual-arm
# grasps on one garment need. The true usable radius is measured by
# test_xarm_teach.py --reach.
XARM_BASE_SEPARATION = 0.66            # m between the two arm bases (measured)
# ⚠️ ASSUMED, NOT MEASURED. The yaw of the right base relative to the left decides
# where every wall and every right-arm target lands, so it must be measured before
# the walls mean anything. Note that a parked right arm reading its TCP at
# x = -0.239 m does NOT settle this either way: a folded pose (J4 -119.8 deg,
# J5 -83.1 deg) puts the TCP behind its own base at any yaw. Measure it with
#     python real_robot/test/test_xarm_teach.py --arm both --separation
# which fits the yaw from several shared marks, then set the value here and flip
# XARM_GEOMETRY_VERIFIED.
XARM_BASE_YAW = np.pi                  # rad, right base yaw relative to the left
# Virtual walls stay OFF until this is True: a boundary derived from an unverified
# frame does not protect anything, it just refuses legitimate motion.
XARM_GEOMETRY_VERIFIED = False
XARM_TABLE_SIZE = (0.80, 1.20)         # m, (across the arms, along the arm line)
# Where the arm line sits on the table, in the LEFT base frame. FRONT IS +y.
# Centred ACROSS the 80 cm width (each base inset 0.07 m from its 120 cm edge),
# but NOT along the length: the arm line is 0.52 m from the front 80 cm edge and
# 0.68 m from the back one. The walls are therefore asymmetric in y, and the two
# arms end up with mirrored (not identical) y limits.
#
# ⚠️ THE SIGN IS MEASURED, and it was misread once. Hand-eye calibration settles
# it: column 1 of xarm-left-calib.yaml is camera +y (image DOWN) -> base -y, so
# the top of the frame is base +y, and the mounted camera shows the front table
# edge at the top with the left arm on the left. Front is +y.
#
# It was set to -y on 2026-08-04 on the strength of a fling that threw "toward the
# back", which looked decisive and was not: that run E-STOPPED mid-motion on the
# J4 fault, and the FIRST y move in get_base_fling_poses is the wind-up to
# y = -stroke. What the operator watched go backwards was the wind-up, not the
# throw. A direction read off a motion that never finished is not a measurement.
#
# The consequence of the wrong sign was a plan to MIRROR the camera frame in
# software, on the argument that a down-looking camera cannot show both "left arm
# on the left" and "front at the top" (right-handed image frame: u -> +x forces
# v -> -y). True, but only when front is at -y. With front at +y the physical
# mounting gives both, and nothing flips the image anywhere.
XARM_BASE_INSET = (XARM_TABLE_SIZE[0] - XARM_BASE_SEPARATION) / 2.0   # 0.07 m
XARM_BASE_TO_FRONT = 0.52              # m, arm line to the front 80 cm edge

# --- Safety ---
XARM_COLLISION_SENSITIVITY = 3     # xArm 0(off)-5(most sensitive)
XARM_COLLISION_THRESHOLD = 0.12    # m, min inter-arm distance for dual moves

# --- Virtual walls -----------------------------------------------------------
# A box the TCP may never leave, in TABLE coordinates (= the LEFT arm's base
# frame: x from 0 at the left base across the short axis to XARM_BASE_SEPARATION
# at the right base, y along the table's long axis, z up with the table at
# XARM_TABLE_Z). Derived from the cell geometry above so the two stay consistent.
#
# Enforced twice, by real_robot/utils/xarm_walls.py + XArmLite6:
#   * the controller's own safety boundary (set_reduced_tcp_boundary +
#     set_fence_mode), which no code path can bypass; and
#   * a check on every waypoint in the driver, so a violation is rejected with a
#     legible message instead of a bare controller error 35.
# NOTE: this bounds the TCP only -- elbows can still leave the box, and it does
# not prevent arm-vs-arm collision (that is check_trajectories_close plus the
# controller's collision detection).
XARM_WALL_MARGIN = 0.05            # m, inset from each table edge
XARM_WALL_CEILING = 0.50           # m above the table (> the ~0.44 m reach, so
                                   # it never binds; lower it if anything hangs)

# The table rectangle itself, in the LEFT base frame (metres):
#   x: across the arms   -- left edge is XARM_BASE_INSET behind the left base
#   y: along the arm line -- front (+y) is XARM_BASE_TO_FRONT away, back is the rest
# With the measured cell that is x [-0.07, +0.73], y [-0.68, +0.52].
XARM_TABLE_RECT = {
    'x': (-XARM_BASE_INSET, XARM_TABLE_SIZE[0] - XARM_BASE_INSET),
    'y': (XARM_BASE_TO_FRONT - XARM_TABLE_SIZE[1], XARM_BASE_TO_FRONT),
}

_m = XARM_WALL_MARGIN
XARM_WALLS = {
    'x': (XARM_TABLE_RECT['x'][0] + _m, XARM_TABLE_RECT['x'][1] - _m),
    'y': (XARM_TABLE_RECT['y'][0] + _m, XARM_TABLE_RECT['y'][1] - _m),
    'z': (XARM_TABLE_Z, XARM_TABLE_Z + XARM_WALL_CEILING),
}
del _m
# NOTE: the arms reach only ~0.44 m, so most of these walls sit BEYOND the reach
# envelope and will never trigger. The ones that actually bite are the z floor
# (driving the gripper into the tabletop) and, near the base, the x wall. That is
# the honest situation, not a bug: the box is a containment guarantee, and the
# reach envelope does most of the work on this table.

# --- Ready poses (JOINT space, radians). ⚠️ MUST be verified on hardware. ---
# Generic "reach forward, elbow bent, TCP pointing down" pose. Override via the
# driver kwargs; the per-arm taught values below are what the scenes actually use.
XARM_HOME_JOINT = np.deg2rad([6.4, -2.2, 87.4, -176.1, -88.6, 16.8]).tolist()

# --- Photo (out-of-scene) pose -----------------------------------------------
# Before every camera frame the arms move out of the way. The pose is NOT written
# out by hand: it is the arm's OWN home with joint 1 swung to that arm's left.
#
# Deriving it from home is the whole point. Joint 1 rotates about the base z axis,
# so changing only J1 leaves the TCP's height and its radius from the base exactly
# as they are at home -- a pose taught on the hardware and known to clear the
# table. A hand-written pose has neither guarantee, and the previous shared
# constant ([45, 20, -40, 0, 20, 0] deg) had no relation to either arm's home.
#
# +J1 is toward base +y, i.e. the arm's own left (confirmed against the URDF's
# forward kinematics). The right base is yawed 180 deg, so "each arm to its own
# left" moves the left arm toward the table's back and the right arm toward its
# front: they separate, and cannot meet.
#
# Measured in the PyBullet cell (real_robot/sim/), the swing pulls each arm out of
# the region above the garment -- the intrusion toward the table centre drops from
# 0.189 m to 0.093 m (left) and from 0.181 m to 0.065 m (right), with no arm-arm
# contact and no change to the lowest link height.
XARM_PHOTO_YAW = np.pi / 2         # rad, J1 offset from home ("to its own left")


def photo_pose_from_home(home_joint, yaw=XARM_PHOTO_YAW):
    """The photo pose for an arm whose home is ``home_joint``.

    Only joint 1 changes, so the TCP keeps home's height and its radius from the
    base: the pose cannot reach the table, whatever home happens to be. Pass the
    home this arm was actually given (which may have come from xarm-cell.yaml)
    rather than a constant, so the two can never drift apart.
    """
    q = [float(v) for v in home_joint]
    q[0] += float(yaw)
    return q


XARM_OUT_SCENE_JOINT = photo_pose_from_home(XARM_HOME_JOINT)

# --- Per-arm overrides -------------------------------------------------------
# The two controllers are configured independently, and a setting on one does not
# imply the same on the other: a leftover 374 mm tcp_offset on the right arm once
# made its measured "gripper offset" read 0.24 m below the left's for the same
# physical table contact. So the quantities that are actually measured per arm are
# stored per arm. Both sides default to the shared values above, and
# test_xarm_teach.py writes the measured ones into
# real_robot/calibration/xarm-cell.yaml under arms.<side>.
XARM_GRIPPER_OFFSET_BY_SIDE = {
    'left': 0.086,
    'right': 0.0857,
}
XARM_TABLE_Z_BY_SIDE = {
    'left': 0,
    'right': 0,
}
XARM_HOME_JOINT_BY_SIDE = {
    'left': np.deg2rad([3.8, -33.6, 24.7, -176.2, -58.2, 2.9]).tolist(),
    'right': np.deg2rad([-4.0, -39.6, 21.9, 6.8, 60.9, 166.8]).tolist(),
}
XARM_WORKSPACE_RADIUS_BY_SIDE = {
    'left': tuple(XARM_WORKSPACE_RADIUS),
    'right': tuple(XARM_WORKSPACE_RADIUS),
}
# Derived, never written by hand -- see photo_pose_from_home above. The driver
# derives its own from whatever home it was constructed with (possibly a taught
# one out of xarm-cell.yaml); this dict is the fallback for callers that only
# have a side.
XARM_OUT_SCENE_JOINT_BY_SIDE = {
    side: photo_pose_from_home(q) for side, q in XARM_HOME_JOINT_BY_SIDE.items()
}


def for_side(mapping, side, default=None):
    """Per-side lookup that tolerates an unknown side (falls back to 'left').

    Scenes name their arms 'left'/'right', but the single-arm arena lets the side
    come from a config, so an unexpected value must not raise deep inside a motion
    primitive.
    """
    if side in mapping:
        return mapping[side]
    return mapping.get('left', default)

# TCP orientation that points the gripper straight down at the table, as an
# axis-angle (rotvec): flip about base X so tool-Z points to -base-Z.
XARM_DOWN_ROTVEC = [np.pi, 0.0, 0.0]

# --- Top-down camera ---------------------------------------------------------
# ⚠️ THESE TWO ARE FALLBACKS ONLY. The camera pose is MEASURED now, by hand-eye
# calibration, and lives in real_robot/calibration/xarm-{left,right}-calib.yaml as
# a 4x4 `camera_to_base` (= T_base_cam, the camera's pose in that arm's base
# frame, from 39 ChArUco samples). Anything with a calibration file loads it --
# load_camera_to_base() in scene_utils.py -- and never reads the numbers below.
# They survive for the synthetic camera in xarm_test_scene.py and for callers on a
# cell that has not been calibrated yet.
#
# The calibration says: camera at (+0.3685, -0.0037, +0.8386) m in the LEFT base
# frame. So the real height is 0.839 m, not the 1.0 m below, and the optical axis
# is 3.7 mm off the arm line rather than exactly on it. It looks straight down at
# very nearly the midpoint of the two BASES -- 0.369 m from the left base against
# a measured half-separation of 0.374, which is an independent consistency check
# on the 0.749 m separation. Note that point is NOT the table's centre: the arm
# line sits 0.52 m from the front edge and 0.68 m from the back, so the table
# midline is y = -0.08.
#
# ORIENTATION. The camera is mounted the natural way round, and hand-eye
# calibration confirms it: image u toward base +x (LEFT arm on the left) and,
# forced by that right-handed frame, image v toward base -y -- which, with the
# front edge at +y, puts the FRONT at the TOP. Both halves of the intended layout
# come out of the mounting, so nothing flips the image. describe_orientation() in
# xarm_camera.py reports what the mounting actually gave you, and now agrees --
# test_xarm_mujoco_camera.py::t_frame_layout asserts it against the real files.
#
# ⚠️ HEIGHT IS NOT A KNOB ANY MORE. It was one while it was a placeholder, and the
# old note here said "raise the bracket to ~1.50 m so a real RealSense sees the
# whole table". That advice is dead: the bracket is up, the calibration measured
# where it is, and 0.839 m is now an input. When the frame is too small the answer
# is a WIDER STREAM -- see the crop block below, which carries the arithmetic.
XARM_CAM_HEIGHT = 1.0              # m above the table plane (fallback; measured 0.8386)
XARM_CAM_CENTRE_Y = 0.0            # m, optical axis on the arm line (measured -0.0037)

# NO VIEW MIRROR. A software flip of the frame was designed on 2026-08-04 and is
# deliberately NOT here. It only existed to rescue the layout from the wrong sign
# of y (see XARM_TABLE_RECT): with front at -y, a down-looking camera genuinely
# cannot show "left arm on the left" and "front at the top" at once, and a
# reflection is the only escape. With front at +y the mounting gives both, and a
# mirrored view would have been a permanent cost -- every garment reading
# mirrored, a left sleeve appearing where a right one is -- paid to hide a
# one-character error. If anyone reaches for a flip again, check the sign of y
# first.
#
# ⚠️ And if a flip is ever genuinely needed, it belongs in the image and the
# INTRINSIC together, never in T_cam: that matrix must stay a proper rotation
# (det = +1). A reflection there still projects the table corners where you expect
# and then quietly breaks every handedness-dependent quantity downstream.

# --- Perception crop ---------------------------------------------------------
# Everything downstream sees a SQUARE window of the table, centred on the midpoint
# of the two arm bases -- (separation/2, 0) in the left base frame. The scene
# crops the frame and shifts its own intrinsic's principal point to match, so the
# pixel <-> metre mapping keeps working with no change at any call site.
#
# ⚠️ THIS IS NOT A TUNING KNOB ANY MORE. The window side IS the base separation,
# so the two arm base centres land exactly on the left and right edges of the
# crop. crop_window() defaults `size_m` to the separation it is given, so a cell
# measured at 0.748 m gets a 0.748 m window with nothing to keep in step; this
# constant is only the fallback for callers that have no measured cell.
#
# It is a length on the TABLE, in metres, not a pixel count, so it keeps meaning
# the same patch of table if the lens changes.
#
# ⚠️ WHETHER IT FITS THE SENSOR IS A REAL CONSTRAINT, and at the calibrated camera
# height it currently does not. Hand-eye puts the camera 0.839 m above the table,
# and that height is a measurement, not a choice. A D435i COLOUR stream
# (69.4 x 42.5 deg, fx ~ 925 px at 1280x720) then sees 1.18 m across but only
# 0.65 m along the arm line -- less than the 0.748 m separation, so a base-to-base
# square needs 825 px of a 720 px frame and crop_window() will CLAMP, silently
# de-centring every pixel handed to a primitive.
#
# The fix is a wider field of view, not a higher camera: the DEPTH stream is
# 87 x 58 deg (fx ~ 674, fy ~ 649 at 1280x720), covering 0.93 m along the arm line
# and making the window 579 px -- still well above the 512 px the arena upsamples
# to. Aligning colour->depth instead of depth->colour keeps that width, and the
# hand-eye result survives it: the device reports the colour->depth extrinsic, so
# T_left_depth = T_left_cam @ T_cam_depth, with no re-calibration.
#
# All of the above is against a NOMINAL intrinsic. Dump the real one first.
XARM_CROP_SIZE = XARM_BASE_SEPARATION   # m, square side on the table plane

# --- Pick-and-fling ----------------------------------------------------------
# These are NOT free parameters. xarm_base_fling_poses() builds the swing in a
# frame centred between the two bases, so with separation S each gripper sits at
#
#     x = (S - width) / 2          from its own base,
#
# and the wind-up waypoint is at (x, -stroke, hang) in that arm's base frame. Two
# constraints follow, and both are violated by a naive port of the UR numbers:
#
#     (1) base keepout:   x >= r_min
#     (2) reach:          x^2 + stroke^2 + hang^2 <= r_max^2
#
# ⚠️ THE TWO CONSTRAINTS PULL IN OPPOSITE DIRECTIONS AS THE CELL WIDENS, which is
# not obvious. A wider cell moves each gripper FURTHER from its own base, so (1)
# gets easier and (2) gets HARDER -- it is tempting to read a wider cell as simply
# more comfortable for the fling, and it is not. Hand-eye then measured the
# separation at 0.7489 m against the 0.66 m these numbers were derived for:
#
#     x = (0.7489 - 0.36)/2 = 0.1945 m,
#     swing radius = sqrt(0.1945^2 + 0.25^2 + 0.25^2) = 0.4035 m,
#     against the MEASURED r_max = 0.410 m (xarm-cell.yaml, per arm).
#
# So it still fits, by 1.6% -- down from the 11% the assumed 0.66 m gave. The
# margin is thin enough that the next change to any of these has to be re-derived
# rather than nudged. Two ways to buy it back if the fling needs room:
#
#     XARM_FLING_WIDTH >= 0.375 m   (a wider cell wants a wider stretch, which
#                                    pulls each gripper back toward its own base)
#     XARM_FLING_STROKE <= 0.244 m  (a shorter wind-up)
#
# ⚠️ Note it does NOT fit against XARM_WORKSPACE_RADIUS's conservative 0.400 m
# below, which is a stale fallback: --reach measured 0.41. Checking a MEASURED
# separation against an UNMEASURED reach mixes two different cells and reports a
# violation that does not exist. test_xarm_walls_offline.py::t_fling_envelope
# takes both from xarm-cell.yaml for exactly that reason.
#
# For scale: the UR cell runs width 0.65, hang 0.35, stroke 0.65 -- but a UR5e
# reaches ~0.85 m. The Lite 6 fling is about half-size, which is geometry, not a
# tuning choice. If the separation or the measured reach changes, RE-DERIVE these.
# Which way the swing STROKES, as a sign on base y in the LEFT base frame. NOT a
# free parameter and NOT a preference: get_base_fling_poses winds up backwards to
# y = -XARM_FLING_STROKE, then strokes forward through touch-down to the drag, so
# the swing needs table BEHIND it to wind up into and table AHEAD to lay onto, and
# it must finish with the garment on the operator's side of the cell.
#
# Front is +y, so forward is +y: wind up into the 0.68 m back, stroke and lay down
# into the 0.52 m front. Both fit.
#
# The direction is DERIVED, which is why this is a sign and not an angle:
# points_to_action_frame takes forward = z_hat x (left_point - right_point), and
# with our left arm at the smaller base x that comes out as +y -- already correct,
# so +1.0 is the identity and the skill's swap branch is the untaken one. It stays
# explicit rather than implicit because the alternative is an unstated assumption,
# and this file has already been bitten by one of those.
#
# ⚠️ This is +1.0 again after a wrong -1.0 on 2026-08-04, which came from reading
# the throw direction off a fling that E-STOPPED during its wind-up. Keep this in
# step with XARM_BASE_TO_FRONT's sign: the two must always say the same thing
# about where the front is.
XARM_FLING_FORWARD_Y = 1.0         # sign on base y; +1 = strokes toward the front

XARM_FLING_WIDTH = 0.36            # m, gripper separation after the stretch
XARM_FLING_HANG = 0.25             # m, hang height = swing height above the table
XARM_FLING_STROKE = 0.25           # m, FORWARD reach of the swing (toward the front)
XARM_FLING_ANGLE = np.pi / 4       # rad, wrist pitch at the swing extremes
XARM_FLING_PLACE_Z = 0.10          # m, touch-down height before the drag
XARM_FLING_MIN_WIDTH = 0.20        # m, never stretch narrower than this

# ⚠️ THE SWING IS DELIBERATELY ASYMMETRIC, and it did not start that way. The first
# version wound up to -STROKE and stroked to +STROKE, so half the motion was
# backwards and the operator watching it read the whole thing as "it flings
# backwards first". The backward half exists only to load the cloth, so it is now a
# SMALL fraction of the forward half and is capped here rather than derived from
# STROKE. Raising this back toward STROKE undoes the fix.
XARM_FLING_WINDUP = 0.06           # m, backward wind-up (24% of the forward stroke)

# Where the garment is LAID DOWN, as base y in the action frame -- which step 2 of
# the skill centres on the base-to-base line, so this is "how far behind that line".
# Slightly behind (negative) on purpose: the arms touch down at +STROKE, out in
# front, then drag back THROUGH the line and finish just past it, so the cloth ends
# up flat and stretched under the grippers rather than piled at the far end. It also
# keeps the release away from the front table edge (0.52 m).
XARM_FLING_PLACE_Y = -0.05         # m, final drag position, just behind the base line
# Swing dynamics. The UR uses 3.0 m/s at 7.0 m/s^2; the Lite 6 is a 0.61 kg-payload
# arm, so this is scaled down. Raise only after the swing looks stable on hardware.
XARM_FLING_SPEED = 1.5             # m/s
XARM_FLING_ACC = 3.0               # m/s^2

# Vertical shake after the stretch, to loosen folds (UR: 3 x 0.03 m, speed 2, acc 4).
XARM_SHAKE_COUNT = 3
XARM_SHAKE_AMPLITUDE = 0.03        # m
XARM_SHAKE_SPEED = 1.0             # m/s
XARM_SHAKE_ACC = 2.0               # m/s^2

# Position-based substitutes for the UR's force-mode stages (no F/T sensor here).
XARM_STRETCH_STEP = 0.01           # m, outward increment per stretch iteration
XARM_STRETCH_MAX_TIME = 5.0        # s, matches the UR's stretch_max_time
XARM_STRETCH_SPEED = 0.10          # m/s, slow -- this is pulling on fabric
XARM_RELEASE_DIST = 0.03           # m, inward move to slacken the cloth before opening
XARM_PROBE_STEP = 0.005            # m, descent increment for the contact probe
# ⚠️ TUNE ON HARDWARE, per arm. The L2 rise in joint effort that counts as "loaded".
# Too low and the stretch stops immediately; too high and it never fires and the
# geometric width cap does all the work (which is the safe failure mode).
XARM_EFFORT_THRESHOLD = {
    'left': 2.0,
    'right': 2.0,
}
# Until the thresholds above are MEASURED, the effort signal is reported and never
# acted on -- the same doctrine as XARM_GEOMETRY_VERIFIED and the virtual walls: a
# bound derived from an unverified number does not protect anything, it just stops
# legitimate motion.
#
# This is not hypothetical. On 2026-08-04 the placeholder 2.0 was below the right
# arm's own noise floor: descending unloaded produced a delta of 2.16, so the
# contact probe "found" the cloth 55 mm above the table and the right gripper
# closed on air, while the left arm (quieter) grasped normally. The failure is
# silent by construction -- both arms report success.
#
# While False: the probe descends to the calibrated grasp height and the stretch
# runs to the geometric width cap, which is exactly the fallback both stages are
# documented to need when the signal is unreadable. Deltas are still printed, so
# one run gives you the numbers to set the thresholds from. Then flip this.
XARM_EFFORT_VERIFIED = False
# Consecutive samples that must exceed the threshold before a stage stops. One
# sample is a noise spike; this is a load.
XARM_EFFORT_CONSECUTIVE = 3
