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
# 0.52 m of table in front of them (-y) and 0.68 m behind (+y).
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
# Where the arm line sits on the table, in the LEFT base frame. FRONT IS -y.
# Centred ACROSS the 80 cm width (each base inset 0.07 m from its 120 cm edge),
# but NOT along the length: the arm line is 0.52 m from the front 80 cm edge and
# 0.68 m from the back one. The walls are therefore asymmetric in y, and the two
# arms end up with mirrored (not identical) y limits.
#
# ⚠️ THE SIGN IS MEASURED, not chosen -- do not flip it to make a view come out
# right. It was briefly set to +y on 2026-08-02 so that a top-down camera could
# show the front edge at the top AND the left arm on the left, and the real cell
# falsified that on 2026-08-04: the fling lays the garment toward base +y (the
# direction is derived, see XARM_FLING_FORWARD_Y below), and it threw toward the
# BACK. That agrees with the mounting -- standing at the front edge, the left
# arm's base +x points across at the right arm, so its +y points away from you.
#
# The camera coupling that motivated the flip is real but is now handled where it
# belongs: a down-looking camera cannot show both "left arm on the left" and
# "front at the top" (right-handed image frame: u -> +x forces v -> -y), so the
# view is MIRRORED in software instead -- see XARM_VIEW_MIRROR. Nothing about the
# table geometry bends to the camera any more.
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
#   y: along the arm line -- front (-y) is XARM_BASE_TO_FRONT away, back is the rest
# With the measured cell that is x [-0.07, +0.73], y [-0.52, +0.68].
XARM_TABLE_RECT = {
    'x': (-XARM_BASE_INSET, XARM_TABLE_SIZE[0] - XARM_BASE_INSET),
    'y': (-XARM_BASE_TO_FRONT, XARM_TABLE_SIZE[1] - XARM_BASE_TO_FRONT),
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
# PLACEHOLDER until hand-eye calibration. The camera looks straight down at the
# midpoint of the two arm BASES -- (separation/2, 0) in the left base frame, i.e.
# on the arm line. Note that is NOT the table's centre: the arm line sits 0.52 m
# from the front edge and 0.68 m from the back, so the table midline is y = +0.08.
#
# ORIENTATION. The camera is mounted the natural way round: image u toward base +x
# (LEFT arm on the left) and, forced by that, image v toward base -y, which puts
# the FRONT edge at the BOTTOM of the raw frame. See XARM_VIEW_MIRROR below for
# how the view then gets the front to the top. describe_orientation() in
# xarm_camera.py reports what the mounting actually gave you.
#
# ⚠️ HEIGHT. At 1.0 m the placeholder intrinsic (f = 500 px, 1280x720) sees
# 2.56 x 1.44 m, so the whole 0.80 x 1.20 m table fits. A real RealSense will NOT:
# a D435i colour stream at 1280x720 has fx ~ 900 px, which at 1.0 m sees only
# 1.42 x 0.80 m -- the table's 1.20 m length is clipped. It needs ~1.50 m of
# height, or the wider depth stream. Check against your actual intrinsics before
# the bracket goes up.
XARM_CAM_HEIGHT = 1.0              # m above the table plane
XARM_CAM_CENTRE_Y = 0.0            # m, optical axis on the arm line

# --- View mirror -------------------------------------------------------------
# ⚠️ THE VIEW IS A REFLECTION OF THE WORLD, ON PURPOSE. A garment in it reads
# mirrored: what looks like a left sleeve is the right one.
#
# The requirement is that the LEFT arm appear on the left of the frame, the RIGHT
# arm on the right, AND the FRONT table edge at the top. No camera can do all
# three. The image frame of a down-looking camera is right-handed with the view
# direction as camera +z, so once u runs toward base +x (arms on the correct
# sides) v is forced to run toward base -y -- and the front edge is at -y, so it
# lands at the BOTTOM. Rolling the camera 180 deg fixes the front and breaks the
# arms. The two are one choice, and only a reflection escapes it.
#
# So the frame is flipped TOP-TO-BOTTOM (rows) before anything sees it. Vertical
# rather than horizontal because it leaves the columns -- and therefore the arms,
# which are already correct -- untouched, and because inverting up/down is much
# less disorienting to click through than inverting left/right.
#
# It is carried by ViewMirror in xarm_camera.py, which flips the image and the
# INTRINSIC together (fy -> -fy, ppy -> (H-1) - ppy). Everything downstream --
# point_on_table_base, pixels2base_on_table, base_to_pixel, table_xy_to_pixel, the
# workspace masks -- divides by or multiplies by fy, so it all follows with no
# change to the maths.
#
# ⚠️ NEVER express this in T_cam. That matrix must stay a proper rotation
# (det = +1); a reflection there would still project the table corners where you
# expect and then quietly break every handedness-dependent quantity downstream.
# And hand-eye calibration must be given the PHYSICAL intrinsic, not this one.
XARM_VIEW_MIRROR = True

# --- Perception crop ---------------------------------------------------------
# Everything downstream sees a SQUARE window of the table, centred on the midpoint
# of the two arm bases -- (separation/2, 0) in the left base frame. The scene
# crops the frame and shifts its own intrinsic's principal point to match, so the
# pixel <-> metre mapping keeps working with no change at any call site.
#
# ⚠️ THIS IS THE KNOB TO TUNE ON HARDWARE. It is a length on the TABLE, in metres,
# not a pixel count, so it keeps meaning the same patch of table if the camera
# height or the lens changes. The default is the base separation: literally the
# square spanned between the two arms.
#
# What that buys, against the measured reach (r_max = 0.41 m from each base):
#   * at the midline (x = 0.33) the reachable band is only y = +/- 0.24 m,
#   * near either base it opens out to about +/- 0.39 m,
# so 0.66 m covers the useful area with a little margin and drops the metre of
# floor either side of the 0.80 m table that a full frame includes.
#
# What it costs in resolution -- the crop is upsampled to the arena's 512 px:
#   * sim placeholder (f = 500 px at 1.0 m):        330 px
#   * RealSense colour (fx ~ 900 px at 1.5 m):     ~396 px
# Enlarging it past ~1.1 m starts pulling the arms' own bases into frame.
XARM_CROP_SIZE = 0.66              # m, square side on the table plane

# --- Pick-and-fling ----------------------------------------------------------
# These are NOT free parameters. get_base_fling_poses() builds the swing in a frame
# centred between the two bases, so with separation S each gripper sits at
#
#     x = (S - width) / 2          from its own base,
#
# and the wind-up waypoint is at (x, -stroke, hang) in that arm's base frame. Two
# constraints follow, and both are violated by a naive port of the UR numbers:
#
#     (1) base keepout:   x >= r_min
#     (2) reach:          x^2 + stroke^2 + hang^2 <= r_max^2
#
# With S = 0.66 and the measured (r_min, r_max) = (0.12, 0.41):
#     width <= S - 2*r_min = 0.42 m, and at width 0.36 -> x = 0.150 m,
#     stroke <= sqrt(0.41^2 - 0.150^2 - 0.25^2) = 0.288 m.
# The values below give a swing radius of 0.363 m, ~11% inside the reach limit.
#
# For scale: the UR cell runs width 0.65, hang 0.35, stroke 0.65 -- but a UR5e
# reaches ~0.85 m. The Lite 6 fling is about half-size, which is geometry, not a
# tuning choice. If XARM_BASE_SEPARATION or the measured reach changes, RE-DERIVE
# these; test_xarm_walls_offline.py asserts the two constraints above.
# Which way the swing lays the garment down, as a sign on base y in the LEFT base
# frame. NOT a free parameter and NOT a preference: the swing needs table BEHIND
# it to wind up into (XARM_FLING_STROKE) and table AHEAD of it to lay onto (the
# drag), and it has to finish with the garment on the operator's side of the cell.
# Front is -y, so forward is -y: wind up into the 0.68 m back, lay down into the
# 0.52 m front. Both fit.
#
# The direction is DERIVED, which is why this is a sign and not an angle:
# points_to_action_frame takes forward = z_hat x (left_point - right_point), and
# with our left arm at the smaller base x that comes out as +y. The skill honours
# this constant by swapping the two points it hands over, which reverses the cross
# product. Flip this if XARM_BASE_TO_FRONT's sign ever changes -- on 2026-08-04
# the real cell threw the garment at the back because the two disagreed.
XARM_FLING_FORWARD_Y = -1.0        # sign on base y; -1 = toward the front edge

XARM_FLING_WIDTH = 0.36            # m, gripper separation after the stretch
XARM_FLING_HANG = 0.25             # m, hang height = swing height above the table
XARM_FLING_STROKE = 0.25           # m, wind-up stroke of the swing
XARM_FLING_ANGLE = np.pi / 4       # rad, wrist pitch at the swing extremes
XARM_FLING_PLACE_Z = 0.10          # m, touch-down height before the drag
XARM_FLING_MIN_WIDTH = 0.20        # m, never stretch narrower than this
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
