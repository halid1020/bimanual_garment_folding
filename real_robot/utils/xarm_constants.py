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
# 0.52 m of table in front of them and 0.68 m behind.
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
# Where the arm line sits on the table, in the LEFT base frame. Front is -y.
# Centred ACROSS the 80 cm width (each base inset 0.07 m from its 120 cm edge),
# but NOT along the length: the arm line is 0.52 m from the front 80 cm edge and
# 0.68 m from the back one. The walls are therefore asymmetric in y, and the two
# arms end up with mirrored (not identical) y limits.
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
# Generic "reach forward, elbow bent, TCP pointing down" pose; and an "out of the
# camera view" pose that swings joint-1 aside. Override via the driver kwargs.
XARM_HOME_JOINT = np.deg2rad([6.4, -2.2, 87.4, -176.1, -88.6, 16.8]).tolist()
XARM_OUT_SCENE_JOINT = np.deg2rad([45.0, 20.0, -40.0, 0.0, 20.0, 0.0]).tolist()

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
