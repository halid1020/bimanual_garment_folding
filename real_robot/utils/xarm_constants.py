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
XARM_GRIPPER_OFFSET = 0.02          # m
# Table surface height in each arm's base frame (z). CALIBRATE on hardware.
XARM_TABLE_Z = 0.0                  # m
XARM_MIN_Z = 0.005                  # m, never command below this (safety floor)

# --- Motion (UR-convention units; converted inside the driver) ---
XARM_MOVE_SPEED = 0.20              # m/s, Cartesian
XARM_MOVE_ACC = 0.50               # m/s^2, Cartesian
XARM_JOINT_SPEED = 0.5             # rad/s, joint moves (home/out_scene)
XARM_JOINT_ACC = 5.0               # rad/s^2
XARM_APPROACH_DIST = 0.08          # m, hover above pick/place before descending
XARM_LIFT_DIST = 0.10              # m, lift after grasp
XARM_BLEND_RADIUS = 0.02           # m, path-blend radius for multi-waypoint moves

# --- Workspace (Lite 6 reach ~0.44 m; bases ~0.35 m apart, top-down camera) ---
# Annular reach test in the base XY plane: keep out of a small base cylinder and
# inside the usable reach. TUNE on hardware.
XARM_WORKSPACE_RADIUS = (0.12, 0.40)   # (min, max) m from the arm base Z-axis
XARM_BASE_SEPARATION = 0.35            # m between the two arm bases (reference)

# --- Safety ---
XARM_COLLISION_SENSITIVITY = 3     # xArm 0(off)-5(most sensitive)
XARM_COLLISION_THRESHOLD = 0.12    # m, min inter-arm distance for dual moves

# --- Ready poses (JOINT space, radians). ⚠️ MUST be verified on hardware. ---
# Generic "reach forward, elbow bent, TCP pointing down" pose; and an "out of the
# camera view" pose that swings joint-1 aside. Override via the driver kwargs.
XARM_HOME_JOINT = np.deg2rad([0.0, 20.0, -40.0, 0.0, 20.0, 0.0]).tolist()
XARM_OUT_SCENE_JOINT = np.deg2rad([45.0, 20.0, -40.0, 0.0, 20.0, 0.0]).tolist()

# TCP orientation that points the gripper straight down at the table, as an
# axis-angle (rotvec): flip about base X so tool-Z points to -base-Z.
XARM_DOWN_ROTVEC = [np.pi, 0.0, 0.0]
