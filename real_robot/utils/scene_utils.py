import yaml
import numpy as np

def load_camera_to_gripper(yaml_path):
    """Load 4x4 camera-to-gripper matrix from the YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    mat_list = data.get('camera_to_gripper', {}).get('matrix', None)
    if mat_list is None:
        raise RuntimeError("camera_to_gripper.matrix not found in YAML")
    mat = np.array(mat_list, dtype=float)
    if mat.shape != (4,4):
        raise RuntimeError(f"camera_to_gripper matrix must be 4x4, got {mat.shape}")
    return mat


def load_camera_to_base(yaml_path):
    """Load 4x4 camera-to-base matrix from the YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # MODIFIED: Look for 'camera_to_base' key from Hand-to-Eye script
    mat_list = data.get('camera_to_base', {}).get('matrix', None)
    if mat_list is None:
        raise RuntimeError("camera_to_base.matrix not found in YAML")
    mat = np.array(mat_list, dtype=float)
    if mat.shape != (4,4):
        raise RuntimeError(f"camera_to_base matrix must be 4x4, got {mat.shape}")
    return mat


def cell_geometry_from_calibration(left_yaml, right_yaml):
    """Where the RIGHT arm's base is, measured -> ``(T_left_right, separation, yaw)``.

    Two eye-to-hand calibrations of the SAME fixed camera, one per arm, pin down
    the transform between the two bases without measuring anything between them:
    the camera is the shared reference, so

        T_left_right = T_left_cam @ inv(T_right_cam).

    That is worth having because ``XARM_BASE_SEPARATION`` and ``XARM_BASE_YAW``
    are marked ASSUMED in xarm_constants.py, and the assumption is 88 mm out --
    the calibration measures 0.748 m and 178.7 deg against an assumed 0.66 m and
    180 deg. Everything derived from the constant (the perception crop, the fling
    width limits) inherits that error, while ``XArmDualArmScene`` has always used
    the calibrated product for its right-arm targets. This function exists so the
    two halves read the same number from the same place.

    ``separation`` is the full distance between the base origins, not just its x
    component, and ``yaw`` is about base z. A cell whose arms are not coplanar
    also shows up here as a non-zero z in ``T_left_right``.

    Additive: nothing that already worked calls this. The UR scenes are untouched.
    """
    T_left_cam = load_camera_to_base(left_yaml)
    T_right_cam = load_camera_to_base(right_yaml)
    T_left_right = T_left_cam @ np.linalg.inv(T_right_cam)
    separation = float(np.linalg.norm(T_left_right[:3, 3]))
    yaw = float(np.arctan2(T_left_right[1, 0], T_left_right[0, 0]))
    return T_left_right, separation, yaw
