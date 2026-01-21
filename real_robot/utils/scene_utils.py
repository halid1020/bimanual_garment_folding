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
