from scipy.spatial.transform import Rotation
import numpy as np
from camera_utils import intrinsic_to_params


def pixel_to_camera_point(pixel, depth_m, intr):
    """Convert image pixel + depth (meters) to 3D camera coordinates."""
    u, v = pixel[0], pixel[1]
    fx, fy, cx, cy = intrinsic_to_params(intr)
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z])



def point_on_table_base(u, v, intr, cam2base, table_z):
    """
    Convert a pixel to a 3D point on a planar table at table_z in base frame.
    """
    # Direction vector in camera frame (depth=1)
    p_cam = pixel_to_camera_point((u,v), 1.0, intr)  # returns 3x1 vector

    # Rotate to base frame
    p_dir_base = cam2base[:3,:3] @ p_cam
    cam_origin_base = cam2base[:3,3]

    # Compute scale factor
    s = (table_z - cam_origin_base[2]) / p_dir_base[2]

    # Compute base-frame point
    p_base = cam_origin_base + s * p_dir_base
    return p_base


def transform_point(T, p):
    """Apply 4x4 transform T to 3D point p (len 3). Returns len-3 point."""
    p_h = np.ones(4)
    p_h[:3] = p
    p_t = T @ p_h
    return p_t[:3]

def tcp_pose_to_transform(tcp_pose):
    """
    tcp_pose: [x,y,z, rx,ry,rz] where rx,ry,rz is rotation vector (axis-angle)
    returns 4x4 homogeneous transform from base -> gripper (TCP)
    """
    t = np.array(tcp_pose[:3], dtype=float)
    rvec = np.array(tcp_pose[3:], dtype=float)
    R = Rotation.from_rotvec(rvec).as_matrix()
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def tcp_pose_to_transform(tcp_pose):
    """
    tcp_pose: [x,y,z, rx,ry,rz] where rx,ry,rz is rotation vector (axis-angle)
    returns 4x4 homogeneous transform from base -> gripper (TCP)
    """
    t = np.array(tcp_pose[:3], dtype=float)
    rvec = np.array(tcp_pose[3:], dtype=float)
    R = Rotation.from_rotvec(rvec).as_matrix()
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

def pixels_to_camera_points(pixels, depths_m, intr):
    """
    Convert N image pixels + depths (meters) to 3D camera coordinates.
    pixels: (N, 2)
    depths_m: scalar or (N,)
    intr: 3x3 intrinsic matrix
    Returns (N, 3) array of 3D camera coordinates.
    """
    fx, fy, cx, cy = intrinsic_to_params(intr)

    pixels = np.asarray(pixels)
    u = pixels[:, 0]
    v = pixels[:, 1]
    depths_m = np.asarray(depths_m)

    # Support scalar depth too
    if depths_m.ndim == 0:
        depths_m = np.full_like(u, depths_m, dtype=float)

    x = (u - cx) * depths_m / fx
    y = (v - cy) * depths_m / fy
    z = depths_m

    return np.stack([x, y, z], axis=1)

def pixels2base_on_table(pixels, intr, cam2base, table_z):
    """
    Vectorized: Convert many pixels to 3D base-frame points on a planar table.
    pixels: (N, 2)
    intr: 3x3
    cam2base: 4x4
    table_z: float
    Returns (N, 3) array of 3D points.
    """
    # 1. Compute direction vectors in camera frame (depth=1)
    p_cam = pixels_to_camera_points(pixels, 1.0, intr)  # (N, 3)

    # 2. Rotate all points into base frame
    p_dir_base = (cam2base[:3, :3] @ p_cam.T).T  # (N, 3)
    cam_origin_base = cam2base[:3, 3]

    # 3. Compute scale factor for intersection with plane z=table_z
    s = (table_z - cam_origin_base[2]) / p_dir_base[:, 2]

    # 4. Compute base-frame points
    p_base = cam_origin_base + s[:, None] * p_dir_base  # (N, 3)

    return p_base