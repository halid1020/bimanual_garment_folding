from scipy.spatial.transform import Rotation
import numpy as np
from real_robot.utils.camera_utils import intrinsic_to_params
import cv2

GRIPPER_OFFSET_UR5e = 0.06 #To calibrate: This has to be accurate
GRIPPER_OFFSET_UR16e = 0.015 #To calibrate: This has to be accurate
SURFACE_HEIGHT = 0.03 #This has to be accurate
FLING_LIFT_DIST = 0.1

def matrix_to_pose(T):
    """
    Converts a 4x4 homogeneous transformation matrix to a UR pose vector.
    
    Args:
        T (np.array): 4x4 transformation matrix
        
    Returns:
        list: [x, y, z, rx, ry, rz] where (rx, ry, rz) is the Rodrigues rotation vector.
    """
    # 1. Extract Translation (x, y, z)
    t = T[:3, 3]
    
    # 2. Extract Rotation Matrix (3x3)
    R = T[:3, :3]
    
    # 3. Convert Rotation Matrix to Rodrigues Vector (rx, ry, rz)
    rvec, _ = cv2.Rodrigues(R)
    
    # 4. Flatten the rvec (it comes out as 3x1 usually)
    rvec = rvec.flatten()
    
    # 5. Combine into a single list
    return [t[0], t[1], t[2], rvec[0], rvec[1], rvec[2]]

def pixel_to_camera_point(pixel, depth_m, intr):
    """Convert image pixel + depth (meters) to 3D camera coordinates."""
    u, v = pixel[0], pixel[1]
    fx, fy, cx, cy = intrinsic_to_params(intr)
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z])

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = Rotation.from_rotvec(rotvec)
    return rot

def points_to_action_frame(right_point, left_point):
    """
    Compute transfrom from action frame to world
    Action frame: centered on the mid-point between gripers,
    with the y-axis facing fling direction (i.e. forward)

                (forward)
                    ↑  (y-axis of action frame)
                    |
    left * -------> * right
            (x-axis)
    """
    right_point, left_point = right_point.copy(), left_point.copy()
    center_point = (right_point + left_point) / 2
    # enforce z
    right_point[2] = center_point[2]
    left_point[2] = center_point[2]
    # compute forward direction
    forward_direction = np.cross(
        np.array([0,0,1]), (left_point - right_point))
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    # default facing +y
    rot = rot_from_directions(
        np.array([0,1,0]), forward_direction)
    tx_world_action = pos_rot_to_mat(center_point, rot)
    return tx_world_action


def get_base_fling_poses(
        place_y=0,
        stroke=0.6, 
        lift_height=0.45, 
        swing_angle=np.pi/4,
        place_height=0.05,
        drag_dist=0.2  # <--- NEW: Distance to drag after landing
    ):
    """
    Fling trajectory with a final drag motion.
    
    Trajectory:
    0: Start (Lifted)
    1: Forward Swing
    2: Backward Swing (Wind up)
    3: Landing (Touch down)
    4: Drag (Move forward on table)
    
                  z
    ----stroke----^
    --------------->y
    |             |
    |2     0     1| lift_height
    |             |
    |            3 -> 4 | place_height (Drag 3->4)
    ---------------
    """

    base_fling_pos = np.array([
        [0, 0, lift_height],                  # 0: Center High
        [0, place_y, lift_height],            # 1: Forward Swing High
        [0, place_y - stroke, lift_height],   # 2: Backward Swing High
        [0, place_y, place_height],           # 3: Touch Down
        [0, place_y + drag_dist, place_height]# 4: Drag Forward
    ])

    init_rot = Rotation.from_rotvec([0, np.pi, 0])
    
    # We repeat the last rotation for the drag step so the gripper 
    # doesn't twist while dragging.
    base_fling_rot = Rotation.from_euler('xyz', [
        [0, 0, 0],
        [swing_angle, 0, 0],
        [-swing_angle, 0, 0],
        [swing_angle/8, 0, 0],    # Rotation at touch down
        [swing_angle/8, 0, 0]     # Rotation during drag (Same as above)
    ])

    fling_rot = base_fling_rot * init_rot
    fling_pose = pos_rot_to_pose(base_fling_pos, fling_rot)
    
    return fling_pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def points_to_gripper_pose(right_point, left_point, max_width=None):

    tx_world_action = points_to_action_frame(right_point, left_point)

    width = np.linalg.norm((right_point - left_point)[:2])
    if max_width is not None:
        width = min(width, max_width)
    left_pose_action = np.array([-width/2, 0, 0, 0, np.pi,0])
    right_pose_action = np.array([width/2, 0, 0, 0,np.pi,0])
    left_pose = transform_pose(tx_world_action, left_pose_action)
    right_pose = transform_pose(tx_world_action, right_pose_action)
    return left_pose, right_pose


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