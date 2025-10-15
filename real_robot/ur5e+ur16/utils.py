import select
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from collections import namedtuple
# from dotmap import DotMap   
import yaml
import os
# from ament_index_python.packages import get_package_share_directory
# from geometry_msgs.msg import PoseStamped
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import distance_transform_edt, sobel
from scipy.spatial.transform import Rotation

import math
# import torch
# import signal
# from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
# from segment_anything import sam_model_registry
# from agent_arena.utilities.visualisation_utils import draw_pick_and_place, filter_small_masks

# import subprocess
# import shlex
from scipy.ndimage import rotate, shift
from skimage.measure import label, regionprops

MyPos = namedtuple('Pos', ['pose', 'orien'])



def get_IoU(mask1, mask2):
    """
    Calculate the maximum IoU between two binary mask images,
    allowing for rotation and translation of mask1.
    
    :param mask1: First binary mask (numpy array)
    :param mask2: Second binary mask (numpy array)
    :return: Tuple of (Maximum IoU value, Matched mask)
    """
    
    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    # Get the properties of mask2
    props = regionprops(label(mask2))[0]
    center_y, center_x = props.centroid

    max_iou = 0
    best_mask = None
    
    # Define rotation angles to try
    angles = range(0, 360, 10)  # Rotate from 0 to 350 degrees in 10-degree steps
    
    for angle in angles:
        # Rotate mask1
        rotated_mask = rotate(mask1, angle, reshape=False)
        
        # Get properties of rotated mask
        rotated_props = regionprops(label(rotated_mask))[0]
        rotated_center_y, rotated_center_x = rotated_props.centroid
        
        # Calculate translation
        dy = center_y - rotated_center_y
        dx = center_x - rotated_center_x
        
        # Translate rotated mask
        translated_mask = shift(rotated_mask, (dy, dx))
        
        # Calculate IoU
        iou = calculate_iou(translated_mask, mask2)
        
        # Update max_iou and best_mask if necessary
        if iou > max_iou:
            max_iou = iou
            best_mask = translated_mask

    # Ensure the best_mask is binary
    best_mask = (best_mask > 0.5).astype(int)

    return max_iou, best_mask

def get_mask_v2(mask_generator, rgb):
        """
        Generate a mask for the given RGB image that is most different from the background.
        
        Parameters:
        - rgb: A NumPy array representing the RGB image.
        
        Returns:
        - A binary mask as a NumPy array with the same height and width as the input image.
        """
        # Generate potential masks from the mask generator
        results = mask_generator.generate(rgb)
        
        final_mask = None
        max_color_difference = 0
        print('Processing mask results...')
        save_color(rgb, 'rgb')
        mask_data = []

        # Iterate over each generated mask result
        for i, result in enumerate(results):
            segmentation_mask = result['segmentation']
            mask_shape = rgb.shape[:2]

            ## count no mask corner of the mask
            margin = 5
            mask_corner_value = 1.0*segmentation_mask[margin, margin] + 1.0*segmentation_mask[margin, -margin] + \
                                1.0*segmentation_mask[-margin, margin] + 1.0*segmentation_mask[-margin, -margin]
            
            

            #print('mask corner value', mask_corner_value)
            # Ensure the mask is in the correct format
            orginal_mask = segmentation_mask.copy()
            segmentation_mask = segmentation_mask.astype(np.uint8) * 255
            
            # Calculate the masked region and the background region
            masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask)
            background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask))
            
            # Calculate the average color of the masked region
            masked_pixels = masked_region[segmentation_mask == 255]
            if masked_pixels.size == 0:
                continue
            avg_masked_color = np.mean(masked_pixels, axis=0)
            
            # Calculate the average color of the background region
            background_pixels = background_region[segmentation_mask == 0]
            if background_pixels.size == 0:
                continue
            avg_background_color = np.mean(background_pixels, axis=0)
            
            # Calculate the Euclidean distance between the average colors
            color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
            #print(f'color difference {i} color_difference {color_difference}')
            #save_mask(orginal_mask, f'mask_candidate_{i}')
            
            # Select the mask with the maximum color difference from the background
            mask_region_size = np.sum(segmentation_mask == 255)

            if mask_corner_value >= 2:
                # if the mask has more than 2 corners, the flip the value
                orginal_mask = 1 - orginal_mask

            mask_data.append({
                'mask': orginal_mask,
                'color_difference': color_difference,
                'mask_region_size': mask_region_size,
            })
        
        top_5_masks = sorted(mask_data, key=lambda x: x['color_difference'], reverse=True)[:5]
        final_mask_data = sorted(top_5_masks, key=lambda x: x['mask_region_size'], reverse=True)[0]
        final_mask = final_mask_data['mask']
        
        #save_mask(final_mask, 'final_mask')
        print('Final mask generated.')

        return final_mask

def wait_for_user_input(timeout=1):
    """
    Wait for user input for a specified timeout period.
    Returns True if input is received, False otherwise.
    """
    print("\n\n[User Attention!] Please Press [Enter] to finish, or wait 1 second to continue...\n\n")
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        os.read(sys.stdin.fileno(), 1024)
        #sys.stdin.readline()
        return True
    print("\n\n[User Attention!] Continue to next step...\n\n")
    return False

# def get_mask_generator():
#     DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print('Device {}'.format(DEVICE))

#     ### Masking Model Macros ###
#     MODEL_TYPE = "vit_h"
#     sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth')
#     sam.to(device=DEVICE)
#     return SamAutomaticMaskGenerator(sam)


def get_orientation(point, mask):
    mask = (mask > 0).astype(np.uint8)
    
    # Compute gradients
    grad_y = sobel(mask, axis=0)
    grad_x = sobel(mask, axis=1)
    
    x, y = point
    
    # Calculate orientation
    gx = grad_x[y, x]
    gy = grad_y[y, x]
    orientation_rad = np.arctan2(gy, gx)
    
    # Convert orientation to degrees
    orientation_deg = np.degrees(orientation_rad)
    
    # Normalize to range [0, 360)
    orientation_deg = (orientation_deg - 90 + 360) % 360
    
    return orientation_deg


def visualize_points_and_orientations(mask, points_with_orientations, line_length=10):
    # Ensure mask is 8-bit single-channel
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Convert to BGR
    vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    for x, y, orientation in points_with_orientations:
        rad = orientation*math.pi/180
        cv2.circle(vis_mask, (int(x), int(y)), 3, (0, 255, 0), -1)
        end_x = int(x + line_length * np.cos(rad))
        end_y = int(y + line_length * np.sin(rad))
        cv2.line(vis_mask, (int(x), int(y)), (end_x, end_y), (0, 0, 255), 2)
    
    cv2.imshow('Points and Orientations', vis_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adjust_points(points, mask, min_distance=3):
    """
    Adjust points to be at least min_distance pixels away from the mask border.
    
    :param points: List of (x, y) coordinates
    :param mask: 2D numpy array where 0 is background and 1 is foreground
    :param min_distance: Minimum distance from the border (default: 2)
    :return: List of adjusted (x, y) coordinates
    """
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Compute distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Create a new mask where pixels < min_distance from border are 0
    eroded_mask = (dist_transform >= min_distance).astype(np.uint8)
    
    adjusted_points = []
    for x, y in points:
        if eroded_mask[y, x] == 0:  # If point is too close to border
            # Find the nearest valid point
            y_indices, x_indices = np.where(eroded_mask == 1)
            distances = np.sqrt((x - x_indices)**2 + (y - y_indices)**2)
            nearest_index = np.argmin(distances)
            new_x, new_y = x_indices[nearest_index], y_indices[nearest_index]
            adjusted_points.append((new_x, new_y))
        else:
            adjusted_points.append((x, y))
    
    return adjusted_points, eroded_mask

def imgmsg_to_cv2_custom(img_msg, encoding="bgr8"):
    # Get the image dimensions
    height = img_msg.height
    width = img_msg.width
    #channels = 3  # Assuming BGR format

    # Extract the raw data
    if encoding == "64FC1":
        data = np.frombuffer(img_msg.data, dtype=np.float64).reshape((height, width, -1))
        return data
    
    data = np.frombuffer(img_msg.data, dtype=np.uint8)

    # Reshape the data to match the image dimensions
    image = data.reshape((height, width, -1))

    # If the encoding is not BGR, convert it using OpenCV
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "mono8":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def save_color(img, filename='color', directory=".", rgb2bgr=True):
    if rgb2bgr:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
        
    cv2.imwrite('{}/{}.png'.format(directory, filename), img_bgr)

def save_depth(depth, filename='depth', directory=".", colour=False):
    depth = (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    if colour:
        depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
    else:
        depth = np.uint8(255 * depth)
    os.makedirs(directory, exist_ok=True)
    #print('save')
    cv2.imwrite('{}/{}.png'.format(directory, filename), depth)

def save_mask(mask, filename='mask', directory="."):
    mask = mask.astype(np.int8)*255
    cv2.imwrite('{}/{}.png'.format(directory, filename), mask)

def normalise_quaterion(q):
    q = np.array(q)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return q / norm

def add_quaternions(quat1, quat2):
    """
    Multiplies two quaternions to combine their rotations.
    
    Parameters:
    quat1 (list or np.array): The first quaternion [w, x, y, z].
    quat2 (list or np.array): The second quaternion [w, x, y, z].
    
    Returns:
    np.array: The resulting quaternion after combining the rotations.
    """
    # Convert the input quaternions to Rotation objects
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    
    # Multiply the quaternions
    combined_rotation = r1 * r2
    
    # Return the resulting quaternion
    return combined_rotation.as_quat()


def camera2base(camera_pos: MyPos, particles_camera):
    camera_pose = camera_pos.pose
    camera_orientation_quat = camera_pos.orien
    r = R.from_quat(camera_orientation_quat)
    rotation_matrix = r.as_matrix()
    particles_camera = np.array(particles_camera)
    particle_base = np.matmul(rotation_matrix, particles_camera) + np.array(camera_pose)
    return particle_base

def pixel2camera(pixel_point, depth, intrinsic):
    pixel_point = [int(pixel_point[0]), int(pixel_point[1])]
    return rs.rs2_deproject_pixel_to_point(intrinsic, pixel_point, depth)

def camera2pixel(point_3d, intrinsic):
    pixel = rs.rs2_project_point_to_pixel(intrinsic, point_3d)
    return pixel

def pixel2base(pixel_point, camera_intrinsic, camera_pos:MyPos, depth):
    
    camera_p = pixel2camera(pixel_point, depth, camera_intrinsic)
    
    
    base_p = camera2base(camera_pos, camera_p)

    return base_p

def interpolate_positions(start_pos, target_pos, num_points=100):
    return np.linspace(start_pos, target_pos, num_points)

def bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22):
    """
    Perform bilinear interpolation.
    
    Parameters:
        x, y: Coordinates of the target point.
        x1, y1, x2, y2: Coordinates of the four corners.
        q11, q21, q12, q22: Values at the four corners.
        
    Returns:
        Interpolated value at the target point.
    """
    denom = (x2 - x1) * (y2 - y1)
    w11 = (x2 - x) * (y2 - y) / denom
    w21 = (x - x1) * (y2 - y) / denom
    w12 = (x2 - x) * (y - y1) / denom
    w22 = (x - x1) * (y - y1) / denom
    
    interpolated_value = q11 * w11 + q21 * w21 + q12 * w12 + q22 * w22
    return interpolated_value


def interpolate_image(height, width, corner_values):
    interpolated_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = i/height
            y = j/width
            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1
            y2 = y1 + 1
            q11 = corner_values[(x1, y1)]
            q21 = corner_values[(x2, y1)]
            q12 = corner_values[(x1, y2)]
            q22 = corner_values[(x2, y2)]
            interpolated_image[i, j] = \
                bilinear_interpolation(x, y, x1, y1, x2, y2, q11, q21, q12, q22)
    return interpolated_image

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


def click_points_pick_and_place(window_name, img):
    clicks = []
    clone = img.copy()

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))
            cv2.circle(clone, (int(x), int(y)), 5, (0,255,0), -1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Please click PICK point then PLACE point on the image window.")
    while len(clicks) < 4:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)
    if len(clicks) < 4:
        raise RuntimeError("Four points not selected")
    return clicks[0], clicks[1], clicks[2], clicks[3]

def click_points_pick_and_fling(window_name, img):
    clicks = []
    clone = img.copy()

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))
            cv2.circle(clone, (int(x), int(y)), 5, (0,255,0), -1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Please click PICK point then PLACE point on the image window.")
    while len(clicks) < 2:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)
    if len(clicks) < 4:
        raise RuntimeError("Four points not selected")
    return clicks[0], clicks[1]

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

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

def points_to_action_frame(left_point, right_point):
    """
    Compute transfrom from action frame to world
    Action frame: centered on the mid-point between gripers,
    with the y-axis facing fling direction (i.e. forward)
    * left_point
    |---> y
    * right_point
    |
    x
    """
    left_point, right_point = left_point.copy(), right_point.copy()
    center_point = (left_point + right_point) / 2
    # enforce z
    left_point[2] = center_point[2]
    right_point[2] = center_point[2]
    # compute forward direction
    forward_direction = np.cross(
        np.array([0,0,1]), (right_point - left_point))
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    # default facing +y
    rot = rot_from_directions(
        np.array([0,1,0]), forward_direction)
    tx_world_action = pos_rot_to_mat(center_point, rot)
    return tx_world_action


def points_to_gripper_pose(left_point, right_point, max_width=None):

    tx_world_action = points_to_action_frame(left_point, right_point)

    width = np.linalg.norm((left_point - right_point)[:2])
    if max_width is not None:
        width = min(width, max_width)
    left_pose_action = np.array([-width/2,0,0,0,np.pi,0])
    right_pose_action = np.array([width/2,0,0, 0,np.pi,0])
    left_pose = transform_pose(tx_world_action, left_pose_action)
    right_pose = transform_pose(tx_world_action, right_pose_action)
    return left_pose, right_pose

def get_base_fling_poses(
        place_y=0.0,
        stroke=0.6, 
        lift_height=0.45, 
        swing_angle=np.pi/4,
        place_height=0.05
    ):
    """
    Basic fling trajectory: single trajectory on y-plane.
    From -y to +y, x=0
    Waypoint 1 is at place_y

                  z
    ----stroke----^
    --------------->y
    |             |
    |2     0     1|lift_height
    |             |
    |            3|place_height
    ---------------
    """

    base_fling_pos = np.array([
        [0,0,lift_height],
        [0,place_y,lift_height],
        [0,place_y-stroke,lift_height],
        [0,place_y,place_height]
    ])
    init_rot = Rotation.from_rotvec([0,np.pi,0])
    base_fling_rot = Rotation.from_euler('xyz',[
        [0,0,0],
        [swing_angle,0,0],
        [-swing_angle,0,0],
        [swing_angle/8,0,0]
    ])
    fling_rot = base_fling_rot * init_rot
    fling_pose = pos_rot_to_pose(base_fling_pos, fling_rot)
    return fling_pose

def points_to_fling_path(
        left_point, right_point,
        width=None,   
        swing_stroke=0.6, 
        swing_height=0.45, 
        swing_angle=np.pi/4,
        lift_height=0.4,
        place_height=0.05):
    tx_world_action = points_to_action_frame(left_point, right_point)
    tx_world_fling_base = tx_world_action.copy()
    # height is managed by get_base_fling_poses
    tx_world_fling_base[2,3] = 0
    base_fling = get_base_fling_poses(
        swing_stroke=swing_stroke,
        swing_height=swing_height,
        swing_angle=swing_angle,
        lift_height=lift_height,
        place_height=place_height)
    if width is None:
        width = np.linalg.norm((left_point - right_point)[:2])
    left_path = base_fling.copy()
    left_path[:,0] = -width/2
    right_path = base_fling.copy()
    right_path[:,0] = width/2
    left_path_w = transform_pose(tx_world_fling_base, left_path)
    right_path_w = transform_pose(tx_world_fling_base, right_path)
    return left_path_w, right_path_w

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


def safe_depth_at(depth_img, pixel):
    """Return a usable depth (meters) at or near pixel (u,v). Tries small neighborhood if zero."""
    u, v = pixel[0], pixel[1]
    H, W = depth_img.shape[:2]
    u = int(np.clip(u, 0, W-1))
    v = int(np.clip(v, 0, H-1))
    z = float(depth_img[v, u])
    if z > 0:
        return z
    # try small neighborhood up to 5x5
    for r in range(1,6):
        ys = slice(max(0, v-r), min(H, v+r+1))
        xs = slice(max(0, u-r), min(W, u+r+1))
        patch = depth_img[ys, xs]
        nz = patch[patch>0]
        if nz.size > 0:
            return float(np.median(nz))
    raise RuntimeError("Could not find valid depth near clicked pixel")

def intrinsic_to_params(intr):
    """Convert pyrealsense2 intrinsics object to fx,fy,ppx,ppy."""
    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy
    return fx, fy, cx, cy

def pixel_to_camera_point(pixel, depth_m, intr):
    """Convert image pixel + depth (meters) to 3D camera coordinates."""
    u, v = pixel[0], pixel[1]
    fx, fy, cx, cy = intrinsic_to_params(intr)
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z])

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