#!/usr/bin/env python3
"""
pick_and_place_rtde_EYE_TO_HAND.py

Integrates:
 - RealsenseCamera (fixed)
 - UR16E (rtde)
 - RG2 gripper (optional)

Workflow:
 - Uses a fixed T_camera^base transform (from Eye-to-Hand calibration)
 - Show image, let user click pick and place points (two clicks)
 - Use fixed T_camera^base to compute 3D base coordinates directly
 - Execute approach -> descend -> grasp -> lift -> transport -> release
"""

import yaml
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Import user modules (assuming consistent class names)
from ur import UR_RTDE             # <--- MODIFIED: Assuming you have a class for UR16e
from realsense_camera import RealsenseCamera # The RealsenseCamera class

# --- EYE-TO-HAND CONFIGURATION ---
# Path to camera-to-base YAML (result of your Hand-to-Eye calibration)
CALIB_YAML = "ur16e_eye_to_hand_calib.yaml" 
# The loaded matrix will be T_camera^base

# Robot IP and settings
ROBOT_IP = "192.168.1.102"  # <--- MODIFIED: Your UR16e IP
GRIPPER_TYPE = 'rg2'        
# ---------------------------------

# Motion parameters (safe defaults — reduce for first tests)
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2            # conservative linear speed
MOVE_ACC = 0.2
HOME_AFTER = True
TABLE_OFFSET = 0.025        # Gripper Length (used to adjust z-height)

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

def intrinsic_to_params(intr):
    """Convert pyrealsense2 intrinsics object to fx,fy,ppx,ppy."""
    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy
    return fx, fy, cx, cy

def pixel_to_camera_point(u, v, depth_m, intr):
    """Convert image pixel + depth (meters) to 3D camera coordinates."""
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

def click_two_points(window_name, img):
    """
    Display image and collect two clicks (pick, place).
    Returns pixel coordinates as (u,v) tuples: (pick_uv, place_uv)
    """
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
    if len(clicks) < 2:
        raise RuntimeError("Two points not selected")
    return clicks[0], clicks[1]


def safe_depth_at(depth_img, u, v):
    """Return a usable depth (meters) at or near pixel (u,v). Tries small neighborhood if zero."""
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

def pose_to_rtde_tcp_format(pos3, rotvec3):
    """Return RTDE TCP pose array [x,y,z,rx,ry,rz]"""
    return [float(pos3[0]), float(pos3[1]), float(pos3[2]),
            float(rotvec3[0]), float(rotvec3[1]), float(rotvec3[2])]

def run_pick_and_place():
    # Load calibration: T_camera^base
    T_cam_base = load_camera_to_base(CALIB_YAML)
    print("Loaded T_camera^base transform (Camera in Base frame):\n", T_cam_base)

    # We need T_base^camera to transform points: p_base = T_base^camera @ p_camera
    # MODIFIED: Calculate the inverse transform
    # R_cam = T_cam_base[:3,:3]
    # t_cam = T_cam_base[:3,3]
    # R_cam_inv = R_cam.T
    # t_cam_inv = -R_cam_inv @ t_cam
    # T_base_cam = np.eye(4)
    # T_base_cam[:3,:3] = R_cam_inv
    # T_base_cam[:3,3] = t_cam_inv
    T_base_cam = T_cam_base
    print("\nCalculated T_base^camera (Base in Camera frame):\n", T_base_cam)

    # Initialize robot and gripper
    robot = UR_RTDE(ROBOT_IP, gripper=GRIPPER_TYPE) # <--- Using new UR16ERTDE class
    time.sleep(0.2)
    robot.home()

    ur5e = UR_RTDE("192.168.1.10", gripper='rg2')
    ur5e.home()
    ur5e.camera_state()

    # Start camera
    cam = RealsenseCamera(debug=False)
    time.sleep(0.5)
    color_img, depth_img = cam.take_rgbd()
    intr = cam.get_intrinsic()

    # Show image and pick points
    pick_uv, place_uv = click_two_points("Pick & Place", color_img)
    print("Picked pixels:", pick_uv, place_uv)

    # Compute 3D camera points (p_cam)
    dz_pick = safe_depth_at(cam.depth_img, pick_uv[0], pick_uv[1])
    dz_place = safe_depth_at(cam.depth_img, place_uv[0], place_uv[1])
    print("Depths (m):", dz_pick, dz_place)

    p_cam_pick = pixel_to_camera_point(pick_uv[0], pick_uv[1], dz_pick, intr)
    p_cam_place = pixel_to_camera_point(pick_uv[0], pick_uv[1], dz_place, intr) # FIXED
    print("p_cam_pick:", p_cam_pick, "p_cam_place:", p_cam_place)


    # --- CORE EYE-TO-HAND TRANSFORMATION ---
    # Transform camera points directly to base frame: p_base = T_base^camera @ p_cam
    p_base_pick = transform_point(T_base_cam, p_cam_pick)
    p_base_place = transform_point(T_base_cam, p_cam_place)
    # ---------------------------------------

    print("p_base_pick (pre-offset):", p_base_pick, "p_base_place (pre-offset):", p_base_place)

    # The Z-coordinate is the object height in the base frame.
    # We must add the gripper offset (distance from TCP flange to gripping point)
    # for the robot to reach the correct height.
    p_base_pick[2] = max(p_base_pick[2], 0) + TABLE_OFFSET
    p_base_place[2] = max(p_base_place[2], 0) + TABLE_OFFSET
    
    print("p_base_pick (final):", p_base_pick, "p_base_place (final):", p_base_place)

    try:
        # Use a fixed, vertical orientation for the TCP during pick/place motions.
        # This rotation vector (e.g., [3.1416, 0, 0]) ensures the gripper points straight down.
        # NOTE: If you need to pick a tilted object, this must be calculated relative 
        # to the camera's perspective of the object.
        vertical_rotvec = [3.1416, 0, 0] # Rx=180 deg, Ry=0, Rz=0 (Points straight down)

        # Approach -> descend -> grasp -> lift -> move -> release
        approach_pick = p_base_pick + np.array([0,0,APPROACH_DIST])
        grasp_pick = p_base_pick
        lift_after = grasp_pick + np.array([0,0,LIFT_DIST])
        approach_place = p_base_place + np.array([0,0,APPROACH_DIST])
        place_pose = p_base_place

        # Move robot: home -> approach -> descend -> grasp -> lift -> approach_place -> descend -> open -> lift -> home
        print("Moving to home (safe start)")
        robot.home(speed=1.0, acceleration=0.8, blocking=True)

        print("Approach above pick point:", approach_pick)
        robot.movel(np.concatenate([approach_pick, vertical_rotvec]), speed=MOVE_SPEED, acceleration=MOVE_ACC, blocking=True)

        print("Descending to pick point:", grasp_pick)
        robot.movel(np.concatenate([grasp_pick, vertical_rotvec]), speed=0.08, acceleration=0.05, blocking=True)

        # Close gripper
        print("Closing gripper...")
        robot.close_gripper()
        time.sleep(2)

        print("Lifting object")
        robot.movel(np.concatenate([lift_after, vertical_rotvec]), speed=0.2, acceleration=0.1, blocking=True)

        print("Move to approach above place point")
        robot.movel(np.concatenate([approach_place, vertical_rotvec]), speed=0.2, acceleration=0.1, blocking=True)

        print("Descending to place point")
        robot.movel(np.concatenate([place_pose, vertical_rotvec]), speed=0.08, acceleration=0.05, blocking=True)

        # Open gripper
        print("Opening gripper...")
        robot.open_gripper()
        time.sleep(0.5)

        print("Lifting after releasing")
        robot.movel(np.concatenate([approach_place, vertical_rotvec]), speed=0.2, acceleration=0.1, blocking=True)

        if HOME_AFTER:
            print("Returning home")
            robot.home(speed=1.0, acceleration=0.8, blocking=True)

        print("Pick-and-place sequence finished.")

    except Exception as e:
        print("ERROR during pick-and-place:", e)
    finally:
        try:
            print("Disconnecting robot...")
            robot.disconnect()
        except Exception as e2:
            print("Error disconnecting robot:", e2)
        print("Exiting.")

if __name__ == "__main__":
    run_pick_and_place()