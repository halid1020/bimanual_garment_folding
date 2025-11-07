#!/usr/bin/env python3
"""
pick_and_place_rtde.py

Integrates:
 - RealsenseCamera (user provided)
 - UR5RTDE (user provided)
 - RG2 gripper (user provided)

Workflow:
 - Start camera and robot
 - Show color image, let user click pick and place points (two clicks)
 - Use depth to compute 3D in camera frame
 - Transform using camera_to_gripper (from YAML) and robot TCP pose to base frame
 - Execute approach -> descend -> grasp -> lift -> transport -> release
"""

import yaml
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Import user modules (assumed to be in same folder or pythonpath)
from ur import UR_RTDE              # your UR5RTDE class file (modify import name if different)
from realsense_camera import RealsenseCamera  # the RealsenseCamera class you showed (file realsense_camera.py)
# If your Realsense file has a different name, change the import accordingly.

# Path to camera-to-gripper YAML
CALIB_YAML = "ur5e_eye_in_hand_calib.yaml"

# Robot IP and settings
ROBOT_IP = "192.168.1.10"   # change to your robot IP
GRIPPER_TYPE = 'rg2'        # your UR5RTDE constructor accepts 'rg2' to initialize RG2

# Motion parameters (safe defaults — reduce for first tests)
APPROACH_DIST = 0.08        # meters above target to approach from
LIFT_DIST = 0.08            # meters to lift after grasp
MOVE_SPEED = 0.2            # conservative linear speed
MOVE_ACC = 0.2
HOME_AFTER = True
TABLE_OFFSET = 0.015        # Gripper Length

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

def intrinsic_to_params(intr):
    """Convert pyrealsense2 intrinsics object to fx,fy,ppx,ppy."""
    # intr expected to have attributes fx, fy, ppx, ppy
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
    cv2.resizeWindow(window_name, 1280, 720)  # 👈 make window larger
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
    # Load calibration
    cam2gripper = load_camera_to_gripper(CALIB_YAML)
    print("Loaded camera_to_gripper transform:\n", cam2gripper)

    # Initialize robot and gripper
    robot = UR_RTDE(ROBOT_IP, gripper=GRIPPER_TYPE)
    time.sleep(0.2)

    robot.camera_state()

    # Start camera
    cam = RealsenseCamera(debug=False)
    time.sleep(0.5)
    color_img, depth_img = cam.take_rgbd()
    intr = cam.get_intrinsic()

    # Show image and pick points
    pick_uv, place_uv = click_two_points("Pick & Place", color_img)
    print("Picked pixels:", pick_uv, place_uv)

    # Compute 3D camera points
    dz_pick = safe_depth_at(cam.depth_img, pick_uv[0], pick_uv[1])
    dz_place = safe_depth_at(cam.depth_img, place_uv[0], place_uv[1])
    print("Depths (m):", dz_pick, dz_place)

    p_cam_pick = pixel_to_camera_point(pick_uv[0], pick_uv[1], dz_pick, intr)
    p_cam_place = pixel_to_camera_point(place_uv[0], place_uv[1], dz_place, intr)
    print("p_cam_pick:", p_cam_pick, "p_cam_place:", p_cam_place)

    # Transform camera->gripper
    p_grip_pick = transform_point(cam2gripper, p_cam_pick)
    p_grip_place = transform_point(cam2gripper, p_cam_place)
    print("p_grip_pick:", p_grip_pick, "p_grip_place:", p_grip_place)

    try:
        # Get current TCP pose (base frame)
        tcp_pose = robot.get_tcp_pose()   # [x,y,z,rx,ry,rz] in base frame for TCP
        T_gripper2base = tcp_pose_to_transform(tcp_pose)
        print("Current TCP pose:", tcp_pose)

        # Compute pick/place in base frame: p_base = T_base_grip @ p_gripper
        p_base_pick = transform_point(T_gripper2base, p_grip_pick)
        p_base_place = transform_point(T_gripper2base, p_grip_place)
        print("p_base_pick:", p_base_pick, "p_base_place:", p_base_place)

        # TODO:
        p_base_pick[2] = max(0, p_base_pick[2])
        p_base_place[2] = max(0, p_base_pick[2])

        p_base_pick += np.array([0, 0, TABLE_OFFSET])
        p_base_place +=  np.array([0, 0, TABLE_OFFSET])
        print("after adding gripper offset", "p_base_pick:", p_base_pick, "p_base_place:", p_base_place)

        # Use current orientation for the TCP during motion (keep orientation same)
        current_rotvec = tcp_pose[3:]
        vertical_rotvec = [3.1416, 0, 0]
        # Approach -> descend -> grasp -> lift -> move -> release
        # Approach positions are along base z up direction (simple strategy)
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
