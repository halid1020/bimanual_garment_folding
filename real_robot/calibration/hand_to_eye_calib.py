import argparse
import time
import sys
import yaml
import numpy as np
import cv2
import math
import os  # <--- Added for file path manipulation
from ur import UR_RTDE

# RealSense
try:
    import pyrealsense2 as rs
except Exception as e:
    print("pyrealsense2 import failed. Please install Intel RealSense Python bindings.")
    raise

# UR RTDE interfaces
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    _HAS_RTDE = True
except Exception:
    _HAS_RTDE = False

# --------------------------- Utility functions ---------------------------

def pose_list_to_matrix(p):
    p = np.asarray(p, dtype=float).reshape(6)
    t = p[:3]
    rvec = p[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def ur_pose_to_lists(pose):
    T = pose_list_to_matrix(pose)
    return T[:3, :3], T[:3, 3]

def matrix_to_pose_lists(T):
    return T[:3, :3], T[:3, 3]

def invert_homogeneous_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def load_config(config_path):
    """Load the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)

# --------------------------- Realsense helpers ---------------------------

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

# --------------------------- Capture Samples ---------------------------

def capture_samples(config, args):
    pipeline, align = start_realsense()
    
    # --- Parse Config ---
    robot_ip = config['robot']['ip']
    gripper_type = config['robot'].get('gripper', 'rg2')
    
    # Prioritize YAML board settings if they exist, otherwise use CLI args
    squares_x = config.get('board', {}).get('squares_x', args.board_size[0])
    squares_y = config.get('board', {}).get('squares_y', args.board_size[1])
    square_len = config.get('board', {}).get('square_length', args.square_size)
    marker_len = config.get('board', {}).get('marker_length', args.marker_size)

    # --- ChArUco Board Setup ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), 
        square_len, 
        marker_len, 
        aruco_dict
    )
    
    # --- Connection to Robot ---
    print(f"Connecting to robot at {robot_ip}...")
    robot = UR_RTDE(robot_ip, gripper=gripper_type)

    # --- Load Poses ---
    poses_deg = config['poses']
    # Convert degrees to radians
    poses_rad = [
        [math.radians(a) for a in pose] for pose in poses_deg
    ]

    samples = []
    print(f"\nStarting ChArUco sample collection for {len(poses_rad)} poses...")

    for i, joint_target in enumerate(poses_rad):
        print(f"\nMoving to pose {i+1}/{len(poses_rad)}: {poses_deg[i]}")
        
        # Move robot
        robot.movej(joint_target, speed=0.5, acceleration=0.5)
        time.sleep(2.0) # Wait for settling

        ur_pose = robot.get_tcp_pose()

        # Capture frames
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            print("No frame received.")
            continue
            
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # Get intrinsics
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        dist_coeffs = np.zeros((5, 1), dtype=float) 

        # --- ChArUco Detection ---
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
        
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            
            if charuco_corners is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
                )
                
                if valid:
                    samples.append({'robot_pose': ur_pose, 'rvec': rvec.flatten(), 'tvec': tvec.flatten()})
                    print(f"✅ Sample {i+1} captured.")
                    
                    # Visualization
                    cv2.drawFrameAxes(color, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    cv2.aruco.drawDetectedCornersCharuco(color, charuco_corners, charuco_ids)
                    cv2.imshow('ChArUco Capture', color)
                    cv2.waitKey(500)
                else:
                    print("❌ Pose estimation failed (invalid solution).")
            else:
                print("❌ Not enough ChArUco corners interpolated.")
        else:
            print("❌ No ArUco markers detected.")

    pipeline.stop()
    cv2.destroyAllWindows()
    return samples

# --------------------------- Main calibration logic ---------------------------

def run_hand_eye(samples, args):
    R_gripper2base_list = [] 
    t_gripper2base_list = [] 
    R_target2cam_list = []   
    t_target2cam_list = []   

    for s in samples:
        # A: T_gripper^base
        T_base2gripper = pose_list_to_matrix(s['robot_pose'])
        T_gripper2base = invert_homogeneous_matrix(T_base2gripper)
        R_g, t_g = matrix_to_pose_lists(T_gripper2base)
        R_gripper2base_list.append(R_g)
        t_gripper2base_list.append(t_g)

        # B: T_camera^target
        rvec = np.asarray(s['rvec']).reshape(3,1)
        tvec = np.asarray(s['tvec']).reshape(3,1)
        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_tc)
        t_target2cam_list.append(tvec.reshape(3,))

    if len(R_gripper2base_list) < 3:
        raise RuntimeError('Need at least 3 valid samples.')

    print(f"Running calibration with {len(R_gripper2base_list)} samples...")
    
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list, 
        R_target2cam_list, t_target2cam_list, 
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    X = np.eye(4)
    X[:3, :3] = R_cam2base
    X[:3, 3] = t_cam2base.reshape(3,)

    out = {
        'camera_to_base': {'matrix': X.tolist()},
        'meta': {'samples': len(samples), 'board_type': 'charuco'}
    }
    
    # Save using the dynamically generated filename
    with open(args.output, 'w') as f:
        yaml.dump(out, f)

    np.set_printoptions(precision=6, suppress=True)
    print('\nCalibration Result (T_base^camera):\n')
    print(X)
    print(f'\nWrote calibration to {args.output}')
    return X

# --------------------------- CLI ---------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Hand-to-Eye calibration using RealSense + UR + ChArUco')
    
    # Configuration File
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    
    # Defaults (used if not in YAML)
    parser.add_argument('--board-size', type=int, nargs=2, default=[7, 5], help='Squares X Y (default)')
    parser.add_argument('--square-size', type=float, default=0.04, help='Square size meters (default)')
    parser.add_argument('--marker-size', type=float, default=0.03, help='Marker size meters (default)')
    
    # Output is now Optional/None by default so we can auto-generate it
    parser.add_argument('--output', type=str, default=None, help='Output YAML file (default: {config}-calib.yaml)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    # --- Auto-generate Output Filename ---
    if args.output is None:
        # Extract filename without extension (e.g. 'my_robot.yaml' -> 'my_robot')
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f"{config_name}-calib.yaml"
    
    print(f"Using configuration: {args.config}")
    print(f"Output will be saved to: {args.output}")

    samples = capture_samples(config, args)
    
    if len(samples) < 3:
        print('Not enough samples captured. Exiting.')
        sys.exit(1)
        
    X = run_hand_eye(samples, args)