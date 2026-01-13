import argparse
import time
import sys
import yaml
import numpy as np
import cv2
import math
from ur import UR_RTDE

# RealSense
try:
    import pyrealsense2 as rs
except Exception as e:
    print("pyrealsense2 import failed. Please install Intel RealSense Python bindings.")
    raise

# UR RTDE interfaces (split version)
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    _HAS_RTDE = True
except Exception:
    _HAS_RTDE = False

# --------------------------- Utility functions ---------------------------

def pose_list_to_matrix(p):
    """Convert UR pose [x,y,z,rx,ry,rz] (Rodrigues rotation) into 4x4 matrix (T_base^gripper)."""
    p = np.asarray(p, dtype=float).reshape(6)
    t = p[:3]
    rvec = p[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def ur_pose_to_lists(pose):
    """Convert a UR TCP pose to (R, t) for OpenCV."""
    T = pose_list_to_matrix(pose)
    return T[:3, :3], T[:3, 3]

def matrix_to_pose_lists(T):
    """Convert a 4x4 matrix to (R, t) for OpenCV."""
    return T[:3, :3], T[:3, 3]

def invert_homogeneous_matrix(T):
    """Invert a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

# --------------------------- Realsense helpers (UNMODIFIED) ---------------------------

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

# The rest of start_realsense, get_color_frame, and capture_samples is the same,
# except for the argument parser which limits samples to the predefined list length.

def capture_samples(args):
    # ... (body of your original capture_samples function remains largely the same)
    # The key is that it collects:
    #   - 'robot_pose' (T_base^gripper)
    #   - 'rvec', 'tvec' (T_target^camera)
    
    # --- Start of capture_samples (mostly original) ---
    pipeline, align = start_realsense()

    rtde_r = None
    rtde_c = None

    ur16e =  UR_RTDE("192.168.1.102", gripper='rg2')
    ur16e.home()
    
    ur5e = UR_RTDE("192.168.1.10", gripper='rg2')
    ur5e.home()
    ur5e.camera_state()

    # ---------------- Predefined joint positions (in DEGREES) ----------------
    predefined_joint_positions_deg = [
        [77,  -45.4,  35.0,   -100.0,  -90,  -120],
        [75.2,  -51.4,  33.0,   -103.0,  -81.8,  -122.9],
        # [97.55,  -1.9,  8.7,   -95.6,  -218.7,  -56.9],
        # [100,  -3,  10,   -90,  -200,  -50],
        # [92.7, -41.7, 69.23, -138.7, -152.6, -56.9],
        # [95, -43, 72, -134, -155, -60],
        [91.9, -53.1, 58.6, -148.9, -120.5, -99.5],
        [89.8, -59.3, 74.4, -148.9, -120.5, -99.5],
        [90, -60, 75, -150, -124, -102],
    ]

    # Convert to radians
    predefined_joint_positions = [
        [math.radians(a) for a in pose] for pose in predefined_joint_positions_deg
    ]

    samples = []
    print(f"\nStarting automated sample collection for {len(predefined_joint_positions)} poses...")

    for i, joint_target in enumerate(predefined_joint_positions):
        print(f"\nMoving to pose {i+1}/{len(predefined_joint_positions)} (deg): {predefined_joint_positions_deg[i]}")
        #rtde_c.moveJ(joint_target, speed=0.5, acceleration=0.5)
        ur16e.movej(joint_target, speed=0.5, acceleration=0.5)
        time.sleep(2.0)  # wait for motion to settle

        ur_pose = ur16e.get_tcp_pose() #rtde_r.getActualTCPPose()

        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            print("No color frame, skipping this pose.")
            continue
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]], dtype=float)
        dist_coeffs = np.zeros((5, 1), dtype=float)

        found, corners = cv2.findChessboardCorners(gray, tuple(args.board_size), None)
        if not found:
            print('⚠️  Chessboard not found in the frame. Skipping this pose.')
            cv2.imshow('color', color)
            cv2.waitKey(500)
            continue

        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        )

        objp = np.zeros((args.board_size[0]*args.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:args.board_size[0], 0:args.board_size[1]].T.reshape(-1, 2)
        objp *= args.square_size

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not ret:
            print('solvePnP failed. Try another view.')
            continue

        samples.append({'robot_pose': ur_pose, 'rvec': rvec.reshape(3,), 'tvec': tvec.reshape(3,)})
        print(f"✅ Sample {i+1} captured successfully {samples[-1]}.")

        # Visualization
        imgpts, _ = cv2.projectPoints(
            np.float32([[0,0,0],[0.03,0,0],[0,0.03,0],[0,0,0.03]]),
            rvec, tvec, camera_matrix, dist_coeffs
        )
        color = cv2.drawChessboardCorners(color, tuple(args.board_size), corners2, True)
        for p in imgpts:
            p = tuple(p.ravel().astype(int))
            cv2.circle(color, p, 5, (0,255,0), -1)
        cv2.imshow('capture', color)
        cv2.waitKey(300)

    pipeline.stop()
    cv2.destroyAllWindows()
    return samples

# --------------------------- Main calibration logic (MODIFIED) ---------------------------

def run_hand_eye(samples, args):
    R_gripper2base_list = [] # Rotation (T_gripper^base) - This is 'A'
    t_gripper2base_list = [] # Translation (T_gripper^base) - This is 'A'
    R_target2cam_list = []   # Rotation (T_camera^target) - This is 'B'
    t_target2cam_list = []   # Translation (T_camera^target) - This is 'B'

    for s in samples:
        # 1. Calculate A: T_gripper^base (Inverse of robot TCP pose)
        T_base2gripper = pose_list_to_matrix(s['robot_pose'])
        T_gripper2base = invert_homogeneous_matrix(T_base2gripper)
        R_g, t_g = matrix_to_pose_lists(T_gripper2base)
        R_gripper2base_list.append(R_g)
        t_gripper2base_list.append(t_g)

        # 2. Calculate B: T_camera^target (solvePnP result)
        rvec = np.asarray(s['rvec']).reshape(3,1)
        tvec = np.asarray(s['tvec']).reshape(3,1)
        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_tc)
        t_target2cam_list.append(tvec.reshape(3,))

    if len(R_gripper2base_list) < 3:
        raise RuntimeError('Need at least 3 valid samples for hand-eye calibration.')

    # cv2.calibrateHandEye solves AX=XB where:
    # A = R_gripper2base_list (T_gripper^base)
    # B = R_target2cam_list (T_camera^target)
    # X = R_cam2base (T_camera^base)
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list, method=cv2.CALIB_HAND_EYE_TSAI)


    X = np.eye(4)
    X[:3, :3] = R_cam2base
    X[:3, 3] = t_cam2base.reshape(3,)

    out = {
        'camera_to_base': {'matrix': X.tolist()}, # Renamed key to reflect T_base^camera
        'meta': {'samples': len(samples), 'board_size': args.board_size, 'square_size': args.square_size}
    }
    with open(args.output, 'w') as f:
        yaml.dump(out, f)

    np.set_printoptions(precision=6, suppress=True)
    print('\nCalibration finished. Camera->Base transform (homogeneous 4x4):\n')
    print(X)
    print(f'Wrote calibration to {args.output}')
    return X



# --------------------------- CLI (MODIFIED) ---------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Hand-to-Eye calibration using RealSense + UR (RTDE)')
    parser.add_argument('--robot-ip', type=str, default=None, required=True, help='UR robot IP for RTDE read')
    parser.add_argument('--board-size', type=int, nargs=2, default=[9, 7], help='Chessboard inner corners (cols rows)')
    parser.add_argument('--square-size', type=float, default=0.02, help='Chessboard square size in meters')
    parser.add_argument('--output', type=str, default='ur16e_eye_to_hand_calib.yaml', help='Output YAML file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    samples = capture_samples(args)
    if len(samples) < 3:
        print('Not enough samples captured. Need at least 3. Exiting.')
        sys.exit(1)
    X = run_hand_eye(samples, args)