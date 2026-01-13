"""
Realsense + UR (RTDE) Eye-in-Hand (Hand-Eye) Calibration Script

Features:
- Connects to an Intel RealSense camera using pyrealsense2
- Captures chessboard detections and runs solvePnP to get target->camera poses
- Optionally connects to a UR robot via rtde_control + rtde_receive to read TCP poses
- Collects a user-specified number of samples (interactive: move robot, then press Enter)
- Solves AX = XB using OpenCV calibrateHandEye to compute camera->gripper transform
- Saves results to a YAML file and prints homogeneous transform

Usage examples:
    python realsense_urtd_hand_eye_calibration.py --robot-ip 192.168.0.2 --samples 15
    python realsense_urtd_hand_eye_calibration.py --samples 12 --output calib.yaml

Dependencies:
    pip install pyrealsense2 opencv-python numpy rtde-control rtde-receive pyyaml
"""

import argparse
import time
import sys
import yaml
import numpy as np
import cv2

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
    """Convert UR pose [x,y,z,rx,ry,rz] (Rodrigues rotation) into 4x4 matrix."""
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

# --------------------------- Realsense helpers ---------------------------

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

def get_color_frame(pipeline, align=None, timeout=5000):
    frames = pipeline.wait_for_frames(timeout_ms=timeout)
    if align is not None:
        frames = align.process(frames)
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError('No color frame')
    img = np.asanyarray(color.get_data())
    return img

# --------------------------- Main calibration logic ---------------------------

def capture_samples(args):
    import math
    pipeline, align = start_realsense()

    rtde_r = None
    rtde_c = None

    if args.robot_ip:
        if not _HAS_RTDE:
            print('rtde-control/rtde-receive packages not available. Install them or omit --robot-ip.')
            sys.exit(1)
        print(f'Connecting to UR robot at {args.robot_ip}...')
        rtde_r = RTDEReceiveInterface(args.robot_ip)
        rtde_c = RTDEControlInterface(args.robot_ip)
        time.sleep(0.5)
        print('Connected successfully.')
    else:
        print(f'Failed to connect to UR robot at {args.robot_ip}...')
        sys.exit(1)

    # ---------------- Predefined joint positions (in DEGREES) ----------------
    # You can edit these for your own calibration poses (degrees are easier to read)
    predefined_joint_positions_deg = [
        [73.4,  -78.7,  0,   -29.7,  -82.3,  79.0],
        [55.6,  -52.0,  0,   -32.1,  -71.5,  46.4],
        [83.1,  -56.2,  0,   -34.8,  -101.13,  49.7],
        [75.7,  -52.6,  0,   -48.5,  -84.9,  144.2],
        [79.2,  -50.5,  0,   -46.4,  -89.3,  70.5],
        [76.2,  -36.9,  0,   -63.1,  -80.4,  103.2],
        [73.6, -33.3, 0, -49.5, -72.1, 78.2],
        [73.4, -51.8, 0, -34.4, -84.7, 75.6],
        [73.4, -44.4, 0, -46.7, -91.4, 27.8],
        [73.4, -64.7, 0, -44.9, -97.6, 3.5],
        [73.3, -68.1, 0, -32.9, -87.3, 53.6],
        [73.4, -78.7, 0, -29.7, -82.3, 55.7],
        [73.4, -78.7, 0, -29.7, -82.3, 79.0]

    ]

    # Convert to radians
    predefined_joint_positions = [
        [math.radians(a) for a in pose] for pose in predefined_joint_positions_deg
    ]

    # Limit to desired sample count
    #predefined_joint_positions = predefined_joint_positions[:args.samples]

    samples = []
    print(f"\nStarting automated sample collection for {len(predefined_joint_positions)} poses...")

    for i, joint_target in enumerate(predefined_joint_positions):
        print(f"\nMoving to pose {i+1}/{len(predefined_joint_positions)} (deg): {predefined_joint_positions_deg[i]}")
        rtde_c.moveJ(joint_target, speed=0.5, acceleration=0.5)
        time.sleep(2.0)  # wait for motion to settle

        # Read joint and TCP pose
        ur_joints = rtde_r.getActualQ()
        ur_pose = rtde_r.getActualTCPPose()
        print("Joint positions (radians):", ur_joints)
        print("Joint positions (degrees):", [np.degrees(j) for j in ur_joints])
        print("TCP pose (m, rad):", ur_pose)

        # Capture color frame
        
        # wait for frames and align
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)

        # get color frame object and image
        color_frame = aligned.get_color_frame()
        if not color_frame:
            print("No color frame, skipping this pose.")
            continue
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # --- use real intrinsics from the camera (preferred) ---
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]], dtype=float)

        # distortion: if you don't have a calibration file, use zeros (suboptimal but OK)
        dist_coeffs = np.zeros((5, 1), dtype=float)

        print(f"Using camera intrinsics from RealSense: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


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

        # if args.camera_intrinsics:
        #     with open(args.camera_intrinsics, 'r') as f:
        #         cam = yaml.safe_load(f)
        #     camera_matrix = np.array(cam['camera_matrix'])
        #     dist_coeffs = np.array(cam.get('dist_coeff', [0,0,0,0,0]))
        # else:
        #     h, w = gray.shape
        #     fx = fy = args.focal_est * max(w, h)
        #     cx, cy = w / 2.0, h / 2.0
        #     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        #     dist_coeffs = np.zeros((5,1))
        #     print('No intrinsics provided. Using estimated focal length (not recommended).')

        ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if ret:
            dist = np.linalg.norm(tvec)
            print(f"Board distance: {dist:.3f} m")
        
        if not ret:
            print('solvePnP failed. Try another view.')
            continue

        samples.append({'robot_pose': ur_pose, 'rvec': rvec.reshape(3,), 'tvec': tvec.reshape(3,)})
        print(f"✅ Sample {i+1} captured successfully.")

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
    if rtde_c:
        rtde_c.disconnect()
    if rtde_r:
        rtde_r.disconnect()
    return samples


def run_hand_eye(samples, args):
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for s in samples:
        R_g, t_g = ur_pose_to_lists(s['robot_pose'])
        rvec = np.asarray(s['rvec']).reshape(3,1)
        tvec = np.asarray(s['tvec']).reshape(3,1)
        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam.append(R_tc)
        t_target2cam.append(tvec.reshape(3,))
        R_gripper2base.append(R_g)
        t_gripper2base.append(t_g)

    if len(R_gripper2base) < 3:
        raise RuntimeError('Need at least 3 valid samples for hand-eye calibration.')

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=cv2.CALIB_HAND_EYE_TSAI)

    X = np.eye(4)
    X[:3, :3] = R_cam2gripper
    X[:3, 3] = t_cam2gripper.reshape(3,)

    out = {
        'camera_to_gripper': {'matrix': X.tolist()},
        'meta': {'samples': len(samples), 'board_size': args.board_size, 'square_size': args.square_size}
    }
    with open(args.output, 'w') as f:
        yaml.dump(out, f)

    np.set_printoptions(precision=6, suppress=True)
    print('\nCalibration finished. Camera->Gripper transform (homogeneous 4x4):\n')
    print(X)
    print(f'Wrote calibration to {args.output}')
    return X

# --------------------------- CLI ---------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Eye-in-Hand calibration using RealSense + UR (RTDE)')
    parser.add_argument('--robot-ip', type=str, default=None, help='UR robot IP for RTDE read (optional)')
    #parser.add_argument('--samples', type=int, default=12, help='Number of samples to collect')
    parser.add_argument('--board-size', type=int, nargs=2, default=[9,7], help='Chessboard inner corners (cols rows)')
    parser.add_argument('--square-size', type=float, default=0.02, help='Chessboard square size in meters')
    parser.add_argument('--camera-intrinsics', type=str, default=None, help='YAML file with camera intrinsics')
    parser.add_argument('--focal-est', type=float, default=0.006, help='Estimated focal length fraction of max(image_dim)')
    parser.add_argument('--output', type=str, default='ur5e_eye_in_hand_calib.yaml', help='Output YAML file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    samples = capture_samples(args)
    if len(samples) == 0:
        print('No samples captured. Exiting.')
        sys.exit(1)
    X = run_hand_eye(samples, args)
