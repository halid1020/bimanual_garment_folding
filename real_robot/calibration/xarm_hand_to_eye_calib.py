import os
import time
import argparse
import threading
import math
import numpy as np
import cv2
import yaml
import pyrealsense2 as rs

from real_robot.robot.xarm_lite6 import XArmLite6

# --- Defaults ---
DEFAULT_LEFT_IP = os.environ.get('XARM_LEFT_IP', '192.168.1.155')
DEFAULT_RIGHT_IP = os.environ.get('XARM_RIGHT_IP', '192.168.1.170')
DEFAULT_SQUARES_X = 3
DEFAULT_SQUARES_Y = 2
DEFAULT_SQUARE_SIZE = 0.035
DEFAULT_MARKER_SIZE = 0.026

# --- Math Utilities for Hand-Eye Calibration ---
def pose_list_to_matrix(p):
    p = np.asarray(p, dtype=float).reshape(6)
    t = p[:3]
    rvec = p[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert_homogeneous_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def matrix_to_pose_lists(T):
    return T[:3, :3], T[:3, 3]

# --- Hand-Eye Solver ---
def intrinsics_to_dict(intr):
    """The pyrealsense2 intrinsics we actually streamed, as plain YAML data."""
    return {'fx': float(intr.fx), 'fy': float(intr.fy),
            'ppx': float(intr.ppx), 'ppy': float(intr.ppy),
            'width': int(intr.width), 'height': int(intr.height)}


def merge_into_yaml(output_file, extra):
    """Add keys to a calibration YAML without dropping what is already there.

    The extrinsic is expensive to produce -- 39 poses and a board in the gripper --
    so writing the intrinsics must never be able to clobber it.
    """
    data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = yaml.safe_load(f) or {}
    data.update(extra)
    with open(output_file, 'w') as f:
        yaml.dump(data, f)
    return data


def dump_intrinsics(calib_files):
    """Write the live colour-stream intrinsics into the calibration files.

    Hand-eye reads these off the stream to solve PnP and then throws them away, so
    the calibration files have only ever carried extrinsics. Everything that turns
    a pixel into a position on the table needs the intrinsic too, and until it is
    saved the simulator and the offline tests run on a NOMINAL one -- which is what
    decides, for example, whether the base-to-base perception crop fits the sensor.

    Needs no ChArUco board and moves no arm: it starts the stream, reads the
    profile and stops.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    try:
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().intrinsics)
        block = intrinsics_to_dict(intr)
        print("\nColour stream intrinsics: fx={fx:.2f} fy={fy:.2f} "
              "ppx={ppx:.2f} ppy={ppy:.2f} ({width}x{height})".format(**block))
        for path in calib_files:
            merge_into_yaml(path, {'intrinsics': block})
            print("  wrote intrinsics into {}".format(path))
    finally:
        pipeline.stop()


def run_hand_eye(samples, output_file, intrinsics=None):
    if len(samples) < 3:
        print(f"Not enough samples for {output_file}. Need at least 3 valid detections.")
        return

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    print(f"\nProcessing {len(samples)} samples for {output_file}...")

    for s in samples:
        T_base2gripper = pose_list_to_matrix(s['robot_pose'])
        T_gripper2base = invert_homogeneous_matrix(T_base2gripper)
        R_g, t_g = matrix_to_pose_lists(T_gripper2base)
        
        R_gripper2base_list.append(R_g)
        t_gripper2base_list.append(t_g)

        rvec = np.asarray(s['rvec']).reshape(3, 1)
        tvec = np.asarray(s['tvec']).reshape(3, 1)
        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_tc)
        t_target2cam_list.append(tvec.reshape(3,))

    print("Solving AX=XB (Eye-to-Hand) using Park & Martin...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    X = np.eye(4)
    X[:3, :3] = R_cam2base
    X[:3, 3] = t_cam2base.reshape(3,)

    if X[2, 3] < 0.5:
        print("Warning: Result seems inverted. Inverting matrix...")
        X = invert_homogeneous_matrix(X)

    out = {
        'camera_to_base': {'matrix': X.tolist()},
        'meta': {'samples': len(samples), 'board_type': 'charuco'}
    }
    # Save the intrinsic ALONGSIDE the extrinsic it was solved with. They are one
    # calibration: a pixel only becomes a position on the table through both, and
    # keeping them in separate places is how they drift apart.
    if intrinsics is not None:
        out['intrinsics'] = intrinsics

    with open(output_file, 'w') as f:
        yaml.dump(out, f)

    np.set_printoptions(precision=6, suppress=True)
    print(f"\nCalibration Result (T_base^camera) for {output_file}:\n")
    print(X)
    print(f"\nWrote calibration to {output_file}")

# --- Custom YAML Saver ---
def save_poses_yaml(filename, ip, gripper, sq_x, sq_y, sq_size, mk_size, poses_rad):
    poses_deg = [[round(math.degrees(rad), 4) for rad in pose] for pose in poses_rad]
    
    with open(filename, 'w') as f:
        f.write(f'robot:\n  ip: "{ip}"\n  gripper: "{gripper}"\n\n')
        f.write(f'# ChArUco Board Settings\n')
        f.write(f'board:\n')
        f.write(f'  squares_x: {sq_x}\n')
        f.write(f'  squares_y: {sq_y}\n')
        f.write(f'  square_length: {sq_size}\n')
        f.write(f'  marker_length: {mk_size}\n\n\n')
        f.write(f'# =============================================================================\n')
        f.write(f'# CALIBRATION POSES\n')
        f.write(f'# =============================================================================\n')
        f.write(f'# Format: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3] (Degrees)\n')
        f.write(f'# Total Poses: {len(poses_deg)}\n')
        f.write(f'poses:\n')
        for p in poses_deg:
            f.write(f'  - {p}\n')

# --- Interactive Terminal Thread ---
def input_thread(state_dict):
    while state_dict['running']:
        cmd = input("\n[Terminal] Enter 'R' (Record Pose), 'F' (Finish Recording), or 'Q' (Quit): ").strip().upper()
        if cmd in ['R', 'F', 'Q']:
            state_dict['cmd'] = cmd
            if cmd in ['F', 'Q']:
                state_dict['running'] = False
                break

# --- Background Error Monitor Thread ---
def error_monitor_thread(arms, state_dict):
    """Silently catches Error 37 and forces arms back into free-drive."""
    while state_dict['running']:
        if state_dict.get('freedrive_active', False):
            for r in arms:
                if r.arm.error_code != 0:
                    err = r.arm.error_code
                    r.arm.clean_error()
                    if err == 37:
                        time.sleep(0.1)
                        r.arm.set_mode(2)
                        r.arm.set_state(0)
        time.sleep(0.1)

# --- Execution ---
def main():
    parser = argparse.ArgumentParser(description='Single-arm xArm Hand-to-Eye calibration using RealSense + ChArUco')
    parser.add_argument('--mode', type=str, choices=['auto', 'manual'], default='auto', help='Run mode: auto uses saved poses if available, manual forces recording.')
    parser.add_argument('--arm', type=str, choices=['left', 'right'], default='left', help='Which arm to calibrate (left or right)')
    parser.add_argument('--dump-intrinsics', action='store_true',
                        help='Write the live colour-stream intrinsics into both '
                             'calib YAMLs and exit. No board, no arm motion.')
    args = parser.parse_args()

    calib_dir = "real_robot/calibration"
    os.makedirs(calib_dir, exist_ok=True)

    pose_file = os.path.join(calib_dir, f"xarm-{args.arm}-poses.yaml")
    calib_file = os.path.join(calib_dir, f"xarm-{args.arm}-calib.yaml")

    # Before anything is connected or powered: this needs only the camera.
    if args.dump_intrinsics:
        dump_intrinsics([os.path.join(calib_dir, "xarm-left-calib.yaml"),
                         os.path.join(calib_dir, "xarm-right-calib.yaml")])
        return

    poses_exist = os.path.exists(pose_file)
    recorded_poses = []
    
    # 1. Load configuration (from YAML if it exists, otherwise use defaults)
    if poses_exist:
        print(f"Loading configuration from {pose_file}...")
        with open(pose_file, 'r') as f:
            data = yaml.safe_load(f)
            
        target_ip = data.get('robot', {}).get('ip', DEFAULT_LEFT_IP if args.arm == 'left' else DEFAULT_RIGHT_IP)
        sq_x = data.get('board', {}).get('squares_x', DEFAULT_SQUARES_X)
        sq_y = data.get('board', {}).get('squares_y', DEFAULT_SQUARES_Y)
        sq_size = data.get('board', {}).get('square_length', DEFAULT_SQUARE_SIZE)
        mk_size = data.get('board', {}).get('marker_length', DEFAULT_MARKER_SIZE)
        
        # Load poses only if we aren't forcing a manual overwrite
        if args.mode == 'auto':
            poses_deg = data.get('poses', [])
            recorded_poses = [[math.radians(deg) for deg in p] for p in poses_deg]
    else:
        target_ip = DEFAULT_LEFT_IP if args.arm == 'left' else DEFAULT_RIGHT_IP
        sq_x = DEFAULT_SQUARES_X
        sq_y = DEFAULT_SQUARES_Y
        sq_size = DEFAULT_SQUARE_SIZE
        mk_size = DEFAULT_MARKER_SIZE

    # 2. Start RealSense Pipeline
    print("Starting RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 3. Configure OpenCV ChArUco Detectors
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard((sq_x, sq_y), sq_size, mk_size, aruco_dict)
        charuco_detector = cv2.aruco.CharucoDetector(board)
    except AttributeError:
        print("Error: OpenCV version mismatch. Please ensure opencv-contrib-python is installed.")
        sys.exit(1)

    # 4. Connect to BOTH Arms
    print(f"Connecting to left arm at {DEFAULT_LEFT_IP} and right arm at {DEFAULT_RIGHT_IP}...")
    left_arm = XArmLite6(DEFAULT_LEFT_IP, gripper='lite6', side='left', walls=False)
    right_arm = XArmLite6(DEFAULT_RIGHT_IP, gripper='lite6', side='right', walls=False)

    arms = [left_arm, right_arm]
    target_robot = left_arm if args.arm == 'left' else right_arm

    for r in arms:
        r.arm.clean_error()
        r.arm.clean_warn()
        r.arm.motion_enable(enable=True)
        r.arm.set_mode(0)
        r.arm.set_state(0)

    # 5. Start Background Error Monitor
    # This thread will keep the arms from locking up when payloads change in free-drive
    monitor_state = {'running': True, 'freedrive_active': False}
    err_thread = threading.Thread(target=error_monitor_thread, args=(arms, monitor_state), daemon=True)
    err_thread.start()

    # 6. Pre-Grasp Positioning (BOTH Arms in Free-Drive)
    print(f"\nOpening {args.arm} gripper...")
    target_robot.open_gripper()
    time.sleep(1.0) 

    input("\n>>> FIRMLY HOLD BOTH ARMS to support their weight, then press ENTER to enable Free-Drive: ")

    for r in arms:
        r.arm.clean_error()
        r.arm.set_mode(2)
        r.arm.set_state(0)
    
    # Activate auto-error-clearing
    monitor_state['freedrive_active'] = True

    if args.mode == 'auto' and poses_exist:
        input(f"\n>>> File loaded. Arrange BOTH arms, place the ChArUco board into the {args.arm} gripper, and press ENTER to close it: ")
    else:
        input(f"\n>>> Arrange BOTH arms, place the ChArUco board into the {args.arm} gripper, and press ENTER to close it: ")

    print(f"Closing {args.arm} gripper...")
    target_robot.close_gripper()
    time.sleep(1.0)

    # 7. Manual Recording Mode Logic
    if args.mode == 'manual' or not recorded_poses:
        if args.mode == 'manual':
            print(f"\nManual mode explicitly selected. Entering manual recording mode.")
        else:
            print(f"\nNo poses loaded. Falling back to manual recording mode.")
            
        print(f"\n*** BOTH arms are currently in Free-Drive. Move the {args.arm} arm to desired poses. ***")

        state = {'running': True, 'cmd': None}
        ui_thread = threading.Thread(target=input_thread, args=(state,), daemon=True)
        ui_thread.start()

        try:
            while state['running']:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    continue

                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    cv2.imshow('Camera View (Terminal is Active)', color_img)
                    cv2.waitKey(1)
                
                if state['cmd'] == 'R':
                    code, q = target_robot.arm.get_servo_angle(is_radian=True)
                    if code == 0:
                        recorded_poses.append(np.array(q).tolist())
                        print(f"✅ Pose #{len(recorded_poses)} recorded.")
                    else:
                        print(f"❌ Failed to read joints (code {code}).")
                    state['cmd'] = None

        finally:
            monitor_state['freedrive_active'] = False
            print("\nFree-drive off, position mode restored.")
            for r in arms:
                r.arm.set_mode(0)
                r.arm.set_state(0)
            time.sleep(0.5)
            
        if state['cmd'] == 'Q':
            print("Quitting without sampling.")
            monitor_state['running'] = False
            pipeline.stop()
            cv2.destroyAllWindows()
            target_robot.open_gripper()
            for r in arms:
                r.disconnect()
            return

        if len(recorded_poses) > 0:
            ans = input(f"\nDo you want to save these {len(recorded_poses)} joint poses to {pose_file}? (y/n): ").strip().lower()
            if ans == 'y':
                print(f"Saving configuration and {len(recorded_poses)} poses to {pose_file}...")
                save_poses_yaml(pose_file, target_ip, 'lite6', sq_x, sq_y, sq_size, mk_size, recorded_poses)
        else:
            print("No poses recorded. Exiting.")
            monitor_state['running'] = False
            pipeline.stop()
            cv2.destroyAllWindows()
            target_robot.open_gripper()
            for r in arms:
                r.disconnect()
            return
    else:
        # If we loaded from file, we lock the arms here before sampling
        monitor_state['freedrive_active'] = False
        print("\nFree-drive off, position mode restored.")
        for r in arms:
            r.arm.set_mode(0)
            r.arm.set_state(0)
        time.sleep(0.5)

    # 8. Automatic Execution and Sampling
    print(f"\nStarting automatic sampling. Targets: {len(recorded_poses)}")
    samples = []
    stream_intrinsics = None
    
    for i, q in enumerate(recorded_poses):
        print(f"\nMoving {args.arm} arm to pose {i+1}/{len(recorded_poses)}...")
        target_robot.movej(q, blocking=True)
        
        print("Waiting for arm to settle and camera auto-exposure to adjust...")
        for _ in range(60):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    cv2.imshow('Camera View', color_img)
                    cv2.waitKey(1)
            except RuntimeError:
                pass
        
        pose_tcp = target_robot.get_tcp_pose()
        
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            print("Warning: RealSense frame timeout, skipping capture for this pose.")
            continue

        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            continue
            
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        stream_intrinsics = intrinsics_to_dict(intr)
        camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        dist_coeffs = np.zeros((5, 1), dtype=float) 

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        active_board = board
        
        if marker_corners is not None and len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(color, marker_corners, marker_ids)
            
            # --- AUTO-ROTATION FALLBACK ---
            if charuco_corners is None or len(charuco_corners) < 4:
                swapped_board = cv2.aruco.CharucoBoard((sq_y, sq_x), sq_size, mk_size, aruco_dict)
                swapped_detector = cv2.aruco.CharucoDetector(swapped_board)
                charuco_corners_swap, charuco_ids_swap, _, _ = swapped_detector.detectBoard(gray)
                
                if charuco_corners_swap is not None and len(charuco_corners_swap) >= 4:
                    print("🔄 Automatically swapped squares_x and squares_y to match board orientation.")
                    charuco_corners = charuco_corners_swap
                    charuco_ids = charuco_ids_swap
                    active_board = swapped_board

        if charuco_corners is not None and len(charuco_corners) >= 4:
            all_obj_points = active_board.getChessboardCorners()
            obj_points = all_obj_points[charuco_ids.flatten()]
            
            valid, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, camera_matrix, dist_coeffs)
            if valid:
                samples.append({'robot_pose': pose_tcp, 'rvec': rvec.flatten().tolist(), 'tvec': tvec.flatten().tolist()})
                print(f"✅ Sample {i+1} captured.")
                cv2.drawFrameAxes(color, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                cv2.aruco.drawDetectedCornersCharuco(color, charuco_corners, charuco_ids)
            else:
                print("❌ Pose estimation failed.")
        else:
            num_found = len(charuco_corners) if charuco_corners is not None else 0
            print(f"❌ Not enough ChArUco corners (Found {num_found}, Need 4+).")

        cv2.imshow('Camera View', color)
        cv2.waitKey(1500)

    # 9. Process Solutions
    if samples:
        run_hand_eye(samples, calib_file, intrinsics=stream_intrinsics)

    print("\nCalibration sequence complete.")
    print(f"Opening {args.arm} gripper to release board...")
    target_robot.open_gripper()
    time.sleep(1.0)
    
    monitor_state['running'] = False
    pipeline.stop()
    cv2.destroyAllWindows()
    for r in arms:
        r.disconnect()

if __name__ == "__main__":
    main()