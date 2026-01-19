import cv2
import cv2.aruco
import numpy as np
import pyrealsense2 as rs
import yaml

def test_charuco_detection(config_file='ur5e.yaml'):
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Setup Realsense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(cfg)

    # Setup Board
    squares_x = config['board']['squares_x']
    squares_y = config['board']['squares_y']
    sq_len = config['board']['square_length']
    mrk_len = config['board']['marker_length']
    
    print(f"Testing Board Definition: {squares_x}x{squares_y} squares")
    print("Press 'q' to quit, 'r' to reload config after editing file.")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), sq_len, mrk_len, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)
            
            # Visualization
            if marker_corners:
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            
            if charuco_corners is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
                cv2.putText(frame, f"SUCCESS: {len(charuco_corners)} corners", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Markers found but Grid Failed", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Try swapping squares_x / squares_y in yaml", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Test Board', frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_charuco_detection()