# TODO: pip install pyrealsense2
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get intrinsics from color stream
color_stream = profile.get_stream(rs.stream.color)  # rs.video_stream_profile
intr = color_stream.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0,      0,      1]
])
dist_coeffs = np.array(intr.coeffs)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
