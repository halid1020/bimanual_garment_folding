import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os
from perception.utils import *


"""
In this work, we simulate a tabletop environment in the PyBullet physics engine to generate 
both fused and ground-truth visual representations. A small sphere is used as a simplified 
proxy for the wrist-mounted camera, which is moved through multiple predefined viewpoints above
the table to capture RGB images of randomly placed, colored objects. These images are subsequently 
projected onto a common top-down plane using ray–plane intersection, producing a fused perceptive 
map that approximates a bird’s-eye view of the scene. To provide a quantitative reference, a 
virtual camera is positioned directly above the table to capture a ground-truth top-down image. 
All intermediate and final outputs—including individual wrist-sphere views, the fused top-down 
map, and the ground-truth reference—are stored for subsequent analysis and validation of the fusion 
procedure. This simplified setup enables systematic evaluation of multi-view perception strategies 
in simulated robotic environments while reducing complexity associated with full manipulator models.

"""

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf", [0, 0, 0])

robot0_base = [0, -0.85, 0]
robot1_base = [0,  0.85, 0]

# ur5_path = "ur5/ur5.urdf"

# robot0 = p.loadURDF(ur5_path, robot0_base, useFixedBase=True)
# robot1 = p.loadURDF(ur5_path, robot1_base, useFixedBase=True)

wrist = p.loadURDF(
    "sphere_small.urdf",
    [0, 0.85, 1.0],   # right-arm side
    useFixedBase=True
)

cube_ids = []

colors = [
    [1, 0, 0, 1],   # red
    [0, 1, 0, 1],   # green
    [0, 0, 1, 1],   # blue
    [1, 1, 0, 1],   # yellow
    [1, 0, 1, 1],   # magenta
    [0, 1, 1, 1],   # cyan
]

for i in range(6):
    cube_id = p.loadURDF(
        "cube_small.urdf",
        [np.random.uniform(-0.4, 0.4),
         np.random.uniform(-0.4, 0.4),
         0.65]
    )
    cube_ids.append(cube_id)

    color = colors[i % len(colors)]
    p.changeVisualShape(
        cube_id,
        linkIndex=-1,
        rgbaColor=color
    )





# EE_LINK = 7

# CAMERA_OFFSET = [0, 0, 0.08]
# CAMERA_ROT = p.getQuaternionFromEuler([np.pi, 0, 0])

# perception_poses = [
#     [0.4, 0.0, 1.0],
#     [0.2, 0.3, 1.0],
#     [-0.2, 0.3, 1.0],
#     [-0.4, 0.0, 1.0],
#     [0.2, -0.3, 1.0],
#     [-0.2, -0.3, 1.0],
# ]


NUM_VIEWS = 6
min_height, max_height = 0.9, 1.3  # Varying heights (Z)
min_radius, max_radius = 0.3, 0.6  # Varying distance from center

perception_poses = []

for _ in range(NUM_VIEWS):
    # Random angle around the table (0 to 360 degrees)
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Random distance and height
    r = np.random.uniform(min_radius, max_radius)
    z = np.random.uniform(min_height, max_height)
    
    # Convert polar to cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    perception_poses.append([x, y, z])



os.makedirs("./tmp/captures", exist_ok=True)
views = []

ref_img = capture_reference_view()
cv2.imwrite("./tmp/captures/reference_scene.png", ref_img)

gt = capture_topdown_groundtruth()
cv2.imwrite("./tmp/captures/ground_truth_topdown.png", gt)

# for i, pos in enumerate(perception_poses):
#     move_ee(robot1, pos)
#     rgb, view = capture_wrist_camera(robot1)

#     cv2.imwrite(f"captures/view_{i}.png", rgb)
#     views.append((rgb, view))




for i, pos in enumerate(perception_poses):
    p.resetBasePositionAndOrientation(wrist, pos, [0, 0, 0, 1])
    rgb, view = capture_wrist_camera(wrist)
    cv2.imwrite(f"./tmp/captures/view_{i}.png", rgb)
    fuse(rgb, view)


# for rgb, view in views:
#     fuse(rgb, view)

topdown = accum / np.maximum(weight[..., None], 1)
topdown = topdown.astype(np.uint8)
cv2.imwrite("./tmp/captures/fused_topdown.png", topdown)


# 1. Convert both to float32 and scale to [0, 1]
# Assuming 'gt' is your ground truth image from capture_topdown_groundtruth()
gt_float = gt.astype(np.float32) / 255.0
topdown_float = topdown.astype(np.float32) / 255.0

# 2. Calculate pixel-wise absolute difference
# This results in values between 0.0 (identical) and 1.0 (completely different)
diff_visual = ((gt_float - topdown_float) + 1)/2

# 5. Save the result
# Convert back to uint8 for saving (multiplying by 255)
diff_to_save = (diff_visual * 255).astype(np.uint8)
cv2.imwrite("./tmp/captures/difference_heatmap.png", diff_to_save)