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

# --- SETUP SIMULATION ---
#p.connect(p.DIRECT)  # Use p.GUI to see it visually
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 1. Load Plane
plane = p.loadURDF("plane.urdf")

# 2. Load and Configure Table
# Standard table.urdf in pybullet_data is approx 1.5m long on X-axis.
# We want 2.0m length, so we scale it: 2.0 / 1.5 ≈ 1.333
TARGET_LENGTH = 2.0
DEFAULT_LEN = 1.5
table_scale = TARGET_LENGTH / DEFAULT_LEN

# We rotate the table 90 degrees (PI/2) around Z so its length aligns with the Y-axis
# (where the robots will be positioned).
table_orn = p.getQuaternionFromEuler([0, 0, np.pi/2])

table = p.loadURDF(
    "table/table.urdf",
    basePosition=[0, 0, 0],
    baseOrientation=table_orn,
    globalScaling=table_scale
)



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
         0.65*table_scale]
    )
    cube_ids.append(cube_id)

    color = colors[i % len(colors)]
    p.changeVisualShape(
        cube_id,
        linkIndex=-1,
        rgbaColor=color
    )

# 3. Handle Xacro -> URDF Conversion
project_path = "/home/hcv530/project/universal_robot/ur_description/urdf/"
ur5_xacro  = os.path.join(project_path, "ur5e.xacro")
ur5_urdf   = os.path.join(project_path, "ur5e.urdf")
ur16_xacro = os.path.join(project_path, "ur16e.xacro")
ur16_urdf  = os.path.join(project_path, "ur16e.urdf")

# Helper to convert if missing
def ensure_urdf(xacro, urdf):
    if not os.path.exists(urdf):
        print(f"Generating {urdf} from {xacro}...")
        doc = XacroDoc.from_file(xacro)
        doc.to_urdf_file(urdf)

ensure_urdf(ur5_xacro, ur5_urdf)
ensure_urdf(ur16_xacro, ur16_urdf)

# 4. Load Robots Face-to-Face
p.setAdditionalSearchPath("/home/hcv530/project/universal_robot/ur_description")

# Separation needs to be slightly more than table length (2.0m) to avoid clipping
ROBOT_SEPARATION = 1.7 

# Robot 0: Positioned at negative Y, Facing Positive Y (Center)
pos_0 = [0, -ROBOT_SEPARATION / 2, 0.65*table_scale] # 0.65 matches table height approx
orn_0 = p.getQuaternionFromEuler([0, 0, np.pi/2]) # Rotate 90 deg to face +Y

# Robot 1: Positioned at positive Y, Facing Negative Y (Center)
pos_1 = [0, ROBOT_SEPARATION / 2, 0.65*table_scale]
orn_1 = p.getQuaternionFromEuler([0, 0, -np.pi/2]) # Rotate -90 deg to face -Y

print("Loading Robots...")
robot0 = p.loadURDF(ur5_urdf, pos_0, orn_0, useFixedBase=True)
robot1 = p.loadURDF(ur16_urdf, pos_1, orn_1, useFixedBase=True)

# 5. Set "Home" Configuration
# Standard "Ready" pose for UR robots (Upright with elbow bent)
# Joints: [Shoulder Pan, Shoulder Lift, Elbow, Wrist 1, Wrist 2, Wrist 3]
home_joints = [0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

def reset_robot_home(robot_id):
    for i, angle in enumerate(home_joints):
        # UR robots usually have joints indexed 1-6 (depending on URDF base link)
        # We try setting indices 0-5 or 1-6 based on URDF structure
        p.resetJointState(robot_id, i+1, angle)

reset_robot_home(robot0)
reset_robot_home(robot1)





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


# NUM_VIEWS = 6
# min_height, max_height = 2, 2.5  # Varying heights (Z)
# min_radius, max_radius = 0.3, 0.6  # Varying distance from center

# perception_poses = []

# for _ in range(NUM_VIEWS):
#     # Random angle around the table (0 to 360 degrees)
#     theta = np.random.uniform(0, 2 * np.pi)
    
#     # Random distance and height
#     r = np.random.uniform(min_radius, max_radius)
#     z = np.random.uniform(min_height, max_height)
    
#     # Convert polar to cartesian
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
    
#     perception_poses.append([x, y, z])



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


# --- CONFIGURATION ---
EE_LINK_INDEX = 7 
# Offset so the camera "sees" past the gripper/wrist
CAMERA_OFFSET = [0, 0, 0.1]  # 10cm out from the wrist
# Looking "forward" from the tool's perspective
CAMERA_ORN = p.getQuaternionFromEuler([0, 0, 0]) 

def get_camera_pos_from_ee(robot_id):
    """Calculates world-space camera position based on current EE link state."""
    state = p.getLinkState(robot_id, EE_LINK_INDEX)
    ee_pos, ee_orn = state[0], state[1]
    cam_pos, cam_orn = p.multiplyTransforms(ee_pos, ee_orn, CAMERA_OFFSET, CAMERA_ORN)
    return cam_pos, cam_orn

# --- PERCEPTION LOOP ---
# Instead of moving a sphere, we move robot0 (UR5e)
perception_poses = [
    [0.2, -0.5, 1.2],  # Sample points around home position
    [-0.2, -0.5, 1.2],
    [0.0, -0.3, 1.1],
    [0.3, -0.4, 1.3],
    [-0.3, -0.4, 1.3],
]

# --- REFINED PERCEPTION LOOP ---
# Orientation for "Pointed Straight Down"
# Roll = 180 deg (pi), Pitch = 0, Yaw = 0 (relative to robot base)
DOWNWARD_ORN = p.getQuaternionFromEuler([-np.pi, 0, 0])

for i, target_pos in enumerate(perception_poses):
    # 1. Compute IK with fixed orientation
    # We pass DOWNWARD_ORN to the solver to force the EE to stay level
    joint_poses = p.calculateInverseKinematics(
        robot0, 
        EE_LINK_INDEX, 
        target_pos, 
        targetOrientation=DOWNWARD_ORN,
        residualThreshold=1e-5 # Increase precision for orientation
    )

    # 2. Reset joints to the IK solution
    movable_joint_indices = [j for j in range(p.getNumJoints(robot0)) 
                             if p.getJointInfo(robot0, j)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]
    
    for j, joint_idx in enumerate(movable_joint_indices):
        p.resetJointState(robot0, joint_idx, joint_poses[j])
        
    # 3. Capture image
    cam_pos, cam_orn = get_camera_pos_from_ee(robot0)
    
    # IMPORTANT: Since we want a perfect top-down view, 
    # the target should be directly below the camera.
    # We take cam_pos and just change the Z to the table height.
    #look_at_target = [cam_pos[0], cam_pos[1], 0.65] 

    # 1. Convert orientation to a rotation matrix
    rot_matrix = p.getMatrixFromQuaternion(cam_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # 2. Define 'Forward' (usually -Z in camera convention or +Z in robot tool convention)
    # Let's assume the camera looks along the tool's Z-axis:
    forward_vec = rot_matrix @ [0, 0, 1] 

    # We calculate the target point slightly in front of the camera for the arrow
    # (e.g., 0.3 meters in front)
    arrow_target = cam_pos + (forward_vec * 0.3)

    # VISUALIZE THE ARROW
    # Note: This will only be visible if you are using p.connect(p.GUI)
    visualize_camera(cam_pos, arrow_target, color=[1, 0, 0], lifeTime=5.0)


    # 3. Target is camera position + forward vector
    target_pos = cam_pos + forward_vec

    view = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=[0, 0, 1]
    )

    proj = p.computeProjectionMatrixFOV(FOV, IMG_W / IMG_H, NEAR, FAR)
    img = p.getCameraImage(IMG_W, IMG_H, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (IMG_H, IMG_W, 4))[:, :, :3]
    
    # 4. Save and Fuse
    cv2.imwrite(f"./tmp/captures/robot_view_{i}.png", rgb)
    fuse(rgb, view)

    ref_img = capture_reference_view()
    cv2.imwrite(f"./tmp/captures/reference_scene_{i}.png", ref_img)


# for i, pos in enumerate(perception_poses):
#     p.resetBasePositionAndOrientation(wrist, pos, [0, 0, 0, 1])
#     rgb, view = capture_wrist_camera(wrist)
#     cv2.imwrite(f"./tmp/captures/view_{i}.png", rgb)
#     fuse(rgb, view)


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