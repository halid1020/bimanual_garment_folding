import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import imageio
from scipy.spatial.transform import Rotation as R

# -------------------------------
# 1. Load and fix controller configuration
# -------------------------------
controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["control_delta"] = False      # absolute commands
controller_config["control_ori"] = False
controller_config["kp"] = 200
controller_config["damping_ratio"] = 1.0

# -------------------------------
# 2. Create environment (offscreen)
# -------------------------------
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    control_freq=20,
    camera_names="frontview",
    camera_heights=480,
    camera_widths=640,
)

obs = env.reset()

# -------------------------------
# 3. Define absolute target pose
# -------------------------------
world_pos = env.sim.data.get_site_xpos("gripper0_grip_site")
world_rotmat = env.sim.data.get_site_xmat("gripper0_grip_site")
world_euler = R.from_matrix(world_rotmat).as_euler('xyz', degrees=True)


# Move 10 cm upward
target_pos = world_pos + np.array([0.0, 0.0, 0.01])

# Convert to degrees
target_aa_deg = world_euler

target_euler_rad = np.deg2rad(target_aa_deg).flatten()

print(target_pos, target_euler_rad)

pos_thresh = 1e-3   # meters
ori_thresh = 0.01   # radians (~0.57 deg)

# -------------------------------
# 4. Control loop + video recording
# -------------------------------
frames = []
num_steps = 300
print("\nStarting control loop...\n")

for step in range(num_steps):
    # Action = [x, y, z, ax, ay, az]
     # Get world-frame EE pose from sim
    world_pos = env.sim.data.get_site_xpos("gripper0_grip_site")
    world_rotmat = env.sim.data.get_site_xmat("gripper0_grip_site")
    world_euler = R.from_matrix(world_rotmat).as_euler('xyz', degrees=True)
    ee_pos = obs["robot0_eef_pos"]

    ee_pos = obs["robot0_eef_pos"]
    ee_quat = obs["robot0_eef_quat"]

    # Compute orientation error
    ee_rot = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
    target_rot = R.from_rotvec(target_euler_rad)
    rot_diff = ee_rot.inv() * target_rot
    angle_error = rot_diff.magnitude()

    print(f"Step {step:03d}")
    print(f"  EE (obs) pos:   {ee_pos}")
    print(f"  EE (sim) world pos: {world_pos}")
    print(f"  EE (sim) world ori [deg XYZ]: {world_euler}")
    print("Target position (world):", target_pos)
    print("Target orientation (axis–angle, degree):", target_aa_deg)
    print("-" * 60)

    action = np.concatenate([target_pos, target_euler_rad, [0]])
    obs, reward, done, info = env.step(action)

    

   

    

    frame = env.sim.render(camera_name="frontview", height=480, width=640)
    frame = np.flipud(frame)
    frames.append(frame)

    if np.linalg.norm(ee_pos - target_pos) < pos_thresh and angle_error < ori_thresh:
        print("✅ Target position and orientation reached!")
        break

# -------------------------------
# 5. Save video
# -------------------------------
video_path = "tmp/ee_absolute_axisangle.mp4"
imageio.mimwrite(video_path, frames, fps=20)
env.close()

print(f"\n✅ Video saved to: {video_path}")
