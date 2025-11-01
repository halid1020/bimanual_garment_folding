import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import imageio

# -------------------------------
# 1. Load controller config
# -------------------------------
controller_config = load_controller_config(default_controller="OSC_POSE")

# Use absolute position control, no orientation control
controller_config["control_delta"] = False
controller_config["use_delta"] = False
controller_config["control_ori"] = False  # disable orientation control
# controller_config["kp"] = 200
# controller_config["damping_ratio"] = 1.0

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
# 3. Define target (absolute position)
# -------------------------------
init_pos = obs["robot0_eef_pos"]
target_pos = init_pos + np.array([0.0, 0.0, 0.0])  # same position (EE should not move)

print("Initial position (world):", init_pos)
print("Target position (world):", target_pos)

# -------------------------------
# 4. Control loop + record video
# -------------------------------
frames = []
num_steps = 200

for step in range(num_steps):
    # Action = [x, y, z] (absolute)
    action = target_pos
    obs, reward, done, info = env.step(action)

    ee_pos = obs["robot0_eef_pos"]
    world_pos = env.sim.data.get_site_xpos("gripper0_grip_site")

    print(f"Step {step:03d}: EE pos (obs) = {ee_pos}, world pos = {world_pos}")

    # Render and record
    frame = env.sim.render(camera_name="frontview", height=480, width=640)
    frame = np.flipud(frame)
    frames.append(frame)

video_path = "tmp/ee_fixed_absolute_no_flip.mp4"
imageio.mimwrite(video_path, frames, fps=20)
env.close()

print(f"\n✅ Video saved to: {video_path}")
