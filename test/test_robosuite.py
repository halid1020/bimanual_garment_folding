import numpy as np
import robosuite as suite
import imageio
import argparse
import sys

# --- Valid robosuite tasks ---
VALID_TASKS = [
    "Lift",
    "Door",
    "PickPlace",
    "PickPlaceCan",
    "Wipe",
    "Stack",
    "NutAssembly",
    #   "Cleanup",
    #"PegInHole"
]

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Run a headless robosuite task and record video.")
parser.add_argument(
    "--task",
    type=str,
    default="Lift",
    help=f"Task name. Valid options: {', '.join(VALID_TASKS)}"
)
args = parser.parse_args()

# Normalize task name
task_name = args.task.replace(" ", "")  # remove spaces to match robosuite naming

if task_name not in VALID_TASKS:
    print(f"Error: Invalid task '{args.task}'. Valid options: {', '.join(VALID_TASKS)}")
    sys.exit(1)

print(f"Running task: {task_name}")

# --- Simulation setup (headless) ---
env = suite.make(
    env_name=task_name,
    robots="Panda",               # Robot model
    has_renderer=False,           # 🚫 No on-screen window
    has_offscreen_renderer=True,  # ✅ Offscreen rendering
    use_camera_obs=False,         # We’ll render manually
)

# --- Reset environment ---
obs = env.reset()

# --- Video setup ---
frames = []
num_steps = 200             # Length of simulation
width, height = 640, 480    # Resolution

# --- Run simulation loop ---
for i in range(num_steps):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)
    
    # Render a frame from a camera view
    frame = env.sim.render(
        camera_name="frontview", width=width, height=height
    )
    frames.append(frame)

    if done:
        env.reset()

# --- Save video ---
video_path = f"robosuite_{task_name}.mp4"
imageio.mimsave(video_path, frames, fps=30, format='ffmpeg')
env.close()

print(f"🎬 Video saved to {video_path}")
