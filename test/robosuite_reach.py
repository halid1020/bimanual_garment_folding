import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import imageio
from scipy.spatial.transform import Rotation as R
import os  # Added for creating the video directory

# -------------------------------
# 1. Load and fix controller configuration
# -------------------------------
print("Loading controller config...")
controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["control_delta"] = False      # Use absolute commands
# We REMOVE controller_config["control_ori"] = False
# OSC_POSE defaults to control_ori=True, so it will now expect a 7-dim action
# (pos 3, ori 3, gripper 1)
# --- EDITED: Increased kp for better orientation holding ---
controller_config["kp"] = 1000
controller_config["damping_ratio"] = 1.0

# -------------------------------
# 2. Create environment (offscreen)
# -------------------------------
print("Creating environment...")
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=False,            # We don't need camera obs, we'll render manually
    control_freq=20,
    camera_names="frontview",
    camera_heights=480,
    camera_widths=640,
)

print("Resetting environment...")
obs = env.reset()

# -------------------------------
# 3. Define absolute target pose
# -------------------------------
# Get the initial end-effector pose from the simulation
world_pos = env.sim.data.get_site_xpos("gripper0_grip_site")
world_rotmat = env.sim.data.get_site_xmat("gripper0_grip_site")

# Define a target position (e.g., move up and over)
target_pos = world_pos + np.array([0.2, 0.2, 0.2])

# --- CORRECTED ORIENTATION TARGET ---
# We want to maintain the initial orientation.
# The controller expects an AXIS-ANGLE vector, not Euler angles.
target_rotation = R.from_matrix(world_rotmat)
target_axis_angle = target_rotation.as_rotvec()  # This is the 3D vector we need

# For debugging, let's also store the Euler angles version
target_euler_deg = target_rotation.as_euler('xyz', degrees=True)
# --- END CORRECTION ---

# Define thresholds for success
pos_thresh = 1e-3   # meters
# --- EDITED: Threshold changed for quaternion vector norm (sin(theta/2))
ori_thresh = 0.005  # Quaternion vector norm (approx 0.005 for a 0.01 radian angular error)
# --- END EDITED ---

# -------------------------------
# 4. Control loop + video recording
# -------------------------------
frames = []
num_steps = 300
# Initialize success flags before the loop
pos_reached, ori_reached = False, False
print("\nStarting control loop...\n")

for step in range(num_steps):
    # Get current EE pose from observations
    ee_pos = obs["robot0_eef_pos"]
    ee_quat = obs["robot0_eef_quat"] # This is in (w, x, y, z) format from MuJoCo

    # Get "ground truth" EE pose from sim (for debugging)
    # We keep this just to see the raw sim data if we want, but we won't use it for error
    sim_world_pos = env.sim.data.get_site_xpos("gripper0_grip_site")
    sim_world_rotmat = env.sim.data.get_site_xmat("gripper0_grip_site")

    # --- EDITED: ERROR CALCULATION (QUATERNION SPACE) ---
    # Compute orientation error using quaternion vector norm
    # SciPy's R.from_quat expects (x, y, z, w)
    ee_rot = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
    
    # rot_diff represents q_error = q_current_inverse * q_target
    rot_diff = ee_rot.inv() * target_rotation
    
    # Get the quaternion (x, y, z, w) and extract the vector part (x, y, z)
    q_error_xyzw = rot_diff.as_quat()
    q_error_vector = q_error_xyzw[:3]
    
    # The magnitude of the quaternion error vector is sin(theta_error / 2)
    quaternion_vector_norm_error = np.linalg.norm(q_error_vector)
    
    pos_error_m = np.linalg.norm(ee_pos - target_pos)
    # --- END EDITED ---

    # --- GET OBS-BASED ROTATIONS FOR PRINTING ---
    # Get the rotation from obs (which is used for error calc)
    # and convert it to the same formats we use for the target.
    obs_euler_deg = ee_rot.as_euler('xyz', degrees=True)
    obs_axis_angle = ee_rot.as_rotvec()
    # --- NEW: Quaternions for printing ---
    current_quat_xyzw = ee_rot.as_quat()
    target_quat_xyzw = target_rotation.as_quat()
    # --- END NEW ---
    # --- END GET OBS-BASED ROTATIONS ---

    # --- ENHANCED DEBUGGING MESSAGES ---
    print(f"Step {step:03d}")
    print(f"  EE (obs) pos:   {np.round(ee_pos, 4)}")
    print(f"  Target pos:     {np.round(target_pos, 4)}")
    print(f"  Position error (m): {pos_error_m:.5f}")
    print("-" * 60)
    
    # --- NEW QUATERNION PRINTS ---
    print(f"  Current EE Ori (Quat WXYZ - raw obs): {np.round(ee_quat, 4)}")
    print(f"  Current EE Ori (Quat XYZW):          {np.round(current_quat_xyzw, 4)}")
    # --- END NEW QUATERNION PRINTS ---

    # Current Orientation (FROM OBS, to match error calculation)
    print(f"  Current EE Ori (Euler deg):          {np.round(obs_euler_deg, 2)}")
    print(f"  Current EE Ori (Axis-Angle rad):     {np.round(obs_axis_angle, 3)}")
    
    # Target Orientation
    print(f"  Target EE Ori (Quat XYZW):           {np.round(target_quat_xyzw, 4)}")
    print(f"  Target EE Ori (Euler deg):           {np.round(target_euler_deg, 2)}")
    print(f"  Target EE Ori (Axis-Angle rad):      {np.round(target_axis_angle, 3)}")
    
    # Error
    print(f"  Orientation error (quat norm): {quaternion_vector_norm_error:.5f}") # Changed print label
    print("=" * 60)
    # --- END DEBUGGING MESSAGES ---

    # --- CORRECTED ACTION ---
    # Action = [x, y, z, ax, ay, az, gripper]
    action = np.concatenate([target_pos, target_axis_angle, [0]]) # Gripper closed
    # --- END CORRECTION ---

    obs, reward, done, info = env.step(action)

    # Render frame
    frame = env.sim.render(camera_name="frontview", height=480, width=640)
    frame = np.flipud(frame) # Flip vertically for correct orientation
    frames.append(frame)

    # --- CORRECTED BREAK CONDITION ---
    # Check both position and orientation
    pos_reached = pos_error_m < pos_thresh
    ori_reached = quaternion_vector_norm_error < ori_thresh
    # --- END CORRECTION ---

    if pos_reached and ori_reached:
        print(f"\n✅ Target position and orientation reached at step {step}!")
        break

if not (pos_reached and ori_reached):
    print(f"\n⚠️ Loop finished after {num_steps} steps, target not reached.")

# -------------------------------
# 5. Save video
# -------------------------------
video_path = "tmp/ee_absolute_axisangle.mp4"
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(video_path), exist_ok=True)

print(f"\nSaving video ({len(frames)} frames) to: {video_path}")
imageio.mimwrite(video_path, frames, fps=20, macro_block_size=1)
env.close()

print(f"\n✅ Video saved successfully.")
