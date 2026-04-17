import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==========================================
# 1. Configuration: Set your paths here
# ==========================================
# Replace this with the actual path to your episode directory
EPISODE_DIR = "/users/hcv530/scratch/garment_folding_data/gc_diff_mp_longsleve_canon_align_100_demo_one_hot_pred_sem_key_v5/eval_checkpoint_-2/episode_0" 
STEP_IDX = 0  # The step you want to inspect

step_dir = os.path.join(EPISODE_DIR, f"step_{STEP_IDX}")

if not os.path.exists(step_dir):
    print(f"Error: Directory {step_dir} does not exist.")
else:
    print(f"--- Inspecting {step_dir} ---")

    # ==========================================
    # 2. Load Data Helpers
    # ==========================================
    def safe_load_json(filename):
        path = os.path.join(step_dir, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def safe_load_npy(filename):
        path = os.path.join(step_dir, filename)
        if os.path.exists(path):
            return np.load(path, allow_pickle=True) 
        return None

    # Load Metadata
    info_data = safe_load_json('info.json')
    action_data = safe_load_json('action.json')

    # Load Arrays
    prim_probs = safe_load_npy('primitive_probabilities.npy')
    pred_kpts = safe_load_npy('predicted_keypoints.npy')
    gt_kpts = safe_load_npy('gt_keypoints.npy')
    noise_history = safe_load_npy('noise_actions_history.npy')

    # Load Image
    rgb_path = os.path.join(step_dir, 'rgb.png')
    if os.path.exists(rgb_path):
        img = cv2.imread(rgb_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.zeros((512, 512, 3), dtype=np.uint8) # Blank fallback

    # ==========================================
    # 3. Print Textual Metadata
    # ==========================================
    print("\n[ Step Info ]")
    print(json.dumps(info_data, indent=2) if info_data else "No info.json found.")
    
    print("\n[ Final Executed Action ]")
    print(json.dumps(action_data, indent=2) if action_data else "No action.json found.")
    
    if noise_history is not None:
        print(f"\n[ Denoising History ] Raw Shape: {noise_history.shape} (Steps x Batch x Horizon x Dim)")

    # ==========================================
    # 4. Visualization Dashboard (1x3 Grid)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ------------------------------------------
    # Plot 1: RGB & Keypoints (-1 to 1 Mapping)
    # ------------------------------------------
    ax_img = axes[0]
    ax_img.imshow(img)
    ax_img.set_title(f"Step {STEP_IDX} - RGB Observation")
    ax_img.axis("off")

    H, W = img.shape[:2]

    def plot_kpts(ax, kpts, color, label, marker='x'):
        if kpts is None or kpts.size == 0: return
        
        # Flatten to (-1, 2) to handle batch/horizon shapes automatically
        kpts_flat = kpts.reshape(-1, 2)
        
        for pt in kpts_flat:
            # Map from [-1, 1] to pixel coordinates [0, W] and [0, H]
            x_px = ((pt[0] + 1) / 2.0) * W
            y_px = ((pt[1] + 1) / 2.0) * H
            ax.plot(x_px, y_px, marker=marker, color=color, markersize=8, markeredgewidth=2, label=label)

    # Plot Ground Truth (Green Circles)
    plot_kpts(ax_img, gt_kpts, color='lime', label='GT Keypoints', marker='o')
    # Plot Predicted (Red Crosses)
    plot_kpts(ax_img, pred_kpts, color='red', label='Pred Keypoints', marker='x')

    # Fix duplicate labels in legend
    handles, labels = ax_img.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_img.legend(by_label.values(), by_label.keys(), loc='upper right')

    # ------------------------------------------
    # Plot 2: Primitive Probabilities
    # ------------------------------------------
    ax_prob = axes[1]
    if prim_probs is not None and prim_probs.ndim >= 1:
        probs = prim_probs.flatten()
        classes = [f"Prim {i}" for i in range(len(probs))]
        
        bars = ax_prob.bar(classes, probs, color='skyblue', edgecolor='black')
        ax_prob.set_ylim(0, 1.05)
        ax_prob.set_ylabel("Probability")
        ax_prob.set_title("Predicted Primitive Probabilities")
        ax_prob.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add text values on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    else:
        ax_prob.text(0.5, 0.5, "No Primitive Probabilities Found", ha='center', va='center')
        ax_prob.axis("off")

    # ------------------------------------------
    # Plot 3: Action Denoising Trajectories
    # ------------------------------------------
    ax_traj = axes[2]
    if noise_history is not None and noise_history.size > 0:
        # Standard shape expected: (DiffusionSteps, Batch, Horizon, Dim)
        # Squeeze out the batch dimension if it is 1
        nh = np.squeeze(noise_history, axis=1) if noise_history.shape[1] == 1 else noise_history
        
        # Safely grab dimensions
        num_diff_steps = nh.shape[0]
        step_array = np.arange(num_diff_steps)
        
        if nh.ndim >= 3:
            # We plot the evolution of the FIRST step in the action horizon (index 0)
            horizon_idx = 0 
            action_dims = nh.shape[-1]
            
            for d in range(action_dims):
                ax_traj.plot(step_array, nh[:, horizon_idx, d], label=f'Action Dim {d}', alpha=0.8, linewidth=2)
                
            ax_traj.set_title(f"Action Denoising Trajectory\n(Horizon Step {horizon_idx})")
            ax_traj.set_xlabel("Diffusion Step (0 = Pure Noise, Final = Prediction)")
            ax_traj.set_ylabel("Normalized Action Value")
            
            # Put legend outside if there are many dimensions
            ax_traj.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)
            ax_traj.grid(True, linestyle='--', alpha=0.6)
        else:
            ax_traj.text(0.5, 0.5, f"Unexpected Shape: {nh.shape}", ha='center', va='center')
            ax_traj.axis("off")
    else:
        ax_traj.text(0.5, 0.5, "No Denoising History Found", ha='center', va='center')
        ax_traj.axis("off")

    plt.tight_layout()
    plt.show()