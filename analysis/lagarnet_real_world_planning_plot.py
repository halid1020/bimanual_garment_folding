import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
data_dir = '/media/hcv530/T7/garment_folding_data/real_world/'
method_folder = 'final_lagarnet_reward_v2'
eval_dir = os.path.join(data_dir, method_folder, 'eval_checkpoint_-2') 

# Variable to control the final step to draw (0-indexed)
target_stop_step = 6 
num_steps_to_draw = target_stop_step + 1  # Total number of images per sequence

# Base output directory
output_dir = os.path.join(eval_dir, 'optimization_visualizations')
out_combined = os.path.join(output_dir, 'combined_grids')
os.makedirs(out_combined, exist_ok=True)

# Find and sort all episode folders
episode_folders = [f for f in os.listdir(eval_dir) if f.startswith('episode_')]
episode_folders.sort(key=lambda x: int(x.split('_')[1]))

print(f"Found {len(episode_folders)} episodes. Generating combined big figures...")

# --- Colormaps ---
time_cmap = mcolors.LinearSegmentedColormap.from_list("time_cmap", ["cyan", "magenta"])

# --- Vectorized Drawing Function ---
def draw_actions_vectorized(ax, actions, width, height, colors, alpha=0.9):
    if len(actions) == 0:
        return
        
    py_norm, px_norm = actions[:, 0], actions[:, 1]
    ly_norm, lx_norm = actions[:, 2], actions[:, 3]
    
    px = ((px_norm + 1.0) / 2.0) * width
    py = ((py_norm + 1.0) / 2.0) * height
    lx = ((lx_norm + 1.0) / 2.0) * width
    ly = ((ly_norm + 1.0) / 2.0) * height
    
    dx = lx - px
    dy = ly - py
    
    ax.quiver(px, py, dx, dy, color=colors, angles='xy', scale_units='xy', scale=1, 
              width=0.015, headwidth=4, headlength=6, headaxislength=5, alpha=alpha)

for ep_folder in tqdm(episode_folders, desc="Processing Combined Grids", unit="ep"):
    ep_path = os.path.join(eval_dir, ep_folder)
    
    # Total columns = 2 * num_steps_to_draw (Left half + Right half)
    total_cols = 2 * num_steps_to_draw
    fig_width = 3 * total_cols
    fig_height = 3 * 2 # 2 rows
    
    fig, axes = plt.subplots(2, total_cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is a 2D array even if total_cols == 1
    if total_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    for j in range(num_steps_to_draw):
        # Map subplots to their respective variables
        ax_in = axes[0, j]                          # Top Left: Input RGB
        ax_rec = axes[1, j]                         # Bottom Left: Recon Mask
        ax_int = axes[0, j + num_steps_to_draw]     # Top Right: Internal Pick Mask
        ax_act = axes[1, j + num_steps_to_draw]     # Bottom Right: ROI Actions
        
        # Strip borders for the current 4 axes
        for ax in [ax_in, ax_rec, ax_int, ax_act]:
            ax.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
        
        # Paths for current step
        step_dir = os.path.join(ep_path, f'step_{j}')
        in_rgb_path = os.path.join(step_dir, 'input_obs.npy')
        roi_rgb_path = os.path.join(step_dir, 'roi_rgb.png')
        internal_mask_path = os.path.join(step_dir, 'internal_pick_mask.png')
        step_mask_path = os.path.join(step_dir, 'mask.png')
        recon_path = os.path.join(step_dir, 'recon_obs.npy')
        means_path = os.path.join(step_dir, 'iteration_means.npy')
        roi_ws_mask_path = os.path.join(step_dir, 'roi_workspace_mask.png')
        action_path = os.path.join(step_dir, 'action.json')
        info_path = os.path.join(step_dir, 'info.json') 

        # --- Extract Prior Reward from step j-1 ---
        prior_reward_str = "N/A"
        if j > 0:
            prev_costs_path = os.path.join(ep_path, f'step_{j-1}', 'last_costs.npy')
            if os.path.exists(prev_costs_path):
                try:
                    costs = np.load(prev_costs_path)
                    prior_reward = -np.min(costs)
                    prior_reward_str = f"{prior_reward * 100:.1f}"
                except Exception as e:
                    tqdm.write(f"Error loading prior costs {ep_folder} Step {j-1}: {e}")
        
        # -----------------------------------------
        # 1. Input RGB (.npy) (Top Left Half)
        # -----------------------------------------
        if os.path.exists(in_rgb_path):
            try:
                img_in = np.load(in_rgb_path)
                if img_in.ndim == 3 and img_in.shape[0] in [1, 3, 4]:
                    img_in = np.transpose(img_in, (1, 2, 0))
                ax_in.imshow(img_in)
            except Exception as e:
                tqdm.write(f"Error Input RGB {ep_folder} Step {j}: {e}")

        # -----------------------------------------
        # 2. Recon Mask (.npy) (Bottom Left Half)
        # -----------------------------------------
        if os.path.exists(recon_path):
            try:
                img_recon = np.load(recon_path).squeeze()
                ax_rec.imshow(img_recon, cmap='gray')
            except Exception as e:
                tqdm.write(f"Error Recon Mask {ep_folder} Step {j}: {e}")
                
        # -----------------------------------------
        # 3. Internal Pick Mask (.png) (Top Right Half)
        # -----------------------------------------
        if os.path.exists(internal_mask_path) and os.path.exists(means_path):
            try:
                img_int_mask = Image.open(internal_mask_path)
                img_w, img_h = img_int_mask.size
                ax_int.imshow(img_int_mask, cmap='gray')
                
                # --- Draw trajectories (Only up to target_stop_step - 1) ---
                if j < target_stop_step:
                    means = np.load(means_path).reshape(-1, 4)
                    if len(means) > 0:
                        idx = np.linspace(0, len(means) - 1, min(20, len(means))).astype(int)
                        sampled_means = means[idx]
                        colors = time_cmap(np.linspace(0, 1, len(sampled_means)))
                        draw_actions_vectorized(ax_int, sampled_means, img_w, img_h, colors, alpha=1.0)
            except Exception as e:
                tqdm.write(f"Error Internal Pick Mask {ep_folder} Step {j}: {e}")

        # -----------------------------------------
        # 4. ROI Masked Action (Bottom Right Half)
        # -----------------------------------------
        if os.path.exists(roi_rgb_path) and os.path.exists(roi_ws_mask_path) and os.path.exists(action_path):
            try:
                img_roi_arr = np.array(Image.open(roi_rgb_path).convert('RGB'))
                mask_ws_arr = np.array(Image.open(roi_ws_mask_path).convert('L'))
                
                H, W = img_roi_arr.shape[:2]
                
                # Create the dimmed composite
                img_dimmed = (img_roi_arr * 0.3).astype(np.uint8)
                mask_bool = mask_ws_arr > 128
                img_composite = np.where(mask_bool[..., None], img_roi_arr, img_dimmed)
                
                # --- Convert to Square by Padding the Bottom ---
                square_size = max(H, W)
                img_square = np.zeros((square_size, square_size, 3), dtype=np.uint8)
                img_square[:H, :W] = img_composite # Place image at the top
                
                ax_act.imshow(img_square)
                
                # --- Draw Arrows (Only up to target_stop_step - 1) ---
                if j < target_stop_step:
                    with open(action_path, 'r') as f:
                        action = json.load(f)
                    
                    if len(action) == 4:
                        px_norm, py_norm, lx_norm, ly_norm = action
                        
                        px_sq = ((px_norm + 1.0) / 2.0) * H
                        py_sq = ((py_norm + 1.0) / 2.0) * H
                        lx_sq = ((lx_norm + 1.0) / 2.0) * H
                        ly_sq = ((ly_norm + 1.0) / 2.0) * H
                        
                        x_offset = (W - H) / 2.0
                        px_img = px_sq + x_offset
                        lx_img = lx_sq + x_offset
                        
                        # Note: Y coordinates remain exactly the same because 
                        # the image is aligned to the top (y=0) of the new square.
                        py_img = py_sq
                        ly_img = ly_sq
                        
                        dx = lx_img - px_img
                        dy = ly_img - py_img
                        
                        ax_act.quiver(px_img, py_img, dx, dy, color='yellow', angles='xy', scale_units='xy', scale=1, 
                                      width=0.015, headwidth=4, headlength=6, headaxislength=5, alpha=1.0, zorder=4)
                        
                        ax_act.scatter(px_img, py_img, color='red', s=50, edgecolors='white', linewidth=1.5, zorder=5)
                    
                # --- TEXT OVERLAYS IN THE PADDED AREA ---
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info_data = json.load(f)
                    
                    eval_data = info_data.get('evaluation', {})
                    reward_data = info_data.get('reward', {})
                    
                    nc = eval_data.get('normalised_coverage', 0) * 100
                    ni = eval_data.get('normalised_improvement', 0) * 100
                    iou = eval_data.get('max_IoU_to_flattened', 0) * 100
                    gt_reward = reward_data.get('coverage_alignment', 0) * 100
                    
                    left_text = f"REWARD:{gt_reward:.1f}\nPRIOR:{prior_reward_str}"
                    right_text = f"NC:{nc:.1f}\nNI:{ni:.1f}\nIoU:{iou:.1f}"
                    
                    base_text_kwargs = dict(
                        transform=ax_act.transAxes,
                        fontsize=16,         # <--- Smaller text
                        fontweight='bold',
                        verticalalignment='bottom', 
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')]
                    )
                    
                    # Values applied at the bottom (y=0.02)
                    ax_act.text(0.04, 0.02, left_text, color='hotpink', horizontalalignment='left', **base_text_kwargs)
                    ax_act.text(0.96, 0.02, right_text, color='lime', horizontalalignment='right', **base_text_kwargs)
                    
            except Exception as e:
                tqdm.write(f"Error ROI Masked Action {ep_folder} Step {j}: {e}")

        # -----------------------------------------
        # 5. Add Titles to the First Column of Each Block
        # -----------------------------------------
        if j == 0:
            title_kwargs = dict(
                transform=ax_in.transAxes,
                fontsize=18,   # <--- Slightly smaller title text
                color='white',
                fontweight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground='black')]
            )
            # Top-Left of the entire figure
            ax_in.text(0.04, 0.96, "RGB\nINPUT", verticalalignment='top', **title_kwargs)
            
            # Bottom-Left of the entire figure
            title_kwargs['transform'] = ax_rec.transAxes
            ax_rec.text(0.04, 0.96, "MASK\nRECONSTRUCTION", verticalalignment='top', **title_kwargs)
            
            # Top-Right section
            title_kwargs['transform'] = ax_int.transAxes
            ax_int.text(0.04, 0.96, "PLANNING\nON PICK\nMASK", verticalalignment='top', **title_kwargs)
            
            # Bottom-Right section
            title_kwargs['transform'] = ax_act.transAxes
            ax_act.text(0.04, 0.96, "ROI RGB\nwith ACTIONS", verticalalignment='top', **title_kwargs)

        # -----------------------------------------
        # 6. Step Numbers
        # -----------------------------------------
        # Puts the step number at the top-right of each column block
        step_kwargs = dict(
            transform=ax_in.transAxes, 
            fontsize=20, 
            color='white', 
            fontweight='bold',
            horizontalalignment='right', # <--- Moved to the right
            verticalalignment='top',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')]
        )
        ax_in.text(0.96, 0.96, f"{j}", **step_kwargs)
        
        step_kwargs['transform'] = ax_int.transAxes
        ax_int.text(0.96, 0.96, f"{j}", **step_kwargs)

    # --- Save The Combined Figure ---
    # Removes all whitespace between subplots
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    save_path = os.path.join(out_combined, f'{ep_folder}_combined.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)

print(f"\nDone! Combined visualizations saved in: {out_combined}")