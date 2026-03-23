import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
data_dir = '/media/hcv530/T7/garment_folding_data/real_world/'
method_folder = 'planet_clothpick_single_picker_single_primitive_multi_longsleeve_flattening_longsleeve_all_garment_5k_eps'
eval_dir = os.path.join(data_dir, method_folder, 'eval_checkpoint_-2') 

# Directory to save the output visualizations
output_dir = os.path.join(eval_dir, 'optimization_visualizations')
os.makedirs(output_dir, exist_ok=True)

# Find and sort all episode folders
episode_folders = [f for f in os.listdir(eval_dir) if f.startswith('episode_')]
episode_folders.sort(key=lambda x: int(x.split('_')[1]))

print(f"Found {len(episode_folders)} episodes. Generating seamless grids...")

# --- Colormaps ---
time_cmap = mcolors.LinearSegmentedColormap.from_list("time_cmap", ["cyan", "magenta"])
# 'autumn' maps 0.0 to Red and 1.0 to Yellow
cost_cmap = plt.get_cmap('autumn')

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

for ep_folder in tqdm(episode_folders, desc="Processing Split Grids", unit="ep"):
    ep_path = os.path.join(eval_dir, ep_folder)
    
    # Create TWO separate figures
    fig_rgb, axes_rgb = plt.subplots(2, 11, figsize=(22, 4))
    fig_mask, axes_mask = plt.subplots(2, 11, figsize=(22, 4))
    
    for j in range(22):
        row = 0 if j < 11 else 1
        col = j if j < 11 else j - 11
            
        ax_rgb = axes_rgb[row, col]
        ax_mask = axes_mask[row, col]
        
        # Strip absolutely all borders, ticks, and labels
        ax_rgb.axis('off')
        ax_rgb.margins(0, 0)
        ax_rgb.xaxis.set_major_locator(plt.NullLocator())
        ax_rgb.yaxis.set_major_locator(plt.NullLocator())
        
        ax_mask.axis('off')
        ax_mask.margins(0, 0)
        ax_mask.xaxis.set_major_locator(plt.NullLocator())
        ax_mask.yaxis.set_major_locator(plt.NullLocator())
        
        step_dir = os.path.join(ep_path, f'step_{j}')
        
        rgb_path = os.path.join(step_dir, 'rgb.png')
        mask_path = os.path.join(step_dir, 'internal_pick_mask.png')
        means_path = os.path.join(step_dir, 'iteration_means.npy')
        samples_path = os.path.join(step_dir, 'last_samples.npy')
        costs_path = os.path.join(step_dir, 'last_costs.npy')
        
        # -----------------------------------------
        # 1. Process RGB Figure (Iteration Means)
        # -----------------------------------------
        if os.path.exists(rgb_path) and os.path.exists(means_path):
            try:
                img_rgb = Image.open(rgb_path)
                img_w, img_h = img_rgb.size
                ax_rgb.imshow(img_rgb)
                
                means = np.load(means_path).reshape(-1, 4)
                
                num_means = len(means)
                if num_means > 0:
                    sample_indices = np.linspace(0, num_means - 1, min(20, num_means)).astype(int)
                    sampled_means = means[sample_indices]
                    
                    fractions = np.linspace(0, 1, len(sampled_means))
                    colors = time_cmap(fractions)
                    
                    draw_actions_vectorized(ax_rgb, sampled_means, img_w, img_h, colors, alpha=1.0)
            except Exception as e:
                tqdm.write(f"Error processing RGB {ep_folder} Step {j}: {e}")

        # -----------------------------------------
        # 2. Process Mask Figure (Last Samples)
        # -----------------------------------------
        if os.path.exists(mask_path) and os.path.exists(samples_path) and os.path.exists(costs_path):
            try:
                img_mask = Image.open(mask_path)
                img_w, img_h = img_mask.size
                ax_mask.imshow(img_mask, cmap='gray')
                
                samples = np.load(samples_path).reshape(-1, 4) 
                costs = np.load(costs_path).flatten() 
                
                num_samples = len(samples)
                if num_samples > 0 and len(costs) > 0:
                    idx = np.linspace(0, num_samples - 1, min(20, num_samples)).astype(int)
                    sampled_samples = samples[idx]
                    sampled_costs = costs[idx]

                    # Sort by cost descending (Highest to Lowest)
                    sort_order = np.argsort(sampled_costs)[::-1]
                    sampled_samples = sampled_samples[sort_order]
                    sampled_costs = sampled_costs[sort_order]

                    # ---> THE FIX: Absolute scaling from [-1, 1] to [0.0, 1.0] <---
                    # -1 becomes 0.0 (Red), 1 becomes 1.0 (Yellow)
                    # np.clip ensures values outside [-1, 1] don't crash the colormap
                    norm_costs = np.clip((sampled_costs + 1.0) / 2.0, 0.0, 1.0)
                    
                    colors = cost_cmap(norm_costs)
                    
                    draw_actions_vectorized(ax_mask, sampled_samples, img_w, img_h, colors, alpha=0.9)
                        
            except Exception as e:
                tqdm.write(f"Error processing Mask {ep_folder} Step {j}: {e}")

    # --- Save Both Figures with ZERO borders/gaps ---
    fig_rgb.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    rgb_save_path = os.path.join(output_dir, f'{ep_folder}_optimization_rgb.png')
    fig_rgb.savefig(rgb_save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig_rgb)
    
    fig_mask.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    mask_save_path = os.path.join(output_dir, f'{ep_folder}_optimization_mask.png')
    fig_mask.savefig(mask_save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig_mask)

print(f"\nDone! Seamless split visualizations saved to: {output_dir}")