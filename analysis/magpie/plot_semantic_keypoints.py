import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import actoris_harena.api as ag_ar

# Custom modules
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_sim_task

def process_image_for_mpl(rgb_image):
    """
    Ensures the image array is suitable for matplotlib.imshow.
    Checks if it is channel-first (C, H, W) and transposes to (H, W, C).
    """
    if isinstance(rgb_image, np.ndarray) and rgb_image.shape[0] == 3:
        return np.transpose(rgb_image, (1, 2, 0))
    return rgb_image

def repel_points(x_coords, y_coords, min_dist=8.0, max_iters=100):
    """
    Applies a simple force-directed algorithm to gently push apart 
    points that are too close to each other, preventing overlap.
    """
    x = np.array(x_coords, dtype=float)
    y = np.array(y_coords, dtype=float)
    n = len(x)
    
    for _ in range(max_iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.hypot(dx, dy)
                
                if dist < min_dist:
                    moved = True
                    # If points are perfectly overlapping, give them a deterministic nudge
                    if dist < 1e-4:
                        angle = (i * 2 * np.pi) / n
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    
                    # Calculate how much to push each point
                    overlap = min_dist - dist
                    push_x = (dx / dist) * overlap * 0.5
                    push_y = (dy / dist) * overlap * 0.5
                    
                    x[i] += push_x
                    y[i] += push_y
                    x[j] -= push_x
                    y[j] -= push_y
                    
        # Early exit if no points needed moving in this iteration
        if not moved:
            break
            
    return x, y

def plot_keypoints_on_axis(ax, obs, rotate_180=False):
    """
    Overlay standard matplotlib plots for keypoints on an existing image axis.
    Registers labels for a legend instead of placing text directly on the image.
    """
    semkey2pid = obs.get('semkey2pid')
    norm_pixels_flat = obs.get('flattened_semkey_norm_pixel')
    
    if semkey2pid is None or norm_pixels_flat is None or len(norm_pixels_flat) == 0:
        print("No keypoints found for this observation.")
        return 

    names = list(semkey2pid.keys())
    num_keys = len(names)
    
    # Reshape flattened array
    points = norm_pixels_flat.reshape(num_keys, 2)
    
    image_shape = obs['rgb'].shape
    H_res, W_res = image_shape[0], image_shape[1]

    # Convert Norm [-1, 1] -> Pixel [0, Resolution]
    pixel_x = (points[:, 1] + 1) / 2.0 * W_res
    pixel_y = (points[:, 0] + 1) / 2.0 * H_res
    
    # Apply 180-degree rotation to coordinates to match the rotated image
    if rotate_180:
        pixel_x = W_res - pixel_x
        pixel_y = H_res - pixel_y
        
    # --- BALANCED REPULSION ---
    # min_dist set to 4.0. It gives the larger dots a tiny bit of space 
    # without scattering them away from their true semantic location.
    pixel_x, pixel_y = repel_points(pixel_x, pixel_y, min_dist=4.0)
    
    cmap = plt.cm.get_cmap('tab20')
    
    for i, (x, y, name) in enumerate(zip(pixel_x, pixel_y, names)):
        color = cmap(i % 20) 
        
        clean_name = name.replace('_', ' ')
        
        # --- INCREASED SIZE ---
        # markersize bumped up to 12 for easy visibility.
        # markeredgewidth increased to 1.2 to define the edges better.
        # alpha=0.85 ensures that if two big dots overlap, you can still see both colors.
        ax.plot(x, y, marker='o', markersize=12, color=color, 
                markeredgecolor='white', markeredgewidth=1.2, 
                linestyle='None', alpha=0.85, label=clean_name)
        
    # Create the legend below the subplot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=2, fontsize=10, frameon=False)
    
def main():
    register_agents()
    register_arenas()

    garments = {
        "Longsleeve": "magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace",
        "Trousers": "magpie/multi_trousers_provide_semkey_pixel_no_success_stop_resol_128_workspace",
        "Skirt": "magpie/multi_skirt_provide_semkey_pixel_no_success_stop_resol_128_workspace",
        "Dress": "magpie/multi_dress_provide_semkey_pixel_no_success_stop_resol_128_workspace"
    }

    # 1. Setup Figure 1x4. 
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.01, bottom=0.25) 
    
    episode_id_seed = 0

    # 2. Loop through garments and axes.
    for i, (g_label, arena_name) in enumerate(garments.items()):
        ax = axes[i]
        print(f"\nVisualizing flattened state for {g_label} on plot {i}...")
        
        # a. Initialize environment
        arena_conf_path = os.path.join('./conf/', "arena", f"{arena_name}.yaml")
        arena_cfg = OmegaConf.load(arena_conf_path)
        
        arena_cfg.provide_flattened_semkey_norm_pixel = True
        
        arena = ag_ar.build_arena(arena_cfg.name, arena_cfg)
        arena.set_eval()
        
        task_conf_path = os.path.join('./conf/', "task", "central_alignment.yaml")
        task_cfg = OmegaConf.load(task_conf_path)
        arena.set_task(build_sim_task(task_cfg))
        
        # b. Load environment
        info = arena.reset({'eid': 0})
        
        main_obs = info['observation']
        goal_obs = info['flattened_obs']['observation']
        
        # c. Plot image (Rotated 180 degrees)
        rgb_flat = process_image_for_mpl(goal_obs['rgb'])
        # k=2 rotates the array 90 degrees twice (180 total) along the spatial axes (0, 1)
        rgb_flat = np.rot90(rgb_flat, k=2, axes=(0, 1))
        
        ax.imshow(rgb_flat)
        ax.axis('off')
        
        # d. Inner-Top Garment Label
        ax.text(0.5, 0.95, g_label, transform=ax.transAxes,
                color='white', fontsize=18, fontweight='bold',
                ha='center', va='top', 
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=4))
        
        # e. Overlay keypoints and generate bottom legend (passing rotate_180=True)
        plot_keypoints_on_axis(ax, main_obs, rotate_180=True)
        
        arena.close()

    # 3. Standard Saving Logic
    save_dir = './analysis/magpie'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "flattened_garments_semkey_plot.png")
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"\nSaved combined visualization to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()