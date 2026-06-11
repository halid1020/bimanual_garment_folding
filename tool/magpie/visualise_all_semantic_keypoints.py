import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import actoris_harena.api as ag_ar

# Custom modules
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_sim_task

def plot_keypoints(img, keypoints_norm, key_names, colors, ax, title):
    """
    Plots the RGB image and overlays semantic keypoints with unique colors.
    """
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    
    H, W = img.shape[:2]
    
    if keypoints_norm is not None:
        for i, (name, kp, color) in enumerate(zip(key_names, keypoints_norm, colors)):
            y_norm, x_norm = kp
            
            # De-normalize from [-1, 1] to pixel coordinates [0, W] and [0, H]
            x = (x_norm + 1.0) / 2.0 * W
            y = (y_norm + 1.0) / 2.0 * H
            
            # Draw marker with unique color, use name as label for the legend
            ax.scatter(x, y, color=color, s=50, edgecolors='white', zorder=2, label=name)

def process_and_plot_mode(arena, mode, args, output_dir):
    """
    Sets the environment mode, collects the configs, and generates the plot grid.
    """
    print(f"\n--- Processing Mode: {mode.upper()} ---")
    
    # 1. Set mode and retrieve corresponding configs safely
    try:
        if mode == 'train':
            if hasattr(arena, 'set_train'): arena.set_train()
            configs = arena.get_train_configs()
        elif mode == 'val':
            if hasattr(arena, 'set_val'): arena.set_val()
            # Fallback if get_val_configs doesn't strictly exist in some older API versions
            configs = arena.get_val_configs() if hasattr(arena, 'get_val_configs') else []
        elif mode == 'eval':
            if hasattr(arena, 'set_eval'): arena.set_eval()
            configs = arena.get_eval_configs() if hasattr(arena, 'get_eval_configs') else []
        else:
            print(f"Unknown mode: {mode}")
            return
    except Exception as e:
        print(f"Error retrieving configs for {mode}: {e}")
        return

    # 2. Filter configs based on requested num_episodes
    if args.num_episodes > 0:
        configs = configs[:args.num_episodes]
        
    num_episodes = len(configs)
    
    if num_episodes == 0:
        print(f"No episodes found for mode '{mode}'. Skipping.")
        return

    print(f"Found {num_episodes} episodes to visualize.")

    # 3. Setup grid parameters
    episodes_per_row = 3
    cols = episodes_per_row * 2  # 2 images (init & goal) per episode
    rows = int(np.ceil(num_episodes / episodes_per_row))
    
    # Warning for very large plots
    if rows > 50:
        print(f"Warning: Generating a very large image ({rows} rows). This may take a moment.")
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)  # Ensure axes is 2D even if there's only 1 row
    
    colors = None  # Will be populated on the first iteration

    # 4. Iterate through episodes
    for ep_idx, config in enumerate(configs):
        info = arena.reset(config)
        obs = info['observation']
        
        # Extract Mesh ID
        pkl_path = arena.init_state_params.get('pkl_path', '')
        if pkl_path:
            mesh_id = os.path.splitext(os.path.basename(pkl_path))[0]
        else:
            mesh_id = "unknown"
            
        print(f"[{mode.upper()}] Processing episode {ep_idx + 1}/{num_episodes} (Mesh ID: {mesh_id})...")
        
        # Extract Image Data
        init_rgb = obs['rgb']
        goal_rgb = obs['flattened-rgb'] 
        
        # Extract Keypoint Data
        semkey2pid = obs.get('semkey2pid', {})
        key_names = list(semkey2pid.keys())
        
        # Generate unique colors for each keypoint on the first pass
        if colors is None and len(key_names) > 0:
            cmap = plt.cm.get_cmap('tab20')
            colors = [cmap(i / len(key_names)) for i in range(len(key_names))]
        
        init_kp_norm = obs.get('semkey_norm_pixel')
        goal_kp_norm = obs.get('flattened_semkey_norm_pixel')
        
        # Reshape to (N, 2) arrays for iterating
        if init_kp_norm is not None:
            init_kp_norm = init_kp_norm.reshape(-1, 2)
        if goal_kp_norm is not None:
            goal_kp_norm = goal_kp_norm.reshape(-1, 2)

        # Calculate grid position
        r = ep_idx // episodes_per_row
        c_init = (ep_idx % episodes_per_row) * 2
        c_goal = c_init + 1
        
        # Plot on corresponding axes
        plot_keypoints(init_rgb, init_kp_norm, key_names, colors, axes[r, c_init], f"Ep {ep_idx} | {mesh_id}\nInit")
        plot_keypoints(goal_rgb, goal_kp_norm, key_names, colors, axes[r, c_goal], f"Ep {ep_idx} | {mesh_id}\nGoal")

    # 5. Hide any unused subplots
    for idx in range(num_episodes * 2, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis('off')

    # 6. Add a global legend at the bottom of the figure
    if num_episodes > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(10, len(labels)), 
                   bbox_to_anchor=(0.5, 0.0), fontsize=12, markerscale=1.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save using dynamic filename including the mode
    save_path = os.path.join(output_dir, f"{args.garment}_{args.task}_{mode}_episodes_grid.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Finished processing {mode.upper()}. Saved as {save_path}")

def main():
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Visualize semantic keypoints for Garment environments.")
    parser.add_argument('--garment', type=str, default='dress', 
                        help="The garment type to inspect (e.g., dress, tshirt, pants). Default is 'dress'.")
    parser.add_argument('--task', type=str, default='flattening', 
                        help="The task configuration to inspect (e.g., flattening). Default is 'flattening'.")
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'val', 'eval', 'all'],
                        help="Mode to inspect ('train', 'val', 'eval', 'all'). Default is 'all'.")
    parser.add_argument('--num_episodes', type=int, default=-1, 
                        help="Number of episodes to visualize. Default is -1 (process ALL available).")
    args = parser.parse_args()

    register_agents()
    register_arenas()

    # Dynamically format the arena name using the garment argument
    arena_name = f"magpie/multi_{args.garment}_provide_semkey_pixel_no_success_stop_resol_128_workspace"
    
    print(f"Setting up environment: {arena_name}")
    print(f"Loading task: {args.task}")
    
    # 1. Initialize environment config
    arena_conf_path = os.path.join('./conf/', "arena", f"{arena_name}.yaml")
    if not os.path.exists(arena_conf_path):
        raise FileNotFoundError(f"Arena config not found: {arena_conf_path}")
        
    arena_cfg = OmegaConf.load(arena_conf_path)
    arena_cfg.provide_semkey_norm_pixel = True
    arena_cfg.provide_flattened_semkey_norm_pixel = True
    
    arena = ag_ar.build_arena(arena_cfg.name, arena_cfg)
    
    # 2. Initialize task dynamically using the task argument
    task_conf_path = os.path.join('./conf/', "task", f"{args.task}.yaml")
    if not os.path.exists(task_conf_path):
        raise FileNotFoundError(f"Task config not found: {task_conf_path}")
        
    task_cfg = OmegaConf.load(task_conf_path)
    arena.set_task(build_sim_task(task_cfg))
    
    # Prepare output directory
    output_dir = "./tmp/keypoint_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {output_dir}")

    # 3. Determine which modes to run
    if args.mode == 'all':
        modes_to_run = ['train', 'val', 'eval']
    else:
        modes_to_run = [args.mode]

    # 4. Process each mode
    for m in modes_to_run:
        process_and_plot_mode(arena, m, args, output_dir)

    arena.close()
    print("\nScript completed successfully.")

if __name__ == "__main__":
    main()