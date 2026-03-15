import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import actoris_harena.api as ag_ar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Custom modules
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task

def main():
    register_agents()
    register_arenas()

    garments = {
        "Longsleeve": "multi_longsleeve_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200",
        "Trousers": "multi_trousers_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200",
        "Skirt": "multi_skirt_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200",
        "Dress": "multi_dress_single_picker_readjust_pick_provide_semkey_pixel_no_success_stop_resol_128_train_200"
    }

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    
    quadrants = [
        (0, 0), (0, 2), 
        (2, 0), (2, 2)  
    ]

    for (g_label, arena_name), (row_off, col_off) in zip(garments.items(), quadrants):
        print(f"Processing {g_label}...")
        
        arena_conf_path = os.path.join('./conf/', "arena", f"{arena_name}.yaml")
        arena_cfg = OmegaConf.load(arena_conf_path)
        arena = ag_ar.build_arena(arena_cfg.name, arena_cfg)
        
        task_conf_path = os.path.join('./conf/', "task", "flattening_overstretch_penalty_1_no_big_bonus.yaml")
        task_cfg = OmegaConf.load(task_conf_path)
        arena.set_task(build_task(task_cfg))
        
        episode_configs = arena.get_train_configs()
        for i in range(4):
            info = arena.reset(episode_configs[i])
            rgb = info['observation']['rgb']
            # Correctly fetching goal RGB from info
            goal_rgb = info['goal']['rgb']
            
            # Transpose both images if they are channel-first
            if isinstance(rgb, np.ndarray) and rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            if isinstance(goal_rgb, np.ndarray) and goal_rgb.shape[0] == 3:
                goal_rgb = np.transpose(goal_rgb, (1, 2, 0))
            
            r = row_off + (i // 2)
            c = col_off + (i % 2)
            ax = axes[r, c]
            
            # Display main observation
            ax.imshow(rgb)
            ax.axis('off')
            
            # Create Inset axes
            ax_ins = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=0.5)
            ax_ins.imshow(goal_rgb)

            # --- UPDATED: Remove ticks and labels but keep the border ---
            ax_ins.set_xticks([])
            ax_ins.set_yticks([])
            # ------------------------------------------------------------

            # Set the border color and width
            for spine in ax_ins.spines.values():
                spine.set_visible(True) # Ensure they are visible
                spine.set_edgecolor('white')
                spine.set_linewidth(1.5) # Slightly thicker looks better on 4x4
            
            # Text remains in top-left
            ax.text(0.05, 0.95, f"{g_label}: Train {i}", 
                    transform=ax.transAxes, 
                    color='white', 
                    fontsize=16, 
                    fontweight='bold',
                    va='top', 
                    ha='left',   
                    bbox=dict(facecolor='black', alpha=0.5, lw=0))
        arena.close()

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    
    save_dir = './analysis'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "combined_garments_4x4_with_goals.png")
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved combined visualization to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()