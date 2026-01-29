import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Assuming these are your custom modules
from train.utils import register_agent, registered_arena, build_task
import agent_arena.api as ag_ar

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Agent Arena Environment")
    parser.add_argument('--arena_name', type=str, required=True, help="Name of the arena (e.g., 'standard_arena')")
    return parser.parse_args()

def main():
    args = parse_args()
    
    register_agent()

    # 1. Load the Arena Config
    arena_conf_path = os.path.join('./conf/', "arena", f"{args.arena_name}.yaml")
    
    if not os.path.exists(arena_conf_path):
        raise FileNotFoundError(f"Arena config not found at: {arena_conf_path}")
        
    print(f"Loading arena config from: {arena_conf_path}")
    arena_cfg = OmegaConf.load(arena_conf_path)

    # 2. Build Arena
    if arena_cfg.name not in registered_arena:
        raise ValueError(f"Arena '{arena_cfg.name}' is not registered. Available: {list(registered_arena.keys())}")
        
    arena = registered_arena[arena_cfg.name](arena_cfg)
    
    # 3. Load Task Config
    task_conf_path = os.path.join('./conf/', "task", "flattening_overstretch_penalty_1_no_big_bonus.yaml")
    if not os.path.exists(task_conf_path):
        raise FileNotFoundError(f"Task config not found at: {task_conf_path}")
    
    print(f"Loading task config from: {task_conf_path}")
    task_cfg = OmegaConf.load(task_conf_path)

    task = build_task(task_cfg)
    arena.set_task(task)
    
    # 4. Collect Data
    episode_configs = arena.get_train_configs()
    collected_images = []
    
    num_samples = min(9, len(episode_configs))
    print(f"Visualizing {num_samples} episodes...")
    
    for i in range(num_samples):
        eps_conf = episode_configs[i]
        info = arena.reset(eps_conf)
        
        # Get the observation
        rgb = info['observation']['rgb']
        
        # Handle format (Channel First -> Channel Last)
        if isinstance(rgb, np.ndarray) and rgb.shape[0] == 3 and rgb.shape[2] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))
            
        collected_images.append(rgb)

    # 5. Plot 3x3 Grid
    if not collected_images:
        print("No images found to plot.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(collected_images):
            ax.imshow(collected_images[idx])
            ax.axis('off')
        else:
            # Turn off empty axes
            ax.axis('off')

    # --- REMOVE SPACES ---
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    # ---------------------

    # --- SAVE LOGIC ---
    save_dir = './tmp'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{args.arena_name}_preview.png")
    
    # bbox_inches='tight' and pad_inches=0 remove the final white border around the whole image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to: {save_path}")
    # ------------------

    plt.show()

if __name__ == "__main__":
    main()