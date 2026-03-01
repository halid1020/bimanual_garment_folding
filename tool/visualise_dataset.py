import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset

def denormalize_action(action, img_size):
    """
    Converts normalized action [-1, 1] back to pixel space [0, img_size] and radians.
    Assumes format: [pick_v, pick_u, place_v, place_u, theta_norm]
    """
    pick_v_norm, pick_u_norm, place_v_norm, place_u_norm, theta_norm = action
    
    # Map [-1, 1] -> [0, img_size]
    pick_v = int((pick_v_norm + 1.0) / 2.0 * img_size)
    pick_u = int((pick_u_norm + 1.0) / 2.0 * img_size)
    place_v = int((place_v_norm + 1.0) / 2.0 * img_size)
    place_u = int((place_u_norm + 1.0) / 2.0 * img_size)
    
    # Map [-1, 1] -> [-pi, pi]
    theta = theta_norm * np.pi
    
    return pick_v, pick_u, place_v, place_u, theta

def draw_action_on_image(img, action, img_size):
    """Draws the pick, place, and orientation on the RGB image."""
    pick_v, pick_u, place_v, place_u, theta = denormalize_action(action, img_size)
    
    # Clip to ensure we don't draw outside the image bounds
    pick_v, pick_u = np.clip([pick_v, pick_u], 0, img_size - 1)
    place_v, place_u = np.clip([place_v, place_u], 0, img_size - 1)

    # Convert to contiguous array for OpenCV drawing
    img_draw = np.ascontiguousarray(img.copy())

    # Draw Pick (Red) and Place (Green)
    cv2.circle(img_draw, (pick_u, pick_v), max(1, img_size//40), (255, 0, 0), -1)
    cv2.circle(img_draw, (place_u, place_v), max(1, img_size//40), (0, 255, 0), -1)
    
    # Draw Trajectory Arrow (Yellow)
    cv2.arrowedLine(img_draw, (pick_u, pick_v), (place_u, place_v), (255, 255, 0), 1, tipLength=0.1)
    
    # Draw Orientation at Place (Blue)
    r_len = max(5, img_size // 10)
    end_u = int(place_u + r_len * np.cos(theta))
    end_v = int(place_v + r_len * np.sin(theta))
    cv2.line(img_draw, (place_u, place_v), (end_u, end_v), (0, 0, 255), 1)

    return img_draw

# ---------------------------------------------------------------------------
# CORE LOGIC SEPARATED FROM HYDRA
# ---------------------------------------------------------------------------
def visualize_data(data_dir, data_path, local_obs_config, action_config):
    print("--- Starting Visualization ---")
    
    # Dynamically find the action key and image size from config
    act_key = list(action_config.keys())[0]  # e.g., 'norm-pixel-pick-and-place'
    img_size = local_obs_config['rgb']['shape'][0] # e.g., 128
    
    full_path = os.path.join(data_dir, data_path)
    print(f"Dataset Path: {full_path}")
    print(f"Image Size: {img_size}x{img_size}")
    print(f"Action Key: {act_key}")

    # Initialize Dataset in Read Mode
    dataset = TrajectoryDataset(
        data_path=data_path,
        data_dir=data_dir,
        io_mode='r', # Strictly read-only
        obs_config=local_obs_config,
        act_config=action_config,
        whole_trajectory=True
    )

    total_trajs = dataset.num_trajectories()
    print(f"Total Trajectories Found: {total_trajs}")

    if total_trajs == 0:
        print("No data to visualize. Exiting.")
        return

    # Set up save directory
    save_dir = os.path.join('./tmp', "data_visualizations", data_path)
    os.makedirs(save_dir, exist_ok=True)

    # Number of trajectories to visualize
    num_to_visualize = min(5, total_trajs)

    for traj_idx in range(num_to_visualize):
        print(f"Visualizing Trajectory {traj_idx}...")
        
        if hasattr(dataset, 'get_trajectory'):
            traj_data = dataset.get_trajectory(traj_idx)
        else:
            traj_data = dataset[traj_idx]
            
        if isinstance(traj_data, tuple) and len(traj_data) == 2:
            obs_dict, act_dict = traj_data
        elif isinstance(traj_data, dict):
            obs_dict = traj_data.get('observation', traj_data)
            act_dict = traj_data.get('action', traj_data)
        else:
            raise ValueError(f"Unexpected trajectory data format: {type(traj_data)}")
            
        rgbs = obs_dict['rgb']
        actions = act_dict[act_key]
        num_steps = len(rgbs)

        for step in range(num_steps):
            fig_panels = []
            titles = []

            # 1. Base RGB + Action
            img = rgbs[step]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            if step < len(actions):
                action = actions[step]
            else:
                action = np.zeros_like(actions[0]) 
            
            img_with_action = img.copy()

            fig_panels.append(img_with_action)
            titles.append(f"Step {step}: RGB + Action")

            # 2. Mask (if available)
            if 'mask' in obs_dict:
                mask = obs_dict['mask'][step].squeeze()
                fig_panels.append(mask)
                titles.append("Mask")

            # 3. Goal RGB (if available)
            if 'goal-rgb' in obs_dict:
                goal_img = obs_dict['goal-rgb'][step]
                if goal_img.dtype != np.uint8:
                    goal_img = (goal_img * 255).astype(np.uint8) if goal_img.max() <= 1.0 else goal_img.astype(np.uint8)
                fig_panels.append(goal_img)
                titles.append("Goal RGB")

            n_panels = len(fig_panels)
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
            
            if n_panels == 1:
                axes = [axes] 

            for ax, panel, title in zip(axes, fig_panels, titles):
                cmap = 'gray' if len(panel.shape) == 2 else None
                ax.imshow(panel, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
            
            plt.tight_layout()
            
            traj_dir = os.path.join(save_dir, f"traj_{traj_idx:03d}")
            os.makedirs(traj_dir, exist_ok=True)
            frame_path = os.path.join(traj_dir, f"step_{step:03d}.png")
            plt.savefig(frame_path, dpi=150)
            plt.close(fig)

    print(f"\n✅ Visualization complete! Check the frames in: {save_dir}")


# ---------------------------------------------------------------------------
# HYDRA ENTRY POINT
# ---------------------------------------------------------------------------
@hydra.main(config_path="../conf", config_name="data_collection/collect_dataset_01", version_base=None)
def hydra_main(cfg: DictConfig):
    print("--- Loading Configuration via Hydra ---")
    local_obs_config = OmegaConf.to_container(cfg.dataset.obs_config, resolve=True)
    action_config = OmegaConf.to_container(cfg.dataset.action_config, resolve=True)
    
    visualize_data(cfg.data_dir, cfg.data_path, local_obs_config, action_config)

if __name__ == "__main__":
    # Intercept direct path usage to bypass Hydra
    if len(sys.argv) == 2 and '=' not in sys.argv[1]:
        direct_path = sys.argv[1]
        
        # Clean up the path format
        direct_path = os.path.normpath(os.path.abspath(direct_path))
        data_dir = os.path.dirname(direct_path)
        data_path = os.path.basename(direct_path)
        
        print(f"--- Direct Path Intercepted: {direct_path} ---")
        print("--- Bypassing Hydra entirely ---")
        
        # 1. Try to load the config that Hydra automatically saves in the dataset folder
        saved_config_path = os.path.join(direct_path, ".hydra", "config.yaml")
        
        if os.path.exists(saved_config_path):
            print(f"Found saved config at {saved_config_path}")
            cfg = OmegaConf.load(saved_config_path)
            local_obs_config = OmegaConf.to_container(cfg.dataset.obs_config, resolve=True)
            action_config = OmegaConf.to_container(cfg.dataset.action_config, resolve=True)
        else:
            # 2. Fallback to manual dictionaries if no saved config exists
            print("No saved config found. Using manual fallback configurations.")
            
            # NOTE: Change '128' if your images are a different size (e.g., 256 or 84)
            local_obs_config = {
                'rgb': {'shape': [128, 128, 3], 'output_key': 'rgb'},
                'mask': {'shape': [128, 128, 1], 'output_key': 'mask'} 
            }
            # NOTE: Change the key if your dataset uses a different action name
            action_config = {
                'default': {'shape': [9], 'output_key': 'default'}
            }

        # Run the visualization core logic directly
        visualize_data(data_dir, data_path, local_obs_config, action_config)
        
    else:
        # Fallback to standard Hydra execution if no direct path is provided
        hydra_main()