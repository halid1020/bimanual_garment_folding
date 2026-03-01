import hydra
from hydra import compose
from omegaconf import DictConfig, OmegaConf
import os
import sys
import random
from tqdm import tqdm
import copy
import numpy as np
import cv2  # <-- Added for image resizing

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset

@hydra.main(config_path="../conf", config_name="data_combination/combine_datasets", version_base=None)
def main(cfg: DictConfig):
    print("--- Merging Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("-----------------------------")

    # 1. Setup Target Configuration
    local_obs_config = OmegaConf.to_container(cfg.dataset.obs_config, resolve=True)
    action_config = OmegaConf.to_container(cfg.dataset.action_config, resolve=True)
    reward_names = OmegaConf.to_container(cfg.dataset.reward_names, resolve=True)
    evaluation_names = OmegaConf.to_container(cfg.dataset.evaluation_names, resolve=True)

    # Dynamically append scalar shapes for rewards and evaluations
    for key in reward_names:
        local_obs_config[key] = {'shape': (1,), 'output_key': key}
    for key in evaluation_names:
        local_obs_config[key] = {'shape': (1,), 'output_key': key}

    if not cfg.include_success_key and 'success' in local_obs_config:
        print("Target dataset will NOT include 'success'. Removing from config.")
        del local_obs_config['success']

    # Initialize Target Dataset in write mode
    target_dataset = TrajectoryDataset(
        data_path=cfg.target_data_path,
        data_dir=cfg.target_data_dir,
        io_mode='w', 
        obs_config=local_obs_config,
        act_config=action_config,
        whole_trajectory=True
    )

    trajectory_map = [] 
    source_datasets = []

    print("Scanning source datasets via Hydra Configs...")
    
    # 2. Dynamically Load Source Configs and Map Trajectories
    for i, source_config_name in enumerate(cfg.sources):
        src_cfg = compose(config_name=source_config_name)
        
        src_data_path = src_cfg.data_path
        src_data_dir = src_cfg.data_dir
        
        print(f"\nLoaded Config: {source_config_name}")
        print(f" -> Path: {src_data_path}")
        print(f" -> Dir:  {src_data_dir}")

        src_obs_config = OmegaConf.to_container(src_cfg.dataset.obs_config, resolve=True)
        src_action_config = OmegaConf.to_container(src_cfg.dataset.action_config, resolve=True)
        src_reward_names = OmegaConf.to_container(src_cfg.dataset.reward_names, resolve=True)
        src_evaluation_names = OmegaConf.to_container(src_cfg.dataset.evaluation_names, resolve=True)

        for key in reward_names:
            src_obs_config[key] = {'shape': (1,), 'output_key': key}
        for key in evaluation_names:
            src_obs_config[key] = {'shape': (1,), 'output_key': key}

        if 'mask-biased' in src_data_path and 'success' in src_obs_config:
            del src_obs_config['success']

        src_dataset = TrajectoryDataset(
            data_path=src_data_path,
            data_dir=src_data_dir,
            io_mode='r', 
            obs_config=src_obs_config,
            act_config=src_action_config,
            whole_trajectory=True
        )
        source_datasets.append(src_dataset)
        
        num_trajs = src_dataset.num_trajectories()
        print(f" -> Found {num_trajs} trajectories.")

        for j in range(num_trajs):
            trajectory_map.append((i, j))

    # 3. Shuffle Completely Randomly
    print(f"\nTotal trajectories to merge: {len(trajectory_map)}")
    print("Shuffling trajectories...")
    random.seed(42) 
    random.shuffle(trajectory_map)

    # 4. Write to Target Dataset
    pbar = tqdm(total=len(trajectory_map), desc='Merging Datasets')
    
    # Keys that might require spatial resizing
    image_keys = ['rgb', 'depth', 'mask', 'goal-rgb', 'goal-depth', 'goal-mask']
    
    for src_idx, traj_idx in trajectory_map:
        src_dataset = source_datasets[src_idx]

        # Retrieve the trajectory dictionary
        trajectory_data = src_dataset.get_trajectory(traj_idx) 
        
        obs = trajectory_data['observation']
        act = trajectory_data['action']

        # --- DYNAMIC RESIZING LOGIC ---
        for key in image_keys:
            if key in obs and key in local_obs_config:
                target_shape = local_obs_config[key]['shape']  # e.g., [128, 128, 3] or [128, 128, 1]
                target_h, target_w = target_shape[0], target_shape[1]
                
                # Current array shape: (T, H, W, C) or (T, H, W)
                current_array = obs[key]
                current_h, current_w = current_array.shape[1], current_array.shape[2]
                
                if current_h != target_h or current_w != target_w:
                    resized_traj = []
                    for t_idx in range(current_array.shape[0]):
                        img = current_array[t_idx]
                        
                        # Use Nearest Neighbor for masks to preserve pure binary/integer values
                        if 'mask' in key:
                            original_dtype = img.dtype
                            # cv2 requires uint8 or float for resizing, so cast boolean masks temporarily
                            if original_dtype == bool:
                                img = img.astype(np.uint8)
                            resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                            resized_img = resized_img.astype(original_dtype)
                        else:
                            # Use Linear for continuous spaces (RGB, Depth)
                            resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        
                        # cv2.resize implicitly removes the last dimension if channels == 1 (e.g., depth, mask)
                        # We must restore it if the target config requires it
                        if len(target_shape) == 3 and target_shape[2] == 1 and len(resized_img.shape) == 2:
                            resized_img = np.expand_dims(resized_img, axis=-1)
                            
                        resized_traj.append(resized_img)
                        
                    # Overwrite the observation with the resized stack
                    obs[key] = np.stack(resized_traj, axis=0)
        # ------------------------------

        # Pad missing 'success' key for mask-biased data
        if 'success' in local_obs_config and 'success' not in obs:
            traj_len = len(obs['rgb']) 
            obs['success'] = np.zeros((traj_len, 1), dtype=np.float32) 

        # Drop 'success' key if the target config doesn't want it
        if 'success' not in local_obs_config and 'success' in obs:
            del obs['success']

        if 'terminal' in obs and 'terminal' not in local_obs_config:
            del obs['terminal']

        target_dataset.add_trajectory(obs, act)
        pbar.update(1)

    print(f"\nSuccessfully combined and shuffled datasets into: {cfg.target_data_path}")

if __name__ == '__main__':
    main()