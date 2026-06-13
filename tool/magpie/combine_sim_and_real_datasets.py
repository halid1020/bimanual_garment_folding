"""
Sim-to-Real Trajectory Dataset Merger

This script aggregates a unified simulation dataset with a real-world human 
demonstration dataset. It creates a master dataset for mixed-domain training 
in the Actoris-Harena framework.

Key operations:
1. Reads the 17-keypoint simulation dataset and 15-keypoint real-world dataset.
2. Pads the real-world semantic keypoints to length 17 to match the simulation schema.
3. Injects a continuous `domain` label into every observation timestep:
   - 0.0 : Simulation Data
   - 1.0 : Real-World Data
4. Commits the interleaved/sequential result to a new Zarr store.

Dependencies:
    - numpy
    - tqdm
    - actoris_harena.TrajectoryDataset
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from actoris_harena import TrajectoryDataset

# ==========================================
# Target Dataset Configuration
# ==========================================
COMBINED_DATASET_CONFIG: Dict[str, Any] = {
    "data_path": "sim_and_real_combined",
    "data_dir": "./data/datasets",
    "split_ratios": [0.0, 0.05, 0.95],
    "seq_length": 1,
    "io_mode": "w",
    "cache_in_memory": False,
    "obs_config": {
        "mask": {"shape": [128, 128, 1], "output_key": "mask"},
        "depth": {"shape": [128, 128, 1], "output_key": "depth"},
        "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
        "semkey_norm_pixel": {"shape": [17, 2], "output_key": "semkey_norm_pixel"},
        "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
        "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
        "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
        "flattened_semkey_norm_pixel": {"shape": [17, 2], "output_key": "flattened_goal_semkey_norm_pixel"},
        
        "garment_type": {"shape": [1], "output_key": "garment_type"},
        "num_semkeys": {"shape": [1], "output_key": "num_semkeys"},
        "num_flattened_semkeys": {"shape": [1], "output_key": "num_flattened_semkeys"},
        
        # NEW FIELD: 0 for Simulation, 1 for Real
        "domain": {"shape": [1], "output_key": "domain"}
    },
    "act_config": {
        "default": {"shape": [9], "output_key": "default"}
    }
}

# ==========================================
# Source Configurations
# ==========================================
BASE_ACT_CONFIG = {"default": {"shape": [9], "output_key": "default"}}

SIM_OBS_CONFIG = {
    "mask": {"shape": [128, 128, 1], "output_key": "mask"},
    "depth": {"shape": [128, 128, 1], "output_key": "depth"},
    "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
    "semkey_norm_pixel": {"shape": [17, 2], "output_key": "semkey_norm_pixel"},
    "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
    "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
    "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
    "flattened_semkey_norm_pixel": {"shape": [17, 2], "output_key": "flattened_goal_semkey_norm_pixel"},
    "garment_type": {"shape": [1], "output_key": "garment_type"},
    "num_semkeys": {"shape": [1], "output_key": "num_semkeys"},
    "num_flattened_semkeys": {"shape": [1], "output_key": "num_flattened_semkeys"}
}

REAL_OBS_CONFIG = {
    "mask": {"shape": [128, 128, 1], "output_key": "mask"},
    "depth": {"shape": [128, 128, 1], "output_key": "depth"},
    "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
    "semkey_norm_pixel": {"shape": [15, 2], "output_key": "semkey_norm_pixel"},
    "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
    "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
    "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
    "flattened_semkey_norm_pixel": {"shape": [15, 2], "output_key": "flattened_goal_semkey_norm_pixel"},
    "garment_type": {"shape": [1], "output_key": "garment_type"},
    "num_semkeys": {"shape": [1], "output_key": "num_semkeys"},
    "num_flattened_semkeys": {"shape": [1], "output_key": "num_flattened_semkeys"}
}

def pad_keypoints(kp_array: np.ndarray, max_len: int = 17, pad_val: float = -1.0) -> np.ndarray:
    L, N, dims = kp_array.shape
    if N == max_len:
        return kp_array
        
    padded = np.full((L, max_len, dims), pad_val, dtype=np.float32)
    padded[:, :N, :] = kp_array
    return padded

def process_and_add_trajectory(ds_writer: TrajectoryDataset, ds_reader: TrajectoryDataset, traj_idx: int, domain_label: float, pad_to: int = 17):
    traj_data = ds_reader.get_trajectory(traj_idx)
    obs = traj_data['observation']
    acts = traj_data['action']
    
    L_obs = obs['rgb'].shape[0]
    
    padded_semkeys = pad_keypoints(obs['semkey_norm_pixel'], max_len=pad_to)
    padded_flat_semkeys = pad_keypoints(obs['flattened_goal_semkey_norm_pixel'], max_len=pad_to)
    
    new_obs = {
        'mask': obs['mask'],
        'depth': obs['depth'],
        'rgb': obs['rgb'],
        'goal_rgb': obs['goal_rgb'],
        'goal_depth': obs['goal_depth'],
        'goal_mask': obs['goal_mask'],
        'semkey_norm_pixel': padded_semkeys,
        'flattened_semkey_norm_pixel': padded_flat_semkeys,
        'garment_type': obs['garment_type'],
        'num_semkeys': obs['num_semkeys'],
        'num_flattened_semkeys': obs['num_flattened_semkeys'],
        
        # Inject the domain flag across the entire sequence
        'domain': np.full((L_obs, 1), domain_label, dtype=np.float32)
    }
    
    new_acts = {'default': acts['default']}
    
    ds_writer.add_trajectory(observations=new_obs, actions=new_acts)

def combine_sim_and_real(data_dir: str, sim_name: str, real_name: str, output_name: str):
    print(f"--- Phase 1: Loading Readers ---")
    print(f"Loading Simulation Dataset: {sim_name}")
    
    # Passing data_dir and data_path separately fixes the double-path bug
    ds_sim = TrajectoryDataset(
        data_dir=data_dir,
        data_path=sim_name,
        io_mode="r",
        cache_in_memory=True,
        obs_config=SIM_OBS_CONFIG,
        act_config=BASE_ACT_CONFIG
    )
    
    print(f"Loading Real-World Dataset: {real_name}")
    ds_real = TrajectoryDataset(
        data_dir=data_dir,
        data_path=real_name,
        io_mode="r",
        cache_in_memory=True,
        obs_config=REAL_OBS_CONFIG,
        act_config=BASE_ACT_CONFIG
    )
    
    print(f"\n--- Phase 2: Initializing Writer ---")
    COMBINED_DATASET_CONFIG["data_dir"] = data_dir
    COMBINED_DATASET_CONFIG["data_path"] = output_name
    
    ds_combined = TrajectoryDataset(**COMBINED_DATASET_CONFIG)
    
    sim_count = ds_sim.num_trajectories()
    real_count = ds_real.num_trajectories()
    print(f"Detected {sim_count} Sim trajectories and {real_count} Real trajectories.")

    print(f"\n--- Phase 3: Processing Simulation Data (Domain = 0) ---")
    for i in tqdm(range(sim_count), desc="Sim Episodes"):
        process_and_add_trajectory(ds_combined, ds_sim, traj_idx=i, domain_label=0.0)

    print(f"\n--- Phase 4: Processing Real-World Data (Domain = 1) ---")
    for i in tqdm(range(real_count), desc="Real Episodes"):
        process_and_add_trajectory(ds_combined, ds_real, traj_idx=i, domain_label=1.0)

    print(f"\nMerging complete! Total combined trajectories saved: {ds_combined.num_trajectories()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Simulation and Real-World datasets into a unified Zarr store.")
    
    # Separated data directory from the specific folder names
    parser.add_argument('--data_dir', type=str, default="./data/datasets", 
                        help="Base directory where datasets are stored.")
    parser.add_argument('--sim_name', type=str, default="all_garments_multi_primitive_alignment", 
                        help="Folder name of the simulation dataset.")
    parser.add_argument('--real_name', type=str, default="real_world_longsleeve", 
                        help="Folder name of the real-world dataset.")
    parser.add_argument('--output_name', type=str, default="sim_and_real_combined", 
                        help="Folder name for the new output dataset.")
    
    args = parser.parse_args()
    
    combine_sim_and_real(args.data_dir, args.sim_name, args.real_name, args.output_name)