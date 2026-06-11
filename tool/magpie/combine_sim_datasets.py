"""
Garment Trajectory Dataset Merger

This script aggregates four distinct Zarr-based trajectory datasets 
(longsleeve, trousers, skirt, dress) into a single, unified dataset 
compatible with the Actoris-Harena framework. 

It handles dimensional mismatches in semantic keypoints by dynamically 
padding them to a unified maximum length and interleaves the trajectories 
using a round-robin approach for balanced training distributions.

Dependencies:
    - numpy
    - tqdm
    - actoris_harena.TrajectoryDataset

Usage: python tool/magpie/combine_sim_datasets.py
"""

import numpy as np
from tqdm import tqdm
from actoris_harena import TrajectoryDataset
from typing import Dict, Any

# ==========================================
# Target Dataset Configuration
# ==========================================
# Defines the schema for the final unified dataset. Keypoints are fixed to 
# a shape of [17, 2] to accommodate the maximum length found in the dress dataset.
MERGED_DATASET_CONFIG: Dict[str, Any] = {
    "data_path": "all_garments_multi_primitive_alignment",
    "data_dir": "./data/datasets",
    "split_ratios": [0.0, 0.05, 0.95],
    "seq_length": 1,
    "io_mode": "w",
    "cache_in_memory": False,  # Writing mode requires this to be False
    "obs_config": {
        "mask": {"shape": [128, 128, 1], "output_key": "mask"},
        "depth": {"shape": [128, 128, 1], "output_key": "depth"},
        "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
        "semkey_norm_pixel": {"shape": [17, 2], "output_key": "semkey_norm_pixel"},
        "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
        "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
        "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
        "flattened_semkey_norm_pixel": {"shape": [17, 2], "output_key": "flattened_goal_semkey_norm_pixel"},
        
        # Injected metadata fields for downstream model conditioning
        "garment_type": {"shape": [1], "output_key": "garment_type"},
        "num_semkeys": {"shape": [1], "output_key": "num_semkeys"},
        "num_flattened_semkeys": {"shape": [1], "output_key": "num_flattened_semkeys"}
    },
    "act_config": {
        "default": {"shape": [9], "output_key": "default"}
    }
}

def pad_keypoints(kp_array: np.ndarray, max_len: int = 17, pad_val: float = -1.0) -> np.ndarray:
    """
    Pads a sequence of keypoint arrays to a uniform maximum length.
    
    This ensures that variable-length semantic keypoints across different 
    garment types can be batched together in a unified tensor format.
    
    Args:
        kp_array (np.ndarray): The original keypoint array of shape (L, N, dims),
                               where L is sequence length and N is current keypoints.
        max_len (int): The target number of keypoints to pad up to.
        pad_val (float): The specific value used for padded elements.
        
    Returns:
        np.ndarray: Padded array of shape (L, max_len, dims).
    """
    L, N, dims = kp_array.shape
    padded = np.full((L, max_len, dims), pad_val, dtype=np.float32)
    padded[:, :N, :] = kp_array
    return padded

def combine_datasets() -> None:
    """
    Executes the dataset merging pipeline.
    
    1. Iterates over defined source datasets.
    2. Dynamically generates configurations to load them in read-only mode.
    3. Initializes a new write-mode dataset.
    4. Interleaves trajectories round-robin, pads arrays, and writes to disk.
    """
    
    # Mapping of source directories relative to data_dir
    dataset_paths = {
        "longsleeve": "multi_longsleeve_multi_primitive_alignment_human_demo",
        "trousers": "multi_trousers_multi_primitive_alignment_human_demo",
        "skirt": "multi_skirt_multi_primitive_alignment_human_demo",
        "dress": "multi_dress_multi_primitive_alignment_human_demo"
    }

    # Registry mapping discrete IDs to dataset names and specific keypoint lengths
    # Used to correctly parse the original data before padding
    garment_map = {
        0: {"name": "longsleeve", "orig_sk_len": 15, "orig_flat_len": 15},
        1: {"name": "trousers",   "orig_sk_len": 8,  "orig_flat_len": 8},
        2: {"name": "skirt",      "orig_sk_len": 7,  "orig_flat_len": 7},
        3: {"name": "dress",      "orig_sk_len": 17, "orig_flat_len": 17}
    }

    datasets = {}
    base_act_config = {"default": {"shape": [9], "output_key": "default"}}

    print("--- Phase 1: Loading Source Datasets ---")
    
    for g_id, info in garment_map.items():
        name = info["name"]
        print(f"Loading {name} dataset...")
        
        # Dynamically construct the specific read configuration for this dataset
        # This prevents TrajectoryDataset from throwing shape mismatch or NoneType errors
        source_obs_config = {
            "mask": {"shape": [128, 128, 1], "output_key": "mask"},
            "depth": {"shape": [128, 128, 1], "output_key": "depth"},
            "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
            "semkey_norm_pixel": {"shape": [info["orig_sk_len"], 2], "output_key": "semkey_norm_pixel"},
            "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
            "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
            "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
            "flattened_semkey_norm_pixel": {"shape": [info["orig_flat_len"], 2], "output_key": "flattened_goal_semkey_norm_pixel"}
        }

        datasets[g_id] = TrajectoryDataset(
            data_path=dataset_paths[name],
            data_dir="./data/datasets",
            io_mode="r",
            cache_in_memory=True,  # Recommended for fast sequential reading
            obs_config=source_obs_config,
            act_config=base_act_config
        )

    print("\n--- Phase 2: Initializing Merged Dataset ---")
    merged_dataset = TrajectoryDataset(**MERGED_DATASET_CONFIG)

    # Determine maximum number of loops needed for the round-robin process
    traj_counts = {g_id: ds.num_trajectories() for g_id, ds in datasets.items()}
    max_trajectories = max(traj_counts.values())
    print(f"Detected trajectory counts per garment: {traj_counts}\n")

    print("--- Phase 3: Merging Trajectories ---")
    
    # Wrap the primary loop in tqdm to display a progress bar
    for traj_idx in tqdm(range(max_trajectories), desc="Interleaving Trajectories", unit="cycle"):
        
        # Round-robin insertion logic ensures balanced dataset sampling
        for g_id in [0, 1, 2, 3]: 
            ds = datasets[g_id]
            
            # Exhaustion check: Skip if this specific garment has no more trajectories
            if traj_idx >= traj_counts[g_id]:
                continue
                
            # 1. Fetch raw trajectory
            traj_data = ds.get_trajectory(traj_idx)
            obs = traj_data['observation']
            acts = traj_data['action']
            
            # 2. Extract sequence length for current observation
            L_obs = obs['rgb'].shape[0] 
            
            sk_len = garment_map[g_id]["orig_sk_len"]
            flat_len = garment_map[g_id]["orig_flat_len"]
            
            # 3. Apply uniform padding (max_len=17 supports the dress dataset)
            padded_semkeys = pad_keypoints(obs['semkey_norm_pixel'], max_len=17, pad_val=-1)
            padded_flat_semkeys = pad_keypoints(obs['flattened_goal_semkey_norm_pixel'], max_len=17, pad_val=-1)
            
            # 4. Construct unified observation dictionary
            new_obs = {
                'mask': obs['mask'],
                'depth': obs['depth'],
                'rgb': obs['rgb'],
                'goal_rgb': obs['goal_rgb'],
                'goal_depth': obs['goal_depth'],
                'goal_mask': obs['goal_mask'],
                'semkey_norm_pixel': padded_semkeys,
                'flattened_semkey_norm_pixel': padded_flat_semkeys, 
                
                # Broadcast static metadata across the temporal axis of the observation
                'garment_type': np.full((L_obs, 1), g_id, dtype=np.float32),
                'num_semkeys': np.full((L_obs, 1), sk_len, dtype=np.float32),
                'num_flattened_semkeys': np.full((L_obs, 1), flat_len, dtype=np.float32)
            }
            
            # 5. Passthrough actions
            new_acts = {'default': acts['default']}
            
            # 6. Commit to the new Zarr store
            merged_dataset.add_trajectory(observations=new_obs, actions=new_acts)

    print(f"\nMerging complete! Total unified trajectories written: {merged_dataset.num_trajectories()}")

if __name__ == "__main__":
    combine_datasets()