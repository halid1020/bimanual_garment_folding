"""
Real-World Garment Data to TrajectoryDataset Converter

This script parses a directory tree of real-world human demonstration data
and converts it into the unified Zarr TrajectoryDataset format. 

Key operations:
1. Resizes all RGB, Depth, and Mask images to 128x128.
2. Parses discrete primitive actions into the continuous [9] vector format.
3. Injects -1 arrays for missing semantic keypoints.
4. Broadcasts the final step's images as the episode's "goal" states.
5. Appends a duplicated final observation to satisfy the N+1 observation 
   requirement for N actions in Actoris-Harena.
6. Recursively searches for episode folders to handle nested checkpoints.

Usage:
    python tool/magpie/convert_real_data_to_trajectory_dataset.py --source_dir <raw_real_world_data_path>
"""

import os
import json
import glob
import re
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from actoris_harena import TrajectoryDataset

# ==========================================
# Target Dataset Configuration
# ==========================================
MERGED_DATASET_CONFIG = {
    "data_path": "real_world_longsleeve",
    "data_dir": "./data/datasets",
    "split_ratios": [0.0, 0.05, 0.95],
    "seq_length": 1,
    "io_mode": "w",
    "cache_in_memory": False,
    "obs_config": {
        "mask": {"shape": [128, 128, 1], "output_key": "mask"},
        "depth": {"shape": [128, 128, 1], "output_key": "depth"},
        "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
        "semkey_norm_pixel": {"shape": [15, 2], "output_key": "semkey_norm_pixel"}, # Longsleeve default is 15
        "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
        "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
        "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
        "flattened_semkey_norm_pixel": {"shape": [15, 2], "output_key": "flattened_goal_semkey_norm_pixel"},
        "garment_type": {"shape": [1], "output_key": "garment_type"},
        "num_semkeys": {"shape": [1], "output_key": "num_semkeys"},
        "num_flattened_semkeys": {"shape": [1], "output_key": "num_flattened_semkeys"}
    },
    "act_config": {
        "default": {"shape": [9], "output_key": "default"}
    }
}

# ==========================================
# Primitive Action Mappings
# ==========================================
PRIMITIVES = [
    "norm-pixel-pick-and-fling",
    "norm-pixel-dual-pick-and-place",
    "norm-pixel-single-pick-and-place",
    "no-operation"
]
PRIM_NAME2ID = {name: i for i, name in enumerate(PRIMITIVES)}
K_PRIMITIVES = len(PRIMITIVES)

def natural_sort_key(s):
    """Sorts strings containing numbers numerically (e.g., step_2 before step_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_and_resize_image(path, img_type='rgb', target_size=(128, 128)):
    """Loads an image from disk, resizes it, and returns a float32 numpy array."""
    if img_type == 'rgb':
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img_type == 'depth':
        # IMREAD_ANYDEPTH preserves 16-bit data if your depth maps are 16-bit PNGs
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    else: # mask
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Failed to load image at {path}")

    # Resize using NEAREST interpolation for masks/depths to avoid artifacting, LINEAR for RGB
    interp = cv2.INTER_LINEAR if img_type == 'rgb' else cv2.INTER_NEAREST
    img_resized = cv2.resize(img, target_size, interpolation=interp)
    
    # Ensure correct shape for channel dimensions
    if len(img_resized.shape) == 2:
        img_resized = np.expand_dims(img_resized, axis=-1)
        
    return img_resized.astype(np.float32)

def process_action_file(json_path):
    """Parses action.json into the unified 9-dimensional array format."""
    with open(json_path, 'r') as f:
        action_data = json.load(f)
    
    # Extract primitive name and parameters
    action_name = list(action_data.keys())[0]
    params = action_data[action_name]
    
    # Calculate primitive ID mapping
    prim_id = PRIM_NAME2ID[action_name]
    prim_act = (1.0 * (prim_id + 0.5) / K_PRIMITIVES * 2 - 1)
    
    # Build the 9-dimensional action vector
    action_vector = np.zeros(9, dtype=np.float32)
    action_vector[0] = prim_act
    
    for i, param_val in enumerate(params):
        if i + 1 < 9: # Safeguard to prevent out-of-bounds
            action_vector[i + 1] = float(param_val)
            
    return action_vector

def convert_real_world_data(source_dir: str):
    """Main function to parse directory and write to Zarr."""
    print(f"Initializing real-world TrajectoryDataset from source: {source_dir}")
    dataset = TrajectoryDataset(**MERGED_DATASET_CONFIG)
    
    # RECURSIVE SEARCH: Find all episode directories anywhere inside the source_dir
    search_pattern = os.path.join(source_dir, "**", "episode_*")
    all_matches = glob.glob(search_pattern, recursive=True)
    
    # Filter to only include directories, then sort
    episode_dirs = [d for d in all_matches if os.path.isdir(d)]
    episode_dirs = sorted(episode_dirs, key=natural_sort_key)
    
    if not episode_dirs:
        print(f"Warning: No episode directories found in {source_dir} or its subdirectories.")
        return

    # Garment metadata (0 = longsleeve)
    g_id = 0
    sk_len = 15
    
    for ep_dir in tqdm(episode_dirs, desc="Processing Episodes", unit="ep"):
        step_dirs = glob.glob(os.path.join(ep_dir, "step_*"))
        step_dirs = sorted(step_dirs, key=natural_sort_key)
        
        if not step_dirs:
            continue
            
        ep_obs_rgb = []
        ep_obs_depth = []
        ep_obs_mask = []
        ep_actions = []
        
        # 1. Iterate through every step folder to gather states and actions
        for step_dir in step_dirs:
            rgb = load_and_resize_image(os.path.join(step_dir, "rgb.png"), 'rgb')
            depth = load_and_resize_image(os.path.join(step_dir, "depth.png"), 'depth')
            mask = load_and_resize_image(os.path.join(step_dir, "mask.png"), 'mask')
            action = process_action_file(os.path.join(step_dir, "action.json"))
            
            ep_obs_rgb.append(rgb)
            ep_obs_depth.append(depth)
            ep_obs_mask.append(mask)
            ep_actions.append(action)
            
        # 2. Fulfill N+1 observation requirement
        ep_obs_rgb.append(ep_obs_rgb[-1])
        ep_obs_depth.append(ep_obs_depth[-1])
        ep_obs_mask.append(ep_obs_mask[-1])
        
        # Determine sequence length for this episode (which is N + 1)
        L_obs = len(ep_obs_rgb)
        
        # 3. Extract goals (derived from the final appended state)
        goal_rgb = ep_obs_rgb[-1]
        goal_depth = ep_obs_depth[-1]
        goal_mask = ep_obs_mask[-1]
        
        # 4. Generate dummy semantic keypoints (-1 padding)
        dummy_semkeys = np.full((L_obs, 15, 2), -1.0, dtype=np.float32)
        
        # 5. Package the trajectory 
        new_obs = {
            'rgb': np.array(ep_obs_rgb),
            'depth': np.array(ep_obs_depth),
            'mask': np.array(ep_obs_mask),
            'goal_rgb': np.array([goal_rgb] * L_obs),
            'goal_depth': np.array([goal_depth] * L_obs),
            'goal_mask': np.array([goal_mask] * L_obs),
            'semkey_norm_pixel': dummy_semkeys,
            'flattened_semkey_norm_pixel': dummy_semkeys,
            'garment_type': np.full((L_obs, 1), g_id, dtype=np.float32),
            'num_semkeys': np.full((L_obs, 1), sk_len, dtype=np.float32),
            'num_flattened_semkeys': np.full((L_obs, 1), sk_len, dtype=np.float32)
        }
        
        new_acts = {
            'default': np.array(ep_actions)
        }
        
        # 6. Commit to Zarr
        dataset.add_trajectory(observations=new_obs, actions=new_acts)

    print(f"\nConversion complete! Total real-world trajectories saved: {dataset.num_trajectories()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert real-world human demonstration data to Actoris-Harena TrajectoryDataset format.")
    
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing episode folders"
    )
    
    args = parser.parse_args()
    
    convert_real_world_data(args.source_dir)