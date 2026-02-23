import os
import numpy as np
from tqdm import tqdm
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset

def combine_datasets():
    # -------------------------------------------------------------------------
    # 1. Configuration for the NEW Combined Dataset
    # -------------------------------------------------------------------------
    combined_config = {
        "data_path": "combined_folding_and_flattening_no_goal",
        "data_dir": "./data/datasets",
        "seq_length": 1,
        "io_mode": "w",  # Write mode to create/overwrite
        "cache_in_memory": False, # False for writing to save RAM
        "obs_config": {
            "robot0_mask": {"shape": [128, 128, 1], "output_key": "robot0_mask"},
            "robot1_mask": {"shape": [128, 128, 1], "output_key": "robot1_mask"},
            "mask":        {"shape": [128, 128, 1], "output_key": "mask"},
            "depth":       {"shape": [128, 128, 1], "output_key": "depth"},
            "rgb":         {"shape": [128, 128, 3], "output_key": "rgb"},
        },
        "act_config": {
            "default": {"shape": [9], "output_key": "default"}
        }
    }

    # -------------------------------------------------------------------------
    # 2. Configuration for Source Dataset A (Folding - No Goals)
    # -------------------------------------------------------------------------
    source_A_config = {
        "data_path": "multi_longsleeve_multi_primitive_folding_from_random_flattened_horizon_10_human_demo_100_workspace",
        "data_dir": "./data/datasets",
        "seq_length": 1,
        "io_mode": "r", # Read mode
        "cache_in_memory": True,
        # Same structure as combined
        "obs_config": combined_config["obs_config"],
        "act_config": combined_config["act_config"]
    }

    # -------------------------------------------------------------------------
    # 3. Configuration for Source Dataset B (Flattening - Has Goals)
    # -------------------------------------------------------------------------
    source_B_config = {
        "data_path": "multi_longsleeve_multi_primitive_flattening_horizon_10_workspace_snap_human_demo_100",
        "data_dir": "./data/datasets",
        "seq_length": 1,
        "io_mode": "r",
        "cache_in_memory": True,
        "obs_config": {
            # Has extra goal keys
            "robot0_mask": {"shape": [128, 128, 1], "output_key": "robot0_mask"},
            "robot1_mask": {"shape": [128, 128, 1], "output_key": "robot1_mask"},
            "mask":        {"shape": [128, 128, 1], "output_key": "mask"},
            "depth":       {"shape": [128, 128, 1], "output_key": "depth"},
            "rgb":         {"shape": [128, 128, 3], "output_key": "rgb"},
            "goal_rgb":    {"shape": [128, 128, 3], "output_key": "goal_rgb"},
            "goal_depth":  {"shape": [128, 128, 1], "output_key": "goal_depth"},
            "goal_mask":   {"shape": [128, 128, 1], "output_key": "goal_mask"},
        },
        "act_config": {
            "default": {"shape": [9], "output_key": "default"}
        }
    }

    # -------------------------------------------------------------------------
    # 4. Initialize Datasets
    # -------------------------------------------------------------------------
    print("Initializing Target Dataset...")
    target_ds = TrajectoryDataset(**combined_config)

    print("Initializing Source A (Folding)...")
    source_A = TrajectoryDataset(**source_A_config)

    print("Initializing Source B (Flattening)...")
    source_B = TrajectoryDataset(**source_B_config)

    # List of keys we want to keep (intersection of what we want)
    keep_obs_keys = list(combined_config["obs_config"].keys())
    keep_act_keys = list(combined_config["act_config"].keys())

    # -------------------------------------------------------------------------
    # 5. Copy Loop
    # -------------------------------------------------------------------------
    
    def copy_dataset(source_ds, name):
        num_episodes = source_ds.num_trajectories()
        print(f"Copying {num_episodes} episodes from {name}...")

        for i in tqdm(range(num_episodes)):
            # Load episode from source
            episode = source_ds.get_trajectory(i)
            
            # Extract Observations
            # The get_trajectory method usually returns a dict like {'obs': ..., 'action': ...}
            # or it might return them separately depending on your specific version of TrajectoryDataset.
            # Assuming standard structure where episode is a dict containing keys for obs and actions directly or nested.
            
            # Let's handle the specific return structure of TrajectoryDataset.
            # Usually it allows access by keys.
            
            new_obs = {}
            new_act = {}

            # Filter Observations
            for key in keep_obs_keys:
                if key in episode:
                    new_obs[key] = episode[key]
                elif 'observation' in episode and key in episode['observation']:
                     new_obs[key] = episode['observation'][key]
                else:
                    # Fallback: try to find it in the raw dict
                    # Some implementations flatten the dict, others nest it.
                    # We assume the source keys match the config keys.
                    if key in source_ds.keys: # Check if key exists in hdf5 group
                        new_obs[key] = episode[key]

            # Filter Actions
            for key in keep_act_keys:
                 if key in episode:
                    new_act[key] = episode[key]
                 elif 'action' in episode and key in episode['action']:
                    new_act[key] = episode['action'][key]

            # Add to target
            target_ds.add_trajectory(new_obs, new_act)

    # Run Copy
    copy_dataset(source_A, "Source A (No Goal)")
    copy_dataset(source_B, "Source B (Has Goal - Stripping)")

    print(f"Success! Combined dataset created at: {os.path.join(combined_config['data_dir'], combined_config['data_path'])}")
    print(f"Total trajectories: {target_ds.num_trajectories()}")

if __name__ == "__main__":
    combine_datasets()