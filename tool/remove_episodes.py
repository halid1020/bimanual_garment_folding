import os
import zarr
import argparse
import numpy as np
import shutil
from tqdm import tqdm

def remove_episodes(input_path, output_path, episodes_to_remove):
    print(f"Loading original dataset from: {input_path}")
    store_in = zarr.DirectoryStore(input_path)
    root_in = zarr.open(store_in, mode='r')

    if os.path.exists(output_path):
        print(f"Output path '{output_path}' already exists. Removing it...")
        shutil.rmtree(output_path)

    print(f"Creating new dataset at: {output_path}")
    store_out = zarr.DirectoryStore(output_path)
    root_out = zarr.open(store_out, mode='w')

    # Copy root attributes
    root_out.attrs.update(root_in.attrs)

    # 1. Calculate trajectory boundaries
    traj_lengths_in = root_in['trajectory_lengths'][:]
    num_total_episodes = len(traj_lengths_in)
    traj_starts = np.concatenate(([0], np.cumsum(traj_lengths_in)[:-1]))

    # Identify which episodes to keep
    episodes_to_keep = [i for i in range(num_total_episodes) if i not in episodes_to_remove]
    print(f"Total episodes: {num_total_episodes} | Removing {len(episodes_to_remove)} | Keeping {len(episodes_to_keep)}")

    # 2. Process 'trajectory_lengths' (Episode-aligned)
    new_traj_lengths = traj_lengths_in[episodes_to_keep]
    root_out.create_dataset(
        'trajectory_lengths', 
        data=new_traj_lengths, 
        chunks=(1000,),
        dtype=traj_lengths_in.dtype
    )

    # 3. Process data groups (observation, action, goal)
    for group_name in root_in.group_keys():
        group_in = root_in[group_name]
        group_out = root_out.create_group(group_name)
        group_out.attrs.update(group_in.attrs)

        # Check if the group is episode-aligned (goals) or timestep-aligned (obs, act)
        is_episode_aligned = (group_name == 'goal')

        for arr_name in group_in.array_keys():
            arr_in = group_in[arr_name]
            shape = arr_in.shape
            
            # Create an empty elastic dataset in the new store
            arr_out = group_out.create_dataset(
                arr_name,
                shape=(0,) + shape[1:],
                dtype=arr_in.dtype,
                chunks=arr_in.chunks
            )

            # Copy data over iteratively to manage RAM safely
            desc = f"Copying {group_name}/{arr_name}"
            for ep_idx in tqdm(episodes_to_keep, desc=desc, leave=False):
                if is_episode_aligned:
                    # Episode aligned: fetch exactly 1 row corresponding to the episode
                    chunk_data = arr_in[ep_idx : ep_idx + 1]
                else:
                    # Timestep aligned: fetch the slice of timesteps for the episode
                    start_idx = traj_starts[ep_idx]
                    end_idx = start_idx + traj_lengths_in[ep_idx]
                    chunk_data = arr_in[start_idx : end_idx]
                
                # Append the chunk to the new array
                arr_out.append(chunk_data)

    print(f"\nSuccess! Filtered dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove specific episodes from a Zarr trajectory dataset.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the source .zarr directory")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the cleaned .zarr directory")
    parser.add_argument('--remove', type=int, nargs='+', required=True, help="List of episode indices to remove (e.g., --remove 105 126 174)")
    
    args = parser.parse_args()
    
    # Ensure indices are unique
    episodes_to_remove = set(args.remove)
    remove_episodes(args.input_path, args.output_path, episodes_to_remove)