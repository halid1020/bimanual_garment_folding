# Author: Halid Kadi

from tool.lagarnet_utils import obs_config, action_config, reward_names, plot_results, evaluation_names #see from the folder
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
import argparse
import os

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process trajectory datasets.")
    
    # Add arguments
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='mask_biased_random_on_multi_longsleeve_with_single_picker_pick_and_place',
        help='Name of the dataset folder (e.g., mask_biased_random...)'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data/datasets',
        help='Root directory where datasets are stored'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to plot'
    )

    args = parser.parse_args()

    # Use arguments
    print(f"Loading data from: {os.path.join(args.data_dir, args.data_path)}")

    dataset = TrajectoryDataset(
        data_path=args.data_path,
        data_dir=args.data_dir,
        whole_trajectory=True,
        io_mode='r',
        obs_config=obs_config,
        act_config=action_config
    )

    for i in range(args.num_samples):
        try:
            data = dataset[i]
        except IndexError:
            print(f"Index {i} out of range for dataset length {len(dataset)}")
            break

        rgbs = data['observation']['rgb']
        # print('data keys', data['observation'].keys(), data['action'].shape)
        
        # Check if action key exists to avoid errors on different datasets
        if 'norm-pixel-pick-and-place' in data['action']:
            actions = data['action']['norm-pixel-pick-and-place']
        else:
            actions = None # Or handle accordingly
            
        reward_dict = {key: data['observation'][key] for key in reward_names if key in data['observation']}
        evaluation_dict = {key: data['observation'][key] for key in evaluation_names if key in data['observation']}

        depths = data['observation']['depth'] - 1.5
        masks = data['observation']['mask']

        if 'success' in data['observation']:
            # Take the value from the last step
            is_success = data['observation']['success'][-1]
            status_str = "SUCCESS" if is_success else "FAIL"
        else:
            status_str = "UNKNOWN"
        
        print(f"Sample {i}: {status_str}")

        success_data = None
        if 'success' in data['observation']:
            success_data = data['observation']['success']

        plot_results(
            rgbs, depths, masks,    
            actions=actions, 
            reward_dict=reward_dict, 
            evaluation_dict=evaluation_dict, 
            success=success_data, # <--- Pass it here
            filename='data_{}'.format(i)
        )
    
    print("Done.")

if __name__ == '__main__':
    main()