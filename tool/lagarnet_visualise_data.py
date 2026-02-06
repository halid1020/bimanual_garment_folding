# Author: Halid Kadi

from tool.lagarnet_utils import obs_config, action_config, reward_names, plot_results, evaluation_names #see from the folder
from agent_arena.utilities.trajectory_dataset import TrajectoryDataset
import argparse


def main():
    data_path = 'mask_biased_random_on_multi_longsleeve_with_single_picker_pick_and_place'
    data_dir = "./data/datasets"

    dataset = TrajectoryDataset(
        data_path=data_path,
        data_dir=data_dir,
        whole_trajectory=True,
        io_mode='r',
        obs_config=obs_config,
        act_config=action_config
    )

    for i in range(5):
        data = dataset[i]
        rgbs = data['observation']['rgb']
        # print('data keys', data['observation'].keys(), data['action'].shape)
        actions = data['action']['norm-pixel-pick-and-place']
        reward_dict = {key: data['observation'][key] for key in reward_names}
        evaluation_dict = {key: data['observation'][key] for key in evaluation_names}

        depths = data['observation']['depth'] - 1.5
        masks = data['observation']['mask']
        
        plot_results(
            rgbs, depths, masks,    
            actions=actions, reward_dict=reward_dict, 
            evaluation_dict=evaluation_dict, filename='data_{}'.format(i))
    
    

if __name__ == '__main__':
    main()