import argparse
import agent_arena.api as ag_ar
from agent_arena import train_and_evaluate
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os
import torch

from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_folding import GarmentFoldingTask
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
from controllers.data_augmentation.pixel_based_primitives_data_augmenter import PixelBasedPrimitiveDataAugmenter

def main():

    arena_config = {
        'garment_type': 'longsleeve',
        'picker_radius': 0.03, 
        'picker_threshold': 0.007, # 0.05,
        'picker_low': (-5, 0, -5),
        'picker_high': (5, 5, 5),
        'grasp_mode': {'closest': 1.0},
        "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': os.path.join('assets', 'init_states'),
        'disp': False,
        'ray_id': 0,
        'horizon': 4,
        'track_semkey_on_frames': False,
        'init_mode': "flattened"
    }
    
    demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': False}))
    task_config = {
        'num_goals': 10,
        'demonstrator': demonstrator,
        'object': 'longsleeve',
        'asset_dir': 'assets',
        'task_name': 'centre-sleeve-folding',
        'debug': False,
        'alignment': 'deform'
    }

    validation_interval = int(1e3)
    total_update_steps = int(1e5)
    eval_checkpoint = -1
    
    exp_config = ag_ar.retrieve_config_from_path(
        config_path='./train/diffusion.yaml')

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFoldingTask(task_config)
    arena = SingleGarmentFixedInitialEnv(arena_config)
    
    arena.set_task(task)

    agent =  ag_ar.build_agent('diffusion_policy', exp_config)
    ag_ar.register_agent('centre_sleeve_folding_stochastic_policy', CentreSleeveFoldingStochasticPolicy)

    save_dir = os.path.join('/data/hcv530', 'garment_folding', 'test_diffusion_folding_from_flattened')
    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)

    data_augmenter = PixelBasedPrimitiveDataAugmenter(exp_config.data_augmenter.params)
    agent.set_data_augmenter(data_augmenter)
   
    res = ag_ar.train_and_evaluate(agent, arena,
        validation_interval, total_update_steps, eval_checkpoint)


if __name__ == '__main__':
    main()