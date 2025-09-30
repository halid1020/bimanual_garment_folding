import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os

from env.multi_garment_env import MultiGarmentEnv
from env.tasks.garment_folding import GarmentFoldingTask
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
import cv2

def main():

    arena_config = {
        'object': 'longsleeve',
        'picker_radius': 0.03, #0.015,
        # 'particle_radius': 0.00625,
        'picker_threshold': 0.007, # 0.05,
        'picker_low': (-5, 0, -5),
        'picker_high': (5, 5, 5),
        'grasp_mode': {'closest': 1.0},
        "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': os.path.join('assets', 'init_states'),
        #'task': 'centre-sleeve-folding',
        'disp': False,
        'ray_id': 0,
        'horizon': 8,
        'track_semkey_on_frames': True
    }
    
    demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': True}))
    
    task_config = {
        'num_goals': 10,
        'demonstrator': demonstrator,
        'object': 'longsleeve',
        'asset_dir': 'assets',
        'task_name': 'centre-sleeve-folding', # TODO: indicate what kinds of folding,
        'debug': True,
        'alignment': 'simple_rigid'
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFoldingTask(task_config)
    arena = MultiGarmentEnv(arena_config)
    
    arena.set_task(task)

    train_trials_configs = arena.get_train_configs()
    arena.set_train()
    for cfg in train_trials_configs[:1]:
        print('cfg', cfg)
        arena.reset(cfg)

if __name__ == '__main__':
    main()