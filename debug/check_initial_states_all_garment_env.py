import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os

from env.single_garment_subgoal_initial_env import SingleGarmentSubGoalEnv
from env.tasks.garment_folding import GarmentFoldingTask
from env.tasks.garment_flattening import GarmentFlatteningTask
from env.multi_garment_env import MultiGarmentEnv


from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy import WaistLegFoldingStochasticPolicy
from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy import WaistHemAlignmentFoldingStochasticPolicy

from controllers.random.random_multi_primitive import RandomMultiPrimitive
from agent_arena.utilities.perform_single import perform_single
from controllers.human.human_multi_primitive import HumanMultiPrimitive
import cv2

def main():
    task = 'flattening'
    garment_type = 'all'
    mode = 'train'
    reverse_trav = False

    arena_config = {
        'garment_type': garment_type,
        'picker_radius': 0.03, #0.015,
        # 'particle_radius': 0.00625,
        'picker_threshold': 0.0005, # 0.05,
        'picker_low': (-5, 0, -5),
        'picker_high': (5, 5, 5),
        "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': os.path.join('assets', 'init_states'),
        #'task': 'centre-sleeve-folding',
        'disp': False,
        'ray_id': 0,
        'horizon': 10,
        'track_semkey_on_frames': False,
        'readjust_pick': True,
        'provide_semkey_pos': True,
        'provide_flattened_semkey_pos': True,
        'grasp_mode': {'around': 1.0},
        'all_garment_types': ['longsleeve', 'trousers', 'skirt']
    }
    
    task_config = {
        'num_goals': 1,
        'garment_type': garment_type,
        'asset_dir': 'assets',
        'task_name': task,
        'debug': False,
        'alignment': 'simple_rigid',
        'goal_steps': 3
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFlatteningTask(task_config)
    arena = MultiGarmentEnv(arena_config)
    
    arena.set_task(task)

    
    if mode == 'train':
        trials_configs = arena.get_train_configs()
        arena.set_train()
    elif mode == 'val':
        trials_configs = arena.get_val_configs()
        arena.set_val()
    elif mode == 'eval':
        trials_configs = arena.get_eval_configs()
        arena.set_eval()

    agent = RandomMultiPrimitive(DotMap())
    #agent = HumanMultiPrimitive(DotMap())

    print(len(trials_configs))

    for cfg in trials_configs[112:]:
        #print('cfg', cfg)
        cfg['save_video'] = True
        perform_single(arena, agent, mode=mode, 
            episode_config=cfg, collect_frames=True)

if __name__ == '__main__':
    main()