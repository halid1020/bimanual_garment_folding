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
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy import WaistLegFoldingStochasticPolicy

def main():

    task = 'waist-leg-alignment-folding'
    garment_type = 'trousers'
    mode = 'eval'

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
        'disp': True,
        'ray_id': 0,
        'horizon': 2,
        'track_semkey_on_frames': False,
        'readjust_pick': False,
        'grasp_mode': {'around': 1.0},
        'init_mode': 'flattened'
    }
    
    if task == 'centre-sleeve-fodling':
        demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': True})) # TODO: create demonstrator for 'centre-sleeve-folding'
    elif task == 'waist-leg-alignment-folding':
        demonstrator = WaistLegFoldingStochasticPolicy(DotMap({'debug': True}))
    else:
        raise NotImplementedError
    
    task_config = {
        'num_goals': 1,
        'demonstrator': demonstrator,
        'garment_type': garment_type,
        'asset_dir': 'assets',
        'task_name': task,
        'debug': False,
        'alignment': 'simple_rigid'
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFoldingTask(task_config)
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

    agent = arena.task.demonstrator

    for cfg in trials_configs[7:10]:
        agent.reset([arena.id])
        info = arena.reset(cfg)
        print(f"\n\nstep {arena.action_step} evaluation {info['evaluation']}")
        print(f"\nstep {arena.action_step} reward {info['reward']}")
        while not info['done']:
            action = agent.single_act(info)
            info = arena.step(action)
            print(f"\n\nstep {arena.action_step} evaluation {info['evaluation']}")
            print(f"\nstep {arena.action_step} reward {info['reward']}")
    

if __name__ == '__main__':
    main()