import argparse
import actoris_harena.api as ag_ar
from actoris_harena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os

from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_folding import GarmentFoldingTask
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.random.random_pick_and_fling import RandomPickAndFling
from controllers.human.human_multi_primitive import HumanMultiPrimitive
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy import WaistLegFoldingStochasticPolicy


def main():

    task = 'centre-sleeve-folding'
    garment_type = 'longsleeve'

    arena_config = {
        'garment_type': garment_type,
        'picker_radius': 0.03, #0.015,
        # 'particle_radius': 0.00625,
        'picker_threshold': 0.001, # 0.05,
        'picker_low': (-5, 0, -5),
        'picker_high': (5, 5, 5),
        "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': os.path.join('assets', 'init_states'),
        #'task': 'centre-sleeve-folding',
        'disp': False,
        'ray_id': 0,
        'horizon': 1,
        'track_semkey_on_frames': False,
        'readjust_pick': True,
        'grasp_mode': {'around': 1.0}
    }
    
    if task == 'centre-sleeve-folding':
        demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': True})) # TODO: create demonstrator for 'centre-sleeve-folding'
    elif task == 'waist-leg-alignment-folding':
        demonstrator = WaistLegFoldingStochasticPolicy(DotMap({'debug': True}))
    else:
        raise NotImplementedError
    
    task_config = {
        'num_goals': 10,
        'demonstrator': demonstrator,
        'garment_type': garment_type,
        'asset_dir': 'assets',
        'task_name': task,
        'debug': False,
        'alignment': 'simple_rigid',
        'goal_steps': 3
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFoldingTask(task_config)
    arena = SingleGarmentFixedInitialEnv(arena_config)
    arena.set_log_dir('./tmp/test_human')
    
    arena.set_task(task)

    agent = RandomPickAndFling(DotMap())
    
    res = perform_single(arena, agent, mode='eval', 
        episode_config={'eid':0, 'save_video': True}, collect_frames=False, debug=True)
    
    

if __name__ == '__main__':
    main()