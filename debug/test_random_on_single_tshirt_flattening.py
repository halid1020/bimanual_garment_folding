import argparse
import actoris_harena.api as ag_ar
from actoris_harena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os

from env.multi_garment_env import MultiGarmentEnv
from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_flattening import GarmentFlatteningTask
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.human.human_multi_primitive import HumanMultiPrimitive
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy

def main():

    arena_config = {
        'name': 'multi-garment-longsleeve-env',
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
        'track_semkey_on_frames': False
    }
    
    task_config = {
        'num_goals': 1,
        'object': 'longsleeve',
        'asset_dir': 'assets',
        'task_name': 'flattening', # TODO: indicate what kinds of folding,
        'debug': True,
        'alignment': 'simple_rigid'
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFlatteningTask(task_config)
    arena = MultiGarmentEnv(arena_config)
    
    arena.set_task(task)
    

    agent = RandomMultiPrimitive(DotMap())
    
    res = perform_single(arena, agent, mode='eval', 
        episode_config={'eid':0, 'save_video': True}, collect_frames=False, debug=True)
    
    

if __name__ == '__main__':
    main()