import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os

from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_folding import GarmentFoldingTask
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.Image_based_multi_primitive_SAC import ImageBasedMultiPrimitiveSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy

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
        'disp': True,
        'ray_id': 0,
        'horizon': 8,
        'track_semkey_on_frames': True
    }
    
    demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': False})) # TODO: create demonstrator for 'centre-sleeve-folding'
    
    task_config = {
        'num_goals': 1,
        'demonstrator': demonstrator,
        'object': 'longsleeve',
        'asset_dir': 'assets',
        'task_name': 'centre-sleeve-folding',
        'debug': False,
        'alignment': 'deform'
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFoldingTask(task_config)
    arena = SingleGarmentFixedInitialEnv(arena_config)
    
    arena.set_task(task)

    agent = ImageBasedMultiPrimitiveSAC(config=None)
    
    res = perform_single(arena, agent, mode='eval', 
        episode_config={'eid':0, 'save_video': True}, 
        collect_frames=False, debug=True)
    
    

if __name__ == '__main__':
    main()