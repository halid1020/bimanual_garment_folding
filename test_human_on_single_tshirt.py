import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os

from env.single_tshirt_fixed_initial_env import SingleTshirtFixedInitialEnv

from controllers.human.human_multi_primitive import HumanMultiPrimitive

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
        'init_state_path': os.path.join(
            os.environ['AGENT_ARENA_PATH'], '..', 'data', 'cloth_funnel', 'init_states'),
        'task': 'centre-sleeve-folding',
        'disp': False,
        'ray_id': 0,
        'horizon': 10,
    }

    arena_config = DotMap(arena_config)
    arena = SingleTshirtFixedInitialEnv(arena_config)

    agent = HumanMultiPrimitive(DotMap())
    
    res = perform_single(arena, agent, mode='eval', 
        episode_config={'eid':0, 'save_video': False}, collect_frames=False)
    
    

if __name__ == '__main__':
    main()