import argparse
import actoris_harena.api as ag_ar
from actoris_harena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap

def main():

   
    arena_name = 'softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:10'
    agent_name = 'oracle-garment|mask-biased-pick-and-place'


    arena = ag_ar.build_arena(arena_name + ',disp:False')
    agent = ag_ar.build_agent(agent_name, config=DotMap({
        'oracle': True
    }))
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)

    res = perform_single(arena, agent, mode='eval', 
        episode_config=None, collect_frames=False)
    
    

if __name__ == '__main__':
    main()