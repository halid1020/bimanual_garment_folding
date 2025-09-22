import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap

from controllers.random.random_multi_primitive import RandomMultiPrimitive

def main():

   
    arena_name = 'softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:10'


    arena = ag_ar.build_arena(arena_name + ',disp:True')
    agent = RandomMultiPrimitive(DotMap())
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)

    res = perform_single(arena, agent, mode='eval', 
        episode_config=None, collect_frames=False)
    
    

if __name__ == '__main__':
    main()