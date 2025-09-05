import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap

from controllers.human.pixel_multi_primitive import PixelMultiPrimitive

def main():

   
    arena_name = 'softgym|domain:clothfunnels-realadapt-longsleeve,task:flattening,horizon:10'
    agent_name = 'human-pixel-pick-and-place-1'


    arena = ag_ar.build_arena(arena_name + ',disp:False')
    agent = PixelMultiPrimitive(DotMap())
    #logger = ag_ar.build_logger(arena.logger_name, config.save_dir)

    res = perform_single(arena, agent, mode='eval', 
        episode_config=None, collect_frames=False)
    
    

if __name__ == '__main__':
    main()