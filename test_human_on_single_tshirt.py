import argparse
import agent_arena.api as ag_ar
from agent_arena.utilities.perform_single import perform_single
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap

from env.domain_builder import DomainBuilder

from controllers.human.human_multi_primitive import HumanMultiPrimitive

def main():


    arena = DomainBuilder.build_from_config(
        domain='single-tshirt-fixed-initial',
        task='center-sleeve-folding',
        horizon=10,
        disp=True,
    )

    agent = HumanMultiPrimitive(DotMap())
    
    res = perform_single(arena, agent, mode='eval', 
        episode_config={'eid':0, 'save_video': False}, collect_frames=False)
    
    

if __name__ == '__main__':
    main()