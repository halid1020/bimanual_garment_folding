import numpy as np

from statistics import mean

from .utils import *
from .garment_flattening_rewards import *
from .alignment import AlignmentTask

class CanonicalisationAlignmentTask(AlignmentTask):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'canonicalisation-alignment'
    
    def reset(self, arena):
        arena.flattened_obs = None
        arena.get_caon_flattened_obs()
        return super().reset(arena)