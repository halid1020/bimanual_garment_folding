import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from agent_arena import Task
from agent_arena import save_video
from .utils import get_max_IoU
from .folding_rewards import *

class GarmentFlatteningTask(Task):
    def __init__(self, config):
        self.goals = []
        self.config = config
        self.name = 'garment-flattening'
    
    def reset(self, arena):
        #self.cur_coverage = self._get_normalised_coverage(arena)
        #info = self._process_info(arena)
        self.goals = [arena.flattened_obs]
        #self._save_goal(arena)
        return {"goals": self.goals}

    def get_goals(self):
        return self.goals

    def get_goal(self):
        return self.goals[0]

    def reward(self, last_info, action, info):#
        reward = coverage_alignment_reward(last_info, action, info)
        if info['success']:
            reward = info['arena'].horizon - info['observation']['action_step']
        return {
            'coverage_alignment': reward
        }
    
    def evaluate(self, arena):
        return {
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'normalised_improvement': self._get_normalised_impovement(arena),
        }

    def _get_normalised_coverage(self, arena):
        res = arena._get_coverage() / arena.flatten_coverage
        
        # clip between 0 and 1
        return np.clip(res, 0, 1)
    
    def _get_normalised_impovement(self, arena):
        
        res = (arena._get_coverage() - arena.init_coverae) / \
            (max(arena.flatten_coverage - arena.init_coverae, 0) + 1e-3)
        return np.clip(res, 0, 1)
    
    def _get_max_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        IoU, matched_IoU = get_max_IoU(cur_mask, arena.get_flattened_obs()['observation']['mask'], debug=self.config.debug)
        
        return IoU
    
    def success(self, arena):
        cur_eval = self.evaluate(arena)
        IoU = cur_eval['max_IoU_to_flattened']
        coverage = cur_eval['normalised_coverage']
        return IoU > 0.85 and coverage > 0.99