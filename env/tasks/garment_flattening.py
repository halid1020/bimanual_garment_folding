import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


from agent_arena import save_video
from .utils import get_max_IoU
from .folding_rewards import *
from .garment_task import GarmentTask

class GarmentFlatteningTask(GarmentTask):
    def __init__(self, config):
        super().__init__(config)
        self.goals = []
        self.config = config
        self.name = 'garment-flattening'
        self.semkey2pid = None 
    
    def reset(self, arena):
        #self.cur_coverage = self._get_normalised_coverage(arena)
        #info = self._process_info(arena)
        self.semkey2pid = self._load_or_create_keypoints(arena)
        self.goals = [arena.flattened_obs]
        self.ncs = []
        self.nis = []
        self.ious = []
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
        
        reward_ = reward
        
        if info['evaluation']['normalised_coverage'] > 0.7:
            reward_ += (info['evaluation']['normalised_coverage'] - 0.5)

        if info['over_strech']:
            reward_ = 0

        return {
            'coverage_alignment': reward,
            'coverage_alignment_with_strech_penality_high_coverage_bonus': reward_
        }
    
    def evaluate(self, arena):
        eval_dict = {
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'normalised_improvement': self._get_normalised_impovement(arena),
        }

        if arena.action_step == len(self.ncs):
            self.ncs.append(eval_dict['normalised_coverage'])
            self.nis.append(eval_dict['normalised_improvement'])
            self.ious.append(eval_dict['max_IoU_to_flattened'])

        if arena.action_step < len(self.ncs):
            self.ncs[arena.action_step] = eval_dict['normalised_coverage']
            self.nis[arena.action_step] = eval_dict['normalised_improvement']
            self.ious[arena.action_step] = eval_dict['max_IoU_to_flattened']
        
        eval_dict.update({
            'maximum_trj_max_IoU_to_flattened': max(self.ious),
            'maximum_trj_normalised_coverage': max(self.ncs),
            'maximum_trj_normalised_improvement': max(self.nis),
        })
        return eval_dict

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


    