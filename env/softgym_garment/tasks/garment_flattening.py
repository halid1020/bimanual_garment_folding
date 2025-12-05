import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from statistics import mean

from agent_arena import save_video
from .utils import get_max_IoU, IOU_FLATTENING_TRESHOLD, NC_FLATTENING_TRESHOLD
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
        self.goals = [[arena.flattened_obs]]
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
        if info['success'] and self.config.get('big_success_bonus', True):
            reward = info['arena'].action_horizon - info['observation']['action_step']
        
        reward_ = reward
        
        if info['evaluation']['normalised_coverage'] > 0.7:
            reward_ += (info['evaluation']['normalised_coverage'] - 0.5)

        threshold =  self.config.get('overstretch_penalty_threshold', 0)
        if info['overstretch'] > threshold:
           
            reward_ -= self.config.get("overstretch_penalty_scale", 0) * (info['overstretch'] - threshold)
        
        reward_2 = reward_
        aff_score_pen = (1 - info.get('action_affordance_score', 1))
        reward_2 -= self.config.get("affordance_penalty_scale", 0) * aff_score_pen
    
        #print('rev aff score', aff_score_rev)
        return {
            'coverage_alignment': reward,
            'coverage_alignment_with_stretch_penalty_high_coverage_bonus': reward_,
            'coverage_alignment_with_stretch_and_affordance_penalty_high_coverage_bonus': reward_2
        }
    
    def evaluate(self, arena):
        eval_dict = {
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'normalised_improvement': self._get_normalised_impovement(arena),
            'overstretch': arena.overstretch
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
        return IoU > IOU_FLATTENING_TRESHOLD and coverage > NC_FLATTENING_TRESHOLD
    
    def compare(self, results_1, results_2):
        threshold=0.95

        # --- Compute averages for results_1 ---
        avg_nc_1 = mean([ep["normalised_coverage"][-1] for ep in results_1])
        avg_iou_1 = mean([ep["max_IoU_to_flattened"][-1] for ep in results_1])
        avg_len_1 = mean([len(ep["max_IoU_to_flattened"]) for ep in results_1])
        score_1 = avg_nc_1 + avg_iou_1

        # --- Compute averages for results_2 ---
        avg_nc_2 = mean([ep["normalised_coverage"][-1] for ep in results_2])
        avg_iou_2 = mean([ep["max_IoU_to_flattened"][-1] for ep in results_2])
        avg_len_2 = mean([len(ep["max_IoU_to_flattened"]) for ep in results_2])
        score_2 = avg_nc_2 + avg_iou_2

        # --- Both are very good → prefer shorter trajectory ---
        if score_1 > 2 * threshold and score_2 > 2 * threshold:
            if avg_len_1 < avg_len_2:
                return 1
            elif avg_len_1 > avg_len_2:
                return -1
            else:
                return 0

        # --- Otherwise prefer higher score ---
        if score_1 > score_2:
            return 1
        elif score_1 < score_2:
            return -1
        else:
            return 0