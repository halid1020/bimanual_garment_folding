import numpy as np
from statistics import mean

from real_robot.utils.mask_utils import get_max_IoU
from .utils import *


class RealWorldGarmentFlatteningTask():
    def __init__(self, config):
        self.goals = []
        self.config = config
        self.name = 'garment-flattening'
        self.semkey2pid = None 
    
    def reset(self, arena):
        self.goals = [[arena.flattened_obs]]
        self.ncs = []
        self.nis = []
        self.ious = []
        return {"goals": self.goals}

    def get_goals(self):
        return self.goals

    def get_goal(self):
        return self.goals[0]

    def reward(self, last_info, action, info):#
        reward = coverage_alignment_reward(last_info, action, info)
        if info['success']:
            reward = info['arena'].action_horizon - info['observation']['action_step']
        
        reward_ = reward
        
        if info['evaluation']['normalised_coverage'] > 0.7:
            reward_ += (info['evaluation']['normalised_coverage'] - 0.5)

        #print('rev aff score', aff_score_rev)
        return {
            'coverage_alignment': reward,
        }
    
    def evaluate(self, arena):
        eval_dict = {
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'normalised_improvement': self._get_normalised_impovement(arena)
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
            'maximum_trdef resetj_max_IoU_to_flattened': max(self.ious),
            'maximum_trj_normalised_coverage': max(self.ncs),
            'maximum_trj_normalised_improvement': max(self.nis),
        })
        return eval_dict

    def _get_normalised_coverage(self, arena):
        print(f'_get_normlised_coverage {arena.coverage}/{arena.flatten_coverage}')
        res = arena.coverage / arena.flatten_coverage
        
        # clip between 0 and 1
        return np.clip(res, 0, 1)
    
    def _get_normalised_impovement(self, arena):
        
        res = (arena.coverage - arena.init_coverage) / \
            (max(arena.flatten_coverage - arena.init_coverage, 0) + 1e-3)
        return np.clip(res, 0, 1)
    
    def _get_max_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        IoU, matched_IoU = get_max_IoU(cur_mask, arena.get_flattened_obs()['observation']['mask'], debug=self.config.debug)
        
        return IoU
    
    def success(self, arena):
        # return True
    
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