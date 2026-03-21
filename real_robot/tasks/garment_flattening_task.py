import numpy as np
from statistics import mean

from real_robot.utils.mask_utils \
    import get_max_IoU, calculate_iou
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
        self.canon_ious = []
        return {"goals": self.goals}

    def get_goals(self):
        return self.goals

    def get_goal(self):
        return self.goals[0]

    def reward(self, last_info, action, info):#
        #print('rev aff score', aff_score_rev)
        return {
            'coverage_alignment': coverage_alignment_bonus_and_penalty(last_info, action, info, self.config),
        }
    
    def evaluate(self, arena):
        eval_dict = {
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'normalised_improvement': self._get_normalised_improvement(arena),
            'canon_IoU_to_flattened': self._get_canon_IoU_to_flattened(arena),
        }

        if arena.action_step == len(self.ncs):
            self.ncs.append(eval_dict['normalised_coverage'])
            self.nis.append(eval_dict['normalised_improvement'])
            self.ious.append(eval_dict['max_IoU_to_flattened'])
            self.canon_ious.append(eval_dict['canon_IoU_to_flattened'])

        if arena.action_step < len(self.ncs):
            self.ncs[arena.action_step] = eval_dict['normalised_coverage']
            self.nis[arena.action_step] = eval_dict['normalised_improvement']
            self.ious[arena.action_step] = eval_dict['max_IoU_to_flattened']
            self.canon_ious[arena.action_step] = eval_dict['canon_IoU_to_flattened']

        eval_dict.update({
            'maximum_trj_max_IoU_to_flattened': max(self.ious),
            'maximum_trj_canon_IoU_to_flattened': max(self.canon_ious),
            'maximum_trj_normalised_coverage': max(self.ncs),
            'maximum_trj_normalised_improvement': max(self.nis),
        })
        return eval_dict

    def _get_normalised_coverage(self, arena):
        res = arena.coverage / arena.flatten_coverage
        return np.clip(res, 0, 1)
    
    def _get_canon_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        goal_mask = arena.get_flattened_obs()['observation']['mask']
        IoU = calculate_iou(cur_mask, goal_mask)
        return IoU
    
    def _get_normalised_improvement(self, arena):
        print(f'current coverage {arena.coverage}, init coverage {arena.init_coverage}, flatten coverage {arena.flatten_coverage}')
        
        # Cast to float to prevent unsigned integer underflow
        cov = float(arena.coverage)
        init_cov = float(arena.init_coverage)
        flat_cov = float(arena.flatten_coverage)
        
        # Now the math will evaluate to negative numbers correctly
        res = (cov - init_cov) / (max(flat_cov - init_cov, 0.0) + 1e-3)
        
        res = np.clip(res, 0.0, 1.0)
        print(f'normalised improvments', res)
        
        return res
    
    def _get_max_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        IoU, matched_IoU = get_max_IoU(cur_mask, arena.get_flattened_obs()['observation']['mask'], debug=self.config.debug)
        
        return IoU
    

    def _get_canon_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        goal_mask = arena.get_flattened_obs()['observation']['mask']
        IoU = calculate_iou(cur_mask, goal_mask)
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