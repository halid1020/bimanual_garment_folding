import numpy as np
import cv2
from statistics import mean
from termcolor import colored

from .utils import *
from real_robot.utils.mask_utils import calculate_iou
from .garment_flattening_task import RealWorldGarmentFlatteningTask

class RealWorldGarmentAlignmentTask(RealWorldGarmentFlatteningTask):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'algnicalisation-alignment'
        self.randomise_goal = config.get('randomise_goal', False)
    
    def reset(self, arena):
        self.info = super().reset(arena)

        # self.goals[0][0] holds a reference to arena.flattened_obs
        flattened_goal = arena.flattened_obs
        obs = flattened_goal['observation']

        # Interactive drag/rotate UI (behavior unchanged: no mid-line, no extra
        # instructions). Mutates `obs` in place.
        adjust_goal_mask_ui(obs)

        return self.info
    
    def success(self, arena):
        cur_eval = self.evaluate(arena)
        IoU = cur_eval['algn_IoU_to_flattened']
        coverage = cur_eval['normalised_coverage']
        return IoU > IOU_FLATTENING_TRESHOLD and coverage > NC_FLATTENING_TRESHOLD
    
    def compare(self, results_1, results_2):
        threshold=0.95

        # --- Compute averages for results_1 ---
        avg_nc_1 = mean([ep["normalised_coverage"][-1] for ep in results_1])
        avg_iou_1 = mean([ep["algn_IoU_to_flattened"][-1] for ep in results_1])
        avg_len_1 = mean([len(ep["algn_IoU_to_flattened"]) for ep in results_1])
        score_1 = avg_nc_1 + avg_iou_1

        # --- Compute averages for results_2 ---
        avg_nc_2 = mean([ep["normalised_coverage"][-1] for ep in results_2])
        avg_iou_2 = mean([ep["algn_IoU_to_flattened"][-1] for ep in results_2])
        avg_len_2 = mean([len(ep["algn_IoU_to_flattened"]) for ep in results_2])
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