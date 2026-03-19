import numpy as np

from statistics import mean

from .utils import *
from .garment_flattening_rewards import *
from .garment_flattening import GarmentFlatteningTask

class CanonicalisationAlignmentTask(GarmentFlatteningTask):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'canonicalisation-alignment'
        self.randomise_goal = config.get('randomise_goal', False)
    
    def reset(self, arena):
        if self.randomise_goal:
            arena.flattened_obs = None
            arena.get_random_flattened_obs()
        return super().reset(arena)

    def success(self, arena):
        cur_eval = self.evaluate(arena)
        IoU = cur_eval['canon_IoU_to_flattened']
        coverage = cur_eval['normalised_coverage']
        return IoU > IOU_FLATTENING_TRESHOLD and coverage > NC_FLATTENING_TRESHOLD
    
    def compare(self, results_1, results_2):
        threshold=0.95

        # --- Compute averages for results_1 ---
        avg_nc_1 = mean([ep["normalised_coverage"][-1] for ep in results_1])
        avg_iou_1 = mean([ep["canon_IoU_to_flattened"][-1] for ep in results_1])
        avg_len_1 = mean([len(ep["canon_IoU_to_flattened"]) for ep in results_1])
        score_1 = avg_nc_1 + avg_iou_1

        # --- Compute averages for results_2 ---
        avg_nc_2 = mean([ep["normalised_coverage"][-1] for ep in results_2])
        avg_iou_2 = mean([ep["canon_IoU_to_flattened"][-1] for ep in results_2])
        avg_len_2 = mean([len(ep["canon_IoU_to_flattened"]) for ep in results_2])
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