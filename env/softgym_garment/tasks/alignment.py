import numpy as np

from statistics import mean

from .utils import *
from .garment_flattening_rewards import *
from .garment_flattening import GarmentFlatteningTask

class AlignmentTask(GarmentFlatteningTask):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'alignment'
        self.randomise_goal = config.get('randomise_goal', False)
    
    def reset(self, arena):
        # Determine if we are in eval mode and specifically on episode 0
        arena_mode = getattr(arena, 'mode', 'eval')
        arena_eid = getattr(arena, 'eid', None)
        is_eval_episode_zero = (arena_mode == 'eval' and arena_eid == 0)

        if self.randomise_goal:
            # Always clear the old observation first
            arena.flattened_obs = None
            
            if is_eval_episode_zero:
                # Use canonical state: applies hard_shift_x predictably without randomness
                # (Note: keeping the exact spelling 'get_caon_flattened_obs' from your GarmentEnv)
                arena.get_caon_flattened_obs()
            else:
                # Use standard randomized state for all other episodes
                arena.get_random_flattened_obs()
            
        return super().reset(arena)

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