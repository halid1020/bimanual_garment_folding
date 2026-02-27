import numpy as np

from statistics import mean

from .utils import *
from .garment_flattening_rewards import *
from .garment_task import GarmentTask
THRESHOLD_COEFF = 0.3

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
        self.canon_ious = []
        #self._save_goal(arena)
        return {"goals": self.goals}

    def get_goals(self):
        return self.goals

    def get_goal(self):
        return self.goals[0]

    def reward(self, last_info, action, info):#
        ca_reward = coverage_alignment_reward(last_info, action, info) # combination of delta NC and delta IOU
        if info['success'] and self.config.get('big_success_bonus', True):
            ca_reward = info['arena'].action_horizon - info['observation']['action_step']
        
        aug_ca_reward = ca_reward
        if info['evaluation']['normalised_coverage'] > NC_FLATTENING_TRESHOLD and info['evaluation']['max_IoU_to_flattened'] > IOU_FLATTENING_TRESHOLD:
            aug_ca_reward = 1

        threshold =  self.config.get('overstretch_penalty_threshold', 0)
        if info['overstretch'] > threshold:
            stretch_penalty = self.config.get("overstretch_penalty_scale", 0) * (info['overstretch'] - threshold)
            aug_ca_reward_pen = aug_ca_reward - stretch_penalty

            ca_reward_pen = ca_reward - stretch_penalty
        else:
            aug_ca_reward_pen = aug_ca_reward
            ca_reward_pen = ca_reward
        
        reward_2 = aug_ca_reward
        aff_score_pen = (1 - info.get('action_affordance_score', 1))
        reward_2 -= self.config.get("affordance_penalty_scale", 0) * aff_score_pen
    
        #print('rev aff score', aff_score_rev)
        cloth_funnel_weighted_reward, cloth_funnel_tanh_reward = clothfunnel_reward(last_info, action, info)
        
        return {
            'coverage_alignment': ca_reward,
            'augmented_coverage_alignment_with_stretch_penalty': aug_ca_reward_pen,
            'coverage_alignment_with_stretch_and_affordance_penalty': reward_2,
            'coverage_alignment_with_stretch_penalty': ca_reward_pen,

            'clothfunnel_default': cloth_funnel_weighted_reward,
            'clothfunnel_tanh_reward': cloth_funnel_tanh_reward,
            'normalised_coverage': normalised_coverage_reward(last_info, action, info),
            'speedFolding_approx': speedFolding_approx_reward(last_info, action, info),
            'coverage_differance': coverage_differance_reward(last_info, action, info),
            'learningToUnfold_approx': learningTounfold_reward(last_info, action, info),
            'max_IoU': max_IoU_reward(last_info, action, info),
            'max_IoU_differance': max_IoU_differance_reward(last_info, action, info),
            'canon_IoU': canon_IoU_reward(last_info, action, info),
            'canon_IoU_differance': canon_IoU_differance_reward(last_info, action, info),
            'canon_l2_tanh_reward': canon_l2_tanh_reward(last_info, action, info),
            'planet_clothpick_hueristic': planet_clothpick_hueristic_reward(last_info, action, info),
            'coverage_aligment': coverage_alignment_reward(last_info, action, info),
        }
    
    
    
    def evaluate(self, arena):
        eval_dict = {
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'normalised_improvement': self._get_normalised_impovement(arena),
            'overstretch': arena.overstretch,
            'canon_IoU_to_flattened': self._get_canon_IoU_to_flattened(arena),
            'canon_l2_distance': self._get_canon_l2_distance(arena),
            'deform_l2_distance': self._get_deform_l2_distance(arena),
            'rigid_l2_distance': self._get_rigid_l2_distance(arena),
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
    
    def _get_canon_l2_distance(self, arena):
        pos = arena.get_mesh_particles_positions()
        goal_pos = arena.get_canon_mesh_particles_positions()

        flipped_goal_pos = goal_pos.copy()
        flipped_goal_pos[:, 0] = -1 * flipped_goal_pos[:, 0]
        disance_1 = np.mean(np.linalg.norm(pos - goal_pos, axis=1))
        disance_2 = np.mean(np.linalg.norm(pos - flipped_goal_pos, axis=1))
        return min(disance_1, disance_2)
    
    def _get_deform_l2_distance(self, arena):
        cur_verts = arena.get_mesh_particles_positions()
        goal_verts = arena.get_canon_mesh_particles_positions()
        threshold = np.sqrt(arena.get_cloth_area()) * THRESHOLD_COEFF
        return get_deform_distance(cur_verts, goal_verts, threshold=threshold)

    def _get_rigid_l2_distance(self, arena):
        cur_verts = arena.get_mesh_particles_positions()
        goal_verts = arena.get_canon_mesh_particles_positions()
        threshold = np.sqrt(arena.get_cloth_area()) * THRESHOLD_COEFF
        return get_rigid_distance(cur_verts, goal_verts, threshold=threshold)

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
    
    def _get_canon_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        goal_mask = arena.get_flattened_obs()['observation']['mask']
        IoU = calculate_iou(cur_mask, goal_mask)
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