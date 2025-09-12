import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist

from ..task import Task
from ...envs.clothfunnel_utils import get_normalised_hausdorff_distance
from .garment_flattening_rewards import *
from .utils import get_deform_distance, get_rigid_distance
THRESHOLD_COEFF = 0.3

class GarmentFlatteningTask(Task):
    def __init__(self, canonical=False):
        self.canonical = canonical
        self.successes = []
        self.task_name = 'flattening'
    
    def set_reward_fn(self, reward_fn):
        self.reward_fn = reward_fn
    
    def reset(self, arena):
        self.cur_coverage = arena.get_normalised_coverage()
        info = self._process_info(arena)
        self.successes = [info['success']]
        self.goal = self.get_goal(arena)
        self._save_goal(arena)
        return info
    
    def step(self, arena, action):
        info = self._process_info(arena)
        self.last_coverage = self.cur_coverage
        self.cur_coverage = arena.get_normalised_coverage()
        info['reward'] = self.reward(arena, action)
        self.successes.append(info['success'])
        return info
    

    def get_goals(self, arena):
        return [self.get_goal(arena)]
    
    def get_goal(self, arena):
        
        return arena.get_flatten_observation()

    def _process_info(self, arena):
        info = {}
        info['goal'] = self.get_goal(arena)
        info['success'] = self.success(arena)
        return info


    def reward(self, last_info, action, info):

        cloth_funnel_weighted_reward, cloth_funnel_tanh_reward = clothfunnel_reward(last_info, action, info)
        return {
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
    
    def get_steps2sucess(self):

        ## return the index of first success form the self.successes array, if there is no True return -1
        return np.argmax(self.successes) if np.any(self.successes) else -1
    
    def evaluate(self, arena):
        return {
            'normalised_coverage': arena._get_normalised_coverage(),
            'normalised_improvement': arena._get_normalised_impovement(),
            'max_IoU': arena._get_max_IoU(),
            'canon_IoU': arena._get_canon_IoU(),
            'canon_l2_distance': self._get_canon_l2_distance(arena),
            'normalised_hausdorff_distance': self._get_normalised_hausdorff_distance(arena),
            'deform_l2_distance': self._get_deform_l2_distance(arena),
            'rigid_l2_distance': self._get_rigid_l2_distance(arena),
        }
    
    def _get_deform_l2_distance(self, arena):
        cur_verts = arena._get_particle_positions()
        goal_verts = arena.get_canon_particle_position()
        threshold = np.sqrt(arena.get_cloth_area()) * THRESHOLD_COEFF
        return get_deform_distance(cur_verts, goal_verts, threshold=threshold)

    def _get_rigid_l2_distance(self, arena):
        cur_verts = arena._get_particle_positions()
        goal_verts = arena.get_canon_particle_position()
        threshold = np.sqrt(arena.get_cloth_area()) * THRESHOLD_COEFF
        return get_rigid_distance(cur_verts, goal_verts, threshold=threshold)
    
    def _get_canon_l2_distance(self, arena):
        pos = arena._get_particle_positions()
        goal_pos = arena.get_canon_particle_position()
        flipped_goal_pos = goal_pos.copy()
        flipped_goal_pos[:, 0] = -1 * flipped_goal_pos[:, 0]
        disance_1 = np.mean(np.linalg.norm(pos - goal_pos, axis=1))
        disance_2 = np.mean(np.linalg.norm(pos - flipped_goal_pos, axis=1))
        return min(disance_1, disance_2)
    
    def _get_normalised_hausdorff_distance(self, arena):
        cur_mask = arena._get_cloth_mask()
        goal_mask = arena.episode_params['init_rgb'].sum(axis=2) > 0

        val, matched_IoU = get_normalised_hausdorff_distance(cur_mask, goal_mask)#
        return val
    
    def success(self, arena):
        IoU = arena._get_max_IoU()
        coverage = arena._get_normalised_coverage()
        print(f'IoU {IoU:.2f}, NC {coverage:.2f}')
        return IoU > 0.85 and  coverage > 0.99


    def get_flatten_canonical_IoU(self, arena):
        mask = arena.get_cloth_mask(resolution=(128, 128)).reshape(128, 128)
        canonical_mask = arena.get_canonical_mask(resolution=(128, 128)).reshape(128, 128)
        intersection = np.sum(np.logical_and(mask, canonical_mask))
        union = np.sum(np.logical_or(mask, canonical_mask))
        IoU1 = intersection/union

        # Rotate the canonical mask by 90 degrees
        canonical_mask = np.rot90(canonical_mask)
        intersection = np.sum(np.logical_and(mask, canonical_mask))
        union = np.sum(np.logical_or(mask, canonical_mask))
        IoU2 = intersection / union

        return max(IoU1, IoU2)
    
    def get_canonical_hausdorff_distance(self, arena):
        mask = arena.get_cloth_mask(resolution=(128, 128))
        canonical_mask = arena.get_canonical_mask(resolution=(128, 128))
        hausdorff_distance = directed_hausdorff(mask, canonical_mask)[0]

        return hausdorff_distance
    
    def get_canonical_chamfer_distance(self):
        mask1 = self.get_cloth_mask(resolution=(128, 128))
        mask2 = self.get_canonical_mask(resolution=(128, 128))
        points1 = np.transpose(np.where(mask1))
        points2 = np.transpose(np.where(mask2))

        chamfer_distance = np.sum(np.min(cdist(points1, points2), axis=1)) + np.sum(np.min(cdist(points2, points1), axis=1))

        return chamfer_distance

    ## Version 2
    def get_maximum_IoU(self):
        mask = self.get_cloth_mask(resolution=(128, 128))
        canonical_mask = self.get_canonical_mask(resolution=(128, 128))

        x0 = np.array([0, 0, 0])  # Initial guess for rotation angle, translation x, and translation y

        bounds = [(-180, 180), (-63, 63), (-63, 63)]

        result = differential_evolution(self.calculate_IoU, bounds, args=(mask, canonical_mask.copy()),
                                        disp=True, maxiter=100, workers=1)

        optimal_angle = result.x[0]
        optimal_translation = result.x[1:]

        final_mask = self.rotate_and_translate_image(canonical_mask.copy(), optimal_angle, optimal_translation)

        # plt.imshow(final_mask)
        # plt.show()

        return -result.fun

    def calculate_IoU(self, x, mask, target_mask):
        angle = x[0]
        translation = x[1:]

        transformed_mask = self.rotate_and_translate_image(target_mask, angle, translation)

        intersection = np.sum(np.logical_and(mask, transformed_mask))
        union = np.sum(np.logical_or(mask, transformed_mask))
        IoU = intersection / union

        return -IoU  # Minimize the negative IoU

    def rotate_and_translate_image(self, image, angle, translation):
        translated_mask = np.roll(image, (int(translation[0]), int(translation[1])), axis=(0, 1))
        rotated_and_translated_image = self.rotate_image(translated_mask, angle)

        return rotated_and_translated_image

    def rotate_image(self, image, angle):
        angle_rad = np.radians(angle)
        rotated_image = np.zeros_like(image)
        center = (image.shape[0] / 2, image.shape[1] / 2)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x, y = np.dot(rotation_matrix, (i - center[0], j - center[1])) + center
                x = int(round(x))
                y = int(round(y))
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    rotated_image[x, y] = image[i, j]

        return rotated_image

        
    def rotate_and_translate_image(self, image, angle, translation):
        translated_mask = np.roll(image, (int(translation[0]), int(translation[1])), axis=(0, 1))
        rotated_and_translated_image = self.rotate_image(translated_mask, angle)

        return rotated_and_translated_image
    
    def rotate_image(self, image, angle):
        angle_rad = np.radians(angle)
        rotated_image = np.zeros_like(image)
        center = (image.shape[0] / 2, image.shape[1] / 2)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                x, y = np.dot(rotation_matrix, (i - center[0], j - center[1])) + center
                x = int(round(x))
                y = int(round(y))
                if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                    rotated_image[x, y] = image[i, j]

        return rotated_image

         
    def get_wrinkle_pixel_ratio(self, particles=None):
        rgb = self.render(mode='rgb')
        rgb = cv2.resize(rgb, (128, 128))
        mask = self.get_cloth_mask(resolution=(128, 128))

        if mask.dtype != np.uint8:  # Ensure mask has a valid data type (uint8)
            mask = mask.astype(np.uint8)


        # Use cv2 edge detection to get the wrinkle ratio.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # plt.imshow(edges)
        # plt.show()

        masked_edges = cv2.bitwise_and(edges, mask)
        # plt.imshow(masked_edges)
        # plt.show()

        wrinkle_ratio = np.sum(masked_edges) / np.sum(mask)

        return wrinkle_ratio