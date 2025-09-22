import numpy as np
from dotmap import DotMap
import cv2
import os
import random

from agent_arena import Agent


class CentreSleeveFoldingStochasticPolicy(Agent):
    """
    Oracle policy for long-sleeve garment:
    Step 1: Bring right/left sleeves inward
    Step 2: Fold hem upward to collar
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "oracle_garment_agent"
        self.config = config

    def reset(self, arena_ids):
        for aid in arena_ids:
            self.internal_states[aid] = {}
            self.internal_states[aid]['step'] = 0
        return [True for _ in arena_ids]

    # --------------------------
    # ---- Utility methods -----
    # --------------------------

    def get_pixel(self, key, semkey2pid, keypids, key_pixels):
        pid = semkey2pid[key]
        idx = keypids.index(pid)
        pixel = key_pixels[idx].astype(int).copy()
        return pixel

    
    def norm_pixel(self, pixel, H, W):
        """Normalize pixel coords into [-1,1] range."""
        return pixel / np.asarray([H, W]) * 2 - 1

    def random_point_on_line(self, p1, p2, cloth_mask, offset=10, max_tries=30, on_mask=True):
        """Pick a random point around the line segment p1-p2 that lies on cloth."""
        H, W = cloth_mask.shape[:2]

        for _ in range(max_tries):
            alpha = np.random.rand()
            base = (1 - alpha) * p1 + alpha * p2
            noise = np.random.randint(-offset, offset + 1, size=2)
            candidate = np.clip(base + noise, [0, 0], [H - 1, W - 1]).astype(int)
            if (not on_mask) or (cloth_mask[candidate[0], candidate[1]] > 0):
                return candidate
        # fallback: return the nearest cloth pixel
        ys, xs = np.where(cloth_mask > 0)
        idx = np.random.randint(len(xs))
        return np.array([ys[idx], xs[idx]])

    def sample_near_pixel(self, pixel, cloth_mask, radius=10, max_tries=30, on_mask=True):
        """Sample a pixel near the given pixel that lies on cloth."""
        H, W = cloth_mask.shape[:2]
        for _ in range(max_tries):
            noise = np.random.randint(-radius, radius + 1, size=2)
            candidate = np.clip(pixel + noise, [0, 0], [H - 1, W - 1]).astype(int)
            if (not on_mask) or (cloth_mask[candidate[0], candidate[1]] > 0):
                return candidate
        # fallback: return nearest cloth pixel
        ys, xs = np.where(cloth_mask > 0)
        idx = np.random.randint(len(xs))
        return np.array([ys[idx], xs[idx]])


    # --------------------------
    # ---- Step Handlers --------
    # --------------------------

    def act_step0(self, arena_id, key_pixels, semkey2pid, keypids, cloth_mask):
        """Sleeve folding step."""
        higher_left_sleeve = self.get_pixel("higher_left_sleeve", semkey2pid, keypids, key_pixels)
        higher_right_sleeve = self.get_pixel("higher_right_sleeve", semkey2pid, keypids, key_pixels)
        lower_left_sleeve = self.get_pixel("lower_left_sleeve", semkey2pid, keypids, key_pixels)
        lower_right_sleeve = self.get_pixel("lower_right_sleeve", semkey2pid, keypids, key_pixels)

        centre = self.get_pixel("centre", semkey2pid, keypids, key_pixels)
        centre_hem = self.get_pixel("centre_hem", semkey2pid, keypids, key_pixels)

        # randomized picks near sleeves
        left_pick = self.random_point_on_line(lower_left_sleeve, higher_left_sleeve, cloth_mask)
        right_pick = self.random_point_on_line(lower_right_sleeve, higher_right_sleeve, cloth_mask)

        # randomized places along line (centre -> centre_hem)
        left_place = self.random_point_on_line(centre, centre_hem, cloth_mask)
        right_place = self.random_point_on_line(centre, centre_hem, cloth_mask)

        # enforce left is left of right
        if left_place[1] > right_place[1]:
            left_place, right_place = right_place, left_place

        H, W = cloth_mask.shape
        action = {
            'norm-pixel-fold': {
                'pick_0': self.norm_pixel(left_pick, H, W),
                'pick_1': self.norm_pixel(right_pick, H, W),
                'place_0': self.norm_pixel(left_place, H, W),
                'place_1': self.norm_pixel(right_place, H, W)
            }
        }

        self.internal_states[arena_id]['step'] += 1
        return action

    def act_step1(self, arena_id, key_pixels, semkey2pid, keypids, cloth_mask):
        """Hem folding step."""
        #print('Demo step 2')
        left_hem = self.get_pixel("left_hem", semkey2pid, keypids, key_pixels)
        right_hem = self.get_pixel("right_hem", semkey2pid, keypids, key_pixels)
        left_collar = self.get_pixel("left_collar", semkey2pid, keypids, key_pixels)
        right_collar = self.get_pixel("right_collar", semkey2pid, keypids, key_pixels)

        # randomized picks near hems
        left_pick = self.sample_near_pixel(left_hem, cloth_mask)
        right_pick = self.sample_near_pixel(right_hem, cloth_mask)

        # places: near collar line but not below
        left_target = np.array([left_collar[0]-30, left_hem[1]])
        right_target = np.array([right_collar[0]-30, right_hem[1]])

        left_place = self.sample_near_pixel(left_target, cloth_mask, on_mask=False)
        right_place = self.sample_near_pixel(right_target, cloth_mask, on_mask=False)

        # # ensure not below collar line
        # min_y = min(left_collar[1], right_collar[1])
        # left_place[1] = min(left_place[1], min_y)
        # right_place[1] = min(right_place[1], min_y)
        H, W = cloth_mask.shape
        action = {
            'norm-pixel-fold': {
                'pick_0': self.norm_pixel(left_pick, H, W),
                'pick_1': self.norm_pixel(right_pick, H, W),
                'place_0': self.norm_pixel(left_place, H, W),
                'place_1': self.norm_pixel(right_place, H, W)
            }
        }

        self.internal_states[arena_id]['step'] += 1
        return action
    
    def no_op(self):
        action = {
            'norm-pixel-fold': {
                'pick_0': np.ones(2),
                'pick_1': np.ones(2),
                'place_0': np.ones(2),
                'place_1': np.ones(2)
            }
        }

        return action


    # --------------------------
    # ---- Main Control --------
    # --------------------------

    def single_act(self, info, update=False):
        arena_id = info['arena_id']
        semkey2pid = info['observation']['semkey2pid']
        particle_pos = info['observation']['particle_positions']
        rgb = info['observation']['rgb']
        cloth_mask = info['observation']['mask']
        arena = info['arena']

        keypids = list(semkey2pid.values())
        key_particles = particle_pos[keypids]
        key_pixels, visibility = arena.get_visibility(key_particles)

        H, W, _ = rgb.shape
        step = self.internal_states[arena_id]['step']

        if step == 0:
            return self.act_step0(arena_id, key_pixels, semkey2pid, keypids, cloth_mask)
        elif step == 1:
            return self.act_step1(arena_id, key_pixels, semkey2pid, keypids, cloth_mask)
        else:
            return self.no_op()

    def terminate(self):
        return {arena_id: (self.internal_states[arena_id]['step'] >= 2)
                for arena_id in self.internal_states.keys()}
