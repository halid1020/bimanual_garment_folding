import numpy as np
from dotmap import DotMap
import cv2
import os
import random

from actoris_harena import Agent


class CentreSleeveFoldingPolicy(Agent):
    """
    Oracle policy for long-sleeve garment:
    Step 1: Bring right/left sleeves inward
    Step 2: Fold hem upward to collar
    """

    def __init__(self, config):
        super().__init__(config)
        #self.name = "oracle_garment_agent"
        self.config = config
        self.name = "centre_sleeve_folding_stochastic_policy"

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

        centre = self.get_pixel("centre", semkey2pid, keypids, key_pixels)
        centre_hem = self.get_pixel("centre_hem", semkey2pid, keypids, key_pixels)
    
        left_pick = higher_left_sleeve
        right_pick = higher_right_sleeve

        mid_centre = (centre_hem + centre)/2
        
        left_place = mid_centre
        
        
        right_place = mid_centre

    
        self.internal_states[arena_id]['step'] += 1
        H, W = cloth_mask.shape
        action = np.stack([
            self.norm_pixel(left_pick, H, W),
            self.norm_pixel(right_pick, H, W),
            self.norm_pixel(left_place, H, W),
            self.norm_pixel(right_place, H, W)
        ]).flatten()
        return action

    def act_step1(self, arena_id, key_pixels, semkey2pid, keypids, cloth_mask):
        """Hem folding step."""
        #print('Demo step 2')
        left_hem = self.get_pixel("left_hem", semkey2pid, keypids, key_pixels)
        right_hem = self.get_pixel("right_hem", semkey2pid, keypids, key_pixels)
        left_collar = self.get_pixel("left_collar", semkey2pid, keypids, key_pixels)
        right_collar = self.get_pixel("right_collar", semkey2pid, keypids, key_pixels)

       
        left_pick = left_hem
        right_pick = right_hem

        offset = 0.04 * cloth_mask.shape[0]
        left_target = np.array([left_collar[0]-offset, left_hem[1]])
        right_target = np.array([right_collar[0]-offset, right_hem[1]])

        left_place = left_target
        right_place = right_target
        

        H, W = cloth_mask.shape

        self.internal_states[arena_id]['step'] += 1
        action = np.stack([
            self.norm_pixel(left_pick, H, W),
            self.norm_pixel(right_pick, H, W),
            self.norm_pixel(left_place, H, W),
            self.norm_pixel(right_place, H, W)
        ]).flatten()
        return action
    
    def no_op(self, cloth_mask):
        """Pick and place the bottom-left and bottom-right corners of the cloth mask."""
        # Find all y (row) and x (col) coordinates where cloth is present
        ys, xs = np.where(cloth_mask > 0)
        
        # Fallback if the mask is empty for some reason
        if len(ys) == 0:
            return np.ones(8)

        # Bottom-left minimizes x and maximizes y. 
        # Therefore, we want the index that maximizes (y - x)
        bl_idx = np.argmax(ys - xs)
        left_pick = np.array([ys[bl_idx], xs[bl_idx]])
        left_place = left_pick.copy() # Place exactly where we picked

        # Bottom-right maximizes both x and y.
        # Therefore, we want the index that maximizes (y + x)
        br_idx = np.argmax(ys + xs)
        right_pick = np.array([ys[br_idx], xs[br_idx]])
        right_place = right_pick.copy()

        H, W = cloth_mask.shape[:2]

        action = np.stack([
            self.norm_pixel(left_pick, H, W),
            self.norm_pixel(right_pick, H, W),
            self.norm_pixel(left_place, H, W),
            self.norm_pixel(right_place, H, W)
        ]).flatten()
        
        return action


    # --------------------------
    # ---- Main Control --------
    # --------------------------

    def single_act(self, info, update=False):
        arena_id = info['arena_id']
        semkey2pid = info['observation']['semkey2pid']
        particle_pos = info['observation']['particle_positions']
        
        arena = info['arena']

        keypids = list(semkey2pid.values())
        key_particles = particle_pos[keypids]
        key_pixels, visibility = arena.get_visibility(key_particles)
        
        cloth_mask = arena._render(mode='mask')

        step = self.internal_states[arena_id]['step']

        #print('step!!', step)

        if step == 0:
            return self.act_step0(arena_id, key_pixels, semkey2pid, keypids, cloth_mask)
        elif step == 1:
            return self.act_step1(arena_id, key_pixels, semkey2pid, keypids, cloth_mask)
        else:
            # Pass the cloth mask here
            return self.no_op(cloth_mask)

    def terminate(self):
        return {arena_id: (self.internal_states[arena_id]['step'] >= 2)
                for arena_id in self.internal_states.keys()}
