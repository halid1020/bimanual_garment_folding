import numpy as np
from dotmap import DotMap
import cv2
import os
import random

from agent_arena import Agent
from .utils import *


class WaistLegFoldingStochasticPolicy(Agent):
    """
    Oracle policy for trousers folding:
    Step 1: Fold trousers laterally by aligning left and right waist/hem corners.
    Step 2: Fold upwards by aligning bottom hems with waistband.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.name = "waist_leg_folding_stochastic_policy"

    def reset(self, arena_ids):
        for aid in arena_ids:
            self.internal_states[aid] = {'step': 0}
        return [True for _ in arena_ids]


    # --------------------------
    # ---- Step Handlers --------
    # --------------------------

    def act_step0(self, arena_id, key_pixels, semkey2pid, keypids, cloth_mask):
        """Step 1: Lateral fold - bring right half over left half."""
        left_waist = get_pixel("left_waistband", semkey2pid, keypids, key_pixels)
        right_waist = get_pixel("right_waistband", semkey2pid, keypids, key_pixels)
        left_hem = get_pixel("left_hem_left", semkey2pid, keypids, key_pixels)
        right_hem = get_pixel("right_hem_right", semkey2pid, keypids, key_pixels)
        centre = get_pixel("centre", semkey2pid, keypids, key_pixels)

        # random pick along right side (waist->hem)
        right_pick = sample_near_pixel(left_hem, cloth_mask, radius=0, on_mask=True)
        # place near centre line (towards left)
        right_place = right_hem.copy()
        right_place[1] -= 20
        right_place[0] += 15

        # left side for balance (could stay stationary)
        left_pick = sample_near_pixel(left_waist, cloth_mask, radius=0, on_mask=True)
        left_place = right_waist.copy()
        left_place[1] -= 20
        left_place[0] -= 15

        H, W = cloth_mask.shape
        action = {
            'norm-pixel-fold': {
                'pick_0': norm_pixel(right_pick, H, W),
                'pick_1': norm_pixel(left_pick, H, W),
                'place_0': norm_pixel(right_place, H, W),
                'place_1': norm_pixel(left_place, H, W)
            }
        }

        self.internal_states[arena_id]['step'] += 1
        return action

    def act_step1(self, arena_id, key_pixels, semkey2pid, keypids, cloth_mask):
        """Step 2: Vertical fold - bring hems to waistband."""
        left_waist = get_pixel("left_waistband", semkey2pid, keypids, key_pixels)
        right_waist = get_pixel("right_waistband", semkey2pid, keypids, key_pixels)
        centre_waist = get_pixel("centre_waistband", semkey2pid, keypids, key_pixels)
        left_hem = get_pixel("left_hem_left", semkey2pid, keypids, key_pixels)
        right_hem = get_pixel("right_hem_left", semkey2pid, keypids, key_pixels)
        centre_waist = get_pixel("centre_waistband", semkey2pid, keypids, key_pixels)

        # pick near bottom hems
        left_pick = sample_near_pixel(left_waist, cloth_mask, radius=2, on_mask=True)
        right_pick = sample_near_pixel(centre_waist, cloth_mask, radius=2, on_mask=True)


        left_place = sample_near_pixel(left_hem, cloth_mask, radius=2, on_mask=True)
        right_place = sample_near_pixel(right_hem, cloth_mask, radius=2, on_mask=True)

        left_place[0] -= 15
        left_place[1] += 15
        right_place[0] -= 15
        right_place[1] -= 10

        H, W = cloth_mask.shape
        action = {
            'norm-pixel-fold': {
                'pick_0': norm_pixel(left_pick, H, W),
                'pick_1': norm_pixel(right_pick, H, W),
                'place_0': norm_pixel(left_place, H, W),
                'place_1': norm_pixel(right_place, H, W)
            }
        }

        self.internal_states[arena_id]['step'] += 1
        return action

    def no_op(self):
        return {
            'norm-pixel-fold': {
                'pick_0': np.ones(2),
                'pick_1': np.ones(2),
                'place_0': np.ones(2),
                'place_1': np.ones(2)
            }
        }

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
        return {
            arena_id: (self.internal_states[arena_id]['step'] >= 2)
            for arena_id in self.internal_states.keys()
        }
