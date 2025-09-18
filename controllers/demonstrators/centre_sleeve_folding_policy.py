
import numpy as np
from dotmap import DotMap
import cv2
import os

from agent_arena import Agent


class CentreSleeveFoldingPolicy(Agent):
    """
    Oracle policy for long-sleeve garment:
      - first bring right and left sleeves to garment centre (pixel-space)
      - then bring hem to the top of the garment
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

    def single_act(self, info, update=False):
        """
        Produce a single arena action dict (in pixel coords).
        """
        arena = info['arena']
        arena_id = info['arena_id']
        semkey2pid = info['observation']['semkey2pid']   # dict: sem key -> particle id
        particle_pos = info['observation']['particle_position']  # (N, 3)
        rgb = info['observation']['rgb']                 # H x W x 3

        # project semantic particles to pixels
        keypids = list(semkey2pid.values())
        key_particles = particle_pos[keypids]            # (K, 3)
        key_pixels, visibility = arena.get_visibility(key_particles)  # (K,2), (K,)

       

        # helper to get pixel for semantic key
        def get_pixel(key):
            pid = semkey2pid[key]
            idx = keypids.index(pid)
            pixel = key_pixels[idx].astype(int).copy()
            # pixel[0], pixel[1] = pixel[1], pixel[0]
            return pixel

        # get key garment landmarks
        left_sleeve = get_pixel("left_sleeve")
        right_sleeve = get_pixel("right_sleeve")
        left_hem = get_pixel("left_hem")
        right_hem = get_pixel("right_hem")
        centre = get_pixel("centre")


        # --- Debug visualization ---
        if self.config.debug:

             # find left shoulder pixel
            for name in ['left_sleeve', 'right_sleeve', 'left_hem', 'right_hem']:
                left_shoulder_pid = semkey2pid[name]
                idx = keypids.index(left_shoulder_pid)           # map back to key_pixels
                target_pixel = key_pixels[idx].astype(int)       # (u,v)

                debug_img = rgb.copy()
                u, v = int(target_pixel[0]), int(target_pixel[1])
                cv2.circle(debug_img, (v, u), radius=6, color=(0, 0, 255), thickness=-1)

                os.makedirs("./tmp", exist_ok=True)
                out_path = os.path.join("./tmp", f"target_pixel_step_{self.internal_states[arena_id]['step']}_{name}.png")
                cv2.imwrite(out_path, debug_img)

        # image center
        H, W, _ = rgb.shape
        print('single act H, W', H, W)
        mid_x, mid_y = W // 2, H // 2

        if self.internal_states[arena_id]['step'] == 0:
            # Step 1: fold sleeves inward
            left_pick = left_sleeve / np.asarray([H, W]) * 2 - 1
            right_pick = right_sleeve / np.asarray([H, W]) * 2 - 1
            # place both near image center horizontally
            left_place = np.array(centre) / np.asarray([H, W]) * 2 - 1
            right_place = np.array(centre) / np.asarray([H, W]) * 2 - 1

            action = {
                'norm-pixel-fold': {
                   'pick_0': left_pick,
                   'pick_1': right_pick,
                   'place_0': left_place,
                   'place_1': right_place
                }
            }

            self.internal_states[arena_id]['step'] += 1
            return action
        else:
            self.internal_states[arena_id]['step'] += 1
            return {
                'norm-pixel-fold': {
                    'pick_0': np.random.uniform(-1, 1, 2),
                    'place_0': np.random.uniform(-1, 1, 2),
                    'pick_1': np.random.uniform(-1, 1, 2),
                    'place_1': np.random.uniform(-1, 1, 2)
                }
            }
            

        # TODO: convert pick/place into your action format.
        # For now: placeholder fold action
        # return {
        #     'norm-pixel-fold': [u, v, u, v]   # pick and place the same pixel (dummy)
        # }

    def terminate(self):
        return {arena_id: (self.internal_states[arena_id]['step'] >= 2) for arena_id in self.internal_states.keys()}