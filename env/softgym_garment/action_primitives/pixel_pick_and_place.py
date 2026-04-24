import os
import cv2
import numpy as np
from gym.spaces import Dict, Discrete, Box
import random
from scipy.ndimage import binary_erosion

from .world_pick_and_place import WorldPickAndPlace
from ..utils.camera_utils import norm_pixel2world
from .utils import readjust_norm_pixel_pick, norm_pixel_to_index

class PixelPickAndPlace():

    def __init__(self, 
                 action_horizon=20,
                 pick_height=0.025,
                 post_pick_height=0.05,
                 pre_place_height=0.06,
                 place_height=0.06,
                 pick_lower_bound=[-1, -1],
                 pick_upper_bound=[1, 1],
                 place_lower_bound=[-1, -1],
                 place_upper_bound=[1, 1],
                 pregrasp_height=0.05,
                 pre_grasp_vel=0.05,
                 drag_vel=0.05,
                 lift_vel=0.05,
                 readjust_pick_poss=0.0,
                 single_operator=False,
                 **kwargs):
        
        self.action_tool = WorldPickAndPlace(**kwargs) 
        
        self.num_pickers = 2
        pick_lower_bound = np.array(pick_lower_bound)
        pick_upper_bound = np.array(pick_upper_bound)
        place_lower_bound = np.array(place_lower_bound)
        place_upper_bound = np.array(place_upper_bound)
        self.action_space = Dict({
            'pick_0': Box(pick_lower_bound, pick_upper_bound, dtype=np.float32),
            'place_0': Box(place_lower_bound, place_upper_bound, dtype=np.float32),
        })
        if not single_operator:
            self.action_space.spaces['pick_1'] = Box(pick_lower_bound, pick_upper_bound, dtype=np.float32)
            self.action_space.spaces['place_1'] = Box(place_lower_bound, place_upper_bound, dtype=np.float32)
        
        self.pick_height = pick_height
        self.place_height = place_height
        self.action_horizon = action_horizon
        self.kwargs = kwargs
        self.action_mode = 'pixel-pick-and-place'
        self.single_operator = single_operator
        self.pregrasp_height = pregrasp_height
        self.pre_place_height = pre_place_height
        self.pre_grasp_vel = pre_grasp_vel
        self.post_pick_height = post_pick_height
        self.drag_vel = drag_vel
        self.lift_vel = lift_vel
        self.readjust_pick_poss = readjust_pick_poss

    def get_no_op(self):
        return self.no_op
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon
    
    def reset(self, env):
        return self.action_tool.reset(env)
        
    def process(self, env, action):
        r0_mask = getattr(env, 'robot0_mask_crop', env.robot0_mask) if env.apply_workspace else None
        r1_mask = getattr(env, 'robot1_mask_crop', env.robot1_mask) if env.apply_workspace else None
        cloth_mask = getattr(env, 'cloth_mask_crop', env.cloth_mask)

        # # --- DEBUG: SAVE MASKS BEFORE READJUSTMENT ---
        # debug_dir = 'tmp/env_debug/'
        # os.makedirs(debug_dir, exist_ok=True)
        # step_idx = getattr(env, 'action_step', 0)
        
        # if cloth_mask is not None:
        #     cv2.imwrite(os.path.join(debug_dir, f'cloth_mask_step_{step_idx}.png'), (cloth_mask.astype(np.uint8) * 255))
        # if r0_mask is not None:
        #     cv2.imwrite(os.path.join(debug_dir, f'r0_mask_step_{step_idx}.png'), (r0_mask.astype(np.uint8) * 255))
        # if r1_mask is not None:
        #     cv2.imwrite(os.path.join(debug_dir, f'r1_mask_step_{step_idx}.png'), (r1_mask.astype(np.uint8) * 255))
        # # ---------------------------------------------

        pick_0 = np.asarray(action['pick_0'])
        place_0 = np.asarray(action['place_0'])

        adj_pick_0, dist_0 = readjust_norm_pixel_pick(pick_0, cloth_mask)

        if random.random() < self.readjust_pick_poss:
            pick_0 = adj_pick_0
            
        dist_1 = 0
        pick_0_depth = action.get('pick_0_d', self.camera_height - self.pick_height)
        place_0_depth = action.get('place_0_d', self.camera_height - self.place_height)

        if 'pick_1' in action:
            pick_1 = np.asarray(action['pick_1'])
            place_1 = np.asarray(action['place_1'])

            adj_pick_1, dist_1 = readjust_norm_pixel_pick(pick_1, cloth_mask)
            if random.random() < self.readjust_pick_poss:
                pick_1 = adj_pick_1

            pick_1_depth = action.get('pick_1_d', self.camera_height - self.pick_height)
            place_1_depth = action.get('place_1_d', self.camera_height - self.place_height)
        else:
            pick_1 = np.asarray([1.5, 1.5])
            place_1 = np.asarray([1.5, 1.5])
            pick_1_depth = self.camera_height
            place_1_depth = self.camera_height

        self.affordance_score = self._calculate_affordance(dist_0, dist_1)
        
        print('pick 0', pick_0)
        print('pick 1', pick_1)
        if pick_0[1] > pick_1[1]:
            pick_0, pick_1 = pick_1, pick_0
            place_0, place_1 = place_1, place_0
            pick_0_depth, pick_1_depth = pick_1_depth, pick_0_depth
            place_0_depth, place_1_depth = place_1_depth, place_0_depth
        
        if env.apply_workspace and env.readjust_to_workspace:
            if 'pick_1' not in action:
                H_crop, W_crop = r0_mask.shape[:2]
                r_p0, c_p0 = norm_pixel_to_index(pick_0, (W_crop, H_crop))
                r_p0, c_p0 = np.clip(r_p0, 0, H_crop - 1), np.clip(c_p0, 0, W_crop - 1)
                
                active_mask = None
                if r0_mask[r_p0, c_p0]:
                    active_mask = r0_mask
                elif r1_mask[r_p0, c_p0]:
                    active_mask = r1_mask
                
                if active_mask is not None:
                    place_0, _ = readjust_norm_pixel_pick(place_0, active_mask.copy())
            else:
                pick_0, _ = readjust_norm_pixel_pick(pick_0, r0_mask.copy())
                place_0, _ = readjust_norm_pixel_pick(place_0, r0_mask.copy())
                pick_1, _ = readjust_norm_pixel_pick(pick_1, r1_mask.copy())
                place_1, _ = readjust_norm_pixel_pick(place_1, r1_mask.copy())

        # --- TRANSFORMATION LOGIC ---
        W, H = self.camera_size
        def to_full_norm(pt):
            if hasattr(env, 'x1') and hasattr(env, 'y1') and hasattr(env, 'crop_size'):
                # maps [-1, 1] to [0, 720]
                px_crop = (pt + 1.0) / 2.0 * env.crop_size 
                # shifts by x1 (offset) and y1
                px_full = px_crop + np.array([env.y1, env.x1]) 
                # normalizes against [1280, 720]
                return (px_full / np.array([H, W])) * 2.0 - 1.0 
            return pt

        pick_0_full = to_full_norm(pick_0)
        place_0_full = to_full_norm(place_0)
        pick_1_full = to_full_norm(pick_1)
        place_1_full = to_full_norm(place_1)

        action_ = np.concatenate([pick_0_full, place_0_full, pick_1_full, place_1_full]).reshape(-1, 2)
        # ----------------------------

        depths = np.array([pick_0_depth, place_0_depth, pick_1_depth, place_1_depth])

        convert_action = norm_pixel2world(
                action_, np.asarray([H, W]),  
                self.camera_intrinsics, self.camera_pose, depths) 
        convert_action = convert_action.reshape(2, 2, 3)

        world_action =  {
            'pick_0_position': convert_action[0, 0],
            'place_0_position': convert_action[0, 1],
            'pick_1_position': convert_action[1, 0],
            'place_1_position': convert_action[1, 1],
            'tograsp_vel': self.pre_grasp_vel,
            'drag_vel': self.drag_vel,
            'lift_vel': self.lift_vel,
            'pregrasp_height': self.pregrasp_height,
            'pre_place_height': self.pre_place_height,
            'post_pick_height': self.post_pick_height,
        }

        pixel_action = np.stack([pick_0, pick_1, place_0, place_1]).flatten()
        return world_action, pixel_action
    
    def _calculate_affordance(self, dist_0, dist_1):
        return np.min([
            1 - min(dist_0, np.sqrt(8)) / np.sqrt(8),
            1 - min(dist_1, np.sqrt(8)) / np.sqrt(8)
        ])

    def step(self, env, action):
        self.camera_height = env.camera_height
        self.camera_intrinsics = env.camera_intrinsic_matrix
        self.camera_pose = env.camera_extrinsic_matrix
        self.camera_size = env.camera_size

        world_action_ , pixel_action = self.process(env, action)

        reject = False
        if env.apply_workspace:
            r0_mask = getattr(env, 'robot0_mask_crop', env.robot0_mask)
            r1_mask = getattr(env, 'robot1_mask_crop', env.robot1_mask)
            
            H_crop, W_crop = r0_mask.shape[:2]
            
            r0p, c0p = norm_pixel_to_index(pixel_action[:2], (W_crop, H_crop))
            r0l, c0l = norm_pixel_to_index(pixel_action[4:6], (W_crop, H_crop))
            r0p, c0p = np.clip(r0p, 0, H_crop - 1), np.clip(c0p, 0, W_crop - 1)
            r0l, c0l = np.clip(r0l, 0, H_crop - 1), np.clip(c0l, 0, W_crop - 1)

            if 'pick_1' not in action:
                in_robot0 = r0_mask[r0p, c0p]
                in_robot1 = r1_mask[r0p, c0p]

                if not (in_robot0 or in_robot1):
                    reject = True
                else:
                    active_mask = r0_mask if in_robot0 else r1_mask
                    if not active_mask[r0l, c0l]:
                        reject = True
            else:
                if not r0_mask[r0p, c0p] or not r0_mask[r0l, c0l]:
                    reject = True
        
                r1p, c1p = norm_pixel_to_index(pixel_action[2:4], (W_crop, H_crop))
                r1l, c1l = norm_pixel_to_index(pixel_action[6:8], (W_crop, H_crop))
                r1p, c1p = np.clip(r1p, 0, H_crop - 1), np.clip(c1p, 0, W_crop - 1)
                r1l, c1l = np.clip(r1l, 0, H_crop - 1), np.clip(c1l, 0, W_crop - 1)
                
                if not r1_mask[r1p, c1p] or not r1_mask[r1l, c1l]:
                    reject = True
            
        if not reject: 
            info = self.action_tool.step(env, world_action_)
        else:
            info = {}
            
        info['applied_action'] = pixel_action
        info['action_affordance_score'] = self.affordance_score
        return info