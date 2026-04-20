import numpy as np
from gym.spaces import Box
import random

from .world_pick_and_fling import WorldPickAndFling
from ..utils.camera_utils import norm_pixel2world
from .utils import pixel_to_world, readjust_norm_pixel_pick, norm_pixel_to_index

class PixelPickAndFling():

    def __init__(self, 
        lowest_cloth_height=0.1,
        max_grasp_dist=0.7,
        stretch_increment_dist=0.02,
        pregrasp_height=0.3,
        pregrasp_vel=0.1,
        tograsp_vel=0.05,
        prefling_height=0.3, 
        prefling_vel=0.01,
        hang_pos_y=0.3, 
        fling_y=0.5, 
        lift_vel=0.02, 
        action_horizon=20,
        hang_adjust_vel=0.01,
        stretch_adjust_vel=0.01,
        fling_vel= 0.02, 
        release_vel=0.01,
        drag_vel=0.005,
        lower_height=0.06,
        readjust_pick_poss=0.0,
        pick_lower_bound=[-1, -1],
        pick_upper_bound=[1, 1],
        place_lower_bound=[-1, -1],
        place_upper_bound=[1, 1],
        pick_height=0.025, 
        readjust_to_workspace=False,
        **kwargs):
        
        self.action_tool = WorldPickAndFling(**kwargs) 
        self.action_horizon = action_horizon
        self.lowest_cloth_height = lowest_cloth_height
        self.max_grasp_dist = max_grasp_dist
        self.stretch_increment_dist = stretch_increment_dist
        self.fling_vel = fling_vel
        self.pregrasp_height = pregrasp_height
        self.pregrasp_vel = pregrasp_vel
        self.tograsp_vel = tograsp_vel
        self.prefling_height = prefling_height
        self.prefling_vel = prefling_vel
        self.hang_pos_y = hang_pos_y
        self.lift_vel = lift_vel
        self.pick_height = pick_height
        self.hang_adjust_vel = hang_adjust_vel
        self.stretch_adjust_vel = stretch_adjust_vel
        self.release_vel = release_vel
        self.drag_vel = drag_vel
        self.lower_height = lower_height
        self.fling_y = fling_y

        self.num_pickers = 2
        self.readjust_pick_poss = readjust_pick_poss
        self.readjust_to_workspace = readjust_to_workspace

        space_low = np.concatenate([pick_lower_bound, place_lower_bound]*self.num_pickers)\
            .reshape(self.num_pickers, -1).astype(np.float32)
        space_high = np.concatenate([pick_upper_bound, place_upper_bound]*self.num_pickers)\
            .reshape(self.num_pickers, -1).astype(np.float32)
        self.action_space = Box(space_low, space_high, dtype=np.float32)

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
    
    def _calculate_affordance(self, dist_0, dist_1):
        return np.min([
            1 - min(dist_0, np.sqrt(8)) / np.sqrt(8),
            1 - min(dist_1, np.sqrt(8)) / np.sqrt(8)
        ])
    
    def process(self, env, action):
        r0_mask = getattr(env, 'robot0_mask_crop', env.robot0_mask) if env.apply_workspace else None
        r1_mask = getattr(env, 'robot1_mask_crop', env.robot1_mask) if env.apply_workspace else None
        cloth_mask = getattr(env, 'cloth_mask_crop', env.cloth_mask)

        p0 = np.asarray(action['pick_0'])
        p1 = np.asarray(action['pick_1'])

        adj_p0, dist_0 = readjust_norm_pixel_pick(p0, cloth_mask)
        adj_p1, dist_1 = readjust_norm_pixel_pick(p1, cloth_mask)
       
        if random.random() < self.readjust_pick_poss:
            p0 = adj_p0
            p1 = adj_p1
        
        self.affordance_score =  self._calculate_affordance(dist_0, dist_1)

        if p0[1] > p1[1]:
            p0, p1 = p1, p0
        
        if env.apply_workspace and env.readjust_to_workspace:
            p0, _ = readjust_norm_pixel_pick(p0, r0_mask)
            p1, _ = readjust_norm_pixel_pick(p1, r1_mask)

        # --- TRANSFORMATION LOGIC ---
        W, H = self.camera_size
        def to_full_norm(pt):
            if hasattr(env, 'x1') and hasattr(env, 'y1') and hasattr(env, 'crop_size'):
                px_crop = (pt + 1.0) / 2.0 * env.crop_size
                px_full = px_crop + np.array([env.x1, env.y1])
                return (px_full / np.array([W, H])) * 2.0 - 1.0
            return pt

        p0_full = to_full_norm(p0)
        p1_full = to_full_norm(p1)

        action_ = np.concatenate([p0_full, p1_full]).reshape(-1, 2)
        # ----------------------------

        p0_depth = self.camera_height - self.pick_height
        p1_depth = self.camera_height - self.pick_height
        depths = np.array([p0_depth, p1_depth])

        # FIXED: Pass [W, H] instead of [H, W]
        convert_action = norm_pixel2world(
                action_, np.asarray([W, H]),  
                self.camera_intrinsics, self.camera_pose, depths) 
        convert_action = convert_action.reshape(2, 3)

        world_action =  {
            'pick_0_position': convert_action[0],
            'pick_1_position': convert_action[1],
            'pregrasp_height': self.pregrasp_height,
            'pregrasp_vel': self.pregrasp_vel,
            'tograsp_vel': self.tograsp_vel,
            'prefling_height': self.prefling_height,
            'prefling_vel': self.prefling_vel,
            'lift_vel': self.lift_vel,
            'hang_pos_y': self.hang_pos_y,
            'fling_y': self.fling_y,
            'hang_adjust_vel': self.hang_adjust_vel, 
            'stretch_adjust_vel': self.stretch_adjust_vel, 
            'fling_vel': self.fling_vel, 
            'release_vel': self.release_vel, 
            'drag_vel': self.drag_vel, 
            'lower_height': self.lower_height, 
        }

        pixel_action = np.stack([p0, p1]).flatten()

        return world_action, pixel_action
    
    def step(self, env, action):
        self.camera_height = env.camera_height
        self.camera_intrinsics = env.camera_intrinsic_matrix
        self.camera_pose = env.camera_extrinsic_matrix
        self.camera_size = env.camera_size
        world_action_, pixel_action = self.process(env, action)
        
        reject = False
        if env.apply_workspace:
            r0_mask = getattr(env, 'robot0_mask_crop', env.robot0_mask)
            r1_mask = getattr(env, 'robot1_mask_crop', env.robot1_mask)

            H_crop, W_crop = r0_mask.shape[:2]
            r0p, c0p = norm_pixel_to_index(pixel_action[:2], (W_crop, H_crop))
            r1p, c1p = norm_pixel_to_index(pixel_action[2:4], (W_crop, H_crop))
            r0p, c0p = np.clip(r0p, 0, H_crop - 1), np.clip(c0p, 0, W_crop - 1)
            r1p, c1p = np.clip(r1p, 0, H_crop - 1), np.clip(c1p, 0, W_crop - 1)

            if not r0_mask[r0p, c0p]:
                reject = True
            
            if not r1_mask[r1p, c1p]:
                reject = True

        if not reject: 
            info = self.action_tool.step(env, world_action_)
        else:
            info = {}

        info['applied_action'] = pixel_action
        info['action_affordance_score'] = self.affordance_score
        return info