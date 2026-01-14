# import numpy as np
# from gym.spaces import Box

# from .world_pick_and_fling \
#     import WorldPickAndFling
# from ..utils.camera_utils import norm_pixel2world
# from .utils import pixel_to_world, readjust_norm_pixel_pick, norm_pixel_to_index
# import random

# class PixelPickAndFling():

#     def __init__(self, 
#         lowest_cloth_height=0.1,
#         max_grasp_dist=0.7,
#         stretch_increment_dist=0.02,
        
#         pregrasp_height=0.3,
#         pregrasp_vel=0.1,
#         tograsp_vel=0.05,
#         prefling_height=0.3, #7,#
#         prefling_vel=0.01,
#         hang_pos_y=0.3, #0 when workspace
#         fling_y=0.5, # 0.3 when workspace
#         lift_vel=0.02, #0.01,
#         #adaptive_fling_momentum=1.0,
#         action_horizon=20,
#         hang_adjust_vel=0.01,
#         stretch_adjust_vel=0.01,
#         fling_vel= 0.02, #0.03 when workspace
#         release_vel=0.01,
#         drag_vel=0.005,
#         lower_height=0.06,
#         readjust_pick_poss=0.0,

#         pick_lower_bound=[-1, -1],
#         pick_upper_bound=[1, 1],
#         place_lower_bound=[-1, -1],
#         place_upper_bound=[1, 1],
#         pick_height=0.025, #0.02,

#         **kwargs):
        
#         ### Environment has to be WorldPickAndFlingWrapper
#         self.action_tool = WorldPickAndFling(**kwargs) 
        
#         self.action_horizon = action_horizon
#         self.lowest_cloth_height = lowest_cloth_height
#         self.max_grasp_dist = max_grasp_dist
#         self.stretch_increment_dist = stretch_increment_dist
#         self.fling_vel = fling_vel
#         self.pregrasp_height = pregrasp_height
#         self.pregrasp_vel = pregrasp_vel
#         self.tograsp_vel = tograsp_vel
#         self.prefling_height = prefling_height
#         self.prefling_vel = prefling_vel
#         self.hang_pos_y = hang_pos_y
#         self.lift_vel = lift_vel
#         #self.adaptive_fling_momentum = adaptive_fling_momentum
#         self.pick_height = pick_height
#         self.hang_adjust_vel = hang_adjust_vel
#         self.stretch_adjust_vel = stretch_adjust_vel
#         self.release_vel = release_vel
#         self.drag_vel = drag_vel
#         self.lower_height = lower_height
#         self.fling_y = fling_y

#         self.num_pickers = 2
#         self.readjust_pick_poss = readjust_pick_poss

        

#         space_low = np.concatenate([pick_lower_bound, place_lower_bound]*self.num_pickers)\
#             .reshape(self.num_pickers, -1).astype(np.float32)
#         space_high = np.concatenate([pick_upper_bound, place_upper_bound]*self.num_pickers)\
#             .reshape(self.num_pickers, -1).astype(np.float32)
#         self.action_space = Box(space_low, space_high, dtype=np.float32)

    
#     def get_no_op(self):
#         return self.no_op
        
#     def sample_random_action(self):
#         return self.action_space.sample()

#     def get_action_space(self):
#         return self.action_space
    
#     def get_action_horizon(self):
#         return self.action_horizon
    
#     def reset(self, env):
#         return self.action_tool.reset(env)

    
#     def _calculate_affordance(self, dist_0, dist_1):

#         return np.min([
#             1 - min(dist_0, np.sqrt(8)) / np.sqrt(8),
#             1 - min(dist_1, np.sqrt(8)) / np.sqrt(8)
#         ])
    
#     def process(self, env, action):
#         #action = action['norm_pixel_pick_and_fling']

#         p0 = np.asarray(action['pick_0'])
#         p1 = np.asarray(action['pick_1'])

#         mask = env._get_cloth_mask()
#         adj_p0, dist_0 = readjust_norm_pixel_pick(p0, mask)
#         adj_p1, dist_1 = readjust_norm_pixel_pick(p1, mask)
       

#         if random.random() < self.readjust_pick_poss:
#             p0 = adj_p0
#             p1 = adj_p1

#            #print('Fling readjust picks')
        
#         self.affordance_score =  self._calculate_affordance(dist_0, dist_1)
           

#         ref_a = np.array([-1, 1])
#         ref_b = np.array([1, 1])

#         if p0[1] > p1[1]:
#             p0, p1 = p1, p0
      
#         action_ = np.concatenate([p0, p1]).reshape(-1, 2)
#         # convert to world coordinate
#         # print('p0', p0)
#         # print('p1', p1)
#         W, H = self.camera_size
#         p0_depth = self.camera_height  - self.pick_height
#         p1_depth = self.camera_height  - self.pick_height
#         depths = np.array([p0_depth, p1_depth])

#         convert_action = norm_pixel2world(
#                 action_, np.asarray([H, W]),  
#                 self.camera_intrinsics, self.camera_pose, depths) 
#         convert_action = convert_action.reshape(2, 3)

#         p0_ = convert_action[0]
#         p1_ = convert_action[1]

#         # print('p0:', p0)
#         # print('p1:', p1)

        

#         world_action =  {
#             'pick_0_position': p0_,
#             'pick_1_position': p1_,
#             'pregrasp_height': self.pregrasp_height,
#             'pregrasp_vel': self.pregrasp_vel,
#             'tograsp_vel': self.tograsp_vel,
#             'prefling_height': self.prefling_height,
#             'prefling_vel': self.prefling_vel,
#             'lift_vel': self.lift_vel,

#             'hang_pos_y': self.hang_pos_y,
#             'fling_y': self.fling_y,
#             'hang_adjust_vel': self.hang_adjust_vel, # for hang and stretch
#             'stretch_adjust_vel': self.stretch_adjust_vel, # for hang and stretch
#             'fling_vel': self.fling_vel, # for fling 
#             'release_vel': self.release_vel, # for fling and release
#             'drag_vel': self.drag_vel, # for  fling and release
#             'lower_height': self.lower_height, # for lower the picker before release
#         }

#         # pixel_action = {
#         #     'pick_0': p0,
#         #     'pick_1': p1
#         # }

#         pixel_action = np.stack([p0, p1]).flatten()
        

#         return world_action, pixel_action
    
#     ## It accpet action has shape (num_picker, 2, 3), where num_picker can be 1 or 2
#     def step(self, env, action):
#         self.camera_height = env.camera_height
#         # self.camera_to_world_ratio = env.pixel_to_world_ratio
#         self.camera_intrinsics = env.camera_intrinsic_matrix
#         self.camera_pose = env.camera_extrinsic_matrix
#         self.camera_size = env.camera_size
#         world_action_, pixel_action = self.process(env, action)
#         W, H = self.camera_size
#         reject=False
#         if env.apply_workspace:

#             # -------- robot 0 --------
#             r0p, c0p = norm_pixel_to_index(pixel_action[:2], (W, H))
#             r1p, c1p = norm_pixel_to_index(pixel_action[2:4], (W, H))

#             if not env.robot0_mask[r0p, c0p]:
#                 print('[PixelPickAndPlace] Reject: pick_0 outside robot0 workspace')
#                 reject=True
#                 info = {}
            

#             if not env.robot1_mask[r1p, c1p]:
#                 print('[PixelPickAndPlace] Reject: pick_1 outside robot1 workspace')
#                 reject=True
#                 info = {}

   
#         if not reject: 
#             info = self.action_tool.step(env, world_action_)
#         info['applied_action'] = pixel_action
#         info['action_affordance_score'] = self.affordance_score
#         return info

import numpy as np
from gym.spaces import Box

from .world_pick_and_fling \
    import WorldPickAndFling
from ..utils.camera_utils import norm_pixel2world
from .utils import pixel_to_world, readjust_norm_pixel_pick, norm_pixel_to_index
import random

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
        
        readjust_to_workspace=False, # <--- Added flag here

        **kwargs):
        
        ### Environment has to be WorldPickAndFlingWrapper
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
        self.readjust_to_workspace = readjust_to_workspace # <--- Store flag

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

    def _snap_to_mask(self, norm_pos, mask):
        """Helper to snap a normalized pixel to the nearest valid value in a mask."""
        H, W = mask.shape
        r, c = norm_pixel_to_index(norm_pos, (W, H))
        
        # Clip indices to be safe
        r = np.clip(r, 0, H - 1)
        c = np.clip(c, 0, W - 1)

        # If already valid, return original
        if mask[r, c]:
            return norm_pos

        # Find all valid indices in the mask
        valid_indices = np.argwhere(mask)
        
        # If mask is empty, return original (cannot snap)
        if len(valid_indices) == 0:
            return norm_pos

        # Find the nearest valid index (Euclidean distance)
        dists = np.sum((valid_indices - np.array([r, c]))**2, axis=1)
        nearest_idx = valid_indices[np.argmin(dists)]
        
        nr, nc = nearest_idx
        
        # Convert index back to normalized coordinates [-1, 1]
        new_x = (nc / W) * 2 - 1.0
        new_y = (nr / H) * 2 - 1.0
        
        return np.array([new_x, new_y])
    
    def _calculate_affordance(self, dist_0, dist_1):
        return np.min([
            1 - min(dist_0, np.sqrt(8)) / np.sqrt(8),
            1 - min(dist_1, np.sqrt(8)) / np.sqrt(8)
        ])
    
    def process(self, env, action):
        p0 = np.asarray(action['pick_0'])
        p1 = np.asarray(action['pick_1'])

        mask = env._get_cloth_mask()
        adj_p0, dist_0 = readjust_norm_pixel_pick(p0, mask)
        adj_p1, dist_1 = readjust_norm_pixel_pick(p1, mask)
       
        if random.random() < self.readjust_pick_poss:
            p0 = adj_p0
            p1 = adj_p1
        
        self.affordance_score =  self._calculate_affordance(dist_0, dist_1)

        # Ensure correct arm assignment based on Y position (standard assumption in this env)
        if p0[1] > p1[1]:
            p0, p1 = p1, p0
        
        # --- NEW LOGIC: Readjust to workspace ---
        # Must happen AFTER swap so we know p0 is robot0 and p1 is robot1
        if env.apply_workspace and env.readjust_to_workspace:
            
            p0, _ = readjust_norm_pixel_pick(p0,  env.robot0_mask)
            p1, _ = readjust_norm_pixel_pick(p1, env.robot1_mask)

            # p0 = self._snap_to_mask(p0, env.robot0_mask)
            # p1 = self._snap_to_mask(p1, env.robot1_mask)
        # ----------------------------------------

        action_ = np.concatenate([p0, p1]).reshape(-1, 2)
        
        W, H = self.camera_size
        p0_depth = self.camera_height  - self.pick_height
        p1_depth = self.camera_height  - self.pick_height
        depths = np.array([p0_depth, p1_depth])

        convert_action = norm_pixel2world(
                action_, np.asarray([H, W]),  
                self.camera_intrinsics, self.camera_pose, depths) 
        convert_action = convert_action.reshape(2, 3)

        p0_ = convert_action[0]
        p1_ = convert_action[1]

        world_action =  {
            'pick_0_position': p0_,
            'pick_1_position': p1_,
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
        W, H = self.camera_size
        reject=False
        if env.apply_workspace:

            # -------- robot 0 --------
            r0p, c0p = norm_pixel_to_index(pixel_action[:2], (W, H))
            r1p, c1p = norm_pixel_to_index(pixel_action[2:4], (W, H))

            if not env.robot0_mask[r0p, c0p]:
                print('[PixelPickAndFling] Reject: pick_0 outside robot0 workspace')
                reject=True
                info = {}
            
            if not env.robot1_mask[r1p, c1p]:
                print('[PixelPickAndFling] Reject: pick_1 outside robot1 workspace')
                reject=True
                info = {}

        if not reject: 
            info = self.action_tool.step(env, world_action_)
        info['applied_action'] = pixel_action
        info['action_affordance_score'] = self.affordance_score
        return info