
import numpy as np
from gym.spaces import Dict, Discrete, Box
import random

from scipy.ndimage import binary_erosion
from .world_pick_and_place \
    import WorldPickAndPlace
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
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.action_tool = WorldPickAndPlace(**kwargs) 
        
        #### Define the action space
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
        
        ### Each parameters has its class variable
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

    def _snap_to_mask(self, norm_pos, mask):
        """
        Helper to snap a normalized pixel to the nearest valid value in an eroded mask.
        Erodes by 2 pixels to keep the point away from the edge.
        """
        # --- NEW LOGIC: Erode the mask ---
        # 2 iterations = 2 pixels (roughly, depending on connectivity)
        eroded_mask = binary_erosion(mask, iterations=1)
        
        # Safety check: if erosion removes the whole mask (area too small), use original
        target_mask = eroded_mask if np.any(eroded_mask) else mask
        # ---------------------------------

        H, W = target_mask.shape
        r, c = norm_pixel_to_index(norm_pos, (W, H))
        
        r = np.clip(r, 0, H - 1)
        c = np.clip(c, 0, W - 1)

        # If the point is valid in the ERODED mask, keep it.
        if target_mask[r, c]:
            return norm_pos

        # Otherwise, snap to the nearest valid point in the ERODED mask
        valid_indices = np.argwhere(target_mask)
        
        if len(valid_indices) == 0:
            return norm_pos

        dists = np.sum((valid_indices - np.array([r, c]))**2, axis=1)
        nearest_idx = valid_indices[np.argmin(dists)]
        
        nr, nc = nearest_idx
        
        new_x = (nc / W) * 2 - 1.0
        new_y = (nr / H) * 2 - 1.0
        
        return np.array([new_x, new_y])
        
    def process(self, env, action):
        pick_0 = np.asarray(action['pick_0'])
        place_0 = np.asarray(action['place_0'])

        mask = env._get_cloth_mask()
        adj_pick_0, dist_0 = readjust_norm_pixel_pick(pick_0, mask)

        if random.random() < self.readjust_pick_poss:
            pick_0 = adj_pick_0
        else:
            pass
            
        dist_1 = 0

        pick_0_depth = action['pick_0_d'] if 'pick_0_d' in action else self.camera_height  - self.pick_height
        place_0_depth = action['place_0_d'] if 'place_0_d' in action else self.camera_height  - self.place_height

        if 'pick_1' in action:
            pick_1 = np.asarray(action['pick_1'])
            place_1 = np.asarray(action['place_1'])

            adj_pick_1, dist_1 = readjust_norm_pixel_pick(pick_1, mask)

            if random.random() < self.readjust_pick_poss:
                pick_1 = adj_pick_1

            pick_1_depth = action['pick_1_d'] if 'pick_1_d' in action else self.camera_height  - self.pick_height
            place_1_depth = action['place_1_d'] if 'place_1_d' in action else self.camera_height - self.place_height
        else:
            pick_1 = np.asarray(np.ones(2)) * 1.5
            place_1 = np.asarray(np.ones(2)) * 1.5
            pick_1_depth = self.camera_height
            place_1_depth = self.camera_height

        self.affordance_score = self._calculate_affordance(dist_0, dist_1)
        
        # Swap logic: ensure pick_0 is "above" or "left" of pick_1 depending on logic
        if pick_0[1] > pick_1[1]:
            pick_0, pick_1 = pick_1, pick_0
            place_0, place_1 = place_1, place_0
            pick_0_depth, pick_1_depth = pick_1_depth, pick_0_depth
            place_0_depth, place_1_depth = place_1_depth, place_0_depth
        
        # --- NEW LOGIC: Readjust to workspace ---
        # Must happen AFTER swap so we know which coords belong to which physical robot
        if env.apply_workspace and env.readjust_to_workspace:
            print('Snap!!!')
            # Snap Robot 0 (pick_0) to robot0_mask
            pick_0, _ = readjust_norm_pixel_pick(pick_0, env.robot0_mask.copy())
            place_0, _ = readjust_norm_pixel_pick(place_0, env.robot0_mask.copy())

            
            # pick_0 = self._snap_to_mask(pick_0, env.robot0_mask.copy())
            # place_0 = self._snap_to_mask(place_0, env.robot0_mask.copy())

            # Snap Robot 1 (pick_1) to robot1_mask
            # Only if we are not in single operator mode (checked via action/bounds check usually, 
            # but picking safe logic here)
            if 'pick_1' in action: 
                # pick_1 = self._snap_to_mask(pick_1, env.robot1_mask.copy())
                # place_1 = self._snap_to_mask(place_1, env.robot1_mask.copy())

                pick_1, _ = readjust_norm_pixel_pick(pick_1, env.robot1_mask.copy())
                place_1, _ = readjust_norm_pixel_pick(place_1, env.robot1_mask.copy())

        # ----------------------------------------

        action_ = np.concatenate([pick_0, place_0, pick_1, place_1]).reshape(-1, 2)
        
        depths = np.array([
            pick_0_depth, place_0_depth, 
            pick_1_depth, place_1_depth])

        W, H = self.camera_size

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
        # 0 is bad, 1 is good

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

        if env.apply_workspace:
            W, H = env.robot0_mask.shape[:2]

        reject = False
        if env.apply_workspace:
            # -------- robot 0 --------
            r0p, c0p = norm_pixel_to_index(pixel_action[:2], (W, H))
            r0l, c0l = norm_pixel_to_index(pixel_action[4:6], (W, H))

            if not env.robot0_mask[r0p, c0p]:
                print('[PixelPickAndPlace] Reject: pick_0 outside robot0 workspace')
                reject = True
                info = {}
            
            if not env.robot0_mask[r0l, c0l]:
                print('[PixelPickAndPlace] Reject: place_0 outside robot0 workspace')
                reject = True
                info = {}
    
            # -------- robot 1 --------
            r1p, c1p = norm_pixel_to_index(pixel_action[2:4], (W, H))
            r1l, c1l = norm_pixel_to_index(pixel_action[6:8], (W, H))
            
            if not env.robot1_mask[r1p, c1p]:
                print('[PixelPickAndPlace] Reject: pick_1 outside robot1 workspace')
                reject = True
                info = {}

            if not env.robot1_mask[r1l, c1l]:
                print('[PixelPickAndPlace] Reject: place_1 outside robot1 workspace')
                reject = True
                info = {}
            
        if not reject: 
            info = self.action_tool.step(env, world_action_)
        info['applied_action'] = pixel_action
        info['action_affordance_score'] = self.affordance_score
        return info