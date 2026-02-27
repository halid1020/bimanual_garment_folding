import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random

from actoris_harena.torch_utils import np_to_ts, ts_to_np
from .utils import preprocess_rgb, postprocess_rgb, gaussian_kernel


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1), indexing='ij')
    h = torch.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

class PickAndPlaceTransformerV1:
    def __init__(self,  config=None):
        self.config = config
        self.process_rgb = self.config.get('rgb_eval_process', False)
        self.process_depth = self.config.get('depth_eval_process', False)
        self.swap_action = self.config.get('swap_action', False)
        self.all_goal_rotate = self.config.get('all_goal_rotate', False)
        self.maskout = self.config.get('maskout', False)
        self.goal_translate = self.config.get('goal_translate', 0)
        self.preserve_goal_mask = self.config.get('preserve_goal_mask', False)
        self.bg_value = self.config.get('bg_value', 0)
        self.apply_mask = lambda x, y: x * y + self.bg_value * (1 - y)
        self.random_rotation = self.config.get('random_rotation', False)
        self.vertical_flip = self.config.get('vertical_flip', False)
        self.depth_flip = self.config.get('depth_flip', False)
        self.apply_depth_noise_on_mask = self.config.get('apply_depth_noise_on_mask', False)
        self.depth_blur = self.config.get('depth_blur', False)
        self.img_dim = (self.config.img_dim[0], self.config.img_dim[1])
        if self.all_goal_rotate:
            self.goal_rotation_degree = self.config.get('goal_rotation_degree', self.config.get('rotation_degree', 360))
        #self.maskout = self.config.get('maskout', False)

        if self.depth_blur:
            kernel_size = self.config.depth_blur_kernel_size
            sigma = 1.0
            self.kernel = gaussian_kernel(kernel_size, sigma).to(self.config.device)
            self.kernel = self.kernel.expand(1, 1, kernel_size, kernel_size)
            if kernel_size % 2 == 1:
                self.padding = (kernel_size - 1) // 2
            else:
                self.padding = kernel_size//2
    

    def __call__(self, sample_in, train=True, to_tensor=True, 
                 single=False):
        #print('before preprocess goal-rgb', sample['goal-rgb'].shape)
        # batch is assumed to have the shape B*T*C*H*W
        #print('transform!!!!')
        allowed_keys = ['rgb', 'depth', 'mask', 'rgbd', 'goal-rgb', 
                        'goal-depth', 'goal-mask', 'action', 'reward', 'terminal']
        if 'observation' in sample_in:
            sample = {k: v for k, v in sample_in['observation'].items() if k in allowed_keys}
        else:
            sample = sample_in
        
        if 'action' in sample_in:
            # if sample action is a dict
            if isinstance(sample_in['action'], dict):
                if 'default' in sample_in['action']:
                    sample['action'] = sample_in['action']['default']
                elif 'norm-pixel-pick-and-place' in sample_in['action']:
                    sample['action'] = sample_in['action']['norm-pixel-pick-and-place']
            ## flatten the last two dimension
            sample['action'] = sample['action'].reshape(sample['action'].shape[0], -1)
            #print('sample action', sample['action'].shape)

        for k, v in sample.items():
            #print(k, v.shape)
            if isinstance(v, np.ndarray):
                sample[k] = np_to_ts(v.copy(), self.config.device)
            else:
                sample[k] = v.to(self.config.device)
            
            #sample[k] = sample[k].unsqueeze(0)
            sample[k] = sample[k].float()


        for obs in ['rgb', 'depth', 'mask', 'rgbd', 'goal-rgb', 
                    'goal-depth', 'goal-mask', 'gc-depth', 'gc-rgb', 'gc-rgbd']:
            if obs in sample:
                #print('obs', obs)
                if sample[obs].shape[-1] <= 10:
                    sample[obs] = sample[obs].permute(0, 3, 1, 2)
                # print('obs', obs, sample[obs].shape)
                # print(' img dim', self.img_dim)
                T, C, H, W = sample[obs].shape
                if (H, W) != self.img_dim:
                    sample[obs] = F.interpolate(
                        sample[obs],
                        size=self.img_dim, mode='bilinear', align_corners=False)\
                            .view(T, C, *self.img_dim)

                    if obs == 'mask':
                        sample[obs] = (sample[obs] > 0.5).float()

        if 'gc-depth' in sample:
            sample['depth'] = sample['gc-depth'][:, :1]
            sample['goal-depth'] = sample['gc-depth'][:, 1:]
        
        if 'gc-rgb' in sample:
            sample['rgb'] = sample['gc-rgb'][:, :3]
            sample['goal-rgb'] = sample['gc-rgb'][:, 3:]
        
        if 'gc-rgbd' in sample:
            sample['rgb'] = sample['gc-rgbd'][:, :3]
            sample['depth'] = sample['gc-rgbd'][:, 3:4]
            sample['goal-rgb'] = sample['gc-rgbd'][:, 4:7]
            sample['goal-depth'] = sample['gc-rgbd'][:, 7:]
            # print('gc-rgb', sample['gc-rgb'].shape)
            # print('rgb', sample['rgb'].shape)

        
        if 'rgbd' in sample:
            #print('Were!!!!!!!!!!!!!!')
            sample['rgb'] = sample['rgbd'][:, :, :3]
            sample['depth'] = sample['rgbd'][:, :, 3:]
   
        if 'rgb' in sample:
            process_rgb = train and self.process_rgb
            sample['rgb'] = preprocess_rgb(
                sample['rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}, 
                noise_factor=(self.config.rgb_noise_factor if process_rgb else 0))
        
        if 'goal-rgb' in sample:
            process_rgb = train and self.process_rgb
            sample['goal-rgb'] = preprocess_rgb(
                sample['goal-rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}, 
                noise_factor=(self.config.rgb_noise_factor if process_rgb else 0))
            #print('after preprocess goal-rgb', sample['goal-rgb'].shape)

        if 'depth' in sample:
            sample['depth'] = self._process_depth(
                sample['depth'], 
                sample.get('mask', None), train)
        
        if 'goal-depth' in sample:
            #print('goal-depth', sample['goal-depth'].shape)
            sample['goal-depth'] = self._process_depth(
                sample['goal-depth'], 
                sample.get('goal-mask', None), train)

        if self.preserve_goal_mask:
            org_goal_mask = sample['goal-mask']

     
            
        if 'action' in sample.keys():
            self.has_orientation = sample['action'].shape[-1] > 4
            
            if self.has_orientation:
                # Split spatial coordinates from orientation
                pixel_actions = sample['action'][:, :4]  # Fixed: act -> sample['action']
                orient_actions = sample['action'][:, 4:] # (B, 1) -> [theta] in [-1, 1]
            else:
                pixel_actions = sample['action'][:, :4]
                orient_actions = None

            if not self.swap_action:
                # Swap [y, x, y, x] to [x, y, x, y] for geometric transformations
                pixel_actions = pixel_actions[:, [1, 0, 3, 2]]

        
        if self.all_goal_rotate and train:
            T, _, H, W = sample['goal-mask'].shape
            samples = T
            degree = self.goal_rotation_degree * \
                torch.randint(int(360 / self.goal_rotation_degree), size=(samples,))
            thetas = torch.deg2rad(degree)
            cos_theta = torch.cos(thetas)
            sin_theta = torch.sin(thetas)
            rot = torch.stack([
                torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(samples, 2, 2)
            ], dim=1).to(self.config.device)

            affine_matrix = torch.zeros(T, 2, 3, device=self.config.device)
            affine_matrix[:, :2, :2] = rot.reshape(T, 2, 2)

            if self.goal_translate != 0:
                #print('goal_translate', self.config.goal_translate)
                affine_matrix[:, 0, 2] = (torch.randn(T, device=self.config.device) * 2 -1) \
                    * self.config.goal_translate
                affine_matrix[:, 1, 2] = (torch.randn(T, device=self.config.device) * 2 -1) \
                    * self.config.goal_translate

            obs_to_rotate = ['goal-rgb', 'goal-depth', 'goal-mask']
            obs_dict = {obs: sample[obs] for obs in obs_to_rotate if obs in sample}

            combined_obs = torch.cat(list(obs_dict.values()), dim=1)
            T, C, H, W = combined_obs.shape

            # Rotate all observations at once
            grid = F.affine_grid(affine_matrix[:, :2], (T, C, H, W), align_corners=True)
            rotated_images = F.grid_sample(combined_obs, grid, align_corners=True)

            start_idx = 0
            for obs in obs_dict.keys():
                end_idx = start_idx + sample[obs].shape[1]
                sample[obs] = rotated_images[:, start_idx:end_idx]
                start_idx = end_idx

    
        # Random Rotation
        if self.random_rotation and train:

            ### Generate torch version of the follow code:
            while True:
                
                degree = self.config.rotation_degree * \
                    torch.randint(int(360 / self.config.rotation_degree), size=(1,))
                thetas = torch.deg2rad(degree)
                cos_theta = torch.cos(thetas)
                sin_theta = torch.sin(thetas)

                rot = torch.stack([
                    torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(2, 2)
                ], dim=0).to(self.config.device)


                # Rotate actions
                T_act = pixel_actions.shape[0]
                # Fixed: Use T_act instead of T-1 to match the batch dimension perfectly
                rotation_matrices_tensor = rot.expand(T_act * 2, 2, 2).reshape(-1, 2, 2)
                rotation_action = pixel_actions.reshape(-1, 1, 2)
               
                rotated_action = torch.bmm(rotation_action, rotation_matrices_tensor)\
                    .reshape(*pixel_actions.shape)
                
                # Fixed: Check the rotated_action for out-of-bounds, not the original
                if torch.abs(rotated_action).max() > 1:
                    continue
                
                pixel_actions = rotated_action

                # apply rotation to the orient action
                if self.has_orientation:
                    # Map degree to [-1, 1] space (Assuming 1.0 = 180 degrees)
                    degree_tensor = degree.to(orient_actions.device)
                    orient_shift = degree_tensor / 180.0
                    
                    # The affine grid rotates the image, so we rotate the angle equivalently
                    orient_actions = orient_actions - orient_shift
                    
                    # Wrap the angle strictly back to the [-1, 1] interval
                    orient_actions = ((orient_actions + 1.0) % 2.0) - 1.0

                # Rotate observations
                affine_matrix = torch.zeros(T, 2, 3, device=self.config.device)
                affine_matrix[:, :2, :2] = rot.expand(T, 2, 2)


                # Rotate observations
                obs_to_rotate = ['rgb', 'depth', 'mask']
                obs_dict = {obs: sample[obs] for obs in obs_to_rotate if obs in sample}

                combined_obs = torch.cat(list(obs_dict.values()), dim=1)
                T, C, H, W = combined_obs.shape

                # Rotate all observations at once
                grid = F.affine_grid(affine_matrix[:, :2], (T, C, H, W), align_corners=True)
                rotated_images = F.grid_sample(combined_obs, grid, align_corners=True)

                start_idx = 0
                for obs in obs_dict.keys():
                    end_idx = start_idx + sample[obs].shape[1]
                    sample[obs] = rotated_images[:, start_idx:end_idx]
                    start_idx = end_idx
    
                break
                # if the max absolute value of the action is more than 1, continue
                
        # Vertical Flip
        if self.vertical_flip and train and (random.random() < 0.5):
            
            # Generate random vertical flip decisions
            obs_dict = {obs: sample[obs] for obs in ['rgb', 'depth', 'mask'] if obs in sample}
            obs_images = torch.cat(list(obs_dict.values()), dim=1)
            flip_obs_images = torch.flip(obs_images, [2])
            start_idx = 0
            for obs in obs_dict.keys():
                end_idx = start_idx + sample[obs].shape[1]
                sample[obs] = flip_obs_images[:, start_idx:end_idx]
                start_idx = end_idx

            T_act = pixel_actions.shape[0]
            new_actions = pixel_actions.reshape(-1, 2)
            new_actions[:, 1] = -new_actions[:, 1]
            pixel_actions = new_actions.reshape(T_act, 4)
            
            # apply flipping to the orient action
            if self.has_orientation:
                # Vertical flip inverts the Y axis, meaning the angle sign flips
                orient_actions = -orient_actions

        if 'action' in sample.keys():
            if not self.swap_action: # Fixed: removed the syntax error ')'
                # swap action back to the original order [y1, x1, y2, x2]
                pixel_actions = pixel_actions[:, [1, 0, 3, 2]]
            
            if self.has_orientation: # Fixed: self.has_orient -> self.has_orientation
                # combine back the pixel actions and orient actions
                sample['action'] = torch.cat([pixel_actions, orient_actions], dim=-1)
            else:
                sample['action'] = pixel_actions

        if self.maskout:
           
            
            # Prepare masks
            mask = sample['mask']
            if 'goal-mask' in sample:
                goal_mask = sample['goal-mask']
            #goal_mask_3ch = goal_mask.repeat(1, 3, 1, 1) if goal_mask is not None else None

            # Define a helper function for masking
            

            # Apply masking to rgb and depth
            if 'rgb' in sample and mask is not None:
                sample['rgb'] = self.apply_mask(sample['rgb'], mask)
            if 'depth' in sample and mask is not None:
                sample['depth'] = self.apply_mask(sample['depth'], mask)

            # Apply masking to goal-rgb and goal-depth
            if 'goal-rgb' in sample and goal_mask is not None:
                sample['goal-rgb'] = self.apply_mask(sample['goal-rgb'], goal_mask)
            if 'goal-depth' in sample and goal_mask is not None:
                sample['goal-depth'] = self.apply_mask(sample['goal-depth'], goal_mask)

                

        if 'rgbd' in sample:
            #print('Here!!!!!!!!!!!!!!')
            sample['rgbd'][:, :3] = sample['rgb']
            sample['rgbd'][:, 3:] = sample['depth']

        if 'gc-depth' in sample:
            sample['gc-depth'] = torch.cat([sample['depth'], sample['goal-depth']], dim=1)

        if 'gc-rgb' in sample:
            sample['gc-rgb'] = torch.cat([sample['rgb'], sample['goal-rgb']], dim=1)
        if 'gc-rgbd' in sample:
            sample['gc-rgbd'] = torch.cat([sample['rgb'], sample['depth'], 
                                           sample['goal-rgb'], sample['goal-depth']], dim=1)

        if self.preserve_goal_mask:
            sample['goal-mask'] = org_goal_mask
        
        
        
        ## check if there is any nan value
        for k, v in sample.items():
            if torch.isnan(v).any():
                print('Transform nan value in', k)
                #print(v)
                raise ValueError('nan value in the transform data {k}')

      
        if not to_tensor:
            for k, v in sample.items():
                sample[k] = ts_to_np(v)
        
        
        return sample
    
    def postprocess(self, sample):
        
        res = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                res[k] = ts_to_np(v)
            else:
                res[k] = v   
            #print(k, res[k].shape)

        if 'rgb' in res:
            res['rgb'] = postprocess_rgb(
                res['rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}) 
        
        if 'rgbd' in res:
            if len(res['rgbd'].shape) == 3:
                rgb = postprocess_rgb(res['rgbd'][:3, :, :], 
                            normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}).astype(np.float32)
                depth = res['rgbd'][3:, :, :].astype(np.float32)

                if self.config.z_norm:
                    depth = depth * self.config.z_norm_std + \
                        self.config.z_norm_mean
                
                if self.config.min_max_norm:
                    depth = \
                        depth * (self.config.depth_max - self.config.depth_min) \
                        + self.config.depth_min
        
                res['rgbd'] = np.concatenate([rgb, depth], axis=0).astype(np.float32)

            else:
                res['rgbd'] = postprocess_rgb(res['rgbd'][:, :3, :, :],
                            normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}) 

        if 'rgbm' in res:
            res['rgbm'] = postprocess_rgb(res['rgbm'][:, :3, :, :])
            
        if 'depth' in res:
            
            if self.config.z_norm:
                res['depth'] = res['depth'] * self.config.z_norm_std + \
                      self.config.z_norm_mean
            
            if self.config.min_max_norm:
                res['depth'] = \
                    res['depth'] * (self.config.depth_max - self.config.depth_min) \
                    + self.config.depth_min

        if 'action' in res:
            res['action'] = res['action'].clip(-1, 1)

        return res
    

    def _process_depth(self, depth, mask=None, train=False):
        T, C, H, W = depth.shape
    


        #print('depth')
        #obs[:,-1,:, :] += torch.randn(obs[:,-1,:, :].shape, device=self.config.device) * (self.config.depth_noise_var if train else 0)
        if self.config.depth_clip:
            depth = depth.clip(self.config.depth_clip_min, self.config.depth_clip_max)
        
        if self.config.z_norm:
            depth = (depth - self.config.z_norm_mean) / self.config.z_norm_std

        elif self.config.min_max_norm:
            # get the min and max of each trajectory
            depth_min = depth.view(T, -1).min(dim=1, keepdim=True).values
            depth_max = depth.view(T, -1).max(dim=1, keepdim=True).values

            ## depth_min compare with self.config.depth_min and get the max ; each trajectory

            if self.config.get('depth_hard_interval', False):
                #print('depth_hard_interval', self.config.depth_hard_interval)
                depth_min = self.config.depth_min
                depth_max = self.config.depth_max
            else:
                depth_min = torch.max(
                    depth_min, 
                    torch.tensor(self.config.depth_min).to(depth_min.device)).view(T, 1, 1, 1)
                
                depth_max = torch.min(
                    depth_max, 
                    torch.tensor(self.config.depth_max).to(depth_max.device)).view(T, 1, 1, 1)



            depth = (depth-depth_min) / (depth_max-depth_min+1e-6)
            if self.depth_flip:
                depth = 1 - depth
        
        depth_process = train and self.process_depth

        depth_noise = \
            torch.randn(depth.shape, device=self.config.device) \
                * (self.config.depth_noise_var if depth_process else 0)

        if self.depth_blur and depth_process:
            ### apply gaussian blur on each image
            T, C, H, W = depth.shape
            
            
            depth_reshaped = depth.view(T, C, H, W)
            blurred_depth = F.conv2d(depth_reshaped, self.kernel, padding=self.padding, groups=C)[:, :, :H, :W]


            # Reshape back to original dimensions
            depth = blurred_depth.reshape(T, C, H, W)

            # remap to [0, 1]
            depth_min = depth.reshape(T, -1).min(dim=1, keepdim=True).values.reshape(T, 1, 1, 1)
            depth_max = depth.reshape(T, -1).max(dim=1, keepdim=True).values.reshape(T, 1, 1, 1)
            # print('depth min shape', depth_min.shape)
            # print('depth max shape', depth_max.shape)
            # print('depth shape', depth.shape)
            depth = (depth-depth_min) / (depth_max-depth_min+1e-6)

        if  self.apply_depth_noise_on_mask \
            and (mask is not None) and depth_process:
            depth_noise *= mask
            
        depth += depth_noise
        
        depth = depth.clip(0, 1)
        
        if self.config.depth_map:
            # At this point we assume depth is between [0, 1]
            map_diff = self.config.depth_map_range[1] - self.config.depth_map_range[0]
            depth = depth*map_diff + self.config.depth_map_range[0]
        
        return depth