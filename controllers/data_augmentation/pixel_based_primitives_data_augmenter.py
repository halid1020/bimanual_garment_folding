import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np
from .utils import preprocess_rgb, postprocess_rgb, gaussian_kernel


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1), indexing='ij')
    h = torch.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

class PixelBasedPrimitiveDataAugmenter:
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
        if self.all_goal_rotate:
            self.rotation_degree = self.config.get('rotation_degree', 360)
            self.goal_rotation_degree = self.config.get('goal_rotation_degree', self.rotation_degree)
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
            sample['action'] = sample['action'].reshape(sample['action'].shape[0], sample['action'].shape[1], -1)
            #print('sample action', sample['action'].shape)

        for k, v in sample.items():
            #print(k, v.shape)
            #print('self.device', self.config.device)
            if isinstance(v, np.ndarray):
                sample[k] = np_to_ts(v.copy(), self.config.device)
            else:
                sample[k] = v.to(self.config.device)
            
            #sample[k] = sample[k].unsqueeze(0)
            sample[k] = sample[k].float()

        #plot pre process action on image
        if self.config.debug:
            if self.config.primitive == 'norm-pixel-fold':
                from agent_arena.utilities.visual_utils import draw_pick_and_place
                import cv2
                rgb = sample['rgb'][0, 0].squeeze(0).cpu().numpy()
                #print('rgb shape', rgb.shape)
                if rgb.shape[0] == 3:
                    rgb = rgb.transpose(1, 2, 0)
                H, W = rgb.shape[:2]
                action = sample['action'][0, 0].cpu().numpy().reshape(2, 4)
                #print('action', action)
                pick = (action[0, :2] + 1)/2 * np.array([W, H])
                pick = (int(pick[0]), int(pick[1]))
                place = (action[0, 2:] + 1)/2 * np.array([W, H])
                place = (int(place[0]), int(place[1]))
                pnp_rgb = draw_pick_and_place(
                    rgb, pick, place, get_ready=True, swap=False, color=(0, 0, 255)
                )
                pick = (action[1, :2] + 1)/2 * np.array([W, H])
                pick = (int(pick[0]), int(pick[1]))
                place = (action[1, 2:] + 1)/2 * np.array([W, H])
                place = (int(place[0]), int(place[1]))
                pnp_rgb = draw_pick_and_place(
                    pnp_rgb, pick, place, get_ready=True, swap=False, color=(255, 0, 0)
                )
                cv2.imwrite('tmp/pre_fold_rgb.png', pnp_rgb)


        for obs in ['rgb', 'depth', 'mask', 'rgbd', 'goal-rgb', 'goal-depth', 'goal-mask']:
            
            if obs in sample:
                if len(sample[obs].shape) == 4:
                    sample[obs] = sample[obs].unsqueeze(0)

                if sample[obs].shape[-1] <= 4:
                    sample[obs] = sample[obs].permute(0, 1, 4, 2, 3)
                #print('obs', obs, sample[obs].shape)
                B, T, C, H, W = sample[obs].shape

                sample[obs] = F.interpolate(
                    sample[obs].view(B*T, C, H, W),
                    size=self.config.img_dim, mode='bilinear', align_corners=False)\
                        .view(B, T, C, *self.config.img_dim)

                if obs == 'mask':
                    sample[obs] = (sample[obs] > 0.5).float()

                #print(f'sample {obs} shape', sample[obs].shape)


        # return sample
        if 'rgb' in sample:
            process_rgb = train and self.process_rgb
            sample['rgb'] = preprocess_rgb(
                sample['rgb'], 
                normalise={'mode': self.config.rgb_norm_mode, 
                           'param': self.config.rgb_norm_param}, 
                noise_factor=(self.config.rgb_noise_factor if process_rgb else 0))
        

        if 'depth' in sample:
            sample['depth'] = self._process_depth(
                sample['depth'].reshape(B*T, 1, *self.config.img_dim), 
                sample.get('mask', None), train).reshape(B, T, 1, *self.config.img_dim)
        
        if self.config.reward_scale and train:
            sample['reward'] *= self.config.reward_scale
        
        # we assume the action is in the correct form
        # if 'action' in sample.keys() and (not self.swap_action):
        #     sample['action'] = sample['action'][:, [1, 0, 3, 2]]

            # if 'swap_action' in self.config and self.config.swap_action:
            # # print('swap action')
            #     sample['action'] = sample['action'][:, [1, 0, 3, 2]]
  
    
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

                rot_inv = rot.transpose(-1, -2) 


                # Rotate actions
                num_points = 2
                if self.config.primitive == 'norm-pixel-fold':
                    num_points = 4
                rotation_matrices_tensor = rot_inv.expand(B*(T-1)*num_points, 2, 2).reshape(-1, 2, 2)
                rotation_action = sample['action'].reshape(-1, 1, 2)
               
                rotated_action = torch.bmm(rotation_action, rotation_matrices_tensor)\
                    .reshape(*sample['action'].shape)
                #print('rotated_action', rotated_action.shape)
                
                if torch.abs(sample['action']).max() > 1:
                    #print('max action', torch.abs(sample['action']).max())
                    continue
                #sample['action'] = rotated_action

                # Rotate observations
                affine_matrix = torch.zeros(B*T, 2, 3, device=self.config.device)
                affine_matrix[:, :2, :2] = rot.expand(B*T, 2, 2)


                # Rotate observations
                obs_to_rotate = ['rgb', 'depth', 'mask']
                obs_dict = {obs: sample[obs] for obs in obs_to_rotate if obs in sample}

                combined_obs = torch.cat(list(obs_dict.values()), dim=2)
                B, T, C, H, W = combined_obs.shape

                # Rotate all observations at once
                grid = F.affine_grid(affine_matrix[:, :2], (B*T, C, H, W), align_corners=True)
                rotated_images = F.grid_sample(
                    combined_obs.reshape(B*T, C, H, W), 
                    grid, align_corners=True).reshape(B, T, C, H, W)

                start_idx = 0
                for obs in obs_dict.keys():
                    end_idx = start_idx + sample[obs].shape[2]
                    sample[obs] = rotated_images[:, :, start_idx:end_idx]
                    start_idx = end_idx
    
               
                sample['action'] = rotated_action
                break
                # if the max absolute value of the action is more than 1, continue
                
        # Vertical Flip
        if self.vertical_flip and train and (random.random() < 0.5):
            
            # Generate random vertical flip decisions
            obs_dict = {obs: sample[obs] for obs in ['rgb', 'depth', 'mask'] if obs in sample}
            obs_images = torch.cat(list(obs_dict.values()), dim=2)
            B, T, C, H, W = obs_images.shape
            flip_obs_images = torch.flip(obs_images.reshape(B*T, C, H, W), [2]).reshape(B, T, C, H, W)
            start_idx = 0
            for obs in obs_dict.keys():
                end_idx = start_idx + sample[obs].shape[2]
                sample[obs] = flip_obs_images[:, :, start_idx:end_idx]
                start_idx = end_idx

            B, T, _ = sample['action'].shape
            new_actions = sample['action'].reshape(-1, 2)
            new_actions[:, 1] = -new_actions[:, 1]
            sample['action'] = new_actions.reshape(B, T, num_points*2)
            # sample['action'] = new_actions.reshape(*sample['action'].shape)
        
        # if 'action' in sample.keys() and (not self.swap_action):
        #     # swap action to the correct order
        #     sample['action'] = sample['action'][:, [1, 0, 3, 2]]
            
            # if self.swap_action:
            #     # print('swap action')
            #     sample['action'] = sample['action'][:, [1, 0, 3, 2]]



        # if self.mask_out:
        #     bg_value = self.config.bg_value \
        #         if 'bg_value' in self.config else 0
        #     # print('rgb', sample['rgb'].shape)
        #     # print('mask', sample['mask'].shape)
        #     if 'rgb' in sample:
        #         sample['rgb'] = sample['rgb'] * sample['mask'] + \
        #             bg_value * (1 - sample['mask'])
        #     if 'depth' in sample:
        #         sample['depth'] = sample['depth'] * sample['mask'] + \
        #             bg_value * (1 - sample['mask'])
            
        #     if 'goal-rgb' in sample:
        #         # make mask three channel
        #         #print('goal-rgb', sample['goal-rgb'].shape)
        #         #print('goal-mask', sample['goal-mask'].shape)
        #         mask_ = sample['goal-mask'].repeat(1, 3, 1, 1)
        #         #print('mask_', mask_.shape)
        #         sample['goal-rgb'] = sample['goal-rgb'] * mask_ + \
        #             bg_value * (1 - mask_)
            
        #     if 'goal-depth' in sample:
        #         sample['goal-depth'] = sample['goal-depth'] * sample['goal-mask'] + \
        #             bg_value * (1 - sample['goal-mask'])

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

                
        
        
        if self.config.debug:
            if self.config.primitive == 'norm-pixel-fold':
                from agent_arena.utilities.visual_utils import draw_pick_and_place
                import cv2
                rgb = (sample['rgb'][0, 0].squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                print('post rgb shape', rgb.shape)
                if rgb.shape[0] == 3:
                    rgb = rgb.transpose(1, 2, 0)
                H, W = rgb.shape[:2]
                action = sample['action'][0, 0].cpu().numpy().reshape(2, 4)
                #print('action', action)
                pick = (action[0, :2] + 1)/2 * np.array([W, H])
                pick = (int(pick[0]), int(pick[1]))
                place = (action[0, 2:] + 1)/2 * np.array([W, H])
                place = (int(place[0]), int(place[1]))
                pnp_rgb = draw_pick_and_place(
                    rgb, pick, place, get_ready=True, swap=False, color=(0, 0, 255)
                )
                pick = (action[1, :2] + 1)/2 * np.array([W, H])
                pick = (int(pick[0]), int(pick[1]))
                place = (action[1, 2:] + 1)/2 * np.array([W, H])
                place = (int(place[0]), int(place[1]))
                pnp_rgb = draw_pick_and_place(
                    pnp_rgb, pick, place, get_ready=True, swap=False, color=(255, 0, 0)
                )
                cv2.imwrite('tmp/post_fold_rgb.png', pnp_rgb)

        
        
        ## check if there is any nan value
        for k, v in sample.items():
            if torch.isnan(v).any():
                print('Transform nan value in', k)
                #print(v)
                raise ValueError('nan value in the transform data {k}')

      
        if not to_tensor:
            for k, v in sample.items():
                sample[k] = ts_to_np(v)
        
        # print all output shape
        # print('HELLOO')
        # for k, v in sample.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        # exit()
        
        return sample
    
    # def apply_mask(self, data, mask):
    #     return data * mask + self.bg_value * (1 - mask)
    
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
        
        # if 'action_heatmap' in res:
        #     # convert action heatmap to action
        #     #print('action heatmap shape', res['action_heatmap'].shape)
        #     B, T, _, H, W = res['action_heatmap'].shape
        #     action_heatmap = res['action_heatmap'].reshape(B*T, 2, H, W)
            
        #     # For pick action
        #     pick_idx = np.argmax(action_heatmap[:, 0].reshape(B*T, -1), axis=1)
        #     pick_idx = np.stack([pick_idx // W, pick_idx % W], axis=1).astype(float)
            
        #     # For place action
        #     place_idx = np.argmax(action_heatmap[:, 1].reshape(B*T, -1), axis=1)
        #     place_idx = np.stack([place_idx // W, place_idx % W], axis=1).astype(float)
            
        #     # Combine pick and place actions
        #     action = np.concatenate([pick_idx, place_idx], axis=1)
        #     res['action'] = action.reshape(B, T, 4)

        #     res['action'] = res['action'] / np.array([H, W, H, W]) * 2 - 1

        #     res['action'] = res['action'].reshape(B, T, 4)

            #print('action', res['action'])

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