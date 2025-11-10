import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1), indexing='ij')
    h = torch.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def add_noise(rgb, noise_factor=0.0, ):
    
    device = rgb.device

    ## Add noise
    if noise_factor != 0:
        rgb = rgb + torch.randn(rgb.shape, device=device)*noise_factor
        rgb = rgb.clip(-0.5, 0.5)
    
    return rgb

class PixelBasedMultiPrimitiveDataAugmenter:
    ## Only for training
    def __init__(self,  config=None):
        self.config = config
        self.process_rgb = self.config.get('rgb_eval_process', False)
        self.process_depth = self.config.get('depth_eval_process', False)
        self.swap_action = self.config.get('swap_action', False)
        self.maskout = self.config.get('maskout', False)
        self.bg_value = self.config.get('bg_value', 0)
        self.apply_mask = lambda x, y: x * y + self.bg_value * (1 - y)
        self.random_rotation = self.config.get('random_rotation', False)
        self.vertical_flip = self.config.get('vertical_flip', False)
        self.depth_flip = self.config.get('depth_flip', False)
        self.apply_depth_noise_on_mask = self.config.get('apply_depth_noise_on_mask', False)
        self.depth_blur = self.config.get('depth_blur', False)
      

        if self.depth_blur:
            kernel_size = self.config.depth_blur_kernel_size
            sigma = 1.0
            self.kernel = gaussian_kernel(kernel_size, sigma).to(self.config.device)
            self.kernel = self.kernel.expand(1, 1, kernel_size, kernel_size)
            if kernel_size % 2 == 1:
                self.padding = (kernel_size - 1) // 2
            else:
                self.padding = kernel_size//2
    

    def __call__(self, sample):
        #Inputs are tensor sampled from replay buffer, observation assumes to rgb
        #Observation value between -0.5 and 0.5 when call

        observation = sample['observation'] # B*C*H*W
        next_observation = sample['next_observation']
        action = sample['action'] # B * (Max Act Dim)

        primitive = action[:, 0].long() # get the first value of each batch
        pixel_actions = action[:, 1:]

        # Preprocess observations
        observation = add_noise(
                observation, 
                noise_factor=self.config.rgb_noise_factor)

        next_observation = add_noise(
                next_observation, 
                noise_factor=self.config.rgb_noise_factor)

        
        # Random Rotation
        if self.random_rotation:

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
                B, A = pixel_actions.shape
                pixel_actions_ =  pixel_actions.reshape(-1, 1, 2)
                N = pixel_actions_.shape[0]
                rotation_matrices_tensor = rot_inv.expand(N, 2, 2).reshape(-1, 2, 2)
                rotation_action = pixel_actions_
               
                rotated_action = torch.bmm(rotation_action, rotation_matrices_tensor)\
                    .reshape(B, A)
                
                if torch.abs(rotated_action).max() > 1:
                    #print('max action', torch.abs(sample['action']).max())
                    continue
                #sample['action'] = rotated_action

                # Rotate observations
                B, C, H, W = observation.shape

                affine_matrix = torch.zeros(B, 2, 3, device=self.config.device)
                affine_matrix[:, :2, :2] = rot.expand(B, 2, 2)


                # Rotate all observations at once
                grid = F.affine_grid(affine_matrix[:, :2], (B, C, H, W), align_corners=True)
                observation = F.grid_sample(
                    observation, 
                    grid, align_corners=True).reshape(B, C, H, W)

                next_observation =  F.grid_sample(
                    next_observation, 
                    grid, align_corners=True).reshape(B, C, H, W)

                pixel_actions = rotated_action
                break
                # if the max absolute value of the action is more than 1, continue
                
        # Vertical Flip
        if self.vertical_flip and (random.random() < 0.5):
            
            # Generate random vertical flip decisions
            B, C, H, W = observation.shape
            flip_obs_images = torch.flip(observation, [2]).reshape(B, C, H, W)

            new_actions = pixel_actions.reshape(-1, 2)
            new_actions[:, 1] = -new_actions[:, 1]
            pixel_actions = new_actions.reshape(B, -1)
        

        sample['observation'] = observation
        sample['next_observation'] = next_observation
        prim_col = primitive.to(pixel_actions.dtype).unsqueeze(1)  # B x 1
        action_out = torch.cat([prim_col, pixel_actions], dim=1)

        sample['action'] = action_out
        return sample