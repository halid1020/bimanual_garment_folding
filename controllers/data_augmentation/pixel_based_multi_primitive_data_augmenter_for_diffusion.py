import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import kornia.augmentation as K

from agent_arena.torch_utils import np_to_ts, ts_to_np
from .utils import randomize_primitive_encoding


def rotate_points_torch(points, R):
    return points @ R.T


# --------------------
# Augmenter class (torch)
# --------------------
class PixelBasedMultiPrimitiveDataAugmenterForDiffusion:
    def __init__(self, config=None):
        config = {} if config is None else config
        self.config = config
        self.random_rotation = config.get("random_rotation", False)
        self.vertical_flip = config.get("vertical_flip", False)
        self.debug = config.get("debug", False)
        self.color_jitter = self.config.get("color_jitter", False)
        self.K = self.config.get("K", 0)
        self.random_channel_permutation = self.config.get("random_channel_permutation", False)
        self.randomise_prim_acts = self.config.get("randomise_prim_acts", False)
        self.use_workspace = self.config.get('use_workspace', False)
        self.use_goal = self.config.get('use_goal', False)
        self.random_crop = self.config.get('random_crop', False)
        
        # New config for RGB Noise
        self.rgb_noise_factor = self.config.get('rgb_noise_factor', 0.0)

        if self.use_goal:
            self.goal_rotation = self.config.get('goal_rotation', False)
            self.goal_translation = self.config.get('goal_translation', False)
            if self.goal_translation:
                self.goal_trans_range = self.config.get('goal_trans_range', [0, 0.2]) 
                # the translation vector has manginitude wihtin 0 and 0.2 on the pixel space ranges from -1 and 1.

        if self.random_crop:
            self.crop_scale = self.config.get('crop_scale', [0.8, 1.0])

        if self.color_jitter:
            self.color_aug = K.ColorJitter(
                brightness=self.config.get("brightness", 0.2),
                contrast=self.config.get("contrast", 0.2),
                saturation=self.config.get("saturation", 0.2),
                hue=self.config.get("hue", 0.05),
                p=1.0,
                keepdim=True,
                same_on_batch=True, 
            )

    def _flatten_bt(self, x):
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:]), B, T

    def _unflatten_bt(self, x, B, T):
        return x.reshape(B, T, *x.shape[1:])

    def _save_debug_image(self, rgb, pts, prefix, step):
        """rgb: (H,W,3) float (cpu numpy), pts: (N,2) in [-1,1]"""
        save_dir = "./tmp/augment_debug"
        os.makedirs(save_dir, exist_ok=True)

        H, W, _ = rgb.shape
        pts = pts.reshape(-1, 2)

        xs = (pts[:, 1] + 1) * W / 2
        ys = (pts[:, 0] + 1) * H / 2

        plt.figure(figsize=(4, 4))
        plt.imshow(rgb)
        plt.scatter(xs, ys, c='red', s=12)
        plt.axis('off')
        plt.savefig(f"{save_dir}/{prefix}_step{step}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    def __call__(self, sample, train=True, device='cpu'):
        """
        Input sample:
            rgb: (B, T, H, W, 3)  (numpy or torch)  uint8 or float
            action: (B, T, A) (numpy or torch)
        Returns sample with augmented rgb and action (torch tensors on device)
        """
        # Convert incoming arrays to torch tensors on device (keep dtype)
        for k, v in list(sample.items()):
            if isinstance(v, np.ndarray):
                t = np_to_ts(v.copy(), device)
                sample[k] = t
            else:
                # assume already a torch tensor
                sample[k] = v.to(device)

        if self.use_workspace and 'rgb-workspace-mask' in sample:
            sample['rgb'] = sample['rgb-workspace-mask'][:, :, :, :, :3]
            sample['robot0_mask'] = sample['rgb-workspace-mask'][:, :, :, :, 3:4]
            sample['robot1_mask'] = sample['rgb-workspace-mask'][:, :, :, :, 4:5]
        
        if self.use_workspace and 'rgb-workspace-mask-goal' in sample:
            sample['rgb'] = sample['rgb-workspace-mask-goal'][:, :, :, :, :3]
            sample['robot0_mask'] = sample['rgb-workspace-mask-goal'][:, :, :, :, 3:4]
            sample['robot1_mask'] = sample['rgb-workspace-mask-goal'][:, :, :, :, 4:5]
            sample['goal_rgb'] = sample['rgb-workspace-mask-goal'][:, :, :, :, 5:8]
        
        if self.use_goal and 'rgb-goal' in sample:
            sample['rgb'] = sample['rgb-goal'][:, :, :, :, :3]
            sample['goal_rgb'] = sample['rgb-goal'][:, :, :, :, 3:6]
        
        if "rgb" not in sample:
            raise KeyError("sample must contain 'rgb'")
        
        if train and "action" not in sample:
            raise KeyError("sample must contain 'action'")


        # Normalize rgb to [0,1] float32
        observation = sample["rgb"].float() / 255.0  # (B, T, H, W, 3)
        obs, BB, TT = self._flatten_bt(observation)  # (B*T, H, W, 3)
        if self.use_goal:
            goal_obs = sample['goal_rgb'].float() / 255.0
            goal_obs, BB, TT = self._flatten_bt(goal_obs)  # (B*T, H, W, 3)
        
        #print('input obs shape', observation.shape)
        
        if self.use_workspace:
            robot0_mask = sample['robot0_mask'].float()
            robot1_mask = sample['robot1_mask'].float()
            
            robot0_mask, _, _ = self._flatten_bt(robot0_mask)  # (B*T,H,W,1)
            robot1_mask, _, _ = self._flatten_bt(robot1_mask)


       

        if train:
            action = sample["action"]  # (B, T, A)
            #print('input act shape', action.shape)
            act, _, _ = self._flatten_bt(action)  # (B*T, A)
            #pixel_actions = act[:, 1:]  # keep continuous part

            if self.K != 0:
                pixel_actions = act[:, 1:]
            else:
                pixel_actions = act


        # Debug save before any augmentation
        if self.debug:
            n_show = min(4, obs.shape[0])
            for b in range(n_show):
                cpu_img = (obs[b].cpu().numpy()).astype(np.float32)
                if train:
                    pa_cpu = pixel_actions[b].cpu().numpy()
                else:
                    pa_cpu = np.zeros((1,2))
                self._save_debug_image(cpu_img, pa_cpu, prefix="diffusion_augment_before_action", step=b)

        
        # =========================
        #       RANDOM CROP
        # =========================
        if self.random_crop and train:
            #print('here!')
            # Current shape of obs is (N, H, W, C) where N = B*T
            _, H, W, _ = obs.shape

            # 1. Determine Crop Parameters (Global for batch to ensure consistency)
            scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
            new_h = int(H * scale)
            new_w = int(W * scale)
            
            # Ensure dimensions are valid
            new_h = max(1, min(new_h, H))
            new_w = max(1, min(new_w, W))
            
            # Sample Top-Left corner
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)

            # 2. Crop the Visual Inputs (Slicing)
            obs = obs[:, top:top+new_h, left:left+new_w, :]

            y_cord, x_cord = 0, 1
            if self.use_goal:
                goal_obs = goal_obs[:, top:top+new_h, left:left+new_w, :]
            
            if self.use_workspace:
                robot0_mask = robot0_mask[:, top:top+new_h, left:left+new_w, :]
                robot1_mask = robot1_mask[:, top:top+new_h, left:left+new_w, :]

            # 3. Adjust Actions (Pixel Space Method)
            # Based on _save_debug_image: index 0 is Y (Height), index 1 is X (Width)
            
            # A. Denormalize: [-1, 1] -> [0, H] or [0, W]
            # pixel = (norm + 1) * (size / 2)
            B, A = pixel_actions.shape
            pixel_actions = pixel_actions.reshape(-1, 2)
            act_y_pixel = (pixel_actions[:, y_cord] + 1.0) * (H / 2.0)
            act_x_pixel = (pixel_actions[:, x_cord] + 1.0) * (W / 2.0)

            # B. Shift: Apply the crop offset
            # The new (0,0) is at (top, left) of the old image
            act_y_pixel_new = act_y_pixel - top
            act_x_pixel_new = act_x_pixel - left

            # C. Renormalize: [0, new_h] -> [-1, 1]
            # norm = pixel / (size / 2) - 1
            pixel_actions[:, y_cord] = act_y_pixel_new / (new_h / 2.0) - 1.0
            pixel_actions[:, x_cord] = act_x_pixel_new / (new_w / 2.0) - 1.0

            # 4. Clip actions to stay within the new valid range
            pixel_actions = torch.clamp(pixel_actions, -1 + 1e-6, 1.0-1e-6).reshape(B, A)
            

        # =========================
        #       Resize
        # =========================
        obs = obs.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        obs = F.interpolate(obs,
            size=tuple(self.config.img_dim), mode='bilinear', align_corners=False)
        
        if self.use_goal:
            goal_obs = goal_obs.permute(0, 3, 1, 2).contiguous()
            goal_obs = F.interpolate(goal_obs,
                size=tuple(self.config.img_dim), mode='bilinear', align_corners=False)
        
        if self.use_workspace:
            #print('[PixelBasedMultiPrimitiveDataAugmenterForDiffusion] robot0_mask shape', robot0_mask.shape)
            robot0_mask = robot0_mask.permute(0, 3, 1, 2).contiguous() 
            robot0_mask = F.interpolate(
                robot0_mask,
                size=tuple(self.config.img_dim),
                mode='nearest'
            )
            robot1_mask = robot1_mask.permute(0, 3, 1, 2).contiguous() 
            robot1_mask = F.interpolate(
                robot1_mask,
                size=tuple(self.config.img_dim),
                mode='nearest'
            )

        # =========================
        #       RANDOM ROTATION
        # =========================
        if self.random_rotation and train:
            
            degree = self.config.rotation_degree * torch.randint(
                int(360 / self.config.rotation_degree), size=(1,)
            )
            thetas = torch.deg2rad(degree)
            cos_theta = torch.cos(thetas)
            sin_theta = torch.sin(thetas)

            rot = torch.stack([
                torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(2, 2)
            ], dim=0).to(device)
            rot_inv = rot.transpose(-1, -2)

            # Rotate actions
            B, A = pixel_actions.shape
            pixel_actions_ = pixel_actions.reshape(-1, 1, 2)
            N = pixel_actions_.shape[0]
            rotation_matrices_tensor = rot_inv.expand(N, 2, 2).reshape(-1, 2, 2)
            rotated_action = torch.bmm(pixel_actions_, rotation_matrices_tensor).reshape(B, A)
            rotated_action = rotated_action.clip(-1+1e-6, 1-1e-6)
            
            B, C, H, W = obs.shape
            affine_matrix = torch.zeros(B, 2, 3, device=device)
            affine_matrix[:, :2, :2] = rot.expand(B, 2, 2)
            grid = F.affine_grid(affine_matrix[:, :2], (B, C, H, W), align_corners=True)
            obs = F.grid_sample(obs, grid, align_corners=True)
            
            pixel_actions = rotated_action.reshape(BB*(TT-1), -1)

            if self.use_workspace:
                robot0_mask = F.grid_sample(
                    robot0_mask, grid, mode='nearest', align_corners=True
                )
                robot1_mask = F.grid_sample(
                    robot1_mask, grid, mode='nearest', align_corners=True
                )
        
       
        # Vertical Flip
        if self.vertical_flip and (random.random() < 0.5) and train:
            #print('Flip!')
            B, C, H, W = obs.shape
            obs = torch.flip(obs, [2])
            
            if self.use_workspace:
                robot0_mask = torch.flip(robot0_mask, [2])
                robot1_mask = torch.flip(robot1_mask, [2])

            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(BB*(TT-1), -1)

        
        # =========================
        #   DEBUG: Goal Before Aug
        # =========================
        if self.debug and self.use_goal:
            n_show = min(4, goal_obs.shape[0])
            for b in range(n_show):
                # goal_obs is (N, C, H, W), convert to (H, W, C) for plotting
                cpu_img = goal_obs[b].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                # Pass dummy points (0,0) as goals usually don't have action points to visualize
                dummy_pts = np.zeros((1, 2)) 
                self._save_debug_image(cpu_img, dummy_pts, prefix="goal_augment_before", step=b)

        # =========================
        #   GOAL AUGMENTATION
        # =========================
        if self.use_goal and train and (self.goal_rotation or self.goal_translation):
            
            # 1. Initialize Affine Matrix (Identity)
            # Shape (1, 2, 3) to broadcast the SAME transformation across the whole batch
            aff_params = torch.eye(2, 3, device=device).unsqueeze(0)

            # 2. Goal Rotation
            if self.goal_rotation:
                # Sample a NEW random index separate from the observation rotation
                # This ensures parameters are different from obs, but shared for all goals
                k_rot = torch.randint(
                    int(360 / self.config.rotation_degree), size=(1,), device=device
                )
                goal_degree = self.config.rotation_degree * k_rot
                theta_g = torch.deg2rad(goal_degree.float())
                
                c_g = torch.cos(theta_g)
                s_g = torch.sin(theta_g)
                
                # Update rotation (2x2) part of the affine matrix
                aff_params[:, 0, 0] = c_g
                aff_params[:, 0, 1] = -s_g
                aff_params[:, 1, 0] = s_g
                aff_params[:, 1, 1] = c_g

            # 3. Goal Translation
            if self.goal_translation:
                # Sample magnitude and angle for the translation vector
                # The magnitude is in normalized pixel space [-1, 1], making it relative to W and H
                r_min, r_max = self.goal_trans_range
                mag = (r_max - r_min) * torch.rand(1, device=device) + r_min
                angle = 2 * np.pi * torch.rand(1, device=device)
                
                tx = mag * torch.cos(angle)
                ty = mag * torch.sin(angle)
                
                # Update translation vector (last column)
                aff_params[:, 0, 2] = tx
                aff_params[:, 1, 2] = ty

            # 4. Apply Affine Grid Sample
            if self.goal_rotation or self.goal_translation:
                # Expand params to match the flattened batch size of goals (N_goal, 2, 3)
                N_goal = goal_obs.shape[0]
                current_aff = aff_params.expand(N_goal, -1, -1)
                
                # Generate grid and sample
                # affine_grid uses normalized coordinates [-1, 1], so the translation 
                # calculated above is automatically relative to W and H.
                grid_g = F.affine_grid(current_aff, goal_obs.shape, align_corners=True)
                goal_obs = F.grid_sample(goal_obs, grid_g, align_corners=True)
        
        # =========================
        #   DEBUG: Goal After Aug
        # =========================
        if self.debug and self.use_goal:
            n_show = min(4, goal_obs.shape[0])
            for b in range(n_show):
                cpu_img = goal_obs[b].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                dummy_pts = np.zeros((1, 2))
                self._save_debug_image(cpu_img, dummy_pts, prefix="goal_augment_after", step=b)

        # =========================
        #     COLOR JITTER (GLOBAL)
        # =========================
        
        if self.color_jitter and train:
            obs = self.color_aug(obs)
        
        # =========================
        #   RANDOM CHANNEL PERMUTATION
        # =========================
        if self.random_channel_permutation:
            # Generate ONE permutation for the whole batch
            perm = torch.randperm(3, device=obs.device)
            obs = obs[:, perm, :, :]

        # =========================
        #       RGB NOISE
        # =========================
        if self.rgb_noise_factor > 0 and train:
            # Generate Gaussian noise N(0, noise_factor)
            # torch.randn_like creates N(0, 1), multiplying by factor scales the std dev
            noise = torch.randn_like(obs) * self.rgb_noise_factor
            obs = obs + noise
            # Clamp result to ensure it stays valid image range [0, 1]
            obs = torch.clamp(obs, 0, 1)

            if self.use_goal:
                noise = torch.randn_like(obs) * self.rgb_noise_factor
                goal_obs = goal_obs + noise
                goal_obs = torch.clamp(goal_obs, 0, 1)

        # =========================
        # debug save after
        # =========================
        if self.debug:
            n_show = min(4, obs.shape[0])
            for b in range(n_show):
                cpu_img = obs[b].permute(1, 2, 0).cpu().numpy()  # H,W,3
                if train:
                    pa_cpu = pixel_actions[b].cpu().numpy()
                else:
                    pa_cpu = np.zeros((1,2))
                self._save_debug_image(cpu_img, pa_cpu, prefix="diffusion_augment_after_action", step=b)
            #exit(1)
        # =========================
        #      RESHAPE BACK
        # =========================
        # # obs -> (N,H,W,3)
        
        obs = self._unflatten_bt(obs, BB, TT)  # (B, T, 3, H, W)
        sample["rgb"] = obs

        if self.use_workspace:
            robot0_mask = self._unflatten_bt(robot0_mask, BB, TT)
            robot1_mask = self._unflatten_bt(robot1_mask, BB, TT)

            sample['robot0_mask'] = robot0_mask
            sample['robot1_mask'] = robot1_mask

            if 'rgb-workspace-mask' in sample:
                sample['rgb-workspace-mask'] = torch.cat([obs, robot0_mask, robot1_mask], dim=2)
            
            
        
        if self.use_goal:
            goal_obs = self._unflatten_bt(goal_obs, BB, TT)
            sample['goal_rgb'] = goal_obs
            
            if 'rgb-goal' in sample:
                sample['rgb-goal'] = torch.cat([obs, goal_obs], dim=2)
            
            if 'rgb-workspace-mask-goal' in sample:
                
                sample['rgb-workspace-mask-goal'] = torch.cat([obs, robot0_mask, robot1_mask, goal_obs], dim=2)


        if train:
            #print('pixel_actions', pixel_actions.shape)
            pixel_actions = self._unflatten_bt(pixel_actions, BB, TT-1)

            if self.K != 0:
                prim_acts = action[..., :1]
                if self.randomise_prim_acts:            
                    prim_acts = randomize_primitive_encoding(prim_acts.reshape(-1, 1), self.config.K).reshape(BB, TT-1, 1)


                full_action = torch.cat([prim_acts, pixel_actions], dim=-1)
            else:
                full_action =  pixel_actions

            sample["action"] = full_action.to(device)

        return sample
    
    def postprocess(self, sample):
        res = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                res[k] = ts_to_np(v)
            else:
                res[k] = v
        if 'rgb' in res:
            res['rgb'] = (res['rgb'].clip(0, 1) * 255).astype(np.int8)
        
        if 'action' in res:
            res['action'] = res['action'].clip(-1, 1)
        return res