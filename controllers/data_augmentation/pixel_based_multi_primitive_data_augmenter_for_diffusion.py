import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import kornia.augmentation as K

from actoris_harena.torch_utils import np_to_ts, ts_to_np
from .utils import randomize_primitive_encoding, gaussian_kernel 


def rotate_points_torch(points, R):
    return points @ R.T

# --------------------
# Augmenter class (torch)
# --------------------
class PixelBasedMultiPrimitiveDataAugmenterForDiffusion:
    def __init__(self, config=None):
        config = {} if config is None else config
        self.config = config
        
        # --- Original Configs ---
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
        self.rgb_noise_factor = self.config.get('rgb_noise_factor', 0.0)
        self.depth_noise_var = self.config.get('depth_noise_var', 0.1)

        if self.use_goal:
            self.goal_rotation = self.config.get('goal_rotation', False)
            self.goal_translation = self.config.get('goal_translation', False)
            if self.goal_translation:
                self.goal_trans_range = self.config.get('goal_trans_range', [0, 0.2]) 

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
            
        # --- Depth Processing Configs ---
        self.process_depth = self.config.get('depth_eval_process', False) 
        self.process_depth_for_eval = self.config.get('process_depth_for_eval', False)
        self.apply_depth_noise_on_mask = self.config.get('apply_depth_noise_on_mask', False)
        self.depth_blur = self.config.get('depth_blur', False)
        self.depth_flip = self.config.get('depth_flip', False)
        
        if self.depth_blur:
            kernel_size = self.config.get('depth_blur_kernel_size', 3)
            sigma = 1.0
            self.kernel = gaussian_kernel(kernel_size, sigma) 
            self.kernel = self.kernel.expand(1, 1, kernel_size, kernel_size)
            if kernel_size % 2 == 1:
                self.padding = (kernel_size - 1) // 2
            else:
                self.padding = kernel_size // 2
        else:
            self.kernel = None

    def _flatten_bt(self, x):
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:]), B, T

    def _unflatten_bt(self, x, B, T):
        return x.reshape(B, T, *x.shape[1:])

    def _save_debug_image(self, img, pts, prefix, step):
        """
        img: (H,W,C) float (cpu numpy). Can be 1 channel (depth) or 3 channel (rgb)
        pts: (N,2) in [-1,1]
        """
        save_dir = "./tmp/augment_debug"
        os.makedirs(save_dir, exist_ok=True)

        H, W, C = img.shape
        pts = pts.reshape(-1, 2)

        xs = (pts[:, 1] + 1) * W / 2
        ys = (pts[:, 0] + 1) * H / 2

        plt.figure(figsize=(4, 4))
        
        if C == 1:
            # Depth image: squeeze channel and use colormap
            plt.imshow(img.squeeze(-1), cmap='viridis')
        else:
            # RGB image
            plt.imshow(img)
            
        plt.scatter(xs, ys, c='red', s=12)
        plt.axis('off')
        plt.savefig(f"{save_dir}/{prefix}_step{step}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    def __call__(self, sample, train=True, device='cpu'):
        # 1. Move to device
        for k, v in list(sample.items()):
            if isinstance(v, np.ndarray):
                t = np_to_ts(v.copy(), device)
                sample[k] = t
            else:
                sample[k] = v.to(device)

        # 2. Extract combined keys (Workspace/Goal/Masks)
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
        
        # if "rgb" not in sample:
        #     raise KeyError("sample must contain 'rgb'")
        
        if train and "action" not in sample:
            raise KeyError("sample must contain 'action'")
        
        # --- Prepare batch info ---
        # We need B and T early to unflatten depth for processing
  
        # =========================
        #    DEPTH PROCESSING (Intensity) - DONE FIRST
        # =========================
        do_process_depth = train or self.process_depth_for_eval
        
        
        use_depth = 'depth' in sample
        if use_depth:
            depth_t = sample['depth'].float()
            if depth_t.ndim == 4: # B,T,H,W -> B,T,H,W,1
                depth_t = depth_t.unsqueeze(-1)
            #print('depth_t shape', depth_t.shape)
            B, T, H, W, C  = depth_t.shape
            
            # Apply processing (Norm, Noise, Blur) BEFORE flattening/geometric augs
            # Expected input to _process_depth is (B, T, C, H, W)
            mask_reshaped = None 
            # Note: depth_t is already (B, T, H, W, 1) or (B, T, H, W). 
            # We need to ensure channel dim is correct for internal logic if it expects (B,T,C,H,W)
            # The current _process_depth implementation expects (B, T, C, H, W).
            # So we permute:
            depth_t = depth_t.permute(0, 1, 4, 2, 3) # (B, T, 1, H, W)
            
            processed_depth = self._process_depth(depth_t, mask=mask_reshaped, train=do_process_depth)
            
            # Permute back to (B, T, H, W, 1) to match the rest of the flattening logic below
            processed_depth = processed_depth.permute(0, 1, 3, 4, 2)
            
            # Flatten for geometric pipeline
            depth_obs, BB, TT = self._flatten_bt(processed_depth) # (B*T, H, W, 1)
        
        use_goal_depth = 'goal-depth' in sample
        if use_goal_depth and self.use_goal:
            goal_depth_t = sample['goal-depth'].float()
            if goal_depth_t.ndim == 4:
                goal_depth_t = goal_depth_t.unsqueeze(-1)
                
            # Process Goal Depth
            goal_depth_t = goal_depth_t.permute(0, 1, 4, 2, 3) # (B, T, 1, H, W)
            goal_mask_reshaped = None
            processed_goal_depth = self._process_depth(goal_depth_t, mask=goal_mask_reshaped, train=do_process_depth)
            
            processed_goal_depth = processed_goal_depth.permute(0, 1, 3, 4, 2)
            goal_depth_obs, _, _ = self._flatten_bt(processed_goal_depth)

        # 3. Flatten and basic prep for RGB
        use_rgb = 'rgb' in sample
        if use_rgb:
            rgb_obs = sample["rgb"].float() / 255.0 
            rgb_obs, BB, TT = self._flatten_bt(rgb_obs) 
            #B = rgb_obs.shape[0]
            B, H, W, _ = rgb_obs.shape
        
        if self.use_goal and 'goal_rgb' in sample:
            goal_obs = sample['goal_rgb'].float() / 255.0
            goal_obs, _, _ = self._flatten_bt(goal_obs)

        if self.use_workspace:
            robot0_mask = sample['robot0_mask'].float()
            robot1_mask = sample['robot1_mask'].float()
            robot0_mask, _, _ = self._flatten_bt(robot0_mask)
            robot1_mask, _, _ = self._flatten_bt(robot1_mask)

        if train:
            action = sample["action"]
            act, _, _ = self._flatten_bt(action)
            pixel_actions = act[:, 1:] if self.K != 0 else act

        # --- DEBUG: Plot Before (Geometric) Augmentation ---
        # Note: Depth is already normalized [0, 1] here due to processing above
        if self.debug:
            n_show = min(4, B)
            for b in range(n_show):
                pa_cpu = pixel_actions[b].cpu().numpy() if train else np.zeros((1,2))
                # RGB
                if use_rgb:
                   
                    cpu_img = (rgb_obs[b].cpu().numpy()).astype(np.float32)
                    
                    self._save_debug_image(cpu_img, pa_cpu, prefix="aug_before_rgb", step=b)
                    
                # Depth
                if use_depth:
                    d_img = depth_obs[b].cpu().numpy().astype(np.float32)
                    # Depth is already [0, 1], so we don't need complex clipping for visualization
                    # Just ensure it's valid for imshow
                    self._save_debug_image(d_img, pa_cpu, prefix="aug_before_depth", step=b)

        # =========================
        #       RANDOM CROP
        # =========================
        if self.random_crop and train:
            
            scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
            new_h = int(H * scale)
            new_w = int(W * scale)
            new_h = max(1, min(new_h, H))
            new_w = max(1, min(new_w, W))
            
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)

            if use_rgb:
                rgb_obs = rgb_obs[:, top:top+new_h, left:left+new_w, :]
                
            if self.use_goal and 'goal_rgb' in sample: 
                goal_obs = goal_obs[:, top:top+new_h, left:left+new_w, :]
            
            if self.use_workspace:
                robot0_mask = robot0_mask[:, top:top+new_h, left:left+new_w, :]
                robot1_mask = robot1_mask[:, top:top+new_h, left:left+new_w, :]
            
            if use_depth:
                depth_obs = depth_obs[:, top:top+new_h, left:left+new_w, :]
            if use_goal_depth and self.use_goal:
                goal_depth_obs = goal_depth_obs[:, top:top+new_h, left:left+new_w, :]

            # Adjust Actions
            B_act, A_act = pixel_actions.shape
            pixel_actions = pixel_actions.reshape(-1, 2)
            act_y_pixel = (pixel_actions[:, 0] + 1.0) * (H / 2.0)
            act_x_pixel = (pixel_actions[:, 1] + 1.0) * (W / 2.0)
            act_y_pixel_new = act_y_pixel - top
            act_x_pixel_new = act_x_pixel - left
            pixel_actions[:, 0] = act_y_pixel_new / (new_h / 2.0) - 1.0
            pixel_actions[:, 1] = act_x_pixel_new / (new_w / 2.0) - 1.0
            pixel_actions = torch.clamp(pixel_actions, -1 + 1e-6, 1.0-1e-6).reshape(B_act, A_act)

        # =========================
        #       RESIZE & PERMUTE
        # =========================
        def resize_tensor(t, mode='bilinear'):
            t = t.permute(0, 3, 1, 2).contiguous()
            # FIX: align_corners must be None if mode is 'nearest'
            align = False if mode != 'nearest' else None
            return F.interpolate(t, size=tuple(self.config.img_dim), mode=mode, align_corners=align)
        
        if use_rgb: rgb_obs = resize_tensor(rgb_obs)
        if self.use_goal and 'goal_rgb' in sample: goal_obs = resize_tensor(goal_obs)
        if self.use_workspace:
            robot0_mask = resize_tensor(robot0_mask, mode='nearest')
            robot1_mask = resize_tensor(robot1_mask, mode='nearest')
        if use_depth:
            depth_obs = resize_tensor(depth_obs, mode='nearest') 
        if use_goal_depth and self.use_goal:
            goal_depth_obs = resize_tensor(goal_depth_obs, mode='nearest')

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

            B_act, A_act = pixel_actions.shape
            pixel_actions_ = pixel_actions.reshape(-1, 1, 2)
            rotation_matrices_tensor = rot_inv.expand(pixel_actions_.shape[0], 2, 2).reshape(-1, 2, 2)
            rotated_action = torch.bmm(pixel_actions_, rotation_matrices_tensor).reshape(B_act, A_act)
            rotated_action = rotated_action.clip(-1+1e-6, 1-1e-6)
            
            if use_rgb:
                B_img, C, H, W = rgb_obs.shape
                affine_matrix = torch.zeros(B_img, 2, 3, device=device)
                affine_matrix[:, :2, :2] = rot.expand(B_img, 2, 2)
                grid = F.affine_grid(affine_matrix[:, :2], (B_img, C, H, W), align_corners=True)
                
                rgb_obs = F.grid_sample(rgb_obs, grid, align_corners=True)
            
            if self.use_workspace:
                robot0_mask = F.grid_sample(robot0_mask, grid, mode='nearest', align_corners=True)
                robot1_mask = F.grid_sample(robot1_mask, grid, mode='nearest', align_corners=True)
            
            if use_depth:
                depth_obs = F.grid_sample(depth_obs, grid, mode='nearest', align_corners=True)
            
            pixel_actions = rotated_action.reshape(BB*(TT-1), -1)

        # =========================
        #      VERTICAL FLIP
        # =========================
        if self.vertical_flip and (random.random() < 0.5) and train:
            if use_rgb: rgb_obs = torch.flip(rgb_obs, [2])
            if self.use_workspace:
                robot0_mask = torch.flip(robot0_mask, [2])
                robot1_mask = torch.flip(robot1_mask, [2])
            if use_depth:
                depth_obs = torch.flip(depth_obs, [2])

            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(BB*(TT-1), -1)

        # =========================
        #   GOAL TRANSFORMATIONS
        # =========================
        if self.use_goal and train and (self.goal_rotation or self.goal_translation):
            aff_params = torch.eye(2, 3, device=device).unsqueeze(0)
            if self.goal_rotation:
                k_rot = torch.randint(int(360 / self.config.rotation_degree), size=(1,), device=device)
                goal_degree = self.config.rotation_degree * k_rot
                theta_g = torch.deg2rad(goal_degree.float())
                c_g, s_g = torch.cos(theta_g), torch.sin(theta_g)
                aff_params[:, 0, 0] = c_g; aff_params[:, 0, 1] = -s_g
                aff_params[:, 1, 0] = s_g; aff_params[:, 1, 1] = c_g

            if self.goal_translation:
                r_min, r_max = self.goal_trans_range
                mag = (r_max - r_min) * torch.rand(1, device=device) + r_min
                angle = 2 * np.pi * torch.rand(1, device=device)
                aff_params[:, 0, 2] = mag * torch.cos(angle)
                aff_params[:, 1, 2] = mag * torch.sin(angle)

            if 'goal_rgb' in sample:
                current_aff = aff_params.expand(goal_obs.shape[0], -1, -1)
                grid_g = F.affine_grid(current_aff, goal_obs.shape, align_corners=True)
                goal_obs = F.grid_sample(goal_obs, grid_g, align_corners=True)

            if use_goal_depth:
                current_aff_d = aff_params.expand(goal_depth_obs.shape[0], -1, -1)
                grid_gd = F.affine_grid(current_aff_d, goal_depth_obs.shape, align_corners=True)
                goal_depth_obs = F.grid_sample(goal_depth_obs, grid_gd, mode='nearest', align_corners=True)

        # =========================
        #     COLOR JITTER & NOISE
        # =========================
        if self.color_jitter and train:
            if self.use_goal and 'goal_rgb' in sample:
                N_obs = rgb_obs.shape[0]
                combined = torch.cat([rgb_obs, goal_obs], dim=0)
                combined = self.color_aug(combined)
                rgb_obs = combined[:N_obs]
                goal_obs = combined[N_obs:]
            elif use_rgb:
                rgb_obs = self.color_aug(rgb_obs)

        if self.random_channel_permutation and train:
            perm = torch.randperm(3, device=rgb_obs.device)
            if use_rgb: rgb_obs = rgb_obs[:, perm, :, :]
            if self.use_goal and 'goal_rgb' in sample: goal_obs = goal_obs[:, perm, :, :]

        if use_rgb and self.rgb_noise_factor > 0 and train:
            noise = torch.randn_like(rgb_obs) * self.rgb_noise_factor
            rgb_obs = torch.clamp(rgb_obs + noise, 0, 1)
            if self.use_goal and 'goal_rgb' in sample:
                noise_g = torch.randn_like(goal_obs) * self.rgb_noise_factor
                goal_obs = torch.clamp(goal_obs + noise_g, 0, 1)

        # --- DEBUG: Plot After Augmentation ---
        if self.debug:
            n_show = min(4, B)
            for b in range(n_show):
                # RGB
                if use_rgb:
                    cpu_img = rgb_obs[b].permute(1, 2, 0).cpu().numpy() # H,W,3
                    pa_cpu = pixel_actions[b].cpu().numpy() if train else np.zeros((1,2))
                    self._save_debug_image(cpu_img, pa_cpu, prefix="aug_after_rgb", step=b)
                    
                # Depth (Note: depth_obs is B*T, C, H, W. Permute to H,W,C)
                if use_depth:
                    d_img = depth_obs[b].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    self._save_debug_image(d_img, pa_cpu, prefix="aug_after_depth", step=b)

        # =========================
        #      RESHAPE BACK
        # =========================
        if use_rgb:
            rgb_obs = self._unflatten_bt(rgb_obs, BB, TT)
            sample["rgb"] = rgb_obs

        if use_depth:
            sample['depth'] = self._unflatten_bt(depth_obs, BB, TT)
        
        if use_goal_depth and self.use_goal:
            sample['goal-depth'] = self._unflatten_bt(goal_depth_obs, BB, TT)

        if self.use_workspace:
            robot0_mask = self._unflatten_bt(robot0_mask, BB, TT)
            robot1_mask = self._unflatten_bt(robot1_mask, BB, TT)
            sample['robot0_mask'] = robot0_mask
            sample['robot1_mask'] = robot1_mask
            if 'rgb-workspace-mask' in sample:
                sample['rgb-workspace-mask'] = torch.cat([rgb_obs, robot0_mask, robot1_mask], dim=2)

        if self.use_goal:
            if 'goal_rgb' in sample:
                goal_obs = self._unflatten_bt(goal_obs, BB, TT)
                sample['goal_rgb'] = goal_obs
                if 'rgb-goal' in sample:
                    sample['rgb-goal'] = torch.cat([rgb_obs, goal_obs], dim=2)
                if 'rgb-workspace-mask-goal' in sample:
                    sample['rgb-workspace-mask-goal'] = torch.cat([rgb_obs, robot0_mask, robot1_mask, goal_obs], dim=2)

        if train:
            pixel_actions = self._unflatten_bt(pixel_actions, BB, TT-1)
            if self.K != 0:
                prim_acts = action[..., :1]
                if self.randomise_prim_acts:            
                    prim_acts = randomize_primitive_encoding(
                        prim_acts.reshape(-1, 1), self.config.K
                    ).reshape(BB, TT-1, 1)
                full_action = torch.cat([prim_acts, pixel_actions], dim=-1)
            else:
                full_action = pixel_actions
            sample["action"] = full_action.to(device)

        return sample

    def _process_depth(self, depth, mask=None, train=False):
        # Implementation same as provided previously...
        # depth shape expected here: (B, T, C, H, W)
        B, T, C, H, W = depth.shape
        if self.config.get('depth_clip', False):
            depth = depth.clip(self.config.depth_clip_min, self.config.depth_clip_max)
        
        if self.config.get('z_norm', False):
            depth = (depth - self.config.z_norm_mean) / self.config.z_norm_std
        elif self.config.get('min_max_norm', False):
            depth_flat = depth.reshape(B, T, -1)
            depth_min = depth_flat.min(dim=2, keepdim=True).values
            depth_max = depth_flat.max(dim=2, keepdim=True).values

            if self.config.get('depth_hard_interval', False):
                depth_min[:] = self.config.depth_min
                depth_max[:] = self.config.depth_max
            else:
                cfg_min = torch.tensor(self.config.depth_min, device=depth.device).view(1, 1, 1)
                cfg_max = torch.tensor(self.config.depth_max, device=depth.device).view(1, 1, 1)
                depth_min = torch.max(depth_min, cfg_min)
                depth_max = torch.min(depth_max, cfg_max)

            depth_min = depth_min.view(B, T, 1, 1, 1)
            depth_max = depth_max.view(B, T, 1, 1, 1)
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
            
            if self.depth_flip:
                depth = 1 - depth
        
        depth_noise = torch.randn(depth.shape, device=depth.device) * \
                      (self.depth_noise_var if train else 0)

        if self.depth_blur and train:
            if self.kernel.device != depth.device:
                self.kernel = self.kernel.to(depth.device)
            depth_reshaped = depth.view(B * T, C, H, W)
            blurred_depth = F.conv2d(depth_reshaped, self.kernel, padding=self.padding, groups=C)
            blurred_depth = blurred_depth[:, :, :H, :W]
            depth = blurred_depth.reshape(B, T, C, H, W)
            
            # Renormalize after blur
            depth_flat = depth.reshape(B, T, -1)
            depth_min = depth_flat.min(dim=2, keepdim=True).values.view(B, T, 1, 1, 1)
            depth_max = depth_flat.max(dim=2, keepdim=True).values.view(B, T, 1, 1, 1)
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)

        if self.apply_depth_noise_on_mask and (mask is not None) and train:
            depth_noise *= mask
            
        depth += depth_noise
        depth = depth.clip(0, 1)
        
        if self.config.get('depth_map', False):
            map_diff = self.config.depth_map_range[1] - self.config.depth_map_range[0]
            depth = depth * map_diff + self.config.depth_map_range[0]
        
        return depth
    
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