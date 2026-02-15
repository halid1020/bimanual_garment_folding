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

    # ... _process_depth and postprocess remain unchanged ...
    def _process_depth(self, depth, mask=None, train=False):
        # (Same as original code)
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

    
    def _save_debug_image(self, img, pts, orients=None, prefix="", step=0):
        """
        img: (H,W,C) float (cpu numpy). 
        pts: (N, 4) -> [pick_y, pick_x, place_y, place_x]  OR (N, 2)
        orients: (N, 1) -> normalized angle [-1, 1] (optional)
        """
        save_dir = "./tmp/augment_debug"
        os.makedirs(save_dir, exist_ok=True)

        H, W, C = img.shape
        # Ensure pts is 2D array
        pts = pts.reshape(-1, pts.shape[-1])
        
        plt.figure(figsize=(4, 4))
        
        if C == 1:
            plt.imshow(img.squeeze(-1), cmap='viridis')
        else:
            plt.imshow(img)
            
        # Helper to denormalize
        def to_pix(y_norm, x_norm):
            x = (x_norm + 1) * W / 2
            y = (y_norm + 1) * H / 2
            return x, y

        # Logic for Pick (Green) + Place (Red) + Orientation (Line)
        if pts.shape[1] == 4:
            # We have [y1, x1, y2, x2]
            pick_y, pick_x = pts[:, 0], pts[:, 1]
            place_y, place_x = pts[:, 2], pts[:, 3]
            
            px_start, py_start = to_pix(pick_y, pick_x)
            px_end, py_end = to_pix(place_y, place_x)
            
            # Plot Pick
            plt.scatter(px_start, py_start, c='lime', s=25, edgecolors='black', label='Pick')
            # Plot Place
            plt.scatter(px_end, py_end, c='red', s=25, edgecolors='black', label='Place')
            
            # Plot Orientation Arrow on Place Point
            if orients is not None:
                # orients is normalized [-1, 1] -> [-pi, pi]
                thetas = orients.flatten() * np.pi
                
                arrow_len = min(H, W) * 0.1 # Arrow length is 10% of image size
                
                for i in range(len(px_end)):
                    # Calculate arrow tip
                    # Note: In image coords, Y is down. Standard trig assumes Y up.
                    # Usually cos/sin works fine relative to the image grid unless coord system is flipped.
                    dx = arrow_len * np.cos(thetas[i])
                    dy = arrow_len * np.sin(thetas[i])
                    
                    plt.plot([px_end[i], px_end[i] + dx], 
                             [py_end[i], py_end[i] + dy], 
                             color='cyan', linewidth=2)
        else:
            # Fallback for simple (N, 2) points
            pts = pts.reshape(-1, 2)
            xs, ys = to_pix(pts[:, 0], pts[:, 1])
            print('len xs', len(xs))
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

        # 2. Extract combined keys
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
        
        if train and "action" not in sample:
            raise KeyError("sample must contain 'action'")
        
        # =========================
        #    DEPTH PROCESSING
        # =========================
        do_process_depth = train or self.process_depth_for_eval
        
        use_depth = 'depth' in sample
        if use_depth:
            depth_t = sample['depth'].float()
            if depth_t.ndim == 4: 
                depth_t = depth_t.unsqueeze(-1)
            B, T, H, W, C  = depth_t.shape
            
            # Permute for process_depth expected input
            depth_t = depth_t.permute(0, 1, 4, 2, 3) 
            mask_reshaped = None 
            processed_depth = self._process_depth(depth_t, mask=mask_reshaped, train=do_process_depth)
            
            processed_depth = processed_depth.permute(0, 1, 3, 4, 2)
            depth_obs, BB, TT = self._flatten_bt(processed_depth) 
        
        use_goal_depth = 'goal-depth' in sample
        if use_goal_depth and self.use_goal:
            goal_depth_t = sample['goal-depth'].float()
            if goal_depth_t.ndim == 4:
                goal_depth_t = goal_depth_t.unsqueeze(-1)
                
            goal_depth_t = goal_depth_t.permute(0, 1, 4, 2, 3) 
            goal_mask_reshaped = None
            processed_goal_depth = self._process_depth(goal_depth_t, mask=goal_mask_reshaped, train=do_process_depth)
            
            processed_goal_depth = processed_goal_depth.permute(0, 1, 3, 4, 2)
            goal_depth_obs, _, _ = self._flatten_bt(processed_goal_depth)

        # 3. Flatten and basic prep for RGB
        use_rgb = 'rgb' in sample
        if use_rgb:
            rgb_obs = sample["rgb"].float() / 255.0 
            rgb_obs, BB, TT = self._flatten_bt(rgb_obs) 
            B, H, W, _ = rgb_obs.shape
        
        if self.use_goal and 'goal_rgb' in sample:
            goal_obs = sample['goal_rgb'].float() / 255.0
            goal_obs, _, _ = self._flatten_bt(goal_obs)

        if self.use_workspace:
            robot0_mask = sample['robot0_mask'].float()
            robot1_mask = sample['robot1_mask'].float()
            robot0_mask, _, _ = self._flatten_bt(robot0_mask)
            robot1_mask, _, _ = self._flatten_bt(robot1_mask)

        # ------------------------------------------------------------------
        # NEW: Handle Action Extraction
        # ------------------------------------------------------------------
        if train:
            action = sample["action"]
            act, _, _ = self._flatten_bt(action)
            
            # Detect 5D action case: [PickY, PickX, PlaceY, PlaceX, Orientation]
            self.has_orientation = (self.K == 0 and act.shape[-1] == 5)
            
            if self.has_orientation:
                # Split spatial coordinates from orientation
                pixel_actions = act[:, :4]  # (B, 4) -> [y1, x1, y2, x2]
                orient_actions = act[:, 4:] # (B, 1) -> [theta] in [-1, 1]
            else:
                pixel_actions = act[:, 1:] if self.K != 0 else act
                orient_actions = None

        # --- DEBUG: Plot Before (Geometric) Augmentation ---
        if self.debug:
            n_show = min(4, B)
            for b in range(n_show):
                pa_cpu = pixel_actions[b].cpu().numpy() if train else np.zeros((1,4))
                print('[augmenter, debug] action', pa_cpu)
                
                # Extract orientation for debug if it exists
                oa_cpu = orient_actions[b].cpu().numpy() if (train and self.has_orientation) else None
                
                if use_rgb:
                    cpu_img = (rgb_obs[b].cpu().numpy()).astype(np.float32)
                    self._save_debug_image(cpu_img, pa_cpu, orients=oa_cpu, prefix="aug_before_rgb", step=b)

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

            # Adjust Actions (Spatial Only)
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
            thetas = torch.deg2rad(degree) # Scalar tensor
            cos_theta = torch.cos(thetas)
            sin_theta = torch.sin(thetas)

            rot = torch.stack([
                torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(2, 2)
            ], dim=0).to(device)
            rot_inv = rot.transpose(-1, -2)

            # Spatial Rotation
            B_act, A_act = pixel_actions.shape
            pixel_actions_ = pixel_actions.reshape(-1, 1, 2).float()
            rotation_matrices_tensor = rot_inv.expand(pixel_actions_.shape[0], 2, 2).reshape(-1, 2, 2).float()
            rotated_action = torch.bmm(pixel_actions_, rotation_matrices_tensor).reshape(B_act, A_act)
            rotated_action = rotated_action.clip(-1+1e-6, 1-1e-6)
            
            # --- Orientation Rotation ---
            if self.has_orientation:
                # Convert normalized [-1, 1] to radians [-pi, pi]
                angle_rad = orient_actions * np.pi
                
                # Update angle
                rotation_rad = thetas.to(device)
                angle_rad = angle_rad - rotation_rad
                
                # Wrap to [-pi, pi]
                angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
                
                # Convert back to normalized [-1, 1]
                orient_actions = angle_rad / np.pi
            # ----------------------------

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
            # Flip Images
            if use_rgb: rgb_obs = torch.flip(rgb_obs, [2])
            if self.use_workspace:
                robot0_mask = torch.flip(robot0_mask, [2])
                robot1_mask = torch.flip(robot1_mask, [2])
            if use_depth:
                depth_obs = torch.flip(depth_obs, [2])

            # Flip Spatial Actions (Invert Y)
            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(BB*(TT-1), -1)
            
            # --- Flip Orientation ---
            if self.has_orientation:
                orient_actions = -orient_actions
            # ------------------------

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
                # Extract orientation for debug if it exists
                oa_cpu = orient_actions[b].cpu().numpy() if (train and self.has_orientation) else None
                
                if use_rgb:
                    cpu_img = rgb_obs[b].permute(1, 2, 0).cpu().numpy() 
                    pa_cpu = pixel_actions[b].cpu().numpy() if train else np.zeros((1,4))
                    self._save_debug_image(cpu_img, pa_cpu, orients=oa_cpu, prefix="aug_after_rgb", step=b)

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
            
            # --- Recombine Spatial + Orientation ---
            if self.has_orientation:
                orient_actions = self._unflatten_bt(orient_actions, BB, TT-1)
                full_action = torch.cat([pixel_actions, orient_actions], dim=-1)
            elif self.K != 0:
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