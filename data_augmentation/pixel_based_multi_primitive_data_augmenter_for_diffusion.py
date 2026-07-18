import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import kornia.augmentation as K

from actoris_harena.torch_utils import np_to_ts, ts_to_np
from actoris_harena.utilities.save_utils import save_mask
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

        self.random_scale = config.get("random_scale", False)
        self.scale_range = config.get("scale_range", [0.7, 1.0])

        if self.use_goal:
            self.goal_aug = self.config.get('goal_aug', 'none')
            self.goal_rotation = self.config.get('goal_rotation', False)
            self.goal_translation = self.config.get('goal_translation', False)

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
        
        self.random_swap_actions = self.config.get("random_swap_actions", False)
        self.swap_action_prob = self.config.get("swap_action_prob", 0.5)
        raw_mapping = self.config.get("swap_action_mapping", {})
        self.swap_action_mapping = {int(k): v for k, v in raw_mapping.items()}

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

        self.not_rotate_primitives = self.config.get("not_rotate_primitives", [])
        self.augment_sem_key = self.config.get("augment_sem_key", False)

    def _flatten_bt(self, x):
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:]), B, T

    def _unflatten_bt(self, x, B, T):
        return x.reshape(B, T, *x.shape[1:])

    def _process_depth(self, depth, mask=None, train=False):
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

    def _save_debug_image(self, img, pts, orients=None, sem_keys=None, prefix="", step=0):
        save_dir = "./tmp/augment_debug"
        os.makedirs(save_dir, exist_ok=True)
        if 'mask' in prefix:
            save_mask(img, f"{prefix}_step{step}", save_dir)
            return

        H, W, C = img.shape
        pts = pts.reshape(-1, pts.shape[-1])
        
        plt.figure(figsize=(4, 4))
        
        if C == 1:
            plt.imshow(img.squeeze(-1), cmap='viridis')
        else:
            plt.imshow(img)
            
        def to_pix(y_norm, x_norm):
            x = (x_norm + 1) * W / 2
            y = (y_norm + 1) * H / 2
            return x, y

        if sem_keys is not None:
            sem_keys = sem_keys.reshape(-1, 2)
            sk_xs, sk_ys = to_pix(sem_keys[:, 0], sem_keys[:, 1])
            plt.scatter(sk_xs, sk_ys, c='lime', s=15, edgecolors='black', marker='x', label='Sem Key')

        if pts.shape[1] == 4:
            pick_y, pick_x = pts[:, 0], pts[:, 1]
            place_y, place_x = pts[:, 2], pts[:, 3]
            
            px_start, py_start = to_pix(pick_y, pick_x)
            px_end, py_end = to_pix(place_y, place_x)
            
            plt.scatter(px_start, py_start, c='orange', s=25, edgecolors='black', label='Pick')
            plt.scatter(px_end, py_end, c='red', s=25, edgecolors='black', label='Place')
            
            if orients is not None:
                thetas = orients.flatten() * np.pi
                arrow_len = min(H, W) * 0.1
                for i in range(len(px_end)):
                    dx = arrow_len * np.cos(thetas[i])
                    dy = arrow_len * np.sin(thetas[i])
                    plt.plot([px_end[i], px_end[i] + dx], [py_end[i], py_end[i] + dy], color='cyan', linewidth=2)
        else:
            pts = pts.reshape(-1, 2)
            xs, ys = to_pix(pts[:, 0], pts[:, 1])
            plt.scatter(xs, ys, c='orange', s=12)

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
        
        if self.use_goal and 'rgb+goal_rgb' in sample:
            sample['rgb'] = sample['rgb+goal_rgb'][:, :, :, :, :3]
            sample['goal_rgb'] = sample['rgb+goal_rgb'][:, :, :, :, 3:6]
        
        if self.use_goal and 'rgb+goal_mask' in sample:
            sample['rgb'] = sample['rgb+goal_mask'][:, :, :, :, :3]
            sample['goal_mask'] = sample['rgb+goal_mask'][:, :, :, :, 3:4]
        
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
            depth_t = depth_t.permute(0, 1, 4, 2, 3) 
            processed_depth = self._process_depth(depth_t, mask=None, train=do_process_depth)
            processed_depth = processed_depth.permute(0, 1, 3, 4, 2)
            depth_obs, BB, TT = self._flatten_bt(processed_depth) 
        
        use_goal_depth = 'goal-depth' in sample
        if use_goal_depth and self.use_goal:
            goal_depth_t = sample['goal-depth'].float()
            if goal_depth_t.ndim == 4:
                goal_depth_t = goal_depth_t.unsqueeze(-1)
            goal_depth_t = goal_depth_t.permute(0, 1, 4, 2, 3) 
            processed_goal_depth = self._process_depth(goal_depth_t, mask=None, train=do_process_depth)
            processed_goal_depth = processed_goal_depth.permute(0, 1, 3, 4, 2)
            goal_depth_obs, _, _ = self._flatten_bt(processed_goal_depth)

        # 3. Flatten and basic prep for RGB
        use_rgb = 'rgb' in sample
        if use_rgb:
            rgb_obs = sample["rgb"].float() / 255.0 
            rgb_obs, BB, TT = self._flatten_bt(rgb_obs) 
            B, H, W, _ = rgb_obs.shape
        
        use_mask = 'mask' in sample
        if use_mask:
            mask_obs = sample["mask"].float()
            mask_obs, BB, TT = self._flatten_bt(mask_obs) 
            
        use_goal_rgb = self.use_goal and 'goal_rgb' in sample
        if use_goal_rgb:
            goal_obs = sample['goal_rgb'].float() / 255.0
            goal_obs, _, _ = self._flatten_bt(goal_obs)
        
        use_goal_mask = self.use_goal and 'goal_mask' in sample
        if use_goal_mask:
            goal_mask_obs = sample['goal_mask'].float()
            goal_mask_obs, _, _ = self._flatten_bt(goal_mask_obs)
        
        # --- Extract Semantic Keypoints (Current) ---
        use_semkey = self.augment_sem_key and 'semkey_norm_pixel' in sample
        if use_semkey:
            semkey_obs = sample['semkey_norm_pixel'].float()
            semkey_obs, BB, TT = self._flatten_bt(semkey_obs)

        # --- Extract Semantic Keypoints (Goal) ---
        # Allow fallback depending on what key is saved in the dataset config
        goal_semkey_key = None
        if 'flattened_goal_semkey_norm_pixel' in sample:
            goal_semkey_key = 'flattened_goal_semkey_norm_pixel'
        elif 'flattened_semkey_norm_pixel' in sample:
            goal_semkey_key = 'flattened_semkey_norm_pixel'
            
        use_goal_semkey = self.augment_sem_key and goal_semkey_key is not None
        if use_goal_semkey:
            goal_semkey_obs = sample[goal_semkey_key].float()
            goal_semkey_obs, _, _ = self._flatten_bt(goal_semkey_obs)

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
            
            self.has_orientation = (self.K == 0 and act.shape[-1] == 5)
            
            if self.has_orientation:
                pixel_actions = act[:, :4] 
                orient_actions = act[:, 4:]
            else:
                pixel_actions = act[:, 1:] if self.K != 0 else act
                orient_actions = None
            
        # --- DEBUG: Plot Before (Geometric) Augmentation ---
        if self.debug:
            n_show = min(4, BB)
            for b in range(n_show):
                img_idx = b * TT
                act_idx = b * (TT - 1) if (TT - 1) > 0 else 0
                
                pa_cpu = pixel_actions[act_idx].cpu().numpy() if train else np.zeros((1,4))
                oa_cpu = orient_actions[act_idx].cpu().numpy() if (train and self.has_orientation) else None
                sk_cpu = semkey_obs[img_idx].cpu().numpy() if use_semkey else None
                gsk_cpu = goal_semkey_obs[img_idx].cpu().numpy() if use_goal_semkey else None
                
                if use_rgb:
                    cpu_img = (rgb_obs[img_idx].cpu().numpy()).astype(np.float32)
                    self._save_debug_image(cpu_img, pa_cpu, orients=oa_cpu, sem_keys=sk_cpu, prefix="aug_before_rgb", step=b)

                if self.use_goal and 'goal_rgb' in sample:
                    cpu_goal_img = (goal_obs[img_idx].cpu().numpy()).astype(np.float32)
                    # Pass the GOAL semantic keypoints to the goal image
                    self._save_debug_image(cpu_goal_img, pa_cpu, orients=oa_cpu, sem_keys=gsk_cpu, prefix="aug_before_goal", step=b)

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

            if use_rgb: rgb_obs = rgb_obs[:, top:top+new_h, left:left+new_w, :]
            if use_goal_rgb: goal_obs = goal_obs[:, top:top+new_h, left:left+new_w, :]
            
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

            # --- Adjust Current Semantic Keypoints ---
            if use_semkey:
                B_sk, N_sk, D_sk = semkey_obs.shape
                semkey_obs = semkey_obs.reshape(-1, 2)
                sk_y_pixel = (semkey_obs[:, 0] + 1.0) * (H / 2.0)
                sk_x_pixel = (semkey_obs[:, 1] + 1.0) * (W / 2.0)
                sk_y_pixel_new = sk_y_pixel - top
                sk_x_pixel_new = sk_x_pixel - left
                semkey_obs[:, 0] = sk_y_pixel_new / (new_h / 2.0) - 1.0
                semkey_obs[:, 1] = sk_x_pixel_new / (new_w / 2.0) - 1.0
                semkey_obs = torch.clamp(semkey_obs, -1 + 1e-6, 1.0-1e-6).reshape(B_sk, N_sk, D_sk)
                
            # --- Adjust Goal Semantic Keypoints ---
            if use_goal_semkey:
                B_gsk, N_gsk, D_gsk = goal_semkey_obs.shape
                goal_semkey_obs = goal_semkey_obs.reshape(-1, 2)
                gsk_y_pixel = (goal_semkey_obs[:, 0] + 1.0) * (H / 2.0)
                gsk_x_pixel = (goal_semkey_obs[:, 1] + 1.0) * (W / 2.0)
                gsk_y_pixel_new = gsk_y_pixel - top
                gsk_x_pixel_new = gsk_x_pixel - left
                goal_semkey_obs[:, 0] = gsk_y_pixel_new / (new_h / 2.0) - 1.0
                goal_semkey_obs[:, 1] = gsk_x_pixel_new / (new_w / 2.0) - 1.0
                goal_semkey_obs = torch.clamp(goal_semkey_obs, -1 + 1e-6, 1.0-1e-6).reshape(B_gsk, N_gsk, D_gsk)

        # =========================
        #       RESIZE & PERMUTE
        # =========================
        def resize_tensor(t, mode='bilinear'):
            t = t.permute(0, 3, 1, 2).contiguous()
            align = False if mode != 'nearest' else None
            return F.interpolate(t, size=tuple(self.config.img_dim), mode=mode, align_corners=align)
        
        if use_rgb: rgb_obs = resize_tensor(rgb_obs)
        if use_goal_rgb: goal_obs = resize_tensor(goal_obs)
        
        # =========================
        #   DETERMINE ROTATION MASK
        # =========================
        if train:
            if self.K != 0 and len(self.not_rotate_primitives) > 0:
                prim_bin = sample["action"][:, 0, 0] 
                prim_ids = torch.clamp((((prim_bin + 1) / 2) * self.K).long(), 0, self.K - 1)
                rotate_mask_batch = torch.ones(BB, dtype=torch.bool, device=device)
                for p in self.not_rotate_primitives:
                    rotate_mask_batch &= (prim_ids != p)
            else:
                rotate_mask_batch = torch.ones(BB, dtype=torch.bool, device=device)

        # =========================
        #   SPATIAL TRANSFORMS
        # =========================
        do_rotation = self.random_rotation and train
        do_scale = self.random_scale and train

        if do_rotation or do_scale:
            if do_scale:
                scale_val = random.uniform(self.scale_range[0], self.scale_range[1])
                S = torch.tensor(scale_val, device=device)
            else:
                S = torch.tensor(1.0, device=device)
            inv_S = 1.0 / S

            if do_rotation:
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
            else:
                thetas = torch.zeros(1, device=device)
                rot = torch.eye(2, device=device).unsqueeze(0)
                rot_inv = torch.eye(2, device=device).unsqueeze(0)

            # Spatial Action Transform
            B_act, A_act = pixel_actions.shape
            num_pts = A_act // 2  
            T_act = B_act // BB
            rotate_mask_act = rotate_mask_batch.unsqueeze(1).expand(BB, T_act).reshape(-1)
            rotate_mask_pts = rotate_mask_act.unsqueeze(1).expand(-1, num_pts).reshape(-1)
            
            pixel_actions_ = pixel_actions.reshape(-1, 1, 2).float()
            total_pts = pixel_actions_.shape[0]  
            rot_action_matrices = torch.eye(2, device=device).unsqueeze(0).expand(total_pts, 2, 2).clone() * S
            
            if do_rotation:
                num_masked_pts = rotate_mask_pts.sum().item()
                if num_masked_pts > 0:
                    rot_action_matrices[rotate_mask_pts] = (rot_inv * S).expand(num_masked_pts, 2, 2).float()
            
            rotated_action = torch.bmm(pixel_actions_, rot_action_matrices).reshape(B_act, A_act)
            pixel_actions = rotated_action.clip(-1+1e-6, 1-1e-6)
            
            if self.has_orientation and do_rotation:
                angle_rad = orient_actions * np.pi
                rotation_rad = thetas.to(device)
                actual_rotation = torch.zeros_like(orient_actions)
                actual_rotation[rotate_mask_act] = rotation_rad 
                angle_rad = angle_rad - actual_rotation
                angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
                orient_actions = angle_rad / np.pi

            B_img = BB * TT
            rotate_mask_img = rotate_mask_batch.unsqueeze(1).expand(BB, TT).reshape(BB * TT)
            C_img, H_img, W_img = rgb_obs.shape[1:]

            affine_matrix = torch.zeros(B_img, 2, 3, device=device)
            affine_matrix[:, 0, 0] = inv_S
            affine_matrix[:, 1, 1] = inv_S
            
            if do_rotation:
                affine_matrix[rotate_mask_img, :2, :2] = (rot * inv_S).expand(rotate_mask_img.sum(), 2, 2)
            
            grid = F.affine_grid(affine_matrix, (B_img, C_img, H_img, W_img), align_corners=True)
            if use_rgb: rgb_obs = F.grid_sample(rgb_obs, grid, align_corners=True)

            if self.use_goal and self.goal_aug == "same_as_obs":
                if use_goal_rgb: goal_obs = F.grid_sample(goal_obs, grid, align_corners=True)
            
            # --- Transform Current Semantic Keypoints ---
            if use_semkey:
                B_sk, N_sk, D_sk = semkey_obs.shape
                rotate_mask_sk = rotate_mask_img.unsqueeze(1).expand(-1, N_sk).reshape(-1)
                semkey_pts_ = semkey_obs.reshape(-1, 1, 2)
                total_sk_pts = semkey_pts_.shape[0]
                rot_sk_matrices = torch.eye(2, device=device).unsqueeze(0).expand(total_sk_pts, 2, 2).clone() * S
                
                if do_rotation:
                    num_masked_sk = rotate_mask_sk.sum().item()
                    if num_masked_sk > 0:
                        rot_sk_matrices[rotate_mask_sk] = (rot_inv * S).expand(num_masked_sk, 2, 2).float()
                
                rotated_semkey = torch.bmm(semkey_pts_, rot_sk_matrices).reshape(B_sk, N_sk, D_sk)
                semkey_obs = rotated_semkey.clip(-1+1e-6, 1-1e-6)
                
            # --- Transform Goal Semantic Keypoints ---
            if use_goal_semkey and self.goal_aug == "same_as_obs":
                B_gsk, N_gsk, D_gsk = goal_semkey_obs.shape
                rotate_mask_gsk = rotate_mask_img.unsqueeze(1).expand(-1, N_gsk).reshape(-1)
                gsemkey_pts_ = goal_semkey_obs.reshape(-1, 1, 2)
                total_gsk_pts = gsemkey_pts_.shape[0]
                rot_gsk_matrices = torch.eye(2, device=device).unsqueeze(0).expand(total_gsk_pts, 2, 2).clone() * S
                
                if do_rotation:
                    num_masked_gsk = rotate_mask_gsk.sum().item()
                    if num_masked_gsk > 0:
                        rot_gsk_matrices[rotate_mask_gsk] = (rot_inv * S).expand(num_masked_gsk, 2, 2).float()
                
                rotated_gsemkey = torch.bmm(gsemkey_pts_, rot_gsk_matrices).reshape(B_gsk, N_gsk, D_gsk)
                goal_semkey_obs = rotated_gsemkey.clip(-1+1e-6, 1-1e-6)
        
        # =========================
        #      VERTICAL FLIP
        # =========================
        if self.vertical_flip and (random.random() < 0.5) and train:
            if use_rgb: rgb_obs = torch.flip(rgb_obs, [2])
            if self.use_goal and self.goal_aug == "same_as_obs":
                if use_goal_rgb: goal_obs = torch.flip(goal_obs, [2])

            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(BB*(TT-1), -1)

            if use_semkey:
                B_sk, N_sk, D_sk = semkey_obs.shape
                semkey_obs = semkey_obs.reshape(-1, 2)
                semkey_obs[:, 0] = -semkey_obs[:, 0]
                semkey_obs = semkey_obs.reshape(B_sk, N_sk, D_sk)
                
            if use_goal_semkey and self.goal_aug == "same_as_obs":
                B_gsk, N_gsk, D_gsk = goal_semkey_obs.shape
                goal_semkey_obs = goal_semkey_obs.reshape(-1, 2)
                goal_semkey_obs[:, 0] = -goal_semkey_obs[:, 0]
                goal_semkey_obs = goal_semkey_obs.reshape(B_gsk, N_gsk, D_gsk)
            
            if self.has_orientation:
                orient_actions = -orient_actions

        # =========================
        #   RANDOM ACTION SWAP
        # =========================
        # Label-symmetry augmentation for bimanual primitives: swapping the two
        # grippers' parameter slots describes the SAME physical action, so with
        # probability `swap_action_prob` (per sample) exchange group-A / group-B
        # indices of `pixel_actions`. Placed AFTER all geometric action transforms
        # (crop, rotation/scale, flip) and before write-back, so it only permutes
        # already-transformed parameter slots. Images/masks/semantic keypoints are
        # untouched, and the primitive scalar (act[:, 0]) is never modified.
        if self.random_swap_actions and train and len(self.swap_action_mapping) > 0:
            prim_bin_swap = act[:, 0]
            prim_ids_swap = torch.clamp(
                (((prim_bin_swap + 1) / 2) * self.K).long(), 0, self.K - 1
            )
            for p, groups in self.swap_action_mapping.items():
                group_A, group_B = groups[0], groups[1]
                p_mask = (prim_ids_swap == p)
                if not torch.any(p_mask):
                    continue
                rand = torch.rand(pixel_actions.shape[0], device=pixel_actions.device)
                swap_mask = p_mask & (rand < self.swap_action_prob)
                if not torch.any(swap_mask):
                    continue
                rows = swap_mask.nonzero(as_tuple=True)[0]
                idx_A = torch.as_tensor(group_A, device=pixel_actions.device, dtype=torch.long)
                idx_B = torch.as_tensor(group_B, device=pixel_actions.device, dtype=torch.long)
                a_vals = pixel_actions[rows][:, idx_A].clone()
                b_vals = pixel_actions[rows][:, idx_B].clone()
                pixel_actions[rows.unsqueeze(1), idx_A.unsqueeze(0)] = b_vals
                pixel_actions[rows.unsqueeze(1), idx_B.unsqueeze(0)] = a_vals

        # =========================
        #     COLOR JITTER & NOISE
        # =========================
        if self.color_jitter and train:
            if use_goal_rgb:
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
            n_show = min(4, BB)
            for b in range(n_show):
                img_idx = b * TT
                act_idx = b * (TT - 1) if (TT - 1) > 0 else 0

                oa_cpu = orient_actions[act_idx].cpu().numpy() if (train and self.has_orientation) else None
                sk_cpu = semkey_obs[img_idx].cpu().numpy() if use_semkey else None
                gsk_cpu = goal_semkey_obs[img_idx].cpu().numpy() if use_goal_semkey else None

                if use_rgb:
                    cpu_img = rgb_obs[img_idx].permute(1, 2, 0).cpu().numpy() 
                    pa_cpu = pixel_actions[act_idx].cpu().numpy() if train else np.zeros((1,4))
                    self._save_debug_image(cpu_img, pa_cpu, orients=oa_cpu, sem_keys=sk_cpu, prefix="aug_after_rgb", step=b)

                if use_goal_rgb:
                    cpu_goal_img = goal_obs[img_idx].permute(1, 2, 0).cpu().numpy()
                    self._save_debug_image(cpu_goal_img, pa_cpu, orients=oa_cpu, sem_keys=gsk_cpu, prefix="aug_after_goal", step=b)
                     
        # =========================
        #      RESHAPE BACK
        # =========================
        if use_rgb:
            rgb_obs = self._unflatten_bt(rgb_obs, BB, TT)
            sample["rgb"] = rgb_obs

        if use_semkey:
            sample['semkey_norm_pixel'] = self._unflatten_bt(semkey_obs, BB, TT)
            
        if use_goal_semkey:
            sample[goal_semkey_key] = self._unflatten_bt(goal_semkey_obs, BB, TT)
        
        if use_goal_rgb:
            sample['goal_rgb'] = self._unflatten_bt(goal_obs, BB, TT)

        if 'rgb+goal_rgb' in sample:
            sample['rgb+goal_rgb'] = torch.cat([sample["rgb"], sample['goal_rgb']], dim=2)

        if train:
            pixel_actions = self._unflatten_bt(pixel_actions, BB, TT-1)
            
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