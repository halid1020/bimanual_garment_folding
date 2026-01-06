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
    # points: (..., 2), R: (2,2)
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
        
        if "rgb" not in sample:
            raise KeyError("sample must contain 'rgb'")
        
        if train and "action" not in sample:
            raise KeyError("sample must contain 'action'")


        # Normalize rgb to [0,1] float32
        observation = sample["rgb"].float() / 255.0  # (B, T, H, W, 3)
        #print('input obs shape', observation.shape)
        obs, BB, TT = self._flatten_bt(observation)  # (B*T, H, W, 3)
        if self.use_workspace:
            robot0_mask = sample['robot0_mask'].float()
            robot1_mask = sample['robot1_mask'].float()
            #TODO: apply the same rotation and flipping as the "observation"

            # # Ensure shape (B,T,H,W)
            # if robot0_mask.dim() == 5:
            #     robot0_mask = robot0_mask.squeeze(2)
            #     robot1_mask = robot1_mask.squeeze(2)

            robot0_mask, _, _ = self._flatten_bt(robot0_mask)  # (B*T,H,W,1)
            robot1_mask, _, _ = self._flatten_bt(robot1_mask)

            # # Add channel dim → (N,1,H,W)
            # robot0_mask = robot0_mask.unsqueeze(1)
            # robot1_mask = robot1_mask.unsqueeze(1)

       

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

        # We'll do geometric ops using channels-first tensors
        # Convert (N,H,W,3) -> (N,3,H,W)
        #print('obs shape', obs.shape)
        obs = obs.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        obs = F.interpolate(obs,
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
            rotated_action = rotated_action.clip(-1, 1)
            
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
            B, C, H, W = obs.shape
            obs = torch.flip(obs, [2])
            if self.use_workspace:
                robot0_mask = torch.flip(robot0_mask, [2])
                robot1_mask = torch.flip(robot1_mask, [2])

            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(BB*(TT-1), -1)

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

        # =========================
        #      RESHAPE BACK
        # =========================
        # # obs -> (N,H,W,3)
        # 
        #obs = obs.permute(0, 2, 3, 1).contiguous()
        obs = self._unflatten_bt(obs, BB, TT)  # (B, T, 3, H, W)
        #print('[diffusion augmenter] obs.shape', obs.shape)
        
       
    
        sample["rgb"] = obs

        if self.use_workspace:
            robot0_mask = self._unflatten_bt(robot0_mask, BB, TT)
            robot1_mask = self._unflatten_bt(robot1_mask, BB, TT)

            sample['robot0_mask'] = robot0_mask
            sample['robot1_mask'] = robot1_mask

            if 'rgb-workspace-mask' in sample:
                sample['rgb-workspace-mask'] = torch.cat([obs, robot0_mask, robot1_mask], dim=2)


        #print('[augmenter] rgb stats', sample['rgb'].max(), sample['rgb'].min())
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

        #print('[diffusion augmenter] after action', sample["action"].shape)

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
