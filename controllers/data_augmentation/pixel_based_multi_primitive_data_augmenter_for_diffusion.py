import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from agent_arena.agent.utilities.torch_utils import np_to_ts, ts_to_np

# --------------------
# Helper functions (torch)
# --------------------
def jitter_brightness_torch(imgs, delta):
    # imgs: (N,C,H,W)
    return torch.clamp(imgs + delta, 0.0, 1.0)

def jitter_contrast_torch(imgs, factor):
    # imgs: (N,C,H,W)
    mean = imgs.mean(dim=(2, 3), keepdim=True)  # shape (N,C,1,1)
    return torch.clamp((imgs - mean) * factor + mean, 0.0, 1.0)

def rgb_to_hsv_torch(rgb):
    # rgb: (N,H,W,3) in [0,1] channels-last
    # returns (N,H,W,3) hsv with h in [0,1], s,v in [0,1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, _ = rgb.max(dim=-1)
    minc, _ = rgb.min(dim=-1)
    v = maxc
    diff = maxc - minc

    s = torch.where(maxc == 0, torch.zeros_like(maxc), diff / (maxc + 1e-8))

    h = torch.zeros_like(maxc)

    mask = diff > 1e-8

    rc = (maxc - r) / (diff + 1e-8)
    gc = (maxc - g) / (diff + 1e-8)
    bc = (maxc - b) / (diff + 1e-8)

    # for r == max
    pick = (r == maxc) & mask
    h = torch.where(pick, (bc - gc), h)

    # for g == max
    pick = (g == maxc) & mask
    h = torch.where(pick, 2.0 + (rc - bc), h)

    # for b == max
    pick = (b == maxc) & mask
    h = torch.where(pick, 4.0 + (gc - rc), h)

    h = (h / 6.0) % 1.0
    hsv = torch.stack([h, s, v], dim=-1)
    return hsv

def hsv_to_rgb_torch(hsv):
    # hsv: (N,H,W,3) channels-last, h in [0,1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = torch.floor(h * 6.0).to(torch.int64)
    f = (h * 6.0) - i.type(h.dtype)
    i = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    shape = h.shape
    r = torch.zeros(shape, dtype=h.dtype, device=h.device)
    g = torch.zeros(shape, dtype=h.dtype, device=h.device)
    b = torch.zeros(shape, dtype=h.dtype, device=h.device)

    idx = (i == 0)
    r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]
    idx = (i == 1)
    r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]
    idx = (i == 2)
    r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]
    idx = (i == 3)
    r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]
    idx = (i == 4)
    r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]
    idx = (i == 5)
    r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

    return torch.stack([r, g, b], dim=-1)

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
        if "rgb" not in sample:
            raise KeyError("sample must contain 'rgb'")
        if train and "action" not in sample:
            raise KeyError("sample must contain 'action'")


        # Normalize rgb to [0,1] float32
        observation = sample["rgb"].float() / 255.0  # (B, T, H, W, 3)
        #print('input obs shape', observation.shape)
        obs, BB, TT = self._flatten_bt(observation)  # (B*T, H, W, 3)
       

        if train:
            action = sample["action"]  # (B, T, A)
            #print('input act shape', action.shape)
            act, _, _ = self._flatten_bt(action)  # (B*T, A)
            pixel_actions = act[:, 1:]  # keep continuous part

        # Debug save before any augmentation
        if self.debug:
            n_show = min(4, obs.shape[0])
            for b in range(n_show):
                cpu_img = (obs[b].cpu().numpy()).astype(np.float32)
                if train:
                    pa_cpu = pixel_actions[b].cpu().numpy()
                else:
                    pa_cpu = np.zeros((1,2))
                self._save_debug_image(cpu_img, pa_cpu, prefix="before_action", step=b)

        # We'll do geometric ops using channels-first tensors
        # Convert (N,H,W,3) -> (N,3,H,W)
        print('obs shape', obs.shape)
        obs = obs.permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)
        obs = F.interpolate(obs,
            size=tuple(self.config.img_dim), mode='bilinear', align_corners=False)

        # =========================
        #       RANDOM ROTATION
        # =========================
        if self.random_rotation and train:
            while True:
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

                if torch.abs(rotated_action).max() > 1:
                    continue

               
                B, C, H, W = obs.shape
                affine_matrix = torch.zeros(B, 2, 3, device=device)
                affine_matrix[:, :2, :2] = rot.expand(B, 2, 2)
                grid = F.affine_grid(affine_matrix[:, :2], (B, C, H, W), align_corners=True)
                obs = F.grid_sample(obs, grid, align_corners=True)
              
                pixel_actions = rotated_action.reshape(BB*(TT-1), -1)
                break

        # Vertical Flip
        if self.vertical_flip and (random.random() < 0.5) and train:
            B, C, H, W = obs.shape
            obs = torch.flip(obs, [2])
           

            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(BB*(TT-1), -1)

        # =========================
        #     COLOR JITTER (GLOBAL)
        # =========================
        if self.config.get("color_jitter", False) and train:
            brightness = float(np.random.uniform(-self.config.get("brightness", 0.2),
                                                 self.config.get("brightness", 0.2)))
            contrast = float(1.0 + np.random.uniform(-self.config.get("contrast", 0.2),
                                                     self.config.get("contrast", 0.2)))
            saturation = float(1.0 + np.random.uniform(-self.config.get("saturation", 0.2),
                                                       self.config.get("saturation", 0.2)))
            hue = float(np.random.uniform(-self.config.get("hue", 0.05),
                                          self.config.get("hue", 0.05)))

            # Convert back to channels-last for HSV ops (N,H,W,3)
            imgs_cl = obs.permute(0, 2, 3, 1).contiguous()  # (N,H,W,3)

            # Brightness & Contrast (operate on channels-first equivalent)
            imgs_cf = imgs_cl.permute(0, 3, 1, 2).contiguous()
            imgs_cf = jitter_brightness_torch(imgs_cf, brightness)
            imgs_cf = jitter_contrast_torch(imgs_cf, contrast)
            imgs_cl = imgs_cf.permute(0, 2, 3, 1).contiguous()

            # Convert to HSV and apply hue/saturation
            hsv = rgb_to_hsv_torch(imgs_cl)  # (N,H,W,3)
            hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
            hsv[..., 1] = torch.clamp(hsv[..., 1] * saturation, 0.0, 1.0)

            imgs_cl = hsv_to_rgb_torch(hsv)

            # back to channels-first
            obs = imgs_cl.permute(0, 3, 1, 2).contiguous()

        # =========================
        # debug save after
        # =========================
        if self.debug and train:
            n_show = min(4, obs.shape[0])
            for b in range(n_show):
                cpu_img = obs[b].permute(1, 2, 0).cpu().numpy()  # H,W,3
                pa_cpu = pixel_actions[b].cpu().numpy()
                self._save_debug_image(cpu_img, pa_cpu, prefix="after_action", step=b)

        # =========================
        #      RESHAPE BACK
        # =========================
        # # obs -> (N,H,W,3)
        # obs = obs.permute(0, 2, 3, 1).contiguous()
    
        obs = self._unflatten_bt(obs, BB, TT)  # (B, T, 3, H, W)
        

        sample["rgb"] = obs
        if train:
            #print('pixel_actions', pixel_actions.shape)
            pixel_actions = self._unflatten_bt(pixel_actions, BB, TT-1)
            # restore action = (1 discrete action + continuous pixel coords)
            # action is a tensor (B,T,A); keep first column as-is
            full_action = torch.cat([action[..., :1], pixel_actions], dim=-1)
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
