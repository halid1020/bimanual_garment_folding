import torch
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt

from .utils import randomize_primitive_encoding  # or torch version if you have it

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
# =========================
#        AUGMENTER
# =========================

class PixelBasedMultiPrimitiveDataAugmenterForDreamer:
    def __init__(self, config=None):
        self.config = config
        self.random_rotation = config.get("random_rotation", False)
        self.vertical_flip = config.get("vertical_flip", False)
        self.debug = config.get("debug", False)

    def _flatten_bt(self, x):
        B, T = x.shape[:2]
        return x.reshape(B * T, *x.shape[2:]), B, T

    def _unflatten_bt(self, x, B, T):
        return x.reshape(B, T, *x.shape[1:])

    def __call__(self, sample):
        """
        Input:
            image:  (B, T, H, W, 3) uint8
            action: (B, T, A)
        """

        obs = sample["image"] / 255.0
        action = sample["action"]
        self.device = obs.device

        obs, B, T = self._flatten_bt(obs)
        act, _, _ = self._flatten_bt(action)

        pixel_actions = act[:, 1:]

        # =========================
        #   DEBUG (BEFORE AUG)
        # =========================
        if self.debug:
            n_show = min(4, B)
            for b in range(n_show):
                for t in range(min(4, T)):
                    self._save_debug_image(
                        obs[b * T + t],
                        pixel_actions[b * T + t],
                        prefix=f"dreamer_before_batch_{b}",
                        step=t
                    )

        
        obs = obs.permute(0, 3, 1, 2).contiguous()
        #print('obs before rotate shape', obs.shape)
        # =========================
        #       RANDOM ROTATION
        # =========================
        if self.random_rotation:
            while True:
                degree = self.config.rotation_degree * torch.randint(
                    int(360 / self.config.rotation_degree), size=(1,)
                )
                thetas = torch.deg2rad(degree)
                cos_theta = torch.cos(thetas)
                sin_theta = torch.sin(thetas)

                rot = torch.stack([
                    torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1).reshape(2, 2)
                ], dim=0).to(self.device)
                rot_inv = rot.transpose(-1, -2)

                # Rotate actions
                NP, A = pixel_actions.shape
                pixel_actions_ = pixel_actions.reshape(-1, 1, 2)
                N_ = pixel_actions_.shape[0]
                rotation_matrices_tensor = rot_inv.expand(N_, 2, 2).reshape(-1, 2, 2)
                rotated_action = torch.bmm(pixel_actions_, rotation_matrices_tensor).reshape(NP, A)

                if torch.abs(rotated_action).max() > 1:
                    continue

            
                N, C, H, W = obs.shape
                affine_matrix = torch.zeros(N, 2, 3, device=self.device)
                affine_matrix[:, :2, :2] = rot.expand(N, 2, 2)
                grid = F.affine_grid(affine_matrix[:, :2], (N, C, H, W), align_corners=True)
                obs = F.grid_sample(obs, grid, align_corners=True)
            
                pixel_actions = rotated_action.reshape(NP, -1)
                break
        
        #print('obs before flip shape', obs.shape)
        # =========================
        #       VERTICAL FLIP
        # =========================
        if self.vertical_flip and random.random() < 0.5:
            N, C, H, W = obs.shape
            obs = torch.flip(obs, [2])
            NP, A = pixel_actions.shape
            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(NP, A)

        # =========================
        #       COLOR JITTER
        # =========================
        obs = obs.permute(0, 2, 3, 1).contiguous()
        if self.config.get("color_jitter", False):
            brightness = random.uniform(-self.config.get("brightness", 0.2),
                                         self.config.get("brightness", 0.2))
            contrast = 1.0 + random.uniform(-self.config.get("contrast", 0.2),
                                            self.config.get("contrast", 0.2))
            saturation = 1.0 + random.uniform(-self.config.get("saturation", 0.2),
                                              self.config.get("saturation", 0.2))
            hue = random.uniform(-self.config.get("hue", 0.05),
                                 self.config.get("hue", 0.05))

            obs = jitter_brightness_torch(obs, brightness)
            obs = jitter_contrast_torch(obs, contrast)

            hsv = rgb_to_hsv_torch(obs)
            hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
            hsv[..., 1] = torch.clamp(hsv[..., 1] * saturation, 0, 1)
            obs = hsv_to_rgb_torch(hsv)

        
        #print('obs shape before debug after', obs.shape)
        # =========================
        #   DEBUG (AFTER AUG)
        # =========================
        if self.debug:
            n_show = min(4, B)
            for b in range(n_show):
                for t in range(min(4, T)):
                    self._save_debug_image(
                        obs[b * T + t],
                        pixel_actions[b * T + t],
                        prefix=f"dreamer_after_batch_{b}",
                        step=t
                    )
        
        # =========================
        #      RESHAPE BACK
        # =========================
        obs = self._unflatten_bt(obs, B, T)
        pixel_actions = self._unflatten_bt(pixel_actions, B, T)

        

        prim_acts = action[..., :1]
        if self.config.get("randomise_prim_acts", False):            
            prim_acts = randomize_primitive_encoding(prim_acts.reshape(-1, 1), self.config.K).reshape(B, T, 1)


        full_action = torch.cat([prim_acts, pixel_actions], dim=-1)

        sample["image"] = (obs.clamp(0, 1) * 255)
        sample["action"] = full_action.float()

        return sample
    
    def _save_debug_image(self, rgb, pts, prefix, step):
        """
        rgb: (H, W, 3) torch or numpy, float in [0,1]
        pts: (N, 2) torch or numpy in [-1,1]
        """
        save_dir = "./tmp/augment_debug"
        os.makedirs(save_dir, exist_ok=True)

        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        if torch.is_tensor(pts):
            pts = pts.detach().cpu().numpy()

        H, W, _ = rgb.shape
        pts = pts.reshape(-1, 2)

        xs = (pts[:, 1] + 1) * [W-1] / 2
        ys = (pts[:, 0] + 1) * [H-1] / 2

        plt.figure(figsize=(4, 4))
        plt.imshow(rgb)
        plt.scatter(xs, ys, c='red', s=12)
        plt.axis('off')
        plt.savefig(
            f"{save_dir}/{prefix}_step{step}.png",
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()
