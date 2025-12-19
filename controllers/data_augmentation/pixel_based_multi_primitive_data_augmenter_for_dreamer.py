import torch
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
import kornia.augmentation as K

from .utils import randomize_primitive_encoding  # or torch version if you have it

# =========================
#        AUGMENTER
# =========================

class PixelBasedMultiPrimitiveDataAugmenterForDreamer:
    def __init__(self, config=None):
        self.config = config
        self.random_rotation = config.get("random_rotation", False)
        self.vertical_flip = config.get("vertical_flip", False)
        self.debug = config.get("debug", False)
        self.color_jitter = self.config.get("color_jitter", False)

        self.include_state = self.config.get("include_state", False)

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
        if self.include_state:
            state = sample['state']
            state, _, _ = self._flatten_bt(state)
        

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
                
                if self.include_state:
                    pixel_state_ = state.reshape(-1, 1, 2)
                    N = pixel_state_.shape[0]
                    rotation_matrices_tensor = rot_inv.expand(N, 2, 2).reshape(-1, 2, 2)
                    new_state = torch.bmm(pixel_state_, rotation_matrices_tensor).reshape(NP, -1)


                
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

            if self.include_state:
                new_state = new_state.reshape(-1, 2)
                new_state[:, 0] = -new_state[:, 0]
                new_state = new_state.reshape(NP, -1)

        # =========================
        #       COLOR JITTER
        # =========================
        
        if self.color_jitter:
            obs = self.color_aug(obs)
        obs = obs.permute(0, 2, 3, 1).contiguous()
        # if self.config.get("color_jitter", False):
        #     brightness = random.uniform(-self.config.get("brightness", 0.2),
        #                                  self.config.get("brightness", 0.2))
        #     contrast = 1.0 + random.uniform(-self.config.get("contrast", 0.2),
        #                                     self.config.get("contrast", 0.2))
        #     saturation = 1.0 + random.uniform(-self.config.get("saturation", 0.2),
        #                                       self.config.get("saturation", 0.2))
        #     hue = random.uniform(-self.config.get("hue", 0.05),
        #                          self.config.get("hue", 0.05))

        #     obs = jitter_brightness_torch(obs, brightness)
        #     obs = jitter_contrast_torch(obs, contrast)

        #     hsv = rgb_to_hsv_torch(obs)
        #     hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
        #     hsv[..., 1] = torch.clamp(hsv[..., 1] * saturation, 0, 1)
        #     obs = hsv_to_rgb_torch(hsv)

        
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
        if self.include_state:
            new_state = self._unflatten_bt(new_state, B, T)
            sample['state'] = new_state

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
