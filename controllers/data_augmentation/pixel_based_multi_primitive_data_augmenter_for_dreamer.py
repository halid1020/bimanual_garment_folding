import numpy as np
from scipy.ndimage import affine_transform
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from scipy.ndimage import affine_transform
import random
import os
import matplotlib.pyplot as plt


def jitter_brightness(imgs, delta):
    return np.clip(imgs + delta, 0.0, 1.0)


def jitter_contrast(imgs, factor):
    mean = imgs.mean(axis=(1,2), keepdims=True)
    return np.clip((imgs - mean) * factor + mean, 0.0, 1.0)


def rgb_to_hsv_vectorized(rgb):
    # rgb: (N,H,W,3)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    diff = maxc - minc

    # Saturation
    s = np.where(maxc == 0, 0, diff / maxc)

    # Hue
    h = np.zeros_like(maxc)
    mask = diff != 0

    rc = (maxc - r) / (diff + 1e-6)
    gc = (maxc - g) / (diff + 1e-6)
    bc = (maxc - b) / (diff + 1e-6)

    h[..., :] = np.where((r == maxc) & mask, (bc - gc), h)
    h[..., :] = np.where((g == maxc) & mask, 2.0 + (rc - bc), h)
    h[..., :] = np.where((b == maxc) & mask, 4.0 + (gc - rc), h)

    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb_vectorized(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    i = np.floor(h * 6).astype(np.int32)
    f = (h * 6) - i
    i = i % 6

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)

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

    return np.stack([r, g, b], axis=-1)



def rotate_points(points, R):
    """points: (..., 2), R: (2,2)"""
    return points @ R.T


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

    def _save_debug_image(self, rgb, pts, prefix, step):
        """rgb: (H,W,3) float, pts: (N,2) in [-1,1]"""
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

    def __call__(self, sample):
        """
        Input sample:
            observation:       (B, T, H, W, 3) numpy      
            action:            (B, T, 9)

        """
        observation = sample["image"].astype(np.float32) / 255.0
        action = sample["action"].astype(np.float32)

        # flatten (B,T) → (B*T)
        obs, B, T = self._flatten_bt(observation)  
        act, _, _ = self._flatten_bt(action)


        pixel_actions = act[:, 1:]      # keep continuous part

        # Save original (before augmentation)
        if self.debug:
            for b in range(min(4, obs.shape[0])):  # only first few samples
                self._save_debug_image(obs[b], pixel_actions[b], prefix="before_action", step=b)

        # =========================
        #       RANDOM ROTATION
        # =========================
        if self.random_rotation:
            while True:
                deg = self.config.rotation_degree * np.random.randint(
                    0, int(360 / self.config.rotation_degree)
                )
                rad = np.deg2rad(deg)

                cos = np.cos(rad)
                sin = np.sin(rad)

                R = np.array([[cos, -sin],
                              [sin,  cos]], dtype=np.float32)
                Rinv = R.T

                # rotate pixel actions
                pa = pixel_actions.reshape(-1, 2)
                pa_rot = rotate_points(pa, Rinv)
                if np.abs(pa_rot).max() > 1:
                    continue

                pixel_actions = pa_rot.reshape(pixel_actions.shape)


                # rotate images
                for i in range(obs.shape[0]):
                    oy = obs[i]

                    H, W = oy.shape[:2]
                    center = np.array([H/2, W/2])
                    offset = center - R @ center

                    rotated = np.zeros_like(oy)

                    # apply transform per channel (SciPy requires channel-wise)
                    for c in range(3):
                        rotated[..., c] = affine_transform(
                            oy[..., c],
                            R,
                            offset=offset,
                            order=1,
                            mode='nearest'
                        )

                    obs[i] = rotated
                   
                break

        # =========================
        #       VERTICAL FLIP
        # =========================
        if self.vertical_flip and random.random() < 0.5:
            obs = obs[:, ::-1]    # flip Y

            pa = pixel_actions.reshape(-1, 2)
            pa[:, 0] = -pa[:, 0]
            pixel_actions = pa.reshape(pixel_actions.shape)

        
        # =========================
        #     COLOR JITTER (GLOBAL)
        # =========================
        if self.config.get("color_jitter", False):
            brightness = np.random.uniform(-self.config.get("brightness", 0.2),
                                            self.config.get("brightness", 0.2))
            contrast = 1.0 + np.random.uniform(-self.config.get("contrast", 0.2),
                                                self.config.get("contrast", 0.2))
            saturation = 1.0 + np.random.uniform(-self.config.get("saturation", 0.2),
                                                self.config.get("saturation", 0.2))
            hue = np.random.uniform(-self.config.get("hue", 0.05),
                                    self.config.get("hue", 0.05))

            # (N,H,W,3)
            imgs = obs

            # Brightness & Contrast
            imgs = jitter_brightness(imgs, brightness)
            imgs = jitter_contrast(imgs, contrast)

            # Convert to HSV (vectorized)
            hsv = rgb_to_hsv_vectorized(imgs)

            # Apply hue/saturation jitter
            hsv[..., 0] = (hsv[..., 0] + hue) % 1.0
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)

            # Convert back
            obs = hsv_to_rgb_vectorized(hsv)


        # =========================
        #      RESHAPE BACK
        # =========================
        if self.debug:
            for b in range(min(4, obs.shape[0])):
                self._save_debug_image(obs[b], pixel_actions[b], prefix="after_action", step=b)

        observation = self._unflatten_bt(obs, B, T)
        pixel_actions = self._unflatten_bt(pixel_actions, B, T)

        # restore action = (1 discrete action + 2*k pixel coords)
        full_action = np.concatenate([action[..., :1], pixel_actions], axis=-1)

        # =========================
        #      PACK OUTPUT
        # =========================
        sample["image"] = (observation * 255).astype(np.uint8)
        sample["action"] = full_action.astype(np.float32)

        return sample
