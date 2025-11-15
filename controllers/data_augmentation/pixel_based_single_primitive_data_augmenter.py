import torch
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1), indexing='ij')
    h = torch.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def add_noise(rgb, noise_factor=0.0):
    device = rgb.device
    if noise_factor != 0:
        rgb = rgb + torch.randn(rgb.shape, device=device)*noise_factor
        rgb = rgb.clip(-0.5, 0.5)
    return rgb


class PixelBasedSinglePrimitiveDataAugmenter:
    def __init__(self, config=None):
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
        self.debug = self.config.get('debug', False)
        #self.device = self.config.get('device', 'cpu')

        if self.depth_blur:
            kernel_size = self.config.depth_blur_kernel_size
            sigma = 1.0
            self.kernel = gaussian_2d((kernel_size, kernel_size), sigma).to(self.device)
            self.kernel = self.kernel.expand(1, 1, kernel_size, kernel_size)
            self.padding = (kernel_size - 1) // 2 if kernel_size % 2 == 1 else kernel_size // 2

    def _save_debug_image(self, rgb_tensor, state_tensor, prefix, step):
        """Save an RGB tensor with state points drawn on it."""
        save_dir = "./tmp/augment_debug"
        os.makedirs(save_dir, exist_ok=True)

        # Convert from [-0.5, 0.5] to [0, 1] and move to CPU
        rgb = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()

        # Assume state values are normalized between [-1, 1]
        H, W, _ = rgb.shape
        states = state_tensor.detach().cpu().numpy().reshape(-1, 2)
        x = (W / 2) * (states[:, 1] + 1)
        y = (H / 2) * (states[:, 0] + 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(rgb)
        plt.scatter(x, y, c='red', s=20)
        plt.axis('off')

        path = os.path.join(save_dir, f"{prefix}_step{step}.png")
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def __call__(self, sample):
        observation = sample['observation']/255.0  # B*C*H*W, [0, 255]
        #self.device = observation.device
        self.device = observation.device
        state = sample['state']
        next_observation = sample['next_observation']/255.0
        action = sample['action']

        pixel_actions = action

        # Save original (before augmentation)
        if self.debug:
            for b in range(min(4, observation.shape[0])):  # only first few samples
                self._save_debug_image(observation[b], state[b], prefix="before_state", step=b)
                self._save_debug_image(observation[b], pixel_actions[b], prefix="before_action", step=b)

        new_state = state.clone()

        # Random Rotation
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
                B, A = pixel_actions.shape
                pixel_actions_ = pixel_actions.reshape(-1, 1, 2)
                N = pixel_actions_.shape[0]
                rotation_matrices_tensor = rot_inv.expand(N, 2, 2).reshape(-1, 2, 2)
                rotated_action = torch.bmm(pixel_actions_, rotation_matrices_tensor).reshape(B, A)

                if torch.abs(rotated_action).max() > 1:
                    continue

                pixel_state_ = state.reshape(-1, 1, 2)
                N = pixel_state_.shape[0]
                rotation_matrices_tensor = rot_inv.expand(N, 2, 2).reshape(-1, 2, 2)
                new_state = torch.bmm(pixel_state_, rotation_matrices_tensor).reshape(B, -1)

                B, C, H, W = observation.shape
                affine_matrix = torch.zeros(B, 2, 3, device=self.device)
                affine_matrix[:, :2, :2] = rot.expand(B, 2, 2)
                grid = F.affine_grid(affine_matrix[:, :2], (B, C, H, W), align_corners=True)
                observation = F.grid_sample(observation, grid, align_corners=True)
                next_observation = F.grid_sample(next_observation, grid, align_corners=True)
                pixel_actions = rotated_action
                break

        # Vertical Flip
        if self.vertical_flip and (random.random() < 0.5):
            B, C, H, W = observation.shape
            observation = torch.flip(observation, [2])
            next_observation = torch.flip(next_observation, [2])

            pixel_actions = pixel_actions.reshape(-1, 2)
            pixel_actions[:, 0] = -pixel_actions[:, 0]
            pixel_actions = pixel_actions.reshape(B, -1)

            new_state = new_state.reshape(-1, 2)
            new_state[:, 0] = -new_state[:, 0]
            new_state = new_state.reshape(B, -1)

        # Save after augmentation
        if self.debug:
            for b in range(min(4, observation.shape[0])):
                self._save_debug_image(observation[b], new_state[b], prefix="after_state", step=b)
                self._save_debug_image(observation[b], pixel_actions[b], prefix="after_action", step=b)

        sample['observation'] = observation.clip(0, 1)*255
        sample['next_observation'] = next_observation.clip(0, 1)*255
        sample['action'] = pixel_actions
        sample['state'] = new_state
        return sample
