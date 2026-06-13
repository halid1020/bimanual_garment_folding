"""
Magpie Utilities Module.

This module provides a collection of helper functions required for the Magpie 
diffusion agent. It handles configuration parsing, hierarchical action space 
flattening, classification metric computations, vision encoder modifications 
(e.g., swapping BatchNorm for GroupNorm), and debugging visualizations.
"""

import os
import torch
import torch.nn as nn
import torchvision
from typing import Callable
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import imageio

# =============================================================================
# Configuration & Action Space Utilities
# =============================================================================

def omegaconf_to_plain_dict(cfg):
    """
    Strips Hydra/OmegaConf wrappers from a configuration object.

    Many logging frameworks (like wandb) or downstream serialization methods 
    fail when passed DictConfig objects. This ensures a pure Python dictionary.

    Args:
        cfg (DictConfig, ListConfig, or dict): The configuration object.

    Returns:
        dict: A standard, unnested Python dictionary.
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True) 
    return cfg

def dict_to_action_vector(dict_action, action_output_template):
    """
    Flattens a structured, hierarchical action dictionary into a 1D NumPy array.

    In complex robotics tasks (e.g., dual-arm setups with grippers), environments 
    often expect actions as nested dictionaries (e.g., `{'arm0': {'pose': [...], 
    'gripper': [...]}}`). However, the diffusion network requires a flat 1D vector.
    This maps the dictionary values to their predefined index positions in the vector.

    Args:
        dict_action (dict): The nested action dictionary from the environment or dataset.
        action_output_template (dict): A template defining which vector indices 
                                       correspond to which dictionary keys.

    Returns:
        np.ndarray: A flattened 1D action vector of type float.
        
    Raises:
        ValueError: If no valid indices can be found in the template.
    """
    indices = list(_max_index_in_dict(action_output_template))
    if indices:
        max_index = max(indices)
    else:
        raise ValueError(f"No indices found in action_output_template: {action_output_template}")

    # Initialize the flat vector
    action = np.zeros(max_index + 1, dtype=float)

    def fill_action(d_action, template):
        """Recursively traverses the template to place values into the flat array."""
        for k, v in template.items():
            if isinstance(v, dict):
                fill_action(d_action[k], v)
            elif isinstance(v, (list, ListConfig)) and len(v) > 0:
                values = d_action[k]
                action[np.array(v)] = values

    fill_action(dict_action, action_output_template)
    return action

def _max_index_in_dict(d):
    """
    Recursive generator to find all index assignments defined in the template.
    
    Args:
        d (dict): The template dictionary.
        
    Yields:
        int: The maximum index found in the current branch.
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from _max_index_in_dict(v)
        elif isinstance(v, (list, ListConfig)) and len(v) > 0:
            yield max(v)

# =============================================================================
# Training Metrics
# =============================================================================

def compute_classification_metrics(logits, targets, num_classes):
    """
    Computes accuracy, precision, recall, and F1 scores for primitive classification.

    When training the high-level primitive selection head, class imbalance is common 
    (e.g., standard movements vs. rare grasping events). This function computes 
    macro-averaged metrics to ensure minority classes are evaluated fairly.

    Args:
        logits (torch.Tensor): Unnormalized network predictions (Batch, NumClasses).
        targets (torch.Tensor): Ground truth class indices (Batch,).
        num_classes (int): Total number of primitive classes (K).

    Returns:
        dict: A dictionary containing 'accuracy', 'precision_macro', 
              'recall_macro', and 'f1_macro'.
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == targets).float().mean()

        metrics = {
            "accuracy": accuracy.item()
        }

        # Epsilon to prevent division by zero
        eps = 1e-8
        precisions, recalls, f1s = [], [], []

        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        metrics.update({
            "precision_macro": torch.stack(precisions).mean().item(),
            "recall_macro": torch.stack(recalls).mean().item(),
            "f1_macro": torch.stack(f1s).mean().item(),
        })

    return metrics

# =============================================================================
# Network Architecture Utilities
# =============================================================================

def get_resnet(name: str, input_channel=3, weights=None, **kwargs) -> nn.Module:
    """
    Instantiates a standard ResNet vision encoder from torchvision.

    Modifies the first convolutional layer to accommodate non-standard input 
    dimensions (e.g., RGB-D or stacked mask channels) and removes the final 
    classification head to extract raw 512-dimensional embeddings.

    Args:
        name (str): Model name (e.g., 'resnet18', 'resnet34').
        input_channel (int): Number of input channels (default: 3).
        weights (str, optional): Pre-trained weights to load, e.g., 'IMAGENET1K_V1'.

    Returns:
        nn.Module: The modified ResNet encoder.
    """
    func = getattr(torchvision.models, name)
    resnet = func(**kwargs)
    
    # Overwrite the first layer if input channels != 3 (e.g., RGBD = 4, RGB+Masks = 5)
    resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Strip the final fully connected classification layer to use as an embedding extractor.
    # For resnet18, the output embedding dimension will be 512.
    resnet.fc = torch.nn.Identity()
    resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Recursively replaces specific submodules within a PyTorch model.

    Args:
        root_module (nn.Module): The top-level PyTorch module.
        predicate (Callable): Returns True if the submodule should be replaced.
        func (Callable): A factory function that takes the old module and returns the new one.

    Returns:
        nn.Module: The updated root module.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
        
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
            
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
            
        tgt_module = func(src_module)
        
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
            
    # Verification pass to ensure no modules were missed
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0, "Failed to replace all target submodules."
    
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Replaces all BatchNorm2d layers in a network with GroupNorm.

    **Why this is critical for RL & Robotics:** BatchNorm relies on calculating running statistics over the batch dimension. 
    In RL pipelines or diffusion models where VRAM limits batch sizes (e.g., batch size 
    of 8 or 16), these statistics become highly erratic and destabilize training. 
    GroupNorm computes statistics over channel groups independently of batch size, 
    making it significantly more stable for these workloads.

    Args:
        root_module (nn.Module): The network (e.g., a ResNet) to modify.
        features_per_group (int): The number of channels per normalization group.

    Returns:
        nn.Module: The network with all BatchNorm layers replaced.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group), # Ensure at least 1 group
            num_channels=x.num_features)
    )
    return root_module

# =============================================================================
# Debugging & Visualization
# =============================================================================

def save_denoising_gif(image, masks, noise_actions_history, step_idx, save_dir='tmp/denoising_evolution'):
    """
    Generates a GIF visualizing the reverse diffusion process over time.
    
    This plots the iterative refinement of the predicted action trajectory as 
    it evolves from pure noise (at the start of the reverse process) to the 
    clean, final predicted trajectory. It overlays the trajectory on the camera 
    feed and workspace masks.

    Args:
        image (np.ndarray): (C, H, W) or (H, W, C) numpy array, normalized or uint8.
        masks (list): List of two (H, W) numpy arrays representing valid regions [mask_left, mask_right].
        noise_actions_history (list): List of (T, D) numpy arrays representing the trajectory at each diffusion step.
        step_idx (int): Current simulation step (used for filename uniqueness).
        save_dir (str): Output directory for the GIF.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prepare Background Image (Ensure H,W,C and uint8 format)
    if image.shape[0] in [1, 3]: image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0: image = (image * 255).astype(np.uint8)
    
    H, W = image.shape[:2]
    
    # Create figure once to reuse settings and save rendering time
    fig, ax = plt.subplots(figsize=(5, 5))
    
    def render_frame(action_traj, frame_idx):
        """Renders a single frame of the diffusion process."""
        ax.clear()
        
        # A. Plot Base Image
        ax.imshow(image)
        
        # B. Overlay Semantic Masks (Left=Blue tint, Right=Red tint)
        if masks[0] is not None:
            blue_mask = np.zeros((H, W, 4))
            blue_mask[..., 2] = 1.0 # Blue channel
            blue_mask[..., 3] = masks[0] * 0.2 # Alpha transparency
            ax.imshow(blue_mask)
            
        if masks[1] is not None:
            red_mask = np.zeros((H, W, 4))
            red_mask[..., 0] = 1.0 # Red channel
            red_mask[..., 3] = masks[1] * 0.2 # Alpha transparency
            ax.imshow(red_mask)
            
        # C. Plot Action Trajectories
        # Robot 0 (Left Arm) -> Assuming indices [0,1] correspond to X, Y
        r0_traj = action_traj[:, :2]
        # Denormalize from diffusion space [-1, 1] back to pixel space [0, W]
        r0_x = (r0_traj[:, 0] + 1) * W / 2
        r0_y = (r0_traj[:, 1] + 1) * H / 2
        
        ax.plot(r0_x, r0_y, c='cyan', linewidth=2, label='Left (R0)')
        ax.scatter(r0_x[0], r0_y[0], c='cyan', s=30, marker='o') # Trajectory Start
        ax.scatter(r0_x[-1], r0_y[-1], c='cyan', s=30, marker='x') # Trajectory End
        
        # Robot 1 (Right Arm) -> Assuming indices [7,8] correspond to X, Y
        r1_traj = action_traj[:, 7:9]
        r1_x = (r1_traj[:, 0] + 1) * W / 2
        r1_y = (r1_traj[:, 1] + 1) * H / 2
        
        ax.plot(r1_x, r1_y, c='tomato', linewidth=2, label='Right (R1)')
        ax.scatter(r1_x[0], r1_y[0], c='tomato', s=30, marker='o')
        ax.scatter(r1_x[-1], r1_y[-1], c='tomato', s=30, marker='x')
        
        ax.set_title(f"Diffusion Step: {frame_idx}/{len(noise_actions_history)}")
        ax.axis('off')
        
        # Draw canvas and convert to numpy array for imageio
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return frame

    # 2. Generate Frames
    frames = []
    # Reverse history so the GIF plays from High Noise -> Clean Output
    for i, act in enumerate(reversed(noise_actions_history)):
        frames.append(render_frame(act, i))
        
    # 3. Save as looping GIF
    filename = os.path.join(save_dir, f'step_{step_idx}_evolution.gif')
    imageio.mimsave(filename, frames, fps=10)
    plt.close(fig)