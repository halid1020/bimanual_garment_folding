import torch
import torch.nn as nn
import torchvision
from typing import Callable
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def omegaconf_to_plain_dict(cfg):
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)  # nested dict/list
    return cfg

def dict_to_action_vector(dict_action, action_output_template):
    """
    Convert dictionary form of action back into flat vector form.
    """
    #print('action_output_template', action_output_template)
    indices = list(_max_index_in_dict(action_output_template))
    if indices:
        max_index = max(indices)
    else:
        raise ValueError(f"No indices found in action_output_template: {action_output_template}")

    #print('max_index', max_index)
    action = np.zeros(max_index + 1, dtype=float)

    def fill_action(d_action, template):
        for k, v in template.items():
            if isinstance(v, dict):
                fill_action(d_action[k], v)
            elif isinstance(v, (list, ListConfig)) and len(v) > 0:
                values = d_action[k]
                action[np.array(v)] = values

    fill_action(dict_action, action_output_template)
    return action


def _max_index_in_dict(d):
    """Helper to find all indices used in the template dict."""
    for v in d.values():
        #print(type(v))
        if isinstance(v, dict):
            yield from _max_index_in_dict(v)
        elif isinstance(v, (list, ListConfig)) and len(v) > 0:
            yield max(v)

def compute_classification_metrics(logits, targets, num_classes):
    """
    logits: (B, K)
    targets: (B,)
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)

        accuracy = (preds == targets).float().mean()

        metrics = {
            "accuracy": accuracy.item()
        }

        # Per-class precision / recall / F1 (macro-averaged)
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

def get_resnet(name:str, input_channel=3, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(**kwargs)
    resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    #print(resnet)
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
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
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def save_denoising_gif(image, masks, noise_actions_history, step_idx, save_dir='tmp/denoising_evolution'):
    """
    Generates a GIF showing the evolution of the action prediction during diffusion.
    
    Args:
        image: (C, H, W) or (H, W, C) numpy array, normalized or uint8.
        masks: List of two (H, W) numpy arrays [mask_left, mask_right].
        noise_actions_history: List of (T, D) numpy arrays. Each element is the trajectory at a diffusion step.
        step_idx: Integer, current simulation step for filename uniqueness.
        save_dir: Directory to save the GIF.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prepare Background Image (RGB + Masks)
    # Ensure image is H,W,C and uint8
    if image.shape[0] in [1, 3]: image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0: image = (image * 255).astype(np.uint8)
    
    H, W = image.shape[:2]
    
    # Create figure once to reuse settings
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Function to render a single frame
    def render_frame(action_traj, frame_idx):
        ax.clear()
        
        # A. Plot Image
        ax.imshow(image)
        
        # B. Overlay Masks (Left=Blue tint, Right=Red tint)
        # We create a colored overlay where mask is active
        if masks[0] is not None:
            # Create blue overlay
            blue_mask = np.zeros((H, W, 4))
            blue_mask[..., 2] = 1.0 # Blue channel
            blue_mask[..., 3] = masks[0] * 0.2 # Alpha
            ax.imshow(blue_mask)
            
        if masks[1] is not None:
            # Create red overlay
            red_mask = np.zeros((H, W, 4))
            red_mask[..., 0] = 1.0 # Red channel
            red_mask[..., 3] = masks[1] * 0.2 # Alpha
            ax.imshow(red_mask)
            
        # C. Plot Action Trajectories
        # Robot 0 (Left) -> Indices [0,1] -> Light Blue ('cyan')
        r0_traj = action_traj[:, :2]
        # Denormalize [-1, 1] -> [0, W]
        r0_x = (r0_traj[:, 0] + 1) * W / 2
        r0_y = (r0_traj[:, 1] + 1) * H / 2
        
        ax.plot(r0_x, r0_y, c='cyan', linewidth=2, label='Left (R0)')
        ax.scatter(r0_x[0], r0_y[0], c='cyan', s=30, marker='o') # Start
        ax.scatter(r0_x[-1], r0_y[-1], c='cyan', s=30, marker='x') # End
        
        # Robot 1 (Right) -> Indices [7,8] -> Warm Red ('tomato')
        r1_traj = action_traj[:, 7:9]
        r1_x = (r1_traj[:, 0] + 1) * W / 2
        r1_y = (r1_traj[:, 1] + 1) * H / 2
        
        ax.plot(r1_x, r1_y, c='tomato', linewidth=2, label='Right (R1)')
        ax.scatter(r1_x[0], r1_y[0], c='tomato', s=30, marker='o')
        ax.scatter(r1_x[-1], r1_y[-1], c='tomato', s=30, marker='x')
        
        ax.set_title(f"Diffusion Step: {frame_idx}/{len(noise_actions_history)}")
        ax.axis('off')
        
        # Draw canvas to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return frame

    # 2. Generate Frames
    frames = []
    # Reverse history so it goes from Noisy -> Clean
    for i, act in enumerate(reversed(noise_actions_history)):
        frames.append(render_frame(act, i))
        
    # 3. Save GIF
    filename = os.path.join(save_dir, f'step_{step_idx}_evolution.gif')
    imageio.mimsave(filename, frames, fps=10)
    plt.close(fig)
