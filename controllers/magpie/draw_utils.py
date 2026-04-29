import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_rgb_workspace_mask_goal_features(image_tensor, save_dir='tmp'):
    
    """
    Saves a diagnostic plot and individual mask channels to the specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. Prepare Tensor: Handle temporal stack (B, T, C, H, W) or (T, C, H, W)
    # We take the most recent frame [-1]
    if image_tensor.ndim == 5: # Batch included
        img = image_tensor[0, -1]
    elif image_tensor.ndim == 4: # Only Horizon included
        img = image_tensor[-1]
    else:
        img = image_tensor

    # Convert to numpy (C, H, W)
    img = img.detach().cpu().numpy()
    
    # 2. Extract Components (Order: RGB(3), M0(1), M1(1), Goal(3))
    curr_rgb = img[0:3].transpose(1, 2, 0)
    mask0    = img[3] # Robot 0 Mask
    mask1    = img[4] # Robot 1 Mask
    goal_rgb = img[5:8].transpose(1, 2, 0)

    # 3. Normalization for visualization (matplotlib/cv2 expect 0-1 or 0-255)
    def to_vis(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    curr_rgb_vis = to_vis(curr_rgb)
    goal_rgb_vis = to_vis(goal_rgb)

    # 4. Save the logic-check Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(curr_rgb_vis)
    axes[0].set_title("Current RGB")
    
    axes[1].imshow(mask0, cmap='gray')
    axes[1].set_title("Ch3: Robot 0 Mask")
    
    axes[2].imshow(mask1, cmap='gray')
    axes[2].set_title("Ch4: Robot 1 Mask")
    
    axes[3].imshow(goal_rgb_vis)
    axes[3].set_title("Goal RGB")

    for ax in axes: ax.axis('off')
    
    fig_path = os.path.join(save_dir, 'input_breakdown.png')
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved diagnostic figure to {fig_path}")

    # 5. Save Masks by Channel Order (Individual files)
    # We multiply by 255 to make them visible as standard grayscale images
    cv2.imwrite(os.path.join(save_dir, 'mask_ch3_robot0.png'), (mask0 * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, 'mask_ch4_robot1.png'), (mask1 * 255).astype(np.uint8))
    
    print(f"Saved individual mask channels to {save_dir}/")