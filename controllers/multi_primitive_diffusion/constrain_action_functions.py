import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _snap_to_nearest(points_bht, valid_points):
    """
    Helper function to snap predicted points to the nearest valid point.
    points_bht: (Batch, Horizon, 2)
    valid_points: (N, 2)
    """
    if valid_points is None or valid_points.shape[0] == 0:
        return points_bht

    B, H, D = points_bht.shape
    flat_points = points_bht.reshape(-1, 2)
    
    # Calculate pairwise distance: (B*H, N)
    dists = torch.cdist(flat_points, valid_points)
    
    # Find index of nearest valid point
    min_indices = torch.argmin(dists, dim=1) # (B*H,)
    
    # Gather coordinates
    nearest_points = valid_points[min_indices] # (B*H, 2)
    
    return nearest_points.reshape(B, H, 2)

def _visualize_constraint(mask, original_act, constrained_act, robot_name, step_idx):
    """
    Visualizes the before/after of the constraint.
    mask: (H, W) numpy array (binary or float)
    original_act: (Horizon, 4) -> [pick_u, pick_v, place_u, place_v]
    constrained_act: (Horizon, 4) -> [pick_u, pick_v, place_u, place_v]
    """
    save_dir = "./tmp/constrain_debug"
    os.makedirs(save_dir, exist_ok=True)
    
    H, W = mask.shape
    
    plt.figure(figsize=(6, 6))
    
    # 1. Plot Background Mask
    plt.imshow(mask, cmap='gray_r', alpha=0.3)
    plt.title(f"{robot_name} Constraint Step {step_idx}")

    # Helper to denormalize [-1, 1] -> [0, W/H]
    def to_pix(uv_coords):
        # uv_coords: (N, 2)
        u, v = uv_coords[:, 0], uv_coords[:, 1]
        x = (u + 1) * W / 2
        y = (v + 1) * H / 2
        return y, x

    # 2. Plot Pick (First 2 coords)
    orig_px, orig_py = to_pix(original_act[:, 0:2])
    new_px, new_py = to_pix(constrained_act[:, 0:2])
    
    plt.scatter(orig_px, orig_py, c='red', marker='x', s=40, label='Pick Init')
    plt.scatter(new_px, new_py, c='lime', marker='o', s=40, edgecolors='black', label='Pick Snapped')
    
    # Draw connection lines
    for i in range(len(orig_px)):
        plt.plot([orig_px[i], new_px[i]], [orig_py[i], new_py[i]], 'g--', alpha=0.5, linewidth=1)

    # 3. Plot Place (Next 2 coords)
    orig_plx, orig_ply = to_pix(original_act[:, 2:4])
    new_plx, new_ply = to_pix(constrained_act[:, 2:4])
    
    plt.scatter(orig_plx, orig_ply, c='orange', marker='x', s=40, label='Place Init')
    plt.scatter(new_plx, new_ply, c='cyan', marker='o', s=40, edgecolors='black', label='Place Snapped')
    
    for i in range(len(orig_plx)):
        plt.plot([orig_plx[i], new_plx[i]], [orig_ply[i], new_ply[i]], 'c--', alpha=0.5, linewidth=1)

    plt.legend(loc='upper right', fontsize='small')
    plt.axis('off') # Hide axes for cleaner image
    
    # Save with robot name and a counter/random ID to prevent overwrite
    import random
    plt.savefig(f"{save_dir}/deniose_step_{step_idx}_{robot_name}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

# -----------------------------------------------------------------------------
# Main Constraint Functions
# -----------------------------------------------------------------------------

def identity(action, info, t, debug=False):
    return action

def constrain_bimanual_mask(action, info, t, debug=False):
    """
    Constrains Pick/Place coordinates to specific robot masks.
    Expects action shape: (Batch, Horizon, ActionDim)
    """

    if debug:
        print(f'[constrain_bimanual_mask] noise action {t} dim {action.shape}, values {action}')
        
    # --- Config: Action Indices ---
    r0_idx = [0, 1, 4, 5] 
    r1_idx = [2, 3, 6, 7]

    original_action = action.clone()
    device = action.device
    obs = info.get('observation', {})

    # --- 1. Robot 0 Constraint ---
    if 'robot0_mask' in obs:
        mask = obs['robot0_mask']
        # if hasattr(mask, 'cpu'): mask = mask.cpu().numpy()
        if mask.ndim == 3: mask = mask.squeeze(-1)
        
        xs, ys = np.where(mask > 0)
        
        if len(xs) > 0:
            H, W = mask.shape
            # Create valid points tensor
            norm_u = (xs / W) * 2 - 1
            norm_v = (ys / H) * 2 - 1
            valid_r0 = torch.tensor(np.stack([norm_u, norm_v], axis=1), dtype=torch.float32, device=device)
            
            # Snap Pick [0, 1]
            action[:, :, r0_idx[0:2]] = _snap_to_nearest(action[:, :, r0_idx[0:2]], valid_r0)
            # Snap Place [2, 3]
            action[:, :, r0_idx[2:4]] = _snap_to_nearest(action[:, :, r0_idx[2:4]], valid_r0)

    # --- 2. Robot 1 Constraint ---
    if 'robot1_mask' in obs:
        mask = obs['robot1_mask']
        # if hasattr(mask, 'cpu'): mask = mask.cpu().numpy()
        if mask.ndim == 3: mask = mask.squeeze(-1)
        
        xs, ys = np.where(mask > 0)
        
        if len(xs) > 0:
            H, W = mask.shape
            norm_u = (xs / W) * 2 - 1
            norm_v = (ys / H) * 2 - 1
            valid_r1 = torch.tensor(np.stack([norm_u, norm_v], axis=1), dtype=torch.float32, device=device)
            
            # Snap Pick [7, 8]
            action[:, :, r1_idx[0:2]] = _snap_to_nearest(action[:, :, r1_idx[0:2]], valid_r1)
            # Snap Place [9, 10]
            action[:, :, r1_idx[2:4]] = _snap_to_nearest(action[:, :, r1_idx[2:4]], valid_r1)

    # --- 3. Visualization (If Debug) ---
    if debug:
        # We visualize only the first item in the batch (Batch index 0)
        batch_idx = 0
        step_idx = t

        # Robot 0 Vis
        if 'robot0_mask' in obs:
            mask0 = obs['robot0_mask']
            if hasattr(mask0, 'cpu'): mask0 = mask0.detach().cpu().numpy()
            if mask0.ndim == 3: mask0 = mask0.squeeze(-1)
            
            # Get coords for Robot 0 (Pick U,V, Place U,V)
            orig_act_r0 = original_action[batch_idx, :, r0_idx].detach().cpu().numpy()
            new_act_r0 = action[batch_idx, :, r0_idx].detach().cpu().numpy()
            
            _visualize_constraint(mask0, orig_act_r0, new_act_r0, "Robot0", step_idx)

        # Robot 1 Vis
        if 'robot1_mask' in obs:
            mask1 = obs['robot1_mask']
            if hasattr(mask1, 'cpu'): mask1 = mask1.detach().cpu().numpy()
            if mask1.ndim == 3: mask1 = mask1.squeeze(-1)
            
            orig_act_r1 = original_action[batch_idx, :, r1_idx].detach().cpu().numpy()
            new_act_r1 = action[batch_idx, :, r1_idx].detach().cpu().numpy()
            
            _visualize_constraint(mask1, orig_act_r1, new_act_r1, "Robot1", step_idx)

    return action

# -----------------------------------------------------------------------------
# Function Registry
# -----------------------------------------------------------------------------

name2func = {
    'identity': identity,
    'bimanual_mask': constrain_bimanual_mask,
}