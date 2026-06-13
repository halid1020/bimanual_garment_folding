"""
Constrain Action Functions Module.

This module provides physical and geometric constraint logic for the diffusion 
action trajectories. It ensures that predicted end-effector positions (picks/places) 
remain within valid workspace boundaries (defined by segmentation masks). It supports 
both "hard snapping" (forcing points into valid regions) and "soft restoring forces" 
(treating constraints as an ODE during the reverse diffusion process).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# CRITICAL: Adjust this based on how your dataset normalized spatial coordinates.
# Set to True if your network outputs [Y, X] (Row, Col).
# Set to False if your network outputs [X, Y] (Width, Height).
IS_YX_ACTION_SPACE = False 

# -----------------------------------------------------------------------------
# Helper Functions for Mathematics & Assignments
# -----------------------------------------------------------------------------

def _snap_to_nearest(points_bht, valid_points):
    """
    Finds the nearest valid spatial point for a batch of trajectory coordinates.

    Args:
        points_bht (torch.Tensor): Trajectory points of shape (Batch, Horizon, 2).
        valid_points (torch.Tensor): Tensor of valid (X, Y) coordinates in the mask.

    Returns:
        torch.Tensor: Snapped coordinates of shape (Batch, Horizon, 2).
    """
    if valid_points is None or valid_points.shape[0] == 0:
        return points_bht
    
    B, H, D = points_bht.shape
    flat_points = points_bht.reshape(-1, 2)
    
    # Compute pairwise Euclidean distances between predicted points and all valid mask points
    dists = torch.cdist(flat_points, valid_points)
    min_indices = torch.argmin(dists, dim=1)
    
    return valid_points[min_indices].reshape(B, H, 2)

def _get_restoring_force(points_bht, valid_points):
    """
    Calculates a directional vector pointing from the predicted point to the 
    nearest valid mask coordinate.

    Args:
        points_bht (torch.Tensor): Trajectory points of shape (Batch, Horizon, 2).
        valid_points (torch.Tensor): Tensor of valid (X, Y) coordinates.

    Returns:
        torch.Tensor: Force vectors of shape (Batch, Horizon, 2).
    """
    if valid_points is None or valid_points.shape[0] == 0:
        return torch.zeros_like(points_bht)
        
    nearest_points = _snap_to_nearest(points_bht, valid_points)
    return nearest_points - points_bht

def _get_min_dist(points_bht, valid_points):
    """
    Calculates the average minimum distance from a set of points to a valid region.
    Used as a heuristic cost function for arm-to-mask assignment.
    """
    if valid_points is None or valid_points.shape[0] == 0:
        return float('inf')
        
    flat_points = points_bht.reshape(-1, 2)
    dists = torch.cdist(flat_points, valid_points)
    return torch.min(dists, dim=1)[0].mean().item()

def _get_dynamic_assignments(action_b, entities, valid_masks):
    """
    Dynamically assigns robotic arms to specific workspace masks using a 
    bipartite matching heuristic based on proximity.

    For bimanual setups, it prevents both arms from trying to snap to the 
    same valid workspace if there are two distinct valid areas.

    Args:
        action_b (torch.Tensor): Single batch action trajectory (1, Horizon, Dim).
        entities (list): List of dictionaries defining the action dimensions for 
                         each robot's 'pick' and 'place' actions.
        valid_masks (dict): Dictionary mapping mask IDs (0, 1) to valid coordinate tensors.

    Returns:
        dict: A mapping of Entity ID to assigned Mask ID (e.g., {0: 1, 1: 0}).
    """
    assignments = {} 
    
    # Bimanual case: 2 arms, 2 valid masks available
    if len(entities) == 2 and 0 in valid_masks and 1 in valid_masks:
        e0_pick = action_b[:, :, entities[0]['pick']]
        e1_pick = action_b[:, :, entities[1]['pick']]
        
        # Calculate cost (distance) matrix
        d_e0_m0 = _get_min_dist(e0_pick, valid_masks[0])
        d_e0_m1 = _get_min_dist(e0_pick, valid_masks[1])
        d_e1_m0 = _get_min_dist(e1_pick, valid_masks[0])
        d_e1_m1 = _get_min_dist(e1_pick, valid_masks[1])
        
        # Simple bipartite matching: minimize total assignment distance
        if (d_e0_m0 + d_e1_m1) <= (d_e0_m1 + d_e1_m0):
            assignments = {0: 0, 1: 1}
        else:
            assignments = {0: 1, 1: 0}
            
    # Single-arm case: assign the arm to whichever mask is closest
    elif len(entities) == 1 and len(valid_masks) > 0:
        e0_pick = action_b[:, :, entities[0]['pick']]
        best_mask = min(valid_masks.keys(), key=lambda m: _get_min_dist(e0_pick, valid_masks[m]))
        assignments = {0: best_mask}
        
    return assignments

# -----------------------------------------------------------------------------
# Debug Visualization
# -----------------------------------------------------------------------------

def _visualize_debug(original_action, new_action, obs, entities, assignments, prim_name, t, method):
    """
    Generates an overlay plot of the original vs. constrained trajectories on 
    top of the current camera observation and segmentation masks.
    """
    save_dir = "tmp/debug_magpie"
    os.makedirs(save_dir, exist_ok=True)

    rgb = obs.get('rgb', None)
    if rgb is None: return 
        
    # Unpack and normalize RGB for Matplotlib
    if torch.is_tensor(rgb): rgb = rgb.detach().cpu().numpy()
    if rgb.ndim == 4: rgb = rgb[0]
    if rgb.shape[0] == 3: rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.max() > 2.0: rgb = rgb / 255.0

    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    
    t_val = t.item() if torch.is_tensor(t) else t
    ax.set_title(f"Step: {t_val:.2f} | {prim_name} | Method: {method.upper()}", fontsize=10)

    def get_mask(m_key):
        if m_key not in obs: return None
        m = obs[m_key]
        if torch.is_tensor(m): m = m.detach().cpu().numpy()
        if m.ndim == 4: m = m[0]
        if m.ndim == 3: m = m.squeeze(-1) if m.shape[-1] == 1 else m[0]
        return m

    m0 = get_mask('robot0_mask')
    m1 = get_mask('robot1_mask')

    # Overlay workspaces as semi-transparent blue/red hues
    if m0 is not None: ax.imshow(np.ma.masked_where(m0 < 0.5, m0), cmap='Blues', alpha=0.4)
    if m1 is not None: ax.imshow(np.ma.masked_where(m1 < 0.5, m1), cmap='Reds', alpha=0.4)

    def to_pix(uv_coords):
        """Converts normalized [-1, 1] coordinates back to pixel space [0, W]."""
        dim0, dim1 = uv_coords[:, 0], uv_coords[:, 1]
        if IS_YX_ACTION_SPACE:
            y, x = (dim0 + 1) * H / 2, (dim1 + 1) * W / 2
        else:
            x, y = (dim0 + 1) * W / 2, (dim1 + 1) * H / 2
        return x, y

    orig_act = original_action[0, 0].detach().cpu().numpy()
    new_act = new_action[0, 0].detach().cpu().numpy()
    colors = {0: 'blue', 1: 'red'}
    
    lbl_prefix = "Snap" if method == 'snap' else "Force"
    
    for e_idx, m_idx in assignments.items():
        ent = entities[e_idx]
        c = colors.get(m_idx, 'green')

        # Plot PICK trajectory adjustments
        if len(ent['pick']) == 2:
            pu, pv = ent['pick']
            ox, oy = to_pix(np.array([[orig_act[pu], orig_act[pv]]]))
            nx, ny = to_pix(np.array([[new_act[pu], new_act[pv]]]))
            
            ax.plot(ox, oy, marker='x', color='yellow', markersize=8, label='Orig Pick' if e_idx==0 else "")
            ax.plot(nx, ny, marker='o', color=c, markersize=8, markeredgecolor='white', label=f'{lbl_prefix} Pick (R{m_idx})')
            ax.annotate("", xy=(nx[0], ny[0]), xytext=(ox[0], oy[0]), arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8))

        # Plot PLACE trajectory adjustments
        if len(ent['place']) == 2:
            pu, pv = ent['place']
            ox, oy = to_pix(np.array([[orig_act[pu], orig_act[pv]]]))
            nx, ny = to_pix(np.array([[new_act[pu], new_act[pv]]]))
            
            ax.plot(ox, oy, marker='x', color='orange', markersize=8, label='Orig Place' if e_idx==0 else "")
            ax.plot(nx, ny, marker='^', color=c, markersize=8, markeredgecolor='white', label=f'{lbl_prefix} Place (R{m_idx})')
            ax.annotate("", xy=(nx[0], ny[0]), xytext=(ox[0], oy[0]), arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"step_{t_val:05.2f}.png"), bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main Constraint Logic
# -----------------------------------------------------------------------------

def identity(action, info, t, debug=False):
    """No-op constraint function. Returns the action unmodified."""
    return action

def _apply_constraints_core(action, info, t, method, debug=False):
    """
    Core engine for applying workspace constraints to the action tensor.
    
    Args:
        action (torch.Tensor): Current predicted trajectory of shape (Batch, Horizon, Dim).
        info (dict): Environment data containing the semantic masks.
        t (float/Tensor): Current diffusion timestep (for debugging).
        method (str): 'snap' for hard replacement, 'force' for soft ODE addition.
        debug (bool): If True, renders visual debugging plots.
        
    Returns:
        torch.Tensor: The constrained action trajectory.
    """
    device = action.device
    obs = info.get('observation', {})
    original_action = action.clone() 
    
    # 1. Map primitive names to the corresponding dimensions in the action vector
    prim_name = info.get('prim_name', 'norm-pixel-dual-pick-and-place')
    
    entities = []
    if 'pick-and-fling' in prim_name:
        entities.append({'pick': [0, 1], 'place': []})     # Left Arm Pick
        entities.append({'pick': [2, 3], 'place': []})     # Right Arm Pick
    elif 'dual-pick-and-place' in prim_name:
        entities.append({'pick': [0, 1], 'place': [4, 5]}) # Left Arm
        entities.append({'pick': [2, 3], 'place': [6, 7]}) # Right Arm
    elif 'single-pick-and-place' in prim_name:
        entities.append({'pick': [0, 1], 'place': [2, 3]}) # Single Arm
    elif 'no-operation' in prim_name:
        return action

    # 2. Extract Valid Pixel Coordinates from Semantic Masks
    # Converts binary 2D image masks into a list of valid [-1, 1] spatial coordinates.
    valid_masks = {}
    for m_idx, key in enumerate(['robot0_mask', 'robot1_mask']):
        if key in obs:
            mask = obs[key]
            if mask.ndim == 3: mask = mask.squeeze(-1)
            
            # np.where returns (rows, columns) = (Axis 0, Axis 1)
            rows, cols = np.where(mask > 0) 
            
            if len(rows) > 0:
                H, W = mask.shape
                
                # Normalize pixel indices to the [-1, 1] range expected by the diffusion model
                norm_x = (cols / float(W)) * 2 - 1.0
                norm_y = (rows / float(H)) * 2 - 1.0
                
                # Stack coordinates based on the network's architectural output format
                if IS_YX_ACTION_SPACE:
                    valid_pts = np.stack([norm_y, norm_x], axis=1)
                else:
                    valid_pts = np.stack([norm_x, norm_y], axis=1)
                    
                valid_masks[m_idx] = torch.tensor(valid_pts, dtype=torch.float32, device=device)

    # If no valid workspace masks are detected, abort constraints
    if not valid_masks:
        return action

    B = action.shape[0]
    force_vector = torch.zeros_like(action) if method == 'force' else None
    debug_assignments_b0 = {}

    # 3. Apply Constraints per Batch Item
    for b in range(B):
        # Figure out which arm should target which valid workspace region
        assignments = _get_dynamic_assignments(action[b:b+1], entities, valid_masks)
        
        if b == 0:
            debug_assignments_b0 = assignments 
            
        for e_idx, m_idx in assignments.items():
            ent = entities[e_idx]
            pts = valid_masks[m_idx]
            
            # Apply to PICK positions
            if method == 'snap':
                action[b:b+1, :, ent['pick']] = _snap_to_nearest(action[b:b+1, :, ent['pick']], pts)
            elif method == 'force':
                force_vector[b:b+1, :, ent['pick']] = _get_restoring_force(action[b:b+1, :, ent['pick']], pts)
                
            # Apply to PLACE positions
            if len(ent['place']) > 0:
                if method == 'snap':
                    action[b:b+1, :, ent['place']] = _snap_to_nearest(action[b:b+1, :, ent['place']], pts)
                elif method == 'force':
                    force_vector[b:b+1, :, ent['place']] = _get_restoring_force(action[b:b+1, :, ent['place']], pts)

    # 4. Integrate Soft Forces (if applicable)
    if method == 'force':
        eta_dt = 0.5 # Hyperparameter governing the strength of the restorative ODE step
        action = action + (eta_dt * force_vector)
        
    if debug:
        _visualize_debug(original_action, action, obs, entities, debug_assignments_b0, prim_name, t, method)
        
    return action

def _get_t_val(t):
    """Helper to safely extract a float value from t, whether it's a tensor or primitive."""
    if torch.is_tensor(t):
        # If it's a batched tensor of timesteps, take the first one
        return t[0].item() if t.ndim > 0 else t.item()
    return float(t)

def constrain_bimanual_mask(action, info, t, debug=False):
    """
    Hard-snapping implementation.
    
    Hard constraints rigidly overwrite the network's prediction. Applying them too 
    early in diffusion destroys the signal. Therefore, we only activate the hard snap 
    in the final 20% of the reverse diffusion process when the trajectory is nearly clean.
    """
    t_val = _get_t_val(t)
    
    # Assuming t normalizes to [0, 1] representing [Data, Noise] (or vice versa based on scheduler).
    # Prevent snapping if the timestep signifies a highly noisy state.
    if t_val < 0.8: 
        return action
        
    return _apply_constraints_core(action, info, t, method='snap', debug=debug)

def apply_workspace_force(action, info, t, debug=False):
    """
    ODE Kinematic restoring force implementation.
    
    Soft constraints act as a continuous directional pull during diffusion. They can be 
    safely applied earlier in the denoising process than hard snaps, gently guiding 
    the vector field toward valid spatial regions.
    """
    t_val = _get_t_val(t)
    
    # Optional: Disable only in the very earliest pure-noise phase (e.g., t < 0.2) 
    # if you find the force destabilizes the initial vector field geometry.
    # if t_val < 0.2: 
    #     return action
        
    return _apply_constraints_core(action, info, t, method='force', debug=debug)

# -----------------------------------------------------------------------------
# Function Registry
# -----------------------------------------------------------------------------
# Used dynamically by the config system to swap constraint behaviors at runtime.
name2func = {
    'identity': identity,
    'bimanual_workspace_snap': constrain_bimanual_mask,
    'bimanual_workspace_force': apply_workspace_force
}