import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Set to True if your network outputs [Y, X] (Row, Col). 
# Set to False if your network outputs [X, Y] (Width, Height).
IS_YX_ACTION_SPACE = False 

# -----------------------------------------------------------------------------
# Helper Functions for Mathematics & Assignments
# -----------------------------------------------------------------------------

def _snap_to_nearest(points_bht, valid_points):
    if valid_points is None or valid_points.shape[0] == 0:
        return points_bht
    B, H, D = points_bht.shape
    flat_points = points_bht.reshape(-1, 2)
    dists = torch.cdist(flat_points, valid_points)
    min_indices = torch.argmin(dists, dim=1)
    return valid_points[min_indices].reshape(B, H, 2)

def _get_restoring_force(points_bht, valid_points):
    if valid_points is None or valid_points.shape[0] == 0:
        return torch.zeros_like(points_bht)
    nearest_points = _snap_to_nearest(points_bht, valid_points)
    return nearest_points - points_bht

def _get_min_dist(points_bht, valid_points):
    if valid_points is None or valid_points.shape[0] == 0:
        return float('inf')
    flat_points = points_bht.reshape(-1, 2)
    dists = torch.cdist(flat_points, valid_points)
    return torch.min(dists, dim=1)[0].mean().item()

def _get_dynamic_assignments(action_b, entities, valid_masks):
    assignments = {} 
    if len(entities) == 2 and 0 in valid_masks and 1 in valid_masks:
        e0_pick = action_b[:, :, entities[0]['pick']]
        e1_pick = action_b[:, :, entities[1]['pick']]
        
        d_e0_m0 = _get_min_dist(e0_pick, valid_masks[0])
        d_e0_m1 = _get_min_dist(e0_pick, valid_masks[1])
        d_e1_m0 = _get_min_dist(e1_pick, valid_masks[0])
        d_e1_m1 = _get_min_dist(e1_pick, valid_masks[1])
        
        if (d_e0_m0 + d_e1_m1) <= (d_e0_m1 + d_e1_m0):
            assignments = {0: 0, 1: 1}
        else:
            assignments = {0: 1, 1: 0}
            
    elif len(entities) == 1 and len(valid_masks) > 0:
        e0_pick = action_b[:, :, entities[0]['pick']]
        best_mask = min(valid_masks.keys(), key=lambda m: _get_min_dist(e0_pick, valid_masks[m]))
        assignments = {0: best_mask}
        
    return assignments

# -----------------------------------------------------------------------------
# Debug Visualization
# -----------------------------------------------------------------------------

def _visualize_debug(original_action, new_action, obs, entities, assignments, prim_name, t, method):
    save_dir = "tmp/debug_magpie"
    os.makedirs(save_dir, exist_ok=True)

    rgb = obs.get('rgb', None)
    if rgb is None: return 
        
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

    if m0 is not None: ax.imshow(np.ma.masked_where(m0 < 0.5, m0), cmap='Blues', alpha=0.4)
    if m1 is not None: ax.imshow(np.ma.masked_where(m1 < 0.5, m1), cmap='Reds', alpha=0.4)

    def to_pix(uv_coords):
        dim0, dim1 = uv_coords[:, 0], uv_coords[:, 1]
        if IS_YX_ACTION_SPACE:
            y, x = (dim0 + 1) * H / 2, (dim1 + 1) * W / 2
        else:
            x, y = (dim0 + 1) * W / 2, (dim1 + 1) * H / 2
        return x, y

    orig_act = original_action[0, 0].detach().cpu().numpy()
    new_act = new_action[0, 0].detach().cpu().numpy()
    colors = {0: 'blue', 1: 'red'}
    
    # Dynamic prefix for the legend
    lbl_prefix = "Snap" if method == 'snap' else "Force"
    
    for e_idx, m_idx in assignments.items():
        ent = entities[e_idx]
        c = colors.get(m_idx, 'green')

        # Plot PICK
        if len(ent['pick']) == 2:
            pu, pv = ent['pick']
            ox, oy = to_pix(np.array([[orig_act[pu], orig_act[pv]]]))
            nx, ny = to_pix(np.array([[new_act[pu], new_act[pv]]]))
            
            ax.plot(ox, oy, marker='x', color='yellow', markersize=8, label='Orig Pick' if e_idx==0 else "")
            ax.plot(nx, ny, marker='o', color=c, markersize=8, markeredgecolor='white', label=f'{lbl_prefix} Pick (R{m_idx})')
            ax.annotate("", xy=(nx[0], ny[0]), xytext=(ox[0], oy[0]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8))

        # Plot PLACE
        if len(ent['place']) == 2:
            pu, pv = ent['place']
            ox, oy = to_pix(np.array([[orig_act[pu], orig_act[pv]]]))
            nx, ny = to_pix(np.array([[new_act[pu], new_act[pv]]]))
            
            ax.plot(ox, oy, marker='x', color='orange', markersize=8, label='Orig Place' if e_idx==0 else "")
            ax.plot(nx, ny, marker='^', color=c, markersize=8, markeredgecolor='white', label=f'{lbl_prefix} Place (R{m_idx})')
            ax.annotate("", xy=(nx[0], ny[0]), xytext=(ox[0], oy[0]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8))

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
    return action

def _apply_constraints_core(action, info, t, method, debug=False):
    device = action.device
    obs = info.get('observation', {})
    original_action = action.clone() 
    
    prim_name = info.get('prim_name', 'norm-pixel-dual-pick-and-place')
    
    entities = []
    if 'pick-and-fling' in prim_name:
        entities.append({'pick': [0, 1], 'place': []})
        entities.append({'pick': [2, 3], 'place': []})
    elif 'dual-pick-and-place' in prim_name:
        entities.append({'pick': [0, 1], 'place': [4, 5]})
        entities.append({'pick': [2, 3], 'place': [6, 7]})
    elif 'single-pick-and-place' in prim_name:
        entities.append({'pick': [0, 1], 'place': [2, 3]})
    elif 'no-operation' in prim_name:
        return action

    # --- THE CRITICAL FIX: PROPER (ROWS, COLS) -> (Y, X) MAPPING ---
    valid_masks = {}
    for m_idx, key in enumerate(['robot0_mask', 'robot1_mask']):
        if key in obs:
            mask = obs[key]
            if mask.ndim == 3: mask = mask.squeeze(-1)
            
            # np.where returns (rows, columns) = (Axis 0, Axis 1)
            rows, cols = np.where(mask > 0) 
            
            if len(rows) > 0:
                H, W = mask.shape
                norm_x = (cols / float(W)) * 2 - 1.0
                norm_y = (rows / float(H)) * 2 - 1.0
                
                # Stack coordinates based on the network's output format
                if IS_YX_ACTION_SPACE:
                    valid_pts = np.stack([norm_y, norm_x], axis=1)
                else:
                    valid_pts = np.stack([norm_x, norm_y], axis=1)
                    
                valid_masks[m_idx] = torch.tensor(valid_pts, dtype=torch.float32, device=device)

    if not valid_masks:
        return action

    B = action.shape[0]
    force_vector = torch.zeros_like(action) if method == 'force' else None
    debug_assignments_b0 = {}

    for b in range(B):
        assignments = _get_dynamic_assignments(action[b:b+1], entities, valid_masks)
        
        if b == 0:
            debug_assignments_b0 = assignments 
            
        for e_idx, m_idx in assignments.items():
            ent = entities[e_idx]
            pts = valid_masks[m_idx]
            
            if method == 'snap':
                action[b:b+1, :, ent['pick']] = _snap_to_nearest(action[b:b+1, :, ent['pick']], pts)
            elif method == 'force':
                force_vector[b:b+1, :, ent['pick']] = _get_restoring_force(action[b:b+1, :, ent['pick']], pts)
                
            if len(ent['place']) > 0:
                if method == 'snap':
                    action[b:b+1, :, ent['place']] = _snap_to_nearest(action[b:b+1, :, ent['place']], pts)
                elif method == 'force':
                    force_vector[b:b+1, :, ent['place']] = _get_restoring_force(action[b:b+1, :, ent['place']], pts)

    if method == 'force':
        eta_dt = 0.5 
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
    """Hard-snapping implementation."""
    t_val = _get_t_val(t)
    
    # Assuming t goes 0 -> 1 (Noise -> Data). 
    # Only apply the hard snap in the final 20% of the trajectory.
    # (Adjust this logic if your solver integrates 1 -> 0 instead)
    if t_val < 0.8: 
        return action
        
    return _apply_constraints_core(action, info, t, method='snap', debug=debug)

def apply_workspace_force(action, info, t, debug=False):
    """ODE Kinematic restoring force implementation (MEGPIE standard)."""
    t_val = _get_t_val(t)
    
    # ODE forces are typically applied earlier and softer than hard snaps.
    # Here we disable it only in the very earliest pure-noise phase (e.g. t < 0.2)
    # if you find it destabilizes the initial vector field.
    # if t_val < 0.2: 
    #     return action
        
    return _apply_constraints_core(action, info, t, method='force', debug=debug)

# -----------------------------------------------------------------------------
# Function Registry
# -----------------------------------------------------------------------------
name2func = {
    'identity': identity,
    'bimanual_workspace_snap': constrain_bimanual_mask,
    'bimanual_workspace_force': apply_workspace_force
}