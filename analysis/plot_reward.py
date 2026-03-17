import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Reward Function 1: Original Threshold-Based
# ---------------------------------------------------------
def coverage_alignment_bonus_and_penalty(last_info, action, info, config=None):
    if config is None: config = {}
    if last_info is None: last_info = info

    nc_curr = info['evaluation']['normalised_coverage']
    nc_prev = last_info['evaluation']['normalised_coverage']
    iou_curr = info['evaluation']['max_IoU_to_flattened']
    iou_prev = last_info['evaluation']['max_IoU_to_flattened']

    dNC = nc_curr - nc_prev
    dIoU = iou_curr - iou_prev

    alpha = config.get('alpha', 0.5)
    reward = alpha * dNC + (1 - alpha) * dIoU

    NC_success_thresh = config.get('NC_success_threshold', 0.9)
    IoU_success_thresh = config.get('IoU_success_threshold', 0.8)

    is_success_curr = (nc_curr > NC_success_thresh) and (iou_curr > IoU_success_thresh)
    is_success_prev = (nc_prev > NC_success_thresh) and (iou_prev > IoU_success_thresh)

    if config.get('apply_success_bonus', False):
        if is_success_curr: reward = 1.0

    if config.get('apply_mess_up_penalty', False):
        if is_success_prev and not is_success_curr:
            reward -= config.get('mess_up_penalty_value', 1.0)

    return np.clip(reward, a_min=-1.0, a_max=1.0)

# ---------------------------------------------------------
# Reward Function 2: Smooth Polynomial (Steep at top)
# ---------------------------------------------------------
def smooth_blended_reward(last_info, action, info, config=None):
    if config is None: config = {}
    if last_info is None: last_info = info

    nc_curr = info['evaluation']['normalised_coverage']
    nc_prev = last_info['evaluation']['normalised_coverage']
    iou_curr = info['evaluation']['max_IoU_to_flattened']
    iou_prev = last_info['evaluation']['max_IoU_to_flattened']

    alpha = config.get('alpha', 0.5)
    # Higher p = flatter at the bottom, steeper cliff at the top right
    p = config.get('steepness_power', 4.0) 

    # 1. Base Progress (Delta) - Keeps gradients alive when the garment is messy
    dNC = nc_curr - nc_prev
    dIoU = iou_curr - iou_prev
    progress_reward = alpha * dNC + (1 - alpha) * dIoU

    # 2. Smooth Coupled Success Bonus (State) - Approaches 1.0 ONLY when BOTH are high
    state_bonus_curr = ((nc_curr ** (2 * alpha)) * (iou_curr ** (2 * (1 - alpha)))) ** p
    
    # 3. Smooth "Mess Up" Penalty (Delta of Bonus) - Harsher punishment for dropping from high states
    state_bonus_prev = ((nc_prev ** (2 * alpha)) * (iou_prev ** (2 * (1 - alpha)))) ** p
    drop_penalty = 0
    if state_bonus_curr < state_bonus_prev:
        # Scale the penalty by how far the potential dropped
        mess_up_scale = config.get('mess_up_penalty_value', 1.0)
        drop_penalty = (state_bonus_curr - state_bonus_prev) * mess_up_scale

    # Combine them
    reward = progress_reward + state_bonus_curr + drop_penalty

    return np.clip(reward, a_min=-1.0, a_max=1.0)

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
config = {
    'alpha': 0.5,
    'NC_success_threshold': 0.95,
    'IoU_success_threshold': 0.90,
    'apply_success_bonus': True,
    'apply_mess_up_penalty': True,
    'apply_large_action_penalty_when_success': False, # Ignored for now
    'steepness_power': 3.0 # Controls the curve of the smooth reward
}

# The 3 previous states we want to test (Columns)
prev_states = [
    (0.3, 0.3),   # Small
    (0.7, 0.7),   # Medium
    (0.97, 0.95)  # Success
]

# The 2 reward functions we want to compare (Rows)
reward_functions = [
    ("Original Threshold Reward", coverage_alignment_bonus_and_penalty),
    ("Smooth Polynomial Reward (p=3)", smooth_blended_reward)
]

# ---------------------------------------------------------
# Generate the 2x3 Grid
# ---------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.3, right=0.9) 

nc_curr_range = np.linspace(0, 1, 100)
iou_curr_range = np.linspace(0, 1, 100)
NC_grid, IoU_grid = np.meshgrid(nc_curr_range, iou_curr_range)
fixed_levels = np.linspace(-1.0, 1.0, 100)

contour = None

for row_idx, (func_name, reward_func) in enumerate(reward_functions):
    for col_idx, (nc_prev, iou_prev) in enumerate(prev_states):
        ax = axes[row_idx, col_idx]
        Z = np.zeros_like(NC_grid)
        
        # Calculate rewards across the grid
        for i in range(NC_grid.shape[0]):
            for j in range(NC_grid.shape[1]):
                last_info = {'evaluation': {'normalised_coverage': nc_prev, 'max_IoU_to_flattened': iou_prev}}
                info = {'evaluation': {'normalised_coverage': NC_grid[i, j], 'max_IoU_to_flattened': IoU_grid[i, j]}}
                Z[i, j] = reward_func(last_info, action=None, info=info, config=config)
                
        # Plotting
        contour = ax.contourf(NC_grid, IoU_grid, Z, levels=fixed_levels, cmap='RdYlGn', extend='both')
        
        # Mark the previous state
        ax.scatter(nc_prev, iou_prev, color='black', marker='X', s=120, zorder=5)
        
        # Formatting
        ax.set_title(f"Prev NC={nc_prev} | Prev IoU={iou_prev}", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if row_idx == 1: ax.set_xlabel('Current NC')
        if col_idx == 0: ax.set_ylabel(f"{func_name}\n\nCurrent IoU")

# Add a single shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(contour, cax=cbar_ax, label='Reward')

fig.suptitle("Reward Function Comparison: Threshold vs. Smooth Polynomial", fontsize=16, fontweight='bold', y=0.95)

# Save as PNG
plt.savefig('reward_comparison.png', dpi=300, bbox_inches='tight')
print("Successfully saved 'reward_comparison.png'")