import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Vectorized Smooth Blended Reward Function
# ---------------------------------------------------------
def compute_reward_grid(NC_grid, IoU_grid, nc_prev, iou_prev, action_dist, alpha, p, config):
    dNC = NC_grid - nc_prev
    dIoU = IoU_grid - iou_prev
    progress_reward = alpha * dNC + (1 - alpha) * dIoU

    term1_curr = np.maximum(NC_grid, 1e-8) ** (2 * alpha)
    term2_curr = np.maximum(IoU_grid, 1e-8) ** (2 * (1 - alpha))
    state_bonus_curr = (term1_curr * term2_curr) ** p

    term1_prev = max(nc_prev, 1e-8) ** (2 * alpha)
    term2_prev = max(iou_prev, 1e-8) ** (2 * (1 - alpha))
    state_bonus_prev = (term1_prev * term2_prev) ** p

    potential_diff = state_bonus_curr - state_bonus_prev
    drop_penalty = np.where(potential_diff < 0, potential_diff * config.get('mess_up_penalty_value', 1.0), 0)

    # Smooth Continuous Action Penalty
    action_penalty = 0.0
    if config.get('apply_large_action_penalty_when_success', True):
        drag_threshold = config.get('drag_penalty_threshold', 0.15)
        excess_dist = max(0.0, action_dist - drag_threshold)
        
        action_penalty_value = config.get('action_penalty_value', 3.0) 
        action_penalty = -1.0 * action_penalty_value * state_bonus_prev * excess_dist

    reward = progress_reward + state_bonus_curr + drop_penalty + action_penalty
    return np.clip(reward, -1.0, 1.0)

# ---------------------------------------------------------
# Configuration Setup
# ---------------------------------------------------------
config = {
    'mess_up_penalty_value': 1.0,
    'apply_large_action_penalty_when_success': True,
    'drag_penalty_threshold': 0.15,
    'action_penalty_value': 3.0  
}

nc_curr_range = np.linspace(0, 1, 100)
iou_curr_range = np.linspace(0, 1, 100)
NC_grid, IoU_grid = np.meshgrid(nc_curr_range, iou_curr_range)
fixed_levels = np.linspace(-1.0, 1.0, 100)

# ---------------------------------------------------------
# Plotting Setup
# ---------------------------------------------------------
# Set figure background to white
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20), facecolor='white')
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.88)

variations = [
    np.linspace(0, 0.95, 5),     
    np.linspace(0, 0.9, 5),      
    np.linspace(0, 0.5, 5),      
    np.linspace(0, 1, 5),        
    np.linspace(0, 5, 5)         
]

row_labels = [
    "Prev State\n(Diagonal)", 
    "Prev State\n(IoU Only, NC=0.9)", 
    "Action Distance\nPenalty",
    "Alpha Weight\n(NC vs IoU)", 
    "Steepness Power (p)"
]

# ---------------------------------------------------------
# Generate the Grid
# ---------------------------------------------------------
for row in range(5):
    for col in range(5):
        ax = axes[row, col]
        val = variations[row][col]
        
        nc_prev, iou_prev, alpha, p, action_dist = 0.5, 0.5, 0.5, 4.0, 0.1
        
        if row == 0:
            nc_prev, iou_prev = val, val
            title = f"Prev: ({nc_prev:.2f}, {iou_prev:.2f})"
        elif row == 1:
            nc_prev, iou_prev = 0.9, val
            title = f"Prev: ({nc_prev:.2f}, {iou_prev:.2f})"
        elif row == 2:
            nc_prev, iou_prev = 0.9, 0.9 
            action_dist = val
            title = f"Act Dist: {action_dist:.2f}"
        elif row == 3:
            alpha = val
            title = f"Alpha: {alpha:.2f}"
        elif row == 4:
            p = val
            title = f"Power (p): {p:.2f}"

        Z = compute_reward_grid(NC_grid, IoU_grid, nc_prev, iou_prev, action_dist, alpha, p, config)
        
        # 'magma' colormap stays for colorblind accessibility and contrast
        contour = ax.contourf(NC_grid, IoU_grid, Z, levels=fixed_levels, cmap='magma', extend='both')
        
        # High contrast cyan marker, but with a black edge to stand out against bright areas
        ax.scatter(nc_prev, iou_prev, color='#00FFFF', edgecolors='black', linewidth=1.0, marker='X', s=150, zorder=5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # White subtitle boxes with black text
        ax.text(0.04, 0.96, title, transform=ax.transAxes, 
                fontsize=11, fontweight='bold', va='top', ha='left', color='black',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

        # Standard black labels
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=15, fontweight='bold', labelpad=15, color='black')
            
        if row == 4 and col == 4:
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_xlabel('Current NC', fontsize=15, fontweight='bold', color='black')
            ax.set_ylabel('Current IoU', fontsize=15, fontweight='bold', color='black')
            
            ax.tick_params(axis='both', which='major', labelsize=15, colors='black')

# Add Shared Colorbar
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
cbar = fig.colorbar(contour, cax=cbar_ax)

cbar.set_label('Reward', fontsize=18, fontweight='bold', labelpad=15, color='black')
cbar.ax.tick_params(labelsize=15, colors='black')
cbar.outline.set_visible(False) 

fig.suptitle("Comprehensive Smooth Reward Sensitivity Analysis", fontsize=26, fontweight='bold', y=0.94, color='black')

# Save with standard white facecolor
plt.savefig('reward_sensitivity_5x5.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Successfully saved 'reward_sensitivity_5x5.png'")