import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Vectorized Linear Alignment Reward Function (No Bonuses/Penalties)
# ---------------------------------------------------------
def compute_reward_grid(NC_grid, IoU_grid, nc_prev, iou_prev, alpha):
    """
    Computes purely the linear progression reward:
    alpha * dNC + (1 - alpha) * dIoU
    """
    dNC = NC_grid - nc_prev
    dIoU = IoU_grid - iou_prev
    
    reward = alpha * dNC + (1 - alpha) * dIoU
    
    # Clamp rewards between -1 and 1
    return np.clip(reward, -1.0, 1.0)

# ---------------------------------------------------------
# Configuration Setup
# ---------------------------------------------------------
nc_curr_range = np.linspace(0, 1, 100)
iou_curr_range = np.linspace(0, 1, 100)
NC_grid, IoU_grid = np.meshgrid(nc_curr_range, iou_curr_range)
fixed_levels = np.linspace(-1.0, 1.0, 100)

# ---------------------------------------------------------
# Plotting Setup
# ---------------------------------------------------------
# Set figure background to white, now 3 rows instead of 5
ncol = 4
fig, axes = plt.subplots(nrows=3, ncols=ncol, figsize=(ncol*4, 12), facecolor='white')

# Adjusted margins to make room for the top colorbar and expand the right side
plt.subplots_adjust(wspace=0.05, hspace=0.05, right=0.95, top=0.88)



variations = [
    np.linspace(0.1, 0.95, ncol),     # Prev State (Diagonal)
    np.linspace(0.1, 0.9, ncol),      # Prev State (IoU Only, NC=0.9)
    np.linspace(0, 1, ncol),          # Alpha Weight (NC vs IoU)
]

row_labels = [
    "Move Prev NC \n & IoU Diagonally", 
    "Move Prev IoU \n Fix Prev NC=0.9", 
    "Alpha Weight\nPrev (0.5, 0.5)"
]

# ---------------------------------------------------------
# Generate the Grid
# ---------------------------------------------------------
for row in range(3):
    for col in range(ncol):
        ax = axes[row, col]
        val = variations[row][col]
        
        # Defaults
        nc_prev, iou_prev, alpha = 0.5, 0.5, 0.5
        
        if row == 0:
            nc_prev, iou_prev = val, val
            title = f"Prev: ({nc_prev:.2f}, {iou_prev:.2f})"
        elif row == 1:
            nc_prev, iou_prev = 0.9, val
            title = f"Prev: ({nc_prev:.2f}, {iou_prev:.2f})"
        elif row == 2:
            alpha = val
            title = f"Alpha: {alpha:.2f}"

        Z = compute_reward_grid(NC_grid, IoU_grid, nc_prev, iou_prev, alpha)
        
        # 'magma' colormap stays for colorblind accessibility and contrast
        contour = ax.contourf(NC_grid, IoU_grid, Z, levels=fixed_levels, cmap='magma', extend='both')
        
        # High contrast cyan marker, but with a black edge to stand out against bright areas
        ax.scatter(nc_prev, iou_prev, color='#00FFFF', edgecolors='black', linewidth=2.0, marker='X', s=200, zorder=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # White subtitle boxes with black text
        ax.text(0.04, 0.96, title, transform=ax.transAxes, 
                fontsize=20, fontweight='bold', va='top', ha='left', color='black',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

        # Standard black labels for the leftmost column
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=20, fontweight='bold', labelpad=15, color='black')
            
        # Put axis labels and ticks on the top-rightmost plot
        if row == 0 and col == 4:
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            
            # Y-axis to the right
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            
            # X-axis to the top
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            
            ax.set_xlabel('Current NC', fontsize=20, fontweight='bold', color='black', labelpad=10)
            ax.set_ylabel('Current IoU', fontsize=20, fontweight='bold', color='black', labelpad=10)
            
            ax.tick_params(axis='both', which='major', labelsize=20, colors='black')

# Add Shared Colorbar (Horizontally on Top)
# Coordinates: [left, bottom, width, height]
cbar_ax = fig.add_axes([0.25, 0.94, 0.5, 0.025]) 
cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', format='%.1f')

cbar.set_label('Reward', fontsize=20, fontweight='bold', labelpad=10, color='black')
cbar.ax.xaxis.set_label_position('top') # Moves the "Reward" text above the bar
cbar.ax.tick_params(labelsize=20, colors='black')
cbar.outline.set_visible(False) 

# Save with standard white facecolor
plt.savefig('./analysis/reward_sensitivity_3x5.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Successfully saved './analysis/reward_sensitivity_3x5.png'")