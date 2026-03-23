import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
data_dir = '/media/hcv530/T7/garment_folding_data/real_world/'
method_folder = 'planet_clothpick_single_picker_single_primitive_multi_longsleeve_flattening_longsleeve_all_garment_5k_eps'
eval_dir = os.path.join(data_dir, method_folder, 'eval_checkpoint_-2') 

# Directory to save the output visualizations
output_dir = os.path.join(eval_dir, 'action_visualizations')
os.makedirs(output_dir, exist_ok=True)

# Find and sort all episode folders
episode_folders = [f for f in os.listdir(eval_dir) if f.startswith('episode_')]
episode_folders.sort(key=lambda x: int(x.split('_')[1]))

print(f"Found {len(episode_folders)} episodes. Generating action trajectory grids...")

# Helper function to convert [-1, 1] to pixel coordinates [0, width/height]
def denormalize_coords(coord_norm, width, height):
    x_norm, y_norm = coord_norm[0], coord_norm[1]
    
    # Standard mapping: -1 -> 0, 1 -> width/height
    x_pix = ((x_norm + 1.0) / 2.0) * width
    y_pix = ((y_norm + 1.0) / 2.0) * height
    
    # NOTE: If your Y-axis is inverted (i.e., -1 is bottom, 1 is top), 
    # uncomment the following line:
    # y_pix = height - y_pix 
    
    return x_pix, y_pix

for ep_folder in tqdm(episode_folders, desc="Processing Action Grids", unit="ep"):
    ep_path = os.path.join(eval_dir, ep_folder)
    
    # Create a figure with 2 rows and 11 columns
    fig, axes = plt.subplots(2, 11, figsize=(22, 5))
    fig.suptitle(f'Action Trajectory | {ep_folder}', fontsize=16, fontweight='bold')
    
    for j in range(22):
        row = 0 if j < 11 else 1
        col = j if j < 11 else j - 11
            
        ax = axes[row, col]
        ax.axis('off') # Clean look
        
        step_dir = os.path.join(ep_path, f'step_{j}')
        rgb_path = os.path.join(step_dir, 'rgb.png')
        action_path = os.path.join(step_dir, 'action.json')
        
        if os.path.exists(rgb_path) and os.path.exists(action_path):
            try:
                # Load Image
                img = Image.open(rgb_path)
                width, height = img.size
                ax.imshow(img)
                ax.set_title(f'Step {j}')
                
                # Load Action
                with open(action_path, 'r') as f:
                    act_data = json.load(f)
                
                # --- UPDATED PARSING LOGIC ---
                # act_data is a list: [pick_x, pick_y, place_x, place_y]
                if len(act_data) >= 4:
                    pick_norm = [act_data[0], act_data[1]]
                    place_norm = [act_data[2], act_data[3]]
                else:
                    pick_norm = [0, 0]
                    place_norm = [0, 0]
                    tqdm.write(f"Warning: Unexpected action format in {action_path}")
                
                # Convert to pixel space
                px, py = denormalize_coords(pick_norm, width, height)
                lx, ly = denormalize_coords(place_norm, width, height)
                
                # Plot Pick (Green Dot) and Place (Red X)
                ax.plot(px, py, 'go', markersize=6, markeredgecolor='white') # Green circle
                ax.plot(lx, ly, 'rX', markersize=6, markeredgecolor='white') # Red X
                
                # Draw an arrow from Pick to Place
                arrow = patches.FancyArrowPatch(
                    (px, py), (lx, ly),
                    arrowstyle='-|>', mutation_scale=15, 
                    color='cyan', linewidth=2, alpha=0.8
                )
                ax.add_patch(arrow)
                
            except Exception as e:
                tqdm.write(f"Error processing {ep_folder} Step {j}: {e}")
                ax.set_title(f'Step {j}\n(Error)')
        else:
            # Mark missing/unreached steps
            if j == 0 or not os.path.exists(os.path.join(ep_path, f'step_{j-1}')): 
                 ax.set_title(f'Step {j}\n(Not Reached)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.1, wspace=0.1) 
    
    save_path = os.path.join(output_dir, f'{ep_folder}_actions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"\nDone! Action visualizations saved to: {output_dir}")