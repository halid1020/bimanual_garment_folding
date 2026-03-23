import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm  # <--- NEW: Import the progress bar library

# --- Configuration ---
data_dir = '/media/hcv530/T7/garment_folding_data/real_world/'
method_folder = 'planet_clothpick_single_picker_single_primitive_multi_longsleeve_flattening_longsleeve_all_garment_5k_eps'

# Make sure this matches your actual folder name (eval_checkpoint_-2 vs eval_checkpoints_-2)
eval_dir = os.path.join(data_dir, method_folder, 'eval_checkpoint_-2') 

# Directory to save the output visualizations
output_dir = os.path.join(eval_dir, 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# Find all episode folders
episode_folders = [f for f in os.listdir(eval_dir) if f.startswith('episode_')]
# Sort them numerically so episode_2 comes before episode_10
episode_folders.sort(key=lambda x: int(x.split('_')[1]))

print(f"Found {len(episode_folders)} episodes. Generating visualizations...")

# ---> NEW: Wrap the list in tqdm() to generate the progress bar <---
for ep_folder in tqdm(episode_folders, desc="Processing Episodes", unit="ep"):
    ep_path = os.path.join(eval_dir, ep_folder)
    
    # Create a figure with 4 rows and 11 columns
    fig, axes = plt.subplots(4, 11, figsize=(22, 8))
    fig.suptitle(f'PlaNet-ClothPick | {ep_folder}', fontsize=16, fontweight='bold')
    
    # We loop up to 22 steps (0 to 21) to fill 2 rows of 11
    for j in range(22):
        # Determine grid placement
        if j < 11:
            row_raw, row_mask, col = 0, 1, j
        else:
            row_raw, row_mask, col = 2, 3, j - 11
            
        ax_raw = axes[row_raw, col]
        ax_mask = axes[row_mask, col]
        
        # Turn off axis borders/ticks for a cleaner look
        ax_raw.axis('off')
        ax_mask.axis('off')
        
        step_dir = os.path.join(ep_path, f'step_{j}')
        raw_path = os.path.join(step_dir, 'raw_input_obs.npy')
        mask_path = os.path.join(step_dir, 'mask.png')
        
        # Check if the step actually exists (episodes might terminate early)
        if os.path.exists(raw_path) and os.path.exists(mask_path):
            try:
                # --- Process Raw Observation ---
                obs = np.load(raw_path)
                
                # NOTE: If your .npy array is Channel-First (e.g., shape is 3 x 256 x 256 like PyTorch),
                # uncomment the next line to convert it to Channel-Last (256 x 256 x 3) for Matplotlib:
                # obs = np.transpose(obs, (1, 2, 0))
                
                # Normalize float arrays to [0, 1] if they aren't already
                if obs.dtype in [np.float32, np.float64]:
                    obs = np.clip(obs, 0, 1)
                    
                ax_raw.imshow(obs)
                ax_raw.set_title(f'Step {j}')
                
                # --- Process Mask ---
                mask = Image.open(mask_path)
                ax_mask.imshow(mask, cmap='gray') # Displaying mask in grayscale
                
            except Exception as e:
                # Using tqdm.write instead of print so it doesn't break the progress bar visual
                tqdm.write(f"Error loading images for {ep_folder} Step {j}: {e}")
                ax_raw.set_title(f'Step {j}\n(Error)')
        else:
            # Add a subtle label if the episode ended before reaching this step
            # Only add it to the first missing step to keep the grid clean
            if j == 0 or j == 11 or not os.path.exists(os.path.join(ep_path, f'step_{j-1}')): 
                 ax_raw.set_title(f'Step {j}\n(Not Reached)')

    # Tighten layout so titles and images don't overlap
    plt.tight_layout()
    # Adjust top slightly to make room for the main suptitle
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.1) 
    
    save_path = os.path.join(output_dir, f'{ep_folder}_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory

print(f"\nDone! All visualizations saved to: {output_dir}")