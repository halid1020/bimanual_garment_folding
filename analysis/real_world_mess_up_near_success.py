import os
import json
from tqdm import tqdm

# --- Configuration ---
data_dir = '/media/hcv530/T7/garment_folding_data/real_world/'
method_folder = 'lagarnet_single_picker_single_primitive_multi_longsleeve_flattening_sanity_check'
eval_dir = os.path.join(data_dir, method_folder, 'eval_checkpoint_-2') 

# Find all episode folders
all_episode_folders = [f for f in os.listdir(eval_dir) if f.startswith('episode_')]

# --- Filter for episodes 0 through 39 inclusive ---
episode_folders = []
for f in all_episode_folders:
    try:
        ep_num = int(f.split('_')[1])
        if 0 <= ep_num <= 39:
            episode_folders.append(f)
    except ValueError:
        pass # Skip if the folder doesn't have a valid integer after 'episode_'

# Sort them numerically for clean progress tracking
episode_folders.sort(key=lambda x: int(x.split('_')[1]))

# Initialize Counters and Lists
total_near_successes = 0
degraded_after_near_success = 0
failed_episodes = []  # NEW: List to track failed episode IDs

print(f"Analyzing {len(episode_folders)} episodes (0-39)...")

for ep_folder in tqdm(episode_folders, desc="Analyzing Steps"):
    ep_path = os.path.join(eval_dir, ep_folder)
    ep_num = int(ep_folder.split('_')[1])
    
    # Dynamically find how many steps are in this episode to avoid out-of-bounds
    step_folders = [f for f in os.listdir(ep_path) if f.startswith('step_')]
    num_steps = len(step_folders)
    
    episode_success = False  # NEW: Flag to track if this specific episode succeeds
    
    for j in range(num_steps):
        info_path = os.path.join(ep_path, f'step_{j}', 'info.json')
        if not os.path.exists(info_path):
            continue
            
        with open(info_path, 'r') as f:
            info_data = json.load(f)
            
        # Extract metrics (Using .get() safely)
        eval_data = info_data.get('evaluation', {})
        nc = eval_data.get('normalised_coverage', 0)
        iou = eval_data.get('max_IoU_to_flattened', 0)
        
        # --- NEW: Check for overall episode success ---
        # Using 0.85 and 0.75 to match the 0-1 scale of the json data
        if nc > 0.85 and iou > 0.75:
            episode_success = True
        
        # 1. Check if the current step is a "Near Success"
        if nc > 0.8 and iou > 0.7:
            total_near_successes += 1
            
            # 2. Check the NEXT step (if it exists)
            next_info_path = os.path.join(ep_path, f'step_{j+1}', 'info.json')
            if os.path.exists(next_info_path):
                with open(next_info_path, 'r') as f:
                    next_info_data = json.load(f)
                
                next_eval_data = next_info_data.get('evaluation', {})
                next_nc = next_eval_data.get('normalised_coverage', 0)
                next_iou = next_eval_data.get('max_IoU_to_flattened', 0)
                
                # Check if performance degraded
                if next_nc < nc or next_iou < iou:
                    degraded_after_near_success += 1

    # --- NEW: If the episode never succeeded, add it to the failed list ---
    if not episode_success:
        failed_episodes.append(ep_num)

# --- Print Results ---
print("\n" + "="*50)
print("🎯 PERFORMANCE ANALYSIS RESULTS (EPISODES 0-39)")
print("="*50)
print(f"Total 'Near Success' steps (NC > 0.8 and IoU > 0.7) : {total_near_successes}")
print(f"Steps where performance DROPPED on the very next step: {degraded_after_near_success}")
if total_near_successes > 0:
    degradation_rate = (degraded_after_near_success / total_near_successes) * 100
    print(f"Degradation Rate from Near-Success states          : {degradation_rate:.1f}%")
print("-" * 50)
print(f"Total Failed Episodes (Never hit NC>0.85 & IoU>0.75) : {len(failed_episodes)}")
print(f"Failed Episode IDs                                 : {failed_episodes}")
print("="*50)