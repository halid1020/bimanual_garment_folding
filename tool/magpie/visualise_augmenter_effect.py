"""
Actoris-Harena DataLoader Augmentation Visualizer (Dynamic Multi-Batch Grid).

This script fetches multiple training batches, applies augmentations, and outputs a 
single master grid. It processes 'N' batches of size 'M', arranging samples into 
a visual grid. Each sample is a 2x2 block showing:
  [ Original RGB ] [ Original Target ]
  [ Augmented RGB] [ Augmented Target]
"""

import os
import argparse
import copy
import yaml
import numpy as np
import torch
import cv2
from dotmap import DotMap

# Import your dataset and augmentor
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
from data_augmentation.pixel_based_multi_primitive_data_augmenter_for_diffusion import PixelBasedMultiPrimitiveDataAugmenterForDiffusion
from controllers.magpie.hindsight_dataset import HindsightDataset

# Import your existing drawing utilities
from tool.magpie.visualise_data import (
    format_image, draw_keypoints, decode_action_vector, draw_action
)

# ==========================================
# Base Dataset Configuration
# ==========================================
DATASET_CONFIG = {
    "data_path": "PLACEHOLDER_UPDATED_IN_MAIN",
    "data_dir": "./data/datasets",
    "split_ratios": [0.0, 0.05, 0.95],
    "seq_length": 1,
    "io_mode": "r", 
    "cache_in_memory": True,
    "obs_config": {
        "mask": {"shape": [128, 128, 1], "output_key": "mask"},
        "depth": {"shape": [128, 128, 1], "output_key": "depth"},
        "rgb": {"shape": [128, 128, 3], "output_key": "rgb"},
        "semkey_norm_pixel": {"shape": [15, 2], "output_key": "semkey_norm_pixel"},
        "goal_rgb": {"shape": [128, 128, 3], "output_key": "goal_rgb"},
        "goal_depth": {"shape": [128, 128, 1], "output_key": "goal_depth"},
        "goal_mask": {"shape": [128, 128, 1], "output_key": "goal_mask"},
        "flattened_semkey_norm_pixel": {"shape": [15, 2], "output_key": "flattened_goal_semkey_norm_pixel"}
    },
    "act_config": {
        "default": {"shape": [9], "output_key": "default"}
    }
}

# ==========================================
# Custom Datasets for Distance Tracking
# ==========================================
class DistanceAwareTrajectoryDataset(TrajectoryDataset):
    """
    Standard dataset wrapper that calculates how many steps away 
    the end of the trajectory (default target) is.
    """
    def __getitem__(self, idx: int):
        ret = super().__getitem__(idx)
        
        idx_offset = idx + self.start_sample
        if self.whole_trajectory:
            traj_idx = idx_offset
            start_idx = self.traj_starts[traj_idx]
        elif self.cross_trajectory:
            start_idx = idx_offset if self.return_trj_last else self.valid_indices[idx_offset]
            traj_idx = np.searchsorted(self.traj_starts, start_idx, side='right') - 1
        else:
            traj_idx, start_idx = self.flat_ranges[idx_offset]
            
        effective_traj_end = self.traj_starts[traj_idx] + self.traj_lengths[traj_idx] - 2
        dist = max(0, effective_traj_end - start_idx)
        
        ret['goal_distance'] = np.array(dist, dtype=np.int32)
        return ret

# ==========================================
# Helpers
# ==========================================
def map_primitive_name(raw_name):
    """
    Maps raw primitive strings from the decoder to the requested formatted names.
    """
    name_lower = str(raw_name).lower()
    if 'fling' in name_lower:
        return "Pick and Fling"
    elif 'bimanual' in name_lower or 'dual' in name_lower:
        return "Dual-Arm Pick and Place"
    elif 'pick' in name_lower or 'place' in name_lower:
        return "Single-Arm Pick and Place"
    else:
        return "No Operation"

def draw_bold_text(img, text, position=(15, 35), text_color=(255, 255, 255), align='top-left'):
    """
    Draws custom bigger, bolder text with NO background, using a subtle stroke 
    so it remains readable against both dark and light image content.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0       # Bigger text
    thickness = 2     # Bolder line
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    if align == 'bottom-right':
        x = img.shape[1] - tw - 15
        y = img.shape[0] - 15
    elif align == 'bottom-left':
        x = 15
        y = img.shape[0] - 15
    else:
        x, y = position
        
    # Draw dark outline/stroke for readability without a background box
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # Draw actual text
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

def unformat_tensor_image(img_tensor: torch.Tensor, is_normalized: bool = False) -> np.ndarray:
    """
    Converts a PyTorch image tensor (C, H, W) back to a numpy array (H, W, C).
    """
    img_np = img_tensor.detach().cpu().numpy()
    if img_np.shape[0] == 3:  
        img_np = np.transpose(img_np, (1, 2, 0))
        
    if is_normalized or img_np.max() <= 1.0:
        img_np = (img_np * 255.0)
        
    return img_np.clip(0, 255).astype(np.uint8)

# ==========================================
# Main Execution
# ==========================================
def main(args):
    # 1. Update Global Config with CLI Arguments
    DATASET_CONFIG['data_path'] = args.data_path

    # 2. Load Augmentor Configuration via DotMap
    print(f"Loading Augmentor Config from: {args.aug_config_path}")
    with open(args.aug_config_path, 'r') as f:
        raw_aug_config = yaml.safe_load(f)
    aug_config = DotMap(raw_aug_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Initialize Dataset and DataLoader
    print(f"Targeting dataset path: {args.data_path}")
    if args.use_hindsight:
        print("Initializing HindsightDataset...")
        dataset = HindsightDataset(**DATASET_CONFIG, sample_mode='train')
    else:
        print("Initializing DistanceAwareTrajectoryDataset...")
        dataset = DistanceAwareTrajectoryDataset(**DATASET_CONFIG, sample_mode='train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0 
    )

    print("Initializing Augmentor...")
    augmentor = PixelBasedMultiPrimitiveDataAugmenterForDiffusion(config=aug_config)

    # 4. Process Multiple Batches
    img_size = 512
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    all_comp_blocks = []
    dataloader_iter = iter(dataloader)
    total_samples = 0

    for batch_idx in range(args.num_batches):
        print(f"\n--- Fetching Batch {batch_idx + 1}/{args.num_batches} ---")
        try:
            raw_batch = next(dataloader_iter)
        except StopIteration:
            print("DataLoader ran out of data!")
            break
        
        # 5. Replicate MagpieTrainer Batch Restructuring
        obs = raw_batch['observation']
        action = raw_batch['action']['default']
        
        formatted_batch = {k: v for k, v in obs.items()}
        formatted_batch['action'] = action.reshape(*action.shape[:2], -1)

        orig_batch = copy.deepcopy(formatted_batch)

        # 6. Apply Augmentations
        print("Applying Augmentations...")
        aug_batch = augmentor(formatted_batch, train=True, device=device)

        # 7. Visualization Generation Loop for Current Batch
        B = orig_batch['rgb'].shape[0]
        
        for b in range(B):
            total_samples += 1
            print(f"Processing Sample {total_samples} (Batch {batch_idx + 1}, Item {b+1}/{B})...")
            
            t = 0 # Since seq_length=1, there is only 1 action transition per sample
            step_distance = raw_batch['goal_distance'][b].item()
            
            # --- Extract Original ---
            orig_rgb_raw = unformat_tensor_image(orig_batch['rgb'][b, t], is_normalized=False)
            orig_goal_raw = unformat_tensor_image(orig_batch['goal_rgb'][b, t], is_normalized=False)
            
            img_cur_orig = format_image(orig_rgb_raw, img_size)
            img_goal_orig = format_image(orig_goal_raw, img_size)
            
            orig_semkey = orig_batch['semkey_norm_pixel'][b, t].numpy()
            goal_key = 'flattened_goal_semkey_norm_pixel' if 'flattened_goal_semkey_norm_pixel' in orig_batch else 'flattened_semkey_norm_pixel'
            orig_goal_semkey = orig_batch[goal_key][b, t].numpy()
            orig_act = orig_batch['action'][b, t].numpy()

            if args.draw_keypoints:
                draw_keypoints(img_cur_orig, orig_semkey)
                draw_keypoints(img_goal_orig, orig_goal_semkey)

            prim_name_orig, params_orig = decode_action_vector(orig_act)
            draw_action(img_cur_orig, prim_name_orig, params_orig) 
            
            # --- Extract Augmented ---
            aug_rgb_raw = unformat_tensor_image(aug_batch['rgb'][b, t], is_normalized=True)
            aug_goal_raw = unformat_tensor_image(aug_batch['goal_rgb'][b, t], is_normalized=True)
            
            img_cur_aug = format_image(aug_rgb_raw, img_size)
            img_goal_aug = format_image(aug_goal_raw, img_size)
            
            aug_semkey = aug_batch['semkey_norm_pixel'][b, t].detach().cpu().numpy()
            aug_goal_semkey = aug_batch[goal_key][b, t].detach().cpu().numpy()
            aug_act = aug_batch['action'][b, t].detach().cpu().numpy()

            if args.draw_keypoints:
                draw_keypoints(img_cur_aug, aug_semkey)
                draw_keypoints(img_goal_aug, aug_goal_semkey)
                
            prim_name_aug, params_aug = decode_action_vector(aug_act)
            draw_action(img_cur_aug, prim_name_aug, params_aug)
            
            # --- Apply Custom Bold Text Labels to ALL Samples ---
            # Use "Actual Target" if distance is 0, otherwise show the distance
            dist_str = f"Dist: {step_distance}" if step_distance > 0 else "Actual Target"

            draw_bold_text(img_cur_orig, "Original RGB", text_color=(255, 255, 255))
            draw_bold_text(img_goal_orig, f"Original Target ({dist_str})", text_color=(255, 255, 255))
            
            draw_bold_text(img_cur_aug, "Augmented RGB", text_color=(100, 255, 100))
            draw_bold_text(img_goal_aug, f"Augmented Target ({dist_str})", text_color=(100, 255, 100))
            
            # Place requested primitive text on the bottom-left corner of Augmented RGB
            mapped_prim_name = map_primitive_name(prim_name_aug)
            draw_bold_text(img_cur_aug, mapped_prim_name, text_color=(200, 200, 255), align='bottom-left')

            # --- Assemble 2x2 Block ---
            top_row = cv2.hconcat([img_cur_orig, img_goal_orig])
            bottom_row = cv2.hconcat([img_cur_aug, img_goal_aug])
            comp_block = cv2.vconcat([top_row, bottom_row])
            
            # Add a visual border to separate samples clearly
            border_size = 10
            sample_bordered = cv2.copyMakeBorder(
                comp_block, border_size, border_size, border_size, border_size, 
                cv2.BORDER_CONSTANT, value=[100, 100, 100]
            )
            
            all_comp_blocks.append(sample_bordered)

    # 8. Final Master Concatenation (Stitching the accumulated batches)
    if all_comp_blocks:
        cols_per_row = args.grid_cols if args.grid_cols > 0 else args.batch_size
        print(f"\nStitching final master grid using {cols_per_row} columns...")
        master_rows = []
        
        for i in range(0, len(all_comp_blocks), cols_per_row):
            row_blocks = all_comp_blocks[i : i + cols_per_row]
            
            # Pad with blank images if the final row has fewer samples
            while len(row_blocks) < cols_per_row:
                blank = np.zeros_like(all_comp_blocks[0])
                row_blocks.append(blank)
                
            row_img = cv2.hconcat(row_blocks)
            master_rows.append(row_img)
            
        master_grid = cv2.vconcat(master_rows)
        
        output_file = f"{os.path.splitext(args.output_path)[0]}_master_multi_batch.png"
        cv2.imwrite(output_file, master_grid)
        print(f"-> Successfully saved Master Comparison Grid to: {output_file}")
    else:
        print("-> No valid samples to visualize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DataLoader Output and Augmentations across multiple batches.")
    parser.add_argument('--aug_config_path', type=str, required=True, help="Path to the augmentor YAML config file")
    parser.add_argument('--data_path', type=str, default="multi_longsleeve_multi_primitive_alignment_human_demo", help="Path or name of the dataset")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size to request from DataLoader")
    parser.add_argument('--num_batches', type=int, default=5, help="Number of batches to sample and stack")
    parser.add_argument('--grid_cols', type=int, default=0, help="Force number of columns in the final grid. Defaults to batch_size if 0.")
    parser.add_argument('--output_path', type=str, default='./tmp/dataloader_augment.png')
    parser.add_argument('--use_hindsight', action='store_true', help="Use HindsightDataset instead of TrajectoryDataset")
    parser.add_argument('--draw_keypoints', action='store_true', help="Draw semantic keypoints on the images")
    args = parser.parse_args()
    
    main(args)