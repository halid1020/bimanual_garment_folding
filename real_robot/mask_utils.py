import torch
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from save_utils import save_colour
import numpy as np
import os

MASK_THRESHOLD_V2=350000

def get_mask_generator():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device {}'.format(DEVICE))

    ### Masking Model Macros ###
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint="/workspace/bimanual_garment_folding/models/sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)

def get_mask_v2(mask_generator, rgb, mask_treshold_min=10000, mask_treshold_max=360000,
                dark_threshold=80, min_variance=200, max_variance=5000,
                debug=False,
                save_dir="./tmp"):
    """
    Select the best mask likely to be the cloth, and save all generated masks.
    
    Parameters:
    - rgb: HxWx3 RGB image
    - mask_generator: SAM AutomaticMaskGenerator instance
    - dark_threshold: minimum average brightness
    - min_variance: minimum variance to avoid uniform regions
    - max_variance: maximum variance to avoid highly textured regions
    - save_dir: folder to save masks
    
    Returns:
    - final_mask: HxW binary mask
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = mask_generator.generate(rgb)
    mask_data = []

    for idx, result in enumerate(results):
        mask = result['segmentation'].copy()
        mask_region_size = np.sum(mask == 1)

        # Save all masks
        if debug:
            mask_filename = os.path.join(save_dir, f"mask_{idx}.png")
            cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))

        # Size filter
        if mask_region_size < mask_treshold_min or mask_region_size > mask_treshold_max:
            if debug:
                print(f"Filtered out {idx} due to size {mask_region_size}")
            continue

        masked_pixels = rgb[mask == 1]
        if masked_pixels.size == 0:
            continue

        avg_brightness = np.mean(masked_pixels)
        if avg_brightness < dark_threshold:
            if debug:
                print(f"Filtered out {idx} due to darkness {avg_brightness:.2f}")
            continue

        mask_variance = np.var(masked_pixels)
        if mask_variance < min_variance or mask_variance > max_variance:
            if debug:
                print(f"Filtered out {idx} due to variance {mask_variance:.2f}")
            continue

        # Skip masks touching image borders
        # if mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any():
        #     print(f"Filtered out {idx} because it touches image border")
        #     continue

        mask_data.append({
            'mask': mask,
            'mask_region_size': mask_region_size,
            'id': idx,
            'variance': mask_variance
        })
    
    if len(mask_data) == 0:
        print("No suitable masks found.")
        return np.zeros(rgb.shape[:2], dtype=np.uint8)
    
    # Pick the mask with highest size * variance product
    final_mask_data = sorted(mask_data, key=lambda x: x['mask_region_size']*(1 + x['variance']/1000), reverse=True)[0]
    final_mask = final_mask_data['mask']

    # Save the final mask
    final_filename = os.path.join(save_dir, f"final_mask_{final_mask_data['id']}.png")
    if debug:
        cv2.imwrite(final_filename, (final_mask * 255).astype(np.uint8))
        print(f"Selected mask {final_mask_data['id']} as final mask, size {final_mask_data['mask_region_size']}, variance {final_mask_data['variance']:.2f}")

    return final_mask
