import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import os
from scipy.ndimage import rotate, shift
from scipy.signal import fftconvolve

MASK_THRESHOLD_V2=350000

def get_mask_generator():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device {}'.format(DEVICE))

    ### Masking Model Macros ###
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint="../models/sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)

def get_mask_v2(mask_generator, rgb, 
                mask_threshold_min=5000,   # Lowered min size slightly
                mask_threshold_max=800000, 
                min_saturation=30,         # NEW: Filter out white/grey things
                white_value_threshold=210, # NEW: Filter out very bright white things
                min_variance=10,           # CHANGED: Lowered significantly for plain clothes
                max_variance=5000,
                debug=False,
                save_dir="./tmp"):
    """
    Select the best mask likely to be the cloth based on Color Saturation.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Convert to HSV once globally to save time
    # H: 0-179, S: 0-255, V: 0-255
    hsv_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    results = mask_generator.generate(rgb)
    mask_data = []

    print(f"Found {len(results)} candidate masks.")

    for idx, result in enumerate(results):
        mask = result['segmentation'].astype(np.uint8)
        mask_region_size = np.sum(mask) # Sum is faster for binary masks

        # --- A. Size Filter ---
        if mask_region_size < mask_threshold_min or mask_region_size > mask_threshold_max:
            continue

        # --- B. Border Filter (Crucial for removing Background) ---
        # If a mask touches all 4 borders, or a significant portion of the border, it's likely the background.
        h, w = mask.shape
        border_pixels = np.sum(mask[0, :]) + np.sum(mask[-1, :]) + np.sum(mask[:, 0]) + np.sum(mask[:, -1])
        # If it touches more than 10% of the perimeter, kill it
        if border_pixels > (2 * (h + w)) * 0.10: 
            if debug: print(f"ID {idx}: Filtered (Touching Borders)")
            continue

        # Get pixels for this mask
        # Note: Indexing with boolean mask flattens the array
        masked_hsv = hsv_image[mask == 1]
        masked_rgb = rgb[mask == 1]

        # --- C. Color/Saturation Logic ---
        avg_saturation = np.mean(masked_hsv[:, 1]) # Index 1 is Saturation
        avg_value = np.mean(masked_hsv[:, 2])      # Index 2 is Value (Brightness)

        # 1. Reject if it is too White (High Brightness + Low Saturation)
        if avg_value > white_value_threshold and avg_saturation < min_saturation:
            if debug: print(f"ID {idx}: Filtered (Too White: Val={avg_value:.0f}, Sat={avg_saturation:.0f})")
            continue

        # 2. Reject if it is too Grey/Shadowy (Low Saturation)
        # Plain coloured clothes usually have Saturation > 30-40. 
        # White/Grey/Shadows usually have Saturation < 20.
        if avg_saturation < min_saturation:
             if debug: print(f"ID {idx}: Filtered (Too Grey/Shadow: Sat={avg_saturation:.0f})")
             continue

        # --- D. Variance Filter ---
        # We calculate variance on the Value (brightness) channel or Grayscale
        # Plain clothes have LOW variance. High variance means texture or noise.
        mask_variance = np.var(masked_rgb)
        if mask_variance < min_variance or mask_variance > max_variance:
            if debug: print(f"ID {idx}: Filtered (Variance: {mask_variance:.0f})")
            continue

        # If we passed all filters, keep it
        score = mask_region_size * avg_saturation # Heuristic: Big + Colorful = Good
        
        mask_data.append({
            'mask': mask,
            'mask_region_size': mask_region_size,
            'id': idx,
            'variance': mask_variance,
            'saturation': avg_saturation,
            'score': score
        })

        if debug:
            save_name = os.path.join(save_dir, f"candidate_{idx}_sat{int(avg_saturation)}.png")
            cv2.imwrite(save_name, (mask * 255))

    if len(mask_data) == 0:
        print("No suitable masks found after filtering.")
        # Fallback: Return empty mask or handle gracefully
        return np.zeros(rgb.shape[:2], dtype=np.uint8)
    
    # Sort by our new heuristic score (Size * Saturation) or just Size
    # final_mask_data = sorted(mask_data, key=lambda x: x['mask_region_size'], reverse=True)[0]
    final_mask_data = sorted(mask_data, key=lambda x: x['score'], reverse=True)[0]
    
    final_mask = final_mask_data['mask']

    if debug:
        final_filename = os.path.join(save_dir, f"FINAL_mask_{final_mask_data['id']}.png")
        cv2.imwrite(final_filename, (final_mask * 255))
        print(f"Selected ID {final_mask_data['id']}: Size {final_mask_data['mask_region_size']}, Sat {final_mask_data['saturation']:.1f}")

    return final_mask

def calculate_iou(mask1, mask2):
    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(bool)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def best_translation(mask1, mask2):
    corr = fftconvolve(mask2, mask1[::-1, ::-1], mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    shift_y = y - mask2.shape[0] // 2
    shift_x = x - mask2.shape[1] // 2
    return shift(mask1, (shift_y, shift_x), order=0)

def get_max_IoU(mask1, mask2, debug=False):
    # preprocess (pad + resize)
    # mask1, mask2 = preprocess(mask1), preprocess(mask2)
    
    if mask1.shape[0] > mask1.shape[1]:
        pad = (mask1.shape[0] - mask1.shape[1]) // 2
        mask1 = np.pad(mask1, ((0, 0), (pad, pad)), mode='constant')
    elif mask1.shape[1] > mask1.shape[0]:
        pad = (mask1.shape[1] - mask1.shape[0]) // 2
        mask1 = np.pad(mask1, ((pad, pad), (0, 0)), mode='constant')
    
    if mask2.shape[0] > mask2.shape[1]:
        pad = (mask2.shape[0] - mask2.shape[1]) // 2
        mask2 = np.pad(mask2, ((0, 0), (pad, pad)), mode='constant')
    elif mask2.shape[1] > mask2.shape[0]:
        pad = (mask2.shape[1] - mask2.shape[0]) // 2
        mask2 = np.pad(mask2, ((pad, pad), (0, 0)), mode='constant')

    if mask1.shape[0] > 128:
        mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask1 = (mask1 > 0.5).astype(np.uint8)
        

    if mask2.shape[0] > 128:
        mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        mask2 = (mask2 > 0.5).astype(np.uint8)
    
    
    max_iou, best_mask = -1, None
    best_angle = 0
    angles = range(0, 360, 5)  # coarse search
    
    for angle in angles:
        rotated = rotate(mask1, angle, reshape=False, order=0) > 0.5
        aligned = best_translation(rotated, mask2)
        iou = calculate_iou(aligned, mask2)
        
        if iou > max_iou:
            max_iou, best_mask, best_angle = iou, aligned, angle
    
    # refine search around best angle
    for angle in range(best_angle-5, best_angle+6, 1):
        rotated = rotate(mask1, angle, reshape=False, order=0) > 0.5
        aligned = best_translation(rotated, mask2)
        iou = calculate_iou(aligned, mask2)
        
        if iou > max_iou:
            max_iou, best_mask = iou, aligned
    
    return max_iou, best_mask.astype(int)