import torch
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from utils import save_color
import numpy as np

MASK_THRESHOLD_V2=350000

def get_mask_generator():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device {}'.format(DEVICE))

    ### Masking Model Macros ###
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint="/workspace/bimanual_garment_folding/models/sam_vit_h_4b8939.pth")
    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(sam)


def get_mask_v2(mask_generator, rgb, mask_treshold=160000):
        """
        Generate a mask for the given RGB image that is most different from the background.
        
        Parameters:
        - rgb: A NumPy array representing the RGB image.
        
        Returns:
        - A binary mask as a NumPy array with the same height and width as the input image.
        """
        # Generate potential masks from the mask generator
        results = mask_generator.generate(rgb)
        
        final_mask = None
        max_color_difference = 0
        #print('Processing mask results...')
        save_color(rgb, 'rgb', './tmp')
        mask_data = []

        # Iterate over each generated mask result
        for i, result in enumerate(results):
            segmentation_mask = result['segmentation']
            mask_shape = rgb.shape[:2]

            ## count no mask corner of the mask
            margin = 5 #5
            mask_corner_value = 1.0*segmentation_mask[margin, margin] + 1.0*segmentation_mask[margin, -margin] + \
                                1.0*segmentation_mask[-margin, margin] + 1.0*segmentation_mask[-margin, -margin]
            
            

            #print('mask corner value', mask_corner_value)
            # Ensure the mask is in the correct format
            orginal_mask = segmentation_mask.copy()
            segmentation_mask = segmentation_mask.astype(np.uint8) * 255
            
            # Calculate the masked region and the background region
            masked_region = cv2.bitwise_and(rgb, rgb, mask=segmentation_mask)
            background_region = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(segmentation_mask))
            
            # Calculate the average color of the masked region
            masked_pixels = masked_region[segmentation_mask == 255]
            if masked_pixels.size == 0:
                continue
            avg_masked_color = np.mean(masked_pixels, axis=0)
            
            # Calculate the average color of the background region
            background_pixels = background_region[segmentation_mask == 0]
            if background_pixels.size == 0:
                continue
            avg_background_color = np.mean(background_pixels, axis=0)
            
            # Calculate the Euclidean distance between the average colors
            color_difference = np.linalg.norm(avg_masked_color - avg_background_color)
            #print(f'color difference {i} color_difference {color_difference}')
            #save_mask(orginal_mask, f'mask_candidate_{i}')
            
            # Select the mask with the maximum color difference from the background
            #mask_region_size = np.sum(segmentation_mask == 255)
            

            if mask_corner_value >= 2:
                # if the mask has more than 2 corners, the flip the value
                orginal_mask = 1 - orginal_mask
            
            mask_region_size = np.sum(orginal_mask == 1)
            if mask_region_size > mask_treshold:
                continue

            mask_data.append({
                'mask': orginal_mask,
                'color_difference': color_difference,
                'mask_region_size': mask_region_size,
            })
        
        top_num = 3
        top_5_masks = sorted(mask_data, key=lambda x: x['color_difference'], reverse=True)[:top_num]
        final_mask_data = sorted(top_5_masks, key=lambda x: x['mask_region_size'], reverse=True)[0]
        final_mask = final_mask_data['mask']

        ## make the margine of the final mask to be 0
        margin = 5
        final_mask[:margin, :] = 0
        final_mask[-margin:, :] = 0
        final_mask[:, :margin] = 0
        final_mask[:, -margin:] = 0

        ## print the average color of the mask background
        masked_region = np.expand_dims(final_mask, -1) * rgb
        background_region = (1 - np.expand_dims(final_mask, -1)) * rgb
        masked_pixels = masked_region[final_mask == 1]
        avg_masked_color = np.mean(masked_pixels, axis=0)
        background_pixels = background_region[final_mask == 0]
        avg_background_color = np.mean(background_pixels, axis=0)
        #print(f'avg_masked_color {avg_masked_color} avg_background_color {avg_background_color}')
        
        #save_mask(final_mask, 'final_mask')
        #print('Final mask generated.')

        return final_mask