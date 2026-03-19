import h5py
import numpy as np
import cv2
import random
from tqdm import tqdm
from typing import Dict, Tuple, Optional

KEYPOINT_PATH = 'keypoint/dataset/keypoints.hdf5'
OUTPUT_PATH = 'data/keypoints_balanced.hdf5'
IMG_SIZE = (128, 128)

KEYPOINT_NAMES = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
DEFAULT_COORD = (-1, -1)

class SampleProcessor:
    @staticmethod
    def process_positive(sample_group, keypoint_group):
        """Process positive samples (with keypoints)"""
        try:
            action_pos_map = sample_group['action_pos_map'][40]
            image = sample_group['transformed_obs'][40, (0,1,2,4,5),:,:].transpose(1,2,0)
            
            # Keypoint extraction logic
            pos_values = np.unique(action_pos_map)
            temp = {}
            for name, keypoint in keypoint_group.items():
                for val in keypoint:
                    if val in pos_values:
                        temp[name] = val
                        break
            
            raw_keypoints = {}
            for name, value in temp.items():
                matches = np.argwhere(action_pos_map == value)
                if matches.size == 0:
                    continue
                row, col = matches[0][0], matches[0][1]
                offset = 16 if (matches[0][2] == 0) else -16
                raw_keypoints[name] = (row + offset, col)
            
            if not ({"top_left", "top_right"}.issubset(raw_keypoints) or 
                    {"bottom_left", "bottom_right"}.issubset(raw_keypoints)):
                raise KeyError("Insufficient keypoints")
            
            # Adjust keypoints
            keypoints, visibility = SampleProcessor._process_keypoints(raw_keypoints, image.shape[:2])
            return cv2.resize(image, IMG_SIZE), keypoints, visibility
        
        except Exception as e:
            print(f"Failed to process positive sample: {str(e)}")
            return None

    @staticmethod
    def process_negative(sample_group):
        """Process negative samples (images only)"""
        try:
            image = sample_group['transformed_obs'][40, (0,1,2,4,5),:,:].transpose(1,2,0)
            return cv2.resize(image, IMG_SIZE)
        except Exception as e:
            print(f"Failed to process negative sample: {str(e)}")
            return None

    @staticmethod
    def _process_keypoints(raw_points: Dict, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Internal method for processing keypoints"""
        adjusted = SampleProcessor._adjust_keypoints(raw_points, img_size)
        coords = []
        visibility = []
        for name in KEYPOINT_NAMES:
            if name in adjusted and adjusted[name] != DEFAULT_COORD:
                coords.append(adjusted[name])
                visibility.append(1)
            else:
                coords.append(DEFAULT_COORD)
                visibility.append(0)
        return np.array(coords, dtype=np.float32), np.array(visibility, dtype=np.uint8)

    @staticmethod
    def _adjust_keypoints(points: Dict, img_size: Tuple[int, int]) -> Dict:
        """Logic for adjusting keypoints"""
        h, _ = img_size
        if not SampleProcessor._check_keypoint_order(points, h):
            return {
                'top_left': points.get('top_right', DEFAULT_COORD),
                'top_right': points.get('top_left', DEFAULT_COORD),
                'bottom_left': points.get('bottom_right', DEFAULT_COORD),
                'bottom_right': points.get('bottom_left', DEFAULT_COORD)
            }
        return points

    @staticmethod
    def _check_keypoint_order(kps: dict, image_height: int) -> bool:
        """Validate keypoint order"""
        keys = set(kps.keys())
        if {"top_left", "top_right", "bottom_left", "bottom_right"}.issubset(keys):
            top_diff = abs(kps["top_left"][0] - kps["top_right"][0])
            bottom_diff = abs(kps["bottom_left"][0] - kps["bottom_right"][0])
            if top_diff >= bottom_diff:
                return kps["top_left"][0] <= kps["top_right"][0]
            else:
                return kps["bottom_left"][0] <= kps["bottom_right"][0]
        return True

def create_balanced_dataset():
    with h5py.File(ORIGIN_PATH, 'r') as origin_file, \
         h5py.File(OUTPUT_PATH, 'w') as output_file, \
         h5py.File(KEYPOINT_PATH, 'r') as keypoint_file:

        # Phase 1: Scan the dataset
        pos_candidates = []
        neg_candidates = []
        
        print("Scanning the dataset...")
        for category_id in tqdm(origin_file, desc="Categories"):
            for cloth_id in tqdm(origin_file[category_id], desc="Clothes"):
                cloth_group = origin_file[f"{category_id}/{cloth_id}"]
                keypoint_group = keypoint_file.get(f"{category_id}_{cloth_id}")
                
                if not keypoint_group:
                    print(f"Missing keypoint data: {category_id}_{cloth_id}")
                    continue
                
                for sample_name in cloth_group:
                    if sample_name == 'data_dict':
                        continue
                    
                    sample_group = cloth_group[sample_name]
                    pw_dist = sample_group.attrs['preaction_weighted_distance']
                    l2_dist = sample_group.attrs['preaction_l2_distance']
                    
                    if pw_dist < 0.15 and l2_dist > 0.01:
                        pos_candidates.append((category_id, cloth_id, sample_name, keypoint_group))
                    elif pw_dist >= 0.15:
                        neg_candidates.append((category_id, cloth_id, sample_name))

        # Phase 2: Process samples
        output_samples = output_file.create_dataset("samples", shape=(0,), dtype=h5py.string_dtype(), maxshape=(None,))
        sample_index = 0
        
        # Process positive samples
        print(f"Processing {len(pos_candidates)} positive samples...")
        for cat_id, clo_id, sam_name, kp_group in tqdm(pos_candidates, desc="Positive samples"):
            sample_group = origin_file[f"{cat_id}/{clo_id}/{sam_name}"]
            processed = SampleProcessor.process_positive(sample_group, kp_group)
            
            if processed is None:
                continue
            
            image, keypoints, visibility = processed
            sample_id = f"pos_{cat_id}_{clo_id}_{sam_name}"
            
            # Write to HDF5
            grp = output_file.create_group(sample_id)
            grp.create_dataset("image", data=image, dtype=np.float32, compression="gzip")
            grp.create_dataset("keypoints", data=keypoints)
            grp.create_dataset("visibility", data=visibility)
            grp.attrs.update({
                'preaction_weighted_distance': sample_group.attrs['preaction_weighted_distance'],
                'preaction_l2_distance': sample_group.attrs['preaction_l2_distance'],
                'sample_type': 'positive'
            })
            
            # Update index
            output_samples.resize((sample_index + 1,))
            output_samples[sample_index] = sample_id
            sample_index += 1

        # Process negative samples (matching quantity)
        required_neg = min(len(neg_candidates), len(pos_candidates))
        selected_neg = random.sample(neg_candidates, required_neg)
        
        print(f"Processing {required_neg} negative samples...")
        for cat_id, clo_id, sam_name in tqdm(selected_neg, desc="Negative samples"):
            sample_group = origin_file[f"{cat_id}/{clo_id}/{sam_name}"]
            image = SampleProcessor.process_negative(sample_group)
            
            if image is None:
                continue
            
            sample_id = f"neg_{cat_id}_{clo_id}_{sam_name}"
            
            # Write to HDF5 (without keypoints)
            grp = output_file.create_group(sample_id)
            grp.create_dataset("image", data=image, dtype=np.float32, compression="gzip")
            grp.attrs.update({
                'preaction_weighted_distance': sample_group.attrs['preaction_weighted_distance'],
                'preaction_l2_distance': sample_group.attrs['preaction_l2_distance'],
                'sample_type': 'negative'
            })
            
            # Update index
            output_samples.resize((sample_index + 1,))
            output_samples[sample_index] = sample_id
            sample_index += 1

        # Add metadata
        output_file.attrs.create("num_samples", sample_index)
        output_file.attrs.create("positive_samples", len(pos_candidates))
        output_file.attrs.create("negative_samples", required_neg)
        print(f"Dataset creation completed, total samples: {sample_index} (positive: {len(pos_candidates)}, negative: {required_neg})")

if __name__ == "__main__":
    create_balanced_dataset()