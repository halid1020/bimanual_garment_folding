import numpy as np
import cv2

def process_depth(raw_depth):
    """
    Process raw depth image:
    1. Convert to meters
    2. Estimate table height (95th percentile)
    3. Clip to range [table - 0.1m, table + 0.1m]
    4. Normalize (0 = Deepest/Background, 1 = Shallowest/Closest)
    """
    # 1. Convert to Meters (RealSense is usually uint16 mm)
    if raw_depth.dtype == np.uint16:
        depth_m = raw_depth.astype(np.float32) / 1000.0
    else:
        depth_m = raw_depth.astype(np.float32)

    # 2. Estimate Camera Height (Distance to Table)
    # 95th percentile is robust against sensor noise (dropouts)
    camera_height = np.percentile(depth_m, 95)
    
    # 3. Define Clipping Range (+- 0.1m around table)
    # Objects on table will be closer (smaller value) than camera_height
    min_dist = camera_height - 0.1
    max_dist = camera_height + 0.1
    
    clipped_depth = np.clip(depth_m, min_dist, max_dist)
    
    # 4. Min-Max Normalization (Inverted)
    # 0 = Deepest point (max_dist), 1 = Shallowest point (min_dist)
    # Formula: (max - value) / (max - min)
    norm_depth = (max_dist - clipped_depth) / (max_dist - min_dist)
    
    return norm_depth

def get_grasp_rotation(mask, point):
    """
    Calculates rotation angle by finding the strongest edge in the neighborhood.
    """
    x, y = int(point[0]), int(point[1])
    h, w = mask.shape

    r = 15
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r + 1), min(h, y + r + 1)
    
    roi = mask[y1:y2, x1:x2].astype(np.float32)
    
    if np.min(roi) == np.max(roi):
        return 0.0

    roi = cv2.GaussianBlur(roi, (7, 7), 1.0)

    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=5)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    
    if magnitude[max_idx] < 1e-3:
        return 0.0
        
    best_gx = gx[max_idx]
    best_gy = gy[max_idx]
    
    angle = np.arctan2(best_gy, best_gx)
    angle += np.pi / 2 
    
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
        
    return angle


def snap_to_mask(point, mask):
    point = np.array(point, dtype=int)
    h, w = mask.shape
    x, y = np.clip(point[0], 0, w - 1), np.clip(point[1], 0, h - 1)

    if mask[y, x] > 0:
        return np.array([x, y])

    valid_indices = np.argwhere(mask > 0)
    
    if len(valid_indices) == 0:
        print("[Warning] Workspace mask is empty! Cannot snap point.")
        return np.array([x, y])

    current_pos_yx = np.array([y, x])
    distances = np.sum((valid_indices - current_pos_yx) ** 2, axis=1)
    
    nearest_idx = np.argmin(distances)
    nearest_yx = valid_indices[nearest_idx]
    
    return np.array([nearest_yx[1], nearest_yx[0]])