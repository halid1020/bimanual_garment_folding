import numpy as np
from scipy.ndimage import distance_transform_edt

def readjust_norm_pixel_pick(pick_point, mask):
    H, W = mask.shape
    pixel_action = ((pick_point + 1)/2 * np.array([H, W])).astype(np.int32)
    pixel_action = np.clip(pixel_action, 0, [H-1, W-1])
    points = [(pixel_action[0], pixel_action[1])]
    pixel_action = (adjust_points(points, mask)[0][0]/np.array([H, W])) * 2 - 1
    return pixel_action

def adjust_points(points, mask, min_distance=2):
    """
    Adjust points to be at least min_distance pixels away from the mask border.
    
    :param points: List of (x, y) coordinates
    :param mask: 2D numpy array where 0 is background and 1 is foreground
    :param min_distance: Minimum distance from the border (default: 2)
    :return: List of adjusted (x, y) coordinates
    """

    mask = (mask > 0).astype(np.uint8)

    if np.sum(mask) == 0:
        return points, mask
    
    # Compute distance transform
    dist_transform = distance_transform_edt(mask)
    
    # Create a new mask where pixels < min_distance from border are 0
    eroded_mask = (dist_transform >= min_distance).astype(np.uint8)
    while np.sum(eroded_mask) == 0 and min_distance >= 1:
        min_distance -= 1
        eroded_mask = (dist_transform >= min_distance).astype(np.uint8)
    
    if np.sum(eroded_mask) == 0:
        return points, mask
   
    # plt.imshow(eroded_mask.astype(np.float32))
    # plt.savefig('tmp/eroded_mask.png')

    adjusted_points = []
    for x, y in points:
        #print('x, y', x, y)
        if eroded_mask[x, y] == 0:  # If point is too close to border
            # Find the nearest valid point
            x_indices, y_indices = np.where(eroded_mask == 1)
            
            distances = np.sqrt((x - x_indices)**2 + (y - y_indices)**2)
            nearest_index = np.argmin(distances)
            new_x, new_y = x_indices[nearest_index], y_indices[nearest_index]
            adjusted_points.append((new_x, new_y))
        else:
            adjusted_points.append((x, y))
    
    return adjusted_points, eroded_mask

def pixel_to_world(p, depth, cam_intrinsics, cam_pose, cam_size):
    # Normalize pixel coordinates from [-1, 1] to [0, 1]
    p_norm = (p + 1) / 2
    
    # swap y and x
    p_norm = np.array([p_norm[1], p_norm[0]])

    #print('cam_size:', cam_size)
    # Convert to pixel coordinates
    pixel_x = p_norm[0] * cam_size[0]
    pixel_y = p_norm[1] * cam_size[1]

    # Create homogeneous pixel coordinates
    pixel_homogeneous = np.array([pixel_x, pixel_y, 1])

    # Convert to camera coordinates
    cam_coords = np.linalg.inv(cam_intrinsics) @ (depth * pixel_homogeneous)
    #print('cam_coords:', cam_coords)

    # Convert to homogeneous coordinates
    cam_coords_homogeneous = np.append(cam_coords, 1)

    # Transform to world coordinates
    world_coords = cam_pose @ cam_coords_homogeneous

    return world_coords[:3]  # Return only x, y, z