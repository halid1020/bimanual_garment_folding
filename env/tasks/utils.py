# codes are borrow from 
# https://github.com/real-stanford/cloth-funnels/blob/main/cloth_funnels/learning/utils.py#L120
import numpy as np
import open3d as o3d
from copy import deepcopy
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass, shift
from scipy.ndimage import rotate, shift
from scipy.signal import fftconvolve
import cv2
import os

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

# def get_max_IoU(mask1, mask2, debug=False):
#     """
#     Calculate the maximum IoU between two binary mask images,
#     allowing for rotation and translation of mask1.
    
#     :param mask1: First binary mask (numpy array)
#     :param mask2: Second binary mask (numpy array)
#     :return: Tuple of (Maximum IoU value, Matched mask)
#     """

#     ## if mask is rectangular, make it square by padding
#     if mask1.shape[0] > mask1.shape[1]:
#         pad = (mask1.shape[0] - mask1.shape[1]) // 2
#         mask1 = np.pad(mask1, ((0, 0), (pad, pad)), mode='constant')
#     elif mask1.shape[1] > mask1.shape[0]:
#         pad = (mask1.shape[1] - mask1.shape[0]) // 2
#         mask1 = np.pad(mask1, ((pad, pad), (0, 0)), mode='constant')
    
#     if mask2.shape[0] > mask2.shape[1]:
#         pad = (mask2.shape[0] - mask2.shape[1]) // 2
#         mask2 = np.pad(mask2, ((0, 0), (pad, pad)), mode='constant')
#     elif mask2.shape[1] > mask2.shape[0]:
#         pad = (mask2.shape[1] - mask2.shape[0]) // 2
#         mask2 = np.pad(mask2, ((pad, pad), (0, 0)), mode='constant')
    
#     # print('mask1 shape:', mask1.shape)
#     # print('mask2 shape:', mask2.shape)

#     # if resolution above 128, we need to resize the mask to 128
#     if mask1.shape[0] > 128:
#         mask1 = cv2.resize(mask1.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
#         mask1 = (mask1 > 0.5).astype(np.uint8)
        

#     if mask2.shape[0] > 128:
#         mask2 = cv2.resize(mask2.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
#         mask2 = (mask2 > 0.5).astype(np.uint8)
    
    

#     # Get the properties of mask2
#     props = regionprops(label(mask2))[0]
#     center_y, center_x = props.centroid

#     max_iou = 0
#     best_mask = None
    
#     # Define rotation angles to try
#     angles = range(0, 360, 1)  # Rotate from 0 to 350 degrees in 10-degree steps
    
#     for angle in angles:
#         # Rotate mask1
#         rotated_mask = rotate(mask1, angle, reshape=False)

#         # if the mask is blank, skip
#         if np.sum(rotated_mask) == 0:
#             continue
        
#         # Get properties of rotated mask
#         rotated_props = regionprops(label(rotated_mask))[0]
#         rotated_center_y, rotated_center_x = rotated_props.centroid
        
#         # Calculate translation
#         dy = center_y - rotated_center_y
#         dx = center_x - rotated_center_x
        
#         # Translate rotated mask
#         translated_mask = shift(rotated_mask, (dy, dx))

#         if debug:
#             os.makedirs("tmp", exist_ok=True)

#             # Normalize to 0–255 for saving
#             rotated_vis = (rotated_mask > 0.5).astype(np.uint8) * 255
#             translated_vis = (translated_mask > 0.5).astype(np.uint8) * 255
#             mask2_vis = (mask2 > 0.5).astype(np.uint8) * 255

#             cv2.imwrite(f"tmp/rotated.png", rotated_vis)
#             cv2.imwrite(f"tmp/translated.png", translated_vis)
#             cv2.imwrite(f"tmp/mask2_ref.png", mask2_vis)
                
#         #translated_mask = (translated_mask > 0.1).astype(int)
#         # Calculate IoU
#         iou = calculate_iou(translated_mask, mask2)
        
#         # Update max_iou and best_mask if necessary
#         if iou > max_iou:
#             max_iou = iou
#             best_mask = translated_mask

#     # Ensure the best_mask is binary
#     if best_mask is None:
#         return 0, None
    
#     best_mask = (best_mask > 0.5).astype(int)

#     return max_iou, best_mask



def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    # assert centroid_A and centroid_B has no nan
    if np.isnan(centroid_A).any() or np.isnan(centroid_B).any():
        print('A', A)
        print('centroid_A', centroid_A)
        print('B', B)
        print('centroid_B', centroid_B)
    assert not np.isnan(centroid_A).any()
    assert not np.isnan(centroid_B).any()

    t = -R @ centroid_A + centroid_B

    return R, t


def superimpose(current_verts, goal_verts, indices=None, symmetric_goal=False):

    current_verts = current_verts.copy()
    goal_verts = goal_verts.copy()
    # flipped_goal_verts = goal_verts.copy()
    # flipped_goal_verts[:, 0] = 2*np.mean(flipped_goal_verts[:, 0]) - flipped_goal_verts[:, 0]

    if indices is not None:
        #assert len(indices) > 0
        #print('indices', indices)
        R, t = rigid_transform_3D(current_verts[indices].T, goal_verts[indices].T)
    else:
        R, t = rigid_transform_3D(current_verts.T, goal_verts.T)

    ## assert R and t has no nan
    assert not np.isnan(R).any()
    assert not np.isnan(t).any()

    icp_verts = (R @ current_verts.T + t).T

    return icp_verts

def transform_verts(verts, goal_verts, threshold, iteration=1):
    indices = None
    transform_verts = verts.copy()
    #print('threshold', threshold)
    for i in range(iteration):
        transform_verts = superimpose(transform_verts, goal_verts, indices=indices)
        distances = np.linalg.norm(transform_verts - goal_verts, axis=1)
        #print('min distance', np.min(distances))
        indices = distances < threshold
        if np.sum(indices) == 0:
            indices = None
            break
    
    transform_verts = superimpose(transform_verts, goal_verts, indices=indices)

    return transform_verts

def get_deform_distance(current_verts, goal_verts, threshold):
    current_verts[:, 2] = 0
    goal_verts[:, 2] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]    
    transform_verts_ = transform_verts(
        current_verts, goal_verts, threshold, iteration=5)
    deform_distance_regular = np.mean(np.linalg.norm(transform_verts_ - goal_verts, axis=1))
    deform_distance_flipped = np.mean(np.linalg.norm(transform_verts_ - flipped_goal_verts, axis=1))
    deform_l2_distance = min(deform_distance_regular, deform_distance_flipped)
    ## assert is a number
    assert not np.isnan(deform_l2_distance)
    return deform_l2_distance

def get_rigid_distance(current_verts, goal_verts, threshold):
    current_verts[:, 2] = 0
    goal_verts[:, 2] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]
    reverse_goal_verts = goal_verts.copy()
    R, t = rigid_transform_3D(reverse_goal_verts.T, current_verts.T)
    reverse_goal_verts = (R @ reverse_goal_verts.T + t).T
    reverse_goal_verts = transform_verts(
        reverse_goal_verts, current_verts, threshold, iteration=1)
    
    rigid_distance_regular = np.mean(np.linalg.norm(goal_verts - reverse_goal_verts, axis=1))
    rigid_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - reverse_goal_verts, axis=1))
    rigid_distance = min(rigid_distance_regular, rigid_distance_flipped)
    ## assert is a number
    assert not np.isnan(rigid_distance)
    return rigid_distance

def deformable_distance(goal_verts, current_verts, max_coverage, 
        deformable_weight=0.65, flip_x=True, icp_steps=1000, scale=None):

    goal_verts = goal_verts.copy()
    current_verts = current_verts.copy()

    #flatten goals
    #print('shape of goal_verts', goal_verts.shape)
    #print('shape of current_verts', current_verts.shape)
    goal_verts[:, 2] = 0
    current_verts[:, 2] = 0
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]

    real_l2_distance = np.mean(np.linalg.norm(goal_verts - current_verts, axis=1))
    real_l2_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - current_verts, axis=1))
    if real_l2_distance_flipped < real_l2_distance:
        real_l2_distance = real_l2_distance_flipped


    #GOAL is RED
    goal_vert_cloud = o3d.geometry.PointCloud()
    goal_vert_cloud.points = o3d.utility.Vector3dVector(goal_verts.copy())
    goal_vert_cloud.paint_uniform_color([1, 0, 0])

    normal_init_vert_cloud = deepcopy(goal_vert_cloud)

    flipped_goal_vert_cloud = o3d.geometry.PointCloud()
    flipped_goal_vert_cloud.points = o3d.utility.Vector3dVector(flipped_goal_verts.copy())
    flipped_goal_vert_cloud.paint_uniform_color([0, 1, 1])

    goal_vert_cloud += flipped_goal_vert_cloud
    #CURRENT is GREEN
    verts_cloud = o3d.geometry.PointCloud()
    verts_cloud.points = o3d.utility.Vector3dVector(current_verts.copy())
    verts_cloud.paint_uniform_color([0, 1, 0])

    THRESHOLD_COEFF = 0.3
    threshold = np.sqrt(max_coverage) * THRESHOLD_COEFF
    
    #### Get Deomrable Distance
    deform_l2_distance = get_deform_distance(current_verts, goal_verts, threshold)
   
    #### Get Rigid Distance
    rigid_l2_distance = get_rigid_distance(current_verts, goal_verts, threshold)

    #make reward scale invariant
    assert(max_coverage != 0 or scale != 0)
    if scale is None:
        deform_l2_distance /= np.sqrt(max_coverage)
        rigid_l2_distance /= np.sqrt(max_coverage)
        real_l2_distance /= np.sqrt(max_coverage)
    else:
        deform_l2_distance /= scale
        rigid_l2_distance /= scale
        real_l2_distance /= scale

    weighted_distance = deformable_weight * deform_l2_distance + (1 - deformable_weight) * rigid_l2_distance

    return weighted_distance, deform_l2_distance, rigid_l2_distance, real_l2_distance