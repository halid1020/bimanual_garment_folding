## These semantics are borrowed from CLASP paper

import numpy as np
import open3d as o3d
from copy import deepcopy


KEYPOINT_SEMANTICS = {
    'longsleeve':[
        'left_collar',
        'right_collar',
        'centre_collar',
        'left_shoulder',
        'right_shoulder',
        'higher_left_sleeve',
        'higher_right_sleeve',
        'lower_left_sleeve',
        'lower_right_sleeve',
        'left_armpit',
        'right_armpit',
        'centre',
        'left_hem',
        'right_hem',
        'centre_hem'
    ],

    'pants': [
        'left_waistband',
        'left_waistband_centre',
        'right_waistband_centre',
        'right_waistband',
        'left_hem',
        'left_hem_centre',
        'right_hem_centre',
        'right_hem'
    ]
}

def rigid_transform_3D(A, B):
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise ValueError(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise ValueError(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # check for NaNs or Infs
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("NaN or Inf detected in input point sets!")

    # find mean column wise
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # check covariance matrix
    # if not np.isfinite(H).all():
    #     raise ValueError("NaN or Inf detected in covariance matrix H!")
    # if np.linalg.matrix_rank(H) < 3:
    #     raise ValueError("Degenerate point configuration: covariance matrix rank < 3")

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

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




def rigid_align(current_verts, goal_verts, max_coverage, flip_x=True, scale=None):
    goal_verts = goal_verts.copy()
    current_verts = current_verts.copy()

    # flatten (ignore z axis)
    z_goals = goal_verts[:, 2].copy()
    z_cur = current_verts[:, 2].copy()
    current_verts[:, 2] = 0
    goal_verts[:, 2] = 0

    # optional flip along X
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] = -1 * flipped_goal_verts[:, 0]

    # choose better initial alignment
    dist = np.mean(np.linalg.norm(goal_verts - current_verts, axis=1))
    dist_flipped = np.mean(np.linalg.norm(flipped_goal_verts - current_verts, axis=1))
    if dist_flipped < dist:
        goal_verts = flipped_goal_verts

    # superimpose (rigid transform)
    icp_verts = superimpose(current_verts, goal_verts)
    for _ in range(5):
        threshold = 0.3 * np.sqrt(max_coverage)
        indices = np.linalg.norm(icp_verts - goal_verts, axis=1) < threshold
        icp_verts = superimpose(icp_verts, goal_verts, indices=indices)

    goal_verts[:, 2] = z_goals
    icp_verts[:, 2] = z_cur

    return goal_verts, icp_verts


def deformable_align(current_verts, goal_verts, max_coverage,  flip_x=True, scale=None):
    # Get rigid alignment first
    goal_verts, icp_verts = rigid_align(goal_verts, current_verts, max_coverage, flip_x=flip_x, scale=scale)
    
    z_goals = goal_verts[:, 2].copy()
    z_cur = current_verts[:, 2].copy()
    current_verts[:, 2] = 0
    goal_verts[:, 2] = 0

    # Reverse alignment (goal → current)
    reverse_goal_verts = goal_verts.copy()
    R, t = rigid_transform_3D(reverse_goal_verts.T, icp_verts.T)
    reverse_goal_verts = (R @ reverse_goal_verts.T + t).T

    threshold = 0.3 * np.sqrt(max_coverage)
    indices = np.linalg.norm(reverse_goal_verts - icp_verts, axis=1) < threshold
    reverse_goal_verts = superimpose(reverse_goal_verts, icp_verts, indices=indices)

    goal_verts[:, 2] = z_goals
    icp_verts[:, 2] = z_cur


    return icp_verts, reverse_goal_verts

def simple_rigid_align(cur, goal):
    cur_centered = cur - np.mean(cur, axis=0)
    goal_centered = goal - np.mean(goal, axis=0)

    # Compute optimal rotation via SVD
    H = cur_centered.T @ goal_centered
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    aligned_curr = cur_centered @ R
    aligned_goal = goal_centered
    return aligned_curr, aligned_goal