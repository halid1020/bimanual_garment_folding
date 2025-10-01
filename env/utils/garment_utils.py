## These semantics are borrowed from CLASP paper

import numpy as np
import open3d as o3d
from copy import deepcopy
from scipy.spatial import cKDTree

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
    current_verts = current_verts.copy()
    goal_verts = goal_verts.copy()

    # Center both sets
    current_verts -= np.mean(current_verts, axis=0)
    goal_verts -= np.mean(goal_verts, axis=0)

    # Try symmetric flips and pick best
    candidates = [
        goal_verts,
        np.copy(goal_verts) * [-1, 1, 1],   # flip X
        np.copy(goal_verts) * [1, -1, 1],   # flip Y
        np.copy(goal_verts) * [-1, -1, 1],  # flip XY
    ]

    dists = [np.mean(np.linalg.norm(current_verts - g, axis=1)) for g in candidates]
    goal_verts = candidates[np.argmin(dists)]

    # Initial superimpose
    icp_verts = superimpose(current_verts, goal_verts)

    # Multi-pass ICP with decreasing thresholds
    for coeff in [0.8, 0.5, 0.3]:
        threshold = coeff * np.sqrt(max_coverage)
        indices = np.linalg.norm(icp_verts - goal_verts, axis=1) < threshold
        if np.sum(indices) >= 3:
            icp_verts = superimpose(icp_verts, goal_verts, indices=indices)

    return goal_verts, icp_verts


# def rigid_align(current_verts, goal_verts, max_coverage, flip_x=True, scale=None):
#     goal_verts = goal_verts.copy()
#     current_verts = current_verts.copy()
    
#     goal_verts = goal_verts - np.mean(goal_verts, axis=0)

#     # flatten (ignore z axis)
#     z_goals = goal_verts[:, 2].copy()
#     z_cur = current_verts[:, 2].copy()
#     current_verts[:, 2] = 0
#     goal_verts[:, 2] = 0

#     # optional flip along X
#     flipped_goal_verts = goal_verts.copy()
#     #flipped_goal_verts[:, 0] = -1 * flipped_goal_verts[:, 0]
#     flipped_goal_verts[:, 0] = -1 * flipped_goal_verts[:, 0]

#     # choose better initial alignment
#     dist = np.mean(np.linalg.norm(goal_verts - current_verts, axis=1))
#     dist_flipped = np.mean(np.linalg.norm(flipped_goal_verts - current_verts, axis=1))
#     if dist_flipped < dist:
#         goal_verts = flipped_goal_verts

#     # superimpose (rigid transform)
#     icp_verts = superimpose(current_verts, goal_verts)
#     for _ in range(5):
#         threshold = 0.3 * np.sqrt(max_coverage)
#         indices = np.linalg.norm(icp_verts - goal_verts, axis=1) < threshold
#         icp_verts = superimpose(icp_verts, goal_verts, indices=indices)

#     goal_verts[:, 2] = z_goals
#     icp_verts[:, 2] = z_cur

#     return goal_verts, icp_verts


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

    assert np.isfinite(cur).all(), "cur contains NaN or Inf"
    assert np.isfinite(goal).all(), "goal contains NaN or Inf"
    
    try:
        # 1. Center in XY only (preserve Z)
        cur_centered = cur.copy()
        goal_centered = goal.copy()
        cur_centered[:, :2] -= np.mean(cur[:, :2], axis=0)
        goal_centered[:, :2] -= np.mean(goal[:, :2], axis=0)

        # 2. Compute optimal rotation in XY plane
        H = cur_centered[:, :2].T @ goal_centered[:, :2]  # 2x2
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt  # 2x2 rotation matrix

        # 3. Apply rotation in XY only
        aligned_cur = cur_centered.copy()
        aligned_cur[:, :2] = cur_centered[:, :2] @ R
        aligned_goal = goal_centered  # leave goal centered

        return aligned_cur, aligned_goal

    except np.linalg.LinAlgError:
        # fallback: return original (unaligned) points
        print('SVD error')
        return cur, goal

def chamfer_alignment_with_rotation(cur_verts, goal_verts, coarse_steps=36, fine_steps=10, fine_window=np.deg2rad(5)):
    """
    Aligns cur_verts to goal_verts by rotating around z-axis (coarse-to-fine) to minimize Chamfer distance.
    
    Args:
        cur_verts (np.ndarray): (N, 3) current point cloud.
        goal_verts (np.ndarray): (M, 3) goal point cloud.
        coarse_steps (int): number of coarse search steps (default 36 = every 10 degrees).
        fine_steps (int): number of fine search steps around best coarse angle.
        fine_window (float): fine search half-window in radians (default = 5°).
    
    Returns:
        best_distance (float): minimum Chamfer distance.
        best_aligned_cur (np.ndarray): aligned cur_verts (N, 3).
        centered_goal (np.ndarray): centered goal_verts (M, 3).
        paired_cur (np.ndarray): aligned cur_verts matched to nearest goal points (N, 3).
        pairs (list of tuples): [(cur_point, goal_point), ...].
    """
    # 1. Center both clouds
    cur_center = np.mean(cur_verts, axis=0)
    goal_center = np.mean(goal_verts, axis=0)
    cur_verts_centered = cur_verts - cur_center
    goal_verts_centered = goal_verts - goal_center

    # Build KDTree for goal cloud
    goal_tree = cKDTree(goal_verts_centered)

    def compute_chamfer(rotated_cur):
        """Compute Chamfer distance and nearest-neighbor pairs."""
        # cur -> goal
        dists_c2g, idx_c2g = goal_tree.query(rotated_cur)
        chamfer_c2g = np.mean(dists_c2g**2)

        # goal -> cur
        cur_tree = cKDTree(rotated_cur)
        dists_g2c, idx_g2c = cur_tree.query(goal_verts_centered)
        chamfer_g2c = np.mean(dists_g2c**2)

        chamfer = chamfer_c2g + chamfer_g2c
        return chamfer, idx_c2g

    # 2. Coarse search
    angles = np.linspace(0, 2*np.pi, coarse_steps, endpoint=False)
    print('corase angles len', len(angles))
    best_distance = float("inf")
    best_angle = 0
    best_idx = None
    for angle in angles:
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_cur = cur_verts_centered @ R.T
        chamfer, idx_c2g = compute_chamfer(rotated_cur)
        if chamfer < best_distance:
            best_distance = chamfer
            best_angle = angle
            best_idx = idx_c2g

    # 3. Fine search around best_angle
    fine_angles = np.linspace(best_angle - fine_window, best_angle + fine_window, fine_steps)
    print('fine angles len', len(fine_angles))
    for angle in fine_angles:
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_cur = cur_verts_centered @ R.T
        chamfer, idx_c2g = compute_chamfer(rotated_cur)
        if chamfer < best_distance:
            best_distance = chamfer
            best_angle = angle
            best_idx = idx_c2g

    # 4. Compute final best alignment + pairs
    R = np.array([
        [np.cos(best_angle), -np.sin(best_angle), 0],
        [np.sin(best_angle),  np.cos(best_angle), 0],
        [0, 0, 1]
    ])
    best_aligned_cur = cur_verts_centered @ R.T
   

    return best_aligned_cur, goal_verts_centered