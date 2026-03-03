import cv2, numpy as np, open3d as o3d, torch, ray, time, trimesh, pyflex, os, subprocess, imageio
from os import devnull
from copy import deepcopy
from sklearn.cluster import DBSCAN
from Imath import PixelType
from scipy.ndimage import distance_transform_edt
from clothmate.utils.utils import compute_intrinsics, translate2d, scale2d, rot2d, rigid_transform_3D, superimpose, transform
from clothmate.utils.utils import get_transform_matrix, pixel_to_3d
from torchvision import transforms
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

def compute_pose(pos, lookat, up=[0, 0, 1]):
    norm = np.linalg.norm
    if type(lookat) != np.array:
        lookat = np.array(lookat)
    if type(pos) != np.array:
        pos = np.array(pos)
    if type(up) != np.array:
        up = np.array(up)
    f = (lookat - pos)
    f = f/norm(f)
    u = up / norm(up)
    s = np.cross(f, u)
    s = s/norm(s)
    u = np.cross(s, f)
    view_matrix = [
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -np.dot(s, pos), -np.dot(u, pos), np.dot(f, pos), 1
    ]
    view_matrix = np.array(view_matrix).reshape(4, 4).T
    pose_matrix = np.linalg.inv(view_matrix)
    pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
    return pose_matrix


def pixels_to_3d_positions(
        transform_pixels, scale, rotation, pretransform_depth,
        transformed_depth, pose_matrix=None,
        pretransform_pix_only=False, **kwargs):

    # print("\n\n")
    # print("transform rotation: ", rotation)
    # print("transform scale: ", scale)
    # print("original dimensions: ", pretransform_depth.shape[0])
    # print("transformed dimensions: ", transformed_depth.shape[0]) 

    mat = get_transform_matrix(
        original_dim=pretransform_depth.shape[0],
        resized_dim=transformed_depth.shape[0],
        rotation=-rotation,  # TODO bug
        scale=scale)

    # print("Pixels before matmul: ", transform_pixels)
    pixels = np.concatenate((transform_pixels, np.array([[1], [1]])), axis=1)
    pixels = np.matmul(pixels, mat)[:, :2].astype(int)
    max_idx = pretransform_depth.shape[0]
    pixels = np.clip(pixels, 0, max_idx-1) 
    pix_1, pix_2 = pixels
    transformed_depth[transform_pixels[0][0], transform_pixels[0][1]] = 0
    transformed_depth[transform_pixels[1][0], transform_pixels[1][1]] = 1
    
    if (pixels < 0).any() or (pixels >= max_idx).any():
        print("pixels out of bounds", pixels, "\n\n\n")

        return {
            'valid_action': False,
            'p1': None, 'p2': None,
            'pretransform_pixels': np.array([pix_1, pix_2])
        }
    # if pretransform_pix_only:
    #     return {
    #         'valid_action': True,
    #         'pretransform_pixels': np.array([pix_1, pix_2])
    #     }
    # Note this order of x,y is not a bug
    x, y = pix_1
    p1 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)
    # Same here
    x, y = pix_2
    p2 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)

    return {
        'valid_action': p1 is not None and p2 is not None,
        'p1': p1,
        'p2': p2,
        'pretransform_pixels': np.array([pix_1, pix_2])
    }


def generate_keypoint(image):
    binary_mask = (image.sum(axis=2) > 0).astype(np.uint8) * 255  

    gray = np.float32(binary_mask)
    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.05)
    harris_corners = cv2.dilate(harris_corners, None)

    thresh = 0.01 * harris_corners.max()
    keypoints = np.argwhere(harris_corners > thresh) 

    filtered_points = []
    radius = 20
    for y, x in keypoints:
        x1, x2 = max(0, x - radius), min(image.shape[1], x + radius)
        y1, y2 = max(0, y - radius), min(image.shape[0], y + radius)
        neighborhood = binary_mask[y1:y2, x1:x2]
        zero_ratio = np.sum(neighborhood == 0) / neighborhood.size

        if zero_ratio > 0.6:
            filtered_points.append((y, x)) 
    
    filtered_points = np.array(filtered_points)
    if len(filtered_points) == 0:
        return {"left": [], "right": [], "corner_matches": {}}

    clustering = DBSCAN(eps=15, min_samples=2).fit(filtered_points)
    cluster_centers = []
    for label in set(clustering.labels_):
        if label == -1:
            continue
        cluster_points = filtered_points[clustering.labels_ == label]
        mean_point = np.mean(cluster_points, axis=0).astype(int)
        cluster_centers.append(tuple(mean_point))  # (y, x)

    width = binary_mask.shape[1]
    center_x = width // 2
    left_keypoints = []
    right_keypoints = []

    for y, x in cluster_centers:
        if y < center_x: 
            left_keypoints.append((x, y))
        else:  
            right_keypoints.append((x, y))

    left_keypoints.sort()
    right_keypoints.sort()

    matched_left = []
    matched_right = []
    max_x_diff = 30 
    max_y_diff = 30 

    for ly, lx in left_keypoints:
        symmetric_x = 2 * center_x - lx
        closest_right = min(
            right_keypoints,
            key=lambda p: abs(p[1] - symmetric_x) + abs(p[0] - ly),
            default=None
        )
        if closest_right:
            ry, rx = closest_right
            if abs(rx - symmetric_x) <= max_x_diff and abs(ry - ly) <= max_y_diff:
                matched_left.append((ly, lx))
                matched_right.append(closest_right)
                right_keypoints.remove(closest_right)

    keypoints_list = {"left": matched_left, "right": matched_right}

    height, width = binary_mask.shape[:2]
    corners = {
        "bottom": (0, 0),
        "top": (height - 1, 0),
    }

    corner_matches = {}
    for corner_name, corner_pos in corners.items():
        closest_match = min(
            zip(matched_left, matched_right),
            key=lambda pair: np.linalg.norm(np.array(pair[0]) - np.array(corner_pos)),
            default=None
        )
        if closest_match:
            corner_matches[corner_name] = {"left": closest_match[0], "right": closest_match[1]}

    output = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    for (lx, ly), (rx, ry) in zip(matched_left, matched_right):
        cv2.circle(output, (lx, ly), 4, (0, 0, 255), -1)
        cv2.circle(output, (rx, ry), 4, (255, 0, 0), -1)
        cv2.line(output, (lx, ly), (rx, ry), (0, 255, 0), 1)

    for corner_name, match in corner_matches.items():
        lx, ly = match["left"]
        rx, ry = match["right"]
        cv2.circle(output, (lx, ly), 6, (255, 255, 0), -1)
        cv2.circle(output, (rx, ry), 6, (255, 255, 0), -1)
        cv2.putText(output, corner_name, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite('outputs/keypoints.png', output)

    return corner_matches


def front_mask_and_pos_pixel(mesh_faces, pos, depth = None, image_dim = 480):

    if depth is None:
        particle_face = pos[mesh_faces, :].reshape(-1,3,3)
        edge1 = particle_face[:, 1] - particle_face[:, 0]
        edge2 = particle_face[:, 2] - particle_face[:, 0]
        face_normals = np.cross(edge1, edge2)
        face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None] # F*3

        vertex_normals = np.zeros_like(pos) # V*3
        np.add.at(vertex_normals, mesh_faces.flatten(), np.repeat(face_normals, 3, axis=0).reshape(-1, 3))

        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals /= norms
        front_mask = np.dot(vertex_normals, np.array([0, 1, 0])) > 0
    else: 
        front_mask = np.ones(pos.shape[0],dtype=bool)

    cam_pose = compute_pose(pos=[0, 2, 0], lookat=[0, 0, 0], up=[0, 0, 1])
    intrinsics_matrix = compute_intrinsics(39.5978, 480)

    pos_homogeneous = np.concatenate([pos, np.ones((pos.shape[0], 1))], axis=1)
    pos_cam = np.dot(np.linalg.inv(cam_pose), pos_homogeneous.T).T

    pos_pixel = np.dot(intrinsics_matrix, pos_cam[:, :3].T).T
    pos_pixel = pos_pixel[:, :2] / pos_pixel[:, 2:3] 
    pos_pixel[:,0] = image_dim - pos_pixel[:,0] - 1
    pos_pixel = np.round(pos_pixel).astype(int)
    pos_mask = np.all((pos_pixel >= 0) & (pos_pixel < image_dim), axis=-1)
    
    front_mask = front_mask & pos_mask
    if depth is not None:
        depth_mask = np.zeros_like(front_mask)
        depth_mask[front_mask] = ((depth[pos_pixel[front_mask][:,0],pos_pixel[front_mask][:,1]]-pos_cam[front_mask][:,2]) > -0.01)
        front_mask = front_mask & (depth_mask>0)
        
    return front_mask, pos_pixel

def get_action_pixel_map(rgb, scale_factors, rotations, dims=128, pix_grasp_dist=16):
    image_dim = rgb.shape[0]

    bins = len(scale_factors)*len(rotations)

    x = np.arange(bins) 
    y = np.arange(dims) 
    z = np.arange(dims) 

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    indices = np.stack((X, Y, Z), axis=-1)

    p1_indices = indices[:,:,:,1:].copy()
    p2_indices = indices[:,:,:,1:].copy()

    p1_indices[:,:,:,0] += pix_grasp_dist
    p2_indices[:,:,:,0] -= pix_grasp_dist


    num_scales = len(scale_factors)
    rotation_idx = np.floor_divide(x, num_scales)
    scale_idx = x - rotation_idx * num_scales
    scale = scale_factors[scale_idx]
    rotation = rotations[rotation_idx]

    original_dim = image_dim
    resized_dim = dims
    
    resize_mat = scale2d(original_dim/resized_dim)

    # scale
    scale_mat = [np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            scale2d(s),
        ), translate2d(np.ones(2)*(resized_dim//2))) for s in scale]
    
    # rotation
    rot_mat = [np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            rot2d(-r),
        ), translate2d(np.ones(2)*(resized_dim//2))) for r in rotation]
    
    mat = [np.matmul(np.matmul(s, r), resize_mat) for s, r in zip(scale_mat, rot_mat)]# 85*3*3
    
    trans2ori_map = np.concatenate((np.stack((p1_indices, p2_indices), axis=-2),np.ones((bins,dims,dims,2,1))),axis=-1)
    action_map = np.array([np.matmul(p, m)[:,:,:,:2].astype(int) for p,m in zip(trans2ori_map, mat)])# 85*128*128*2*2

    return action_map
    

def get_action_pos_map(pos_to_coords, pos_mask, mask, rgb, action_pixel_map):
    image_dim = rgb.shape[0]
    action_bin = mask.shape[0]
    action_dim = mask.shape[1]

    index_map = -np.ones((image_dim,image_dim),dtype=int)
    index_map[pos_to_coords[pos_mask][:,0], pos_to_coords[pos_mask][:,1]] = np.where(pos_mask)
    cv2.imwrite("outputs/index_map.png",(index_map>=0).astype(np.uint8)*255)

    interpolation_mask = (index_map==-1) & (rgb.sum(axis=-1)>0)
    _, nearest_index = distance_transform_edt(interpolation_mask, return_indices=True)
    interpolation_index_map = index_map[tuple(nearest_index)]

    action_pos_map = -np.ones((action_bin, action_dim, action_dim, 2),dtype=int)

    valid_mask = (mask) & (np.all((action_pixel_map>=0) & (action_pixel_map<image_dim),axis=(-1,-2)))

    action_point = action_pixel_map[np.where(valid_mask)]

    x_coords = action_point[:,:,0].reshape(-1)
    y_coords = action_point[:,:,1].reshape(-1)
    action_pos_map[np.where(valid_mask)] = interpolation_index_map[x_coords,y_coords].reshape(-1,2) 

    return action_pos_map


def visual_value_map(label_value_map, save_path, scale_num=5):

    fig, axs = plt.subplots(label_value_map.shape[0]//scale_num, scale_num, figsize=(10/5*scale_num, 30)) 
    
    for i in range(label_value_map.shape[0]):
        ax = axs[i // scale_num, i % scale_num]

        img = label_value_map[i] 
        
        if img.ndim == 2:
            im = ax.imshow(img, cmap='jet')
        else:
            im = ax.imshow(np.transpose(img[:3], (1, 2, 0))) 
        
        ax.set_title(f"{img.min():.2f},{img.max():.2f}", fontsize=8)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path))
    plt.close()

def get_rotation_matrix(r):
    r = np.deg2rad(r)
    return np.array([[np.cos(r), -np.sin(r)],
                        [np.sin(r), np.cos(r)]])
    
def get_data_dict(mesh_faces, init_pos, init_rgb, init_image_dim=960, init_dim=256, pix_grasp_dist=16, save_dir=None):
    print("[SimEnv] get init obs...")
    start_time = time.time()

    front_mask, init_pos_pixel = front_mask_and_pos_pixel(mesh_faces, init_pos)

    init_mask = init_rgb.sum(axis=-1) > 0
    x_indices, y_indices = np.where(init_mask)
    max_y, min_y = np.max(y_indices), np.min(y_indices)
    max_x, min_x = np.max(x_indices), np.min(x_indices)

    fused_rgb = np.concatenate((init_rgb[min_x:max_x,min_y:max_y],\
                                np.flip(init_rgb[min_x:max_x,min_y:max_y],axis=1)),axis=1)

    size = init_image_dim

    if max(2 * (max_y - min_y), max_x - min_x) > init_image_dim:
        raise Exception("Image size is too large")
    
    pad_rows = (size - (max_x - min_x)) // 2
    pad_cols = (size - 2*(max_y - min_y)) // 2
    pad_delta = [(size - (max_x - min_x)) % 2, (size - 2*(max_y - min_y)) % 2]
    fused_rgb = np.pad(fused_rgb,((pad_rows + pad_delta[0], pad_rows), 
                                        (pad_cols + pad_delta[1], pad_cols), (0,0)),
                                        mode='constant')

    fused_mask = fused_rgb.sum(axis=-1) > 0

    pos_index_to_coords = init_pos_pixel.copy()
    pos_index_to_coords[:,0] += (pad_rows + pad_delta[0] - min_x)
    pos_index_to_coords[:,1] += (pad_cols + pad_delta[1] - min_y)
    pos_index_to_coords[~front_mask] = size - pos_index_to_coords[~front_mask] + pad_delta
    pos_index_to_coords = np.clip(pos_index_to_coords,0,size-1)

    fused_obs = preprocess_obs(fused_rgb.copy(), fused_mask.copy())

    scale_bins = 24
    rotation_bins = 24
    rotations = np.linspace(-180, 180, rotation_bins + 1)

    max_scale_factor = max(2 * (max_y - min_y), max_x - min_x)/init_image_dim*init_dim/(2*pix_grasp_dist)
    min_scale_factor = max(2 * (max_y - min_y), max_x - min_x)/init_image_dim*1.3
    scale_factors = np.linspace(min_scale_factor, max_scale_factor, scale_bins)

    transformed_obs = generate_transformed_obs(fused_obs, init_dim, scale_factors, rotations)

    init_label2coords = pos_index_to_coords
    init_adaptive_scale_factors = scale_factors
    init_rotations = rotations
    init_rotations_matrix = np.array([get_rotation_matrix(-r) for r in init_rotations])

    if save_dir is not None:
        cv2.imwrite(f"{save_dir}/fused_rgb.png", fused_rgb)
        fused_index = np.zeros((size, size))
        fused_index[pos_index_to_coords[:,0], pos_index_to_coords[:,1]] = 1
        cv2.imwrite(f"{save_dir}/fused_index.png", fused_index*255)
        visual_value_map(transformed_obs['transformed_obs'], f"transformed_obs.png", scale_num=scale_bins)
        visual_value_map(transformed_obs['fling_mask'], f"transformed_mask.png", scale_num=scale_bins)

    print("[SimEnv] cost time:", time.time()-start_time)

    return {"obs": transformed_obs['transformed_obs'],
            "mask": transformed_obs['fling_mask'],
            "pos_to_pixel": init_label2coords,
            "scale_factors": init_adaptive_scale_factors,
            "rotations": init_rotations,
            "image_dim": init_image_dim,
            "obs_dim": init_dim,
            "pix_grasp_dist": pix_grasp_dist}
    
def generate_keypoints(rgb, pos, mesh_faces):

    num_particles = pos.shape[0]
    image_dim = rgb.shape[0]

    keypoints = generate_keypoint(rgb)
    front_mask, pos_pixel = front_mask_and_pos_pixel(mesh_faces, pos, image_dim=image_dim)
    index_map = -np.ones((image_dim, image_dim),dtype=int)
    index_map[pos_pixel[:,0], pos_pixel[:,1]] = np.arange(num_particles)
    keypoint_group = {}
    index_range = 50
    for key, item in keypoints.items():
        for id, idx in item.items():
            y, x = idx

            y_min = max(y - index_range, 0)
            y_max = min(y + index_range, 480)
            x_min = max(x - index_range, 0)
            x_max = min(x + index_range, 480)

            region = index_map[y_min:y_max, x_min:x_max]

            coords = np.argwhere(region)  
            coords += [y_min, x_min] 

            distances = np.linalg.norm(coords - np.array([y, x]), axis=1)
            sorted_indices = coords[np.argsort(distances)]

            mapped_values = index_map[sorted_indices[:, 1], sorted_indices[:, 0]]
            valid_mask = mapped_values != -1
            keypoint_group[f'{key}_{id}'] = mapped_values[valid_mask]

    return keypoint_group

def preprocess_obs(rgb, d):
    preprocessed_obs = torch.cat((torch.tensor(rgb).float()/255,
                torch.tensor(d).unsqueeze(dim=2).float()),
               dim=2).permute(2, 0, 1)
    return preprocessed_obs

def deformable_distance(goal_verts, current_verts, max_coverage, deformable_weight=0.65, flip_x=True, icp_steps=1000, scale=None):

    goal_verts = goal_verts.copy()
    current_verts = current_verts.copy()

    #flatten goals
    goal_verts[:, 1] = 0
    current_verts[:, 1] = 0
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
    #superimpose current to goal
    icp_verts = superimpose(current_verts, goal_verts)
    for i in range(5):
        threshold = THRESHOLD_COEFF * np.sqrt(max_coverage)
        indices = np.linalg.norm(icp_verts - goal_verts, axis=1) < threshold
        icp_verts = superimpose(icp_verts, goal_verts, indices=indices)

    #superimpose reverse goal to current
    reverse_goal_verts = goal_verts.copy()
    R, t = rigid_transform_3D(reverse_goal_verts.T, current_verts.T)
    reverse_goal_verts = (R @ reverse_goal_verts.T + t).T
    indices = np.linalg.norm(reverse_goal_verts - current_verts, axis=1) < threshold
    reverse_goal_verts = superimpose(reverse_goal_verts, current_verts, indices=indices)

    reverse_goal_cloud = o3d.geometry.PointCloud()
    reverse_goal_cloud.points = o3d.utility.Vector3dVector(reverse_goal_verts.copy())
    reverse_goal_cloud.paint_uniform_color([1, 0, 1])

    icp_verts_cloud = o3d.geometry.PointCloud()
    icp_verts_cloud.points = o3d.utility.Vector3dVector(icp_verts.copy())
    icp_verts_cloud.paint_uniform_color([0, 0, 1])

    l2_regular = np.mean(np.linalg.norm(icp_verts - goal_verts, axis=1)) # R_C deformable
    l2_flipped = np.mean(np.linalg.norm(icp_verts - flipped_goal_verts, axis=1))
    l2_distance = min(l2_regular, l2_flipped)

    icp_distance_regular = np.mean(np.linalg.norm(goal_verts - reverse_goal_verts, axis=1)) # R_A rigid
    icp_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - reverse_goal_verts, axis=1))
    icp_distance = min(icp_distance_regular, icp_distance_flipped)

    #make reward scale invariant
    assert(max_coverage != 0 or scale != 0)
    if scale is None:
        l2_distance /= np.sqrt(max_coverage)
        icp_distance /= np.sqrt(max_coverage)
        real_l2_distance /= np.sqrt(max_coverage)
    else:
        l2_distance /= scale
        icp_distance /= scale
        real_l2_distance /= scale

    weighted_distance = deformable_weight * l2_distance + (1 - deformable_weight) * icp_distance

    return weighted_distance, l2_distance, icp_distance, real_l2_distance, {"init_vert_cloud": goal_vert_cloud, "normal_init_vert_cloud": normal_init_vert_cloud , "verts_cloud": verts_cloud, 'icp_verts_cloud': icp_verts_cloud, "reverse_init_verts_cloud": reverse_goal_cloud}

def shift_tensor(tensor, offset):
    new_tensor = torch.zeros_like(tensor).bool()
    #shifted up
    if offset > 0:
        new_tensor[:, :-offset, :] = tensor[:, offset:, :]
    #shifted down
    elif offset < 0:
        offset *= -1
        new_tensor[:, offset:, :] = tensor[:, :-offset, :]
    return new_tensor

def generate_workspace_mask(left_mask, right_mask, action_primitives, pix_place_dist, pix_grasp_dist):
                                
    workspace_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':

            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_place_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_place_dist)
            #WORKSPACE CONSTRAINTS (ensures that both the pickpoint and the place points are located within the workspace)
            left_primitive_mask = torch.logical_and(left_mask, lowered_left_primitive_mask)
            right_primitive_mask = torch.logical_and(right_mask, lowered_right_primitive_mask)
            primitive_workspace_mask = torch.logical_or(left_primitive_mask, right_primitive_mask)

        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':

            raised_left_primitive_mask = shift_tensor(left_mask, pix_grasp_dist)
            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_grasp_dist)
            raised_right_primitive_mask = shift_tensor(right_mask, pix_grasp_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_grasp_dist)
            #WORKSPACE CONSTRAINTS
            aligned_workspace_mask = torch.logical_and(raised_left_primitive_mask, lowered_right_primitive_mask)
            opposite_workspace_mask = torch.logical_and(raised_right_primitive_mask, lowered_left_primitive_mask)
            primitive_workspace_mask = torch.logical_or(aligned_workspace_mask, opposite_workspace_mask)
        
        workspace_masks[primitive] = primitive_workspace_mask

    return workspace_masks 

def generate_primitive_cloth_mask(cloth_mask, action_primitives, pix_place_dist, pix_grasp_dist):
    cloth_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':
            primitive_cloth_mask = cloth_mask
        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':
            #CLOTH MASK (both pickers grasp the cloth)
            raised_primitive_cloth_mask = shift_tensor(cloth_mask, pix_grasp_dist)
            lowered_primitive_cloth_mask = shift_tensor(cloth_mask, -pix_grasp_dist)
            primitive_cloth_mask = torch.logical_and(raised_primitive_cloth_mask, lowered_primitive_cloth_mask)
        else:
            raise NotImplementedError
        cloth_masks[primitive] = primitive_cloth_mask
    return cloth_masks

def prepare_image(img, transformations, dim: int,
                  parallelize=False, log=False, orientation_net=None, nocs_mode=None, constant_positional_enc = False, inter_dim=256):

    assert nocs_mode == "collapsed" or nocs_mode == "distribution"

    if orientation_net is not None:

        mask = torch.sum(img[:3,], axis=0) > 0
        mask = torch.unsqueeze(mask, 0)

        #resize to network input shape
        input_img = transforms.functional.resize(img, (128, 128))


        with torch.no_grad():
            prepped_img = torch.unsqueeze(input_img[:3, :, :], 0).cpu()
            #print the type of prepped img
            out = ray.get(orientation_net.forward.remote(prepped_img))[0]
            # out = orientation_net.forward(torch.unsqueeze(input_img[:3, :, :], 0))[0]

        nocs_x_bins = out[:, 0, :, :]
        nocs_y_bins = out[:, 1, :, :]
        n_bins = out.shape[0]

        #out shape: 32, 2, 128, 128
        if nocs_mode == "collapsed":
            # mask = torch.cat(2*[torch.unsqueeze(mask, 0)], dim=0)
            #32 bins
            nocs_x = torch.unsqueeze(torch.argmax(nocs_x_bins, dim=0).type(torch.float32)/(n_bins-1), 0)
            nocs_y = torch.unsqueeze(torch.argmax(nocs_y_bins, dim=0).type(torch.float32)/(n_bins-1), 0)
            #mask out bg
            nocs = torch.cat([nocs_x, nocs_y], dim=0)

        elif nocs_mode == "distribution":
            # mask = torch.cat((n_bins * 2)*[torch.unsqueeze(mask, 0)], dim=0)

            nocs_x = torch.nn.functional.softmax(nocs_x_bins, dim=0)
            nocs_y = torch.nn.functional.softmax(nocs_y_bins, dim=0)

            nocs = torch.cat([nocs_x, nocs_y], dim=0)

            nocs = nocs[::2] + nocs[1::2]
        else:
            raise NotImplementedError 

        #to make things more computationally tractable
        # print("NOCS shape", nocs.shape)
        nocs = transforms.functional.resize(nocs, (img.shape[-1], img.shape[-2])).to(img.device)
        nocs = nocs * mask.int() + (1 - mask.int()) * 0.0


        img = torch.cat([img, nocs], dim=0)

    log = False
    if log:
        start = time()

    img = img.cpu()
    img = transforms.functional.resize(img, (inter_dim, inter_dim))
    imgs = torch.stack([transform(img, *t, dim=dim, constant_positional_encoding=constant_positional_enc) for t in transformations])

    if log:
        print(f'\r prepare_image took {float(time()-start):.02f}s with parallelization {parallelize}')

    return imgs.float()


def get_transformations(rotations, 
                        scale_factors = None):
    return list(product(
        rotations, scale_factors))
    
# @profile(stream=fp)
def generate_transformed_obs(obs,
                             input_dim = None,
                             scale_factors = None,
                             rotations = None,
                             primitive_vmap_indices = None,
                             pix_grasp_dist = 16,
                             pix_place_dist = 10,
                             action_primitives = ['fling', 'place']):
    """
    Generates transformed observations and masks
    """
    input_dim = input_dim
    scale_factors = scale_factors
    rotations = rotations
    retval = {}

    ##GENERATE OBSERVATION

    retval['transformed_obs'] = prepare_image(
                    obs, 
                    get_transformations(rotations, scale_factors), 
                    input_dim,
                    orientation_net=None,
                    parallelize=False,
                    nocs_mode='collapsed',
                    inter_dim=256,
                    constant_positional_enc=True,)   

    def get_cloth_mask(rgb):
        return rgb.sum(axis=0) > 0
    
    ##GENERATE MASKS
    pretransform_cloth_mask = get_cloth_mask(obs[:3])
    pretransform_left_arm_mask = torch.ones_like(pretransform_cloth_mask)
    pretransform_right_arm_mask = torch.ones_like(pretransform_cloth_mask) 

    pretransform_mask = torch.stack([pretransform_cloth_mask, 
                                    pretransform_left_arm_mask, 
                                    pretransform_right_arm_mask], 
                                    dim=0)
    
    transformed_mask = prepare_image(
                    pretransform_mask, 
                    get_transformations(rotations, scale_factors), 
                    input_dim,
                    parallelize=False,
                    nocs_mode='collapsed',
                    inter_dim=256,
                    constant_positional_enc=True,)   

    cloth_mask = transformed_mask[:, 0]
    left_arm_mask = transformed_mask[:, 1]
    right_arm_mask = transformed_mask[:, 2]

    workspace_mask = generate_workspace_mask(left_arm_mask, 
                                            right_arm_mask, 
                                            action_primitives, 
                                            pix_place_dist, 
                                            pix_grasp_dist)
    
    cloth_mask = generate_primitive_cloth_mask(
                            cloth_mask,
                            action_primitives,
                            pix_place_dist,
                            pix_grasp_dist)

    for primitive in action_primitives:

        GUARANTEE_OFFSET=6
        offset = pix_grasp_dist if primitive == 'fling' else pix_place_dist + GUARANTEE_OFFSET
        valid_transforms_mask = torch.zeros_like(cloth_mask[primitive]).bool()
        if primitive_vmap_indices is None:
            valid_transforms_mask[:, offset:-offset, offset:-offset] = True
        else:
            primitive_vmap_indices = primitive_vmap_indices[primitive]
            valid_transforms_mask[primitive_vmap_indices[0]:primitive_vmap_indices[1], 
                                  offset:-offset,
                                  offset:-offset] = True
            
        table_mask = retval['transformed_obs'][:, 3] > 0
        offset_table_mask_up = torch.zeros_like(table_mask).bool()
        offset_table_mask_down = torch.zeros_like(table_mask).bool()
        offset_table_mask_up[:, :-offset, :] = table_mask[:, offset:]
        offset_table_mask_down[:, offset:, :] = table_mask[:, :-offset]
        table_mask = offset_table_mask_up & offset_table_mask_down & table_mask

        primitive_workspace_mask = torch.logical_and(workspace_mask[primitive], table_mask)
        primitive_workspace_mask = torch.logical_and(primitive_workspace_mask, valid_transforms_mask)

        retval[f"{primitive}_cloth_mask"] = cloth_mask[primitive]
        retval[f"{primitive}_workspace_mask"] = primitive_workspace_mask
        retval[f"{primitive}_mask"] = torch.logical_and(cloth_mask[primitive], primitive_workspace_mask)
    return retval
    
def grid_index(x, y, dimx):
    return y*dimx + x


def get_cloth_mesh(
        dimx,
        dimy,
        base_index=0):
    if dimx == -1 or dimy == -1:
        positions = pyflex.get_positions().reshape((-1, 4))
        vertices = positions[:, :3]
        faces = pyflex.get_faces().reshape((-1, 3))
    else:
        positions = pyflex.get_positions().reshape((-1, 4))
        faces = []
        vertices = positions[:, :3]
        for y in range(dimy):
            for x in range(dimx):
                if x > 0 and y > 0:
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y-1, dimx),
                        base_index + grid_index(x, y, dimx)
                    ])
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y, dimx),
                        base_index + grid_index(x-1, y, dimx)])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def blender_render_cloth(cloth_mesh, resolution):
    output_prefix = '/tmp/' + str(os.getpid())
    obj_path = output_prefix + '.obj'
    cloth_mesh.export(obj_path)
    commands = [
        'blender',
        'cloth.blend',
        '-noaudio',
        '-E', 'BLENDER_EEVEE',
        '--background',
        '--python',
        'render_rgbd.py',
        obj_path,
        output_prefix,
        str(resolution)]
    with open(devnull, 'w') as FNULL:
        while True:
            try:
                # render images
                subprocess.check_call(
                    commands,
                    stdout=FNULL)
                break
            except Exception as e:
                print(e)
    # get images
    output_dir = Path(output_prefix)
    color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
    color = color[:, :, :3]
    redstr = depth.channel('R', PixelType(PixelType.FLOAT))
    depth = np.fromstring(redstr, dtype=np.float32)
    depth = depth.reshape(resolution, resolution)
    return color, depth


def pixels_to_3d_positions(
        transform_pixels, scale, rotation, pretransform_depth,
        transformed_depth, pose_matrix=None,
        pretransform_pix_only=False, **kwargs):

    # print("\n\n")
    # print("transform rotation: ", rotation)
    # print("transform scale: ", scale)
    # print("original dimensions: ", pretransform_depth.shape[0])
    # print("transformed dimensions: ", transformed_depth.shape[0]) 

    mat = get_transform_matrix(
        original_dim=pretransform_depth.shape[0],
        resized_dim=transformed_depth.shape[0],
        rotation=-rotation,  # TODO bug
        scale=scale)

    # print("Pixels before matmul: ", transform_pixels)
    pixels = np.concatenate((transform_pixels, np.array([[1], [1]])), axis=1)
    pixels = np.matmul(pixels, mat)[:, :2].astype(int)
    max_idx = pretransform_depth.shape[0]
    pixels = np.clip(pixels, 0, max_idx-1) 
    pix_1, pix_2 = pixels
    transformed_depth[transform_pixels[0][0], transform_pixels[0][1]] = 0
    transformed_depth[transform_pixels[1][0], transform_pixels[1][1]] = 1
    
    if (pixels < 0).any() or (pixels >= max_idx).any():
        print("pixels out of bounds", pixels, "\n\n\n")
        return {
            'valid_action': False,
            'p1': None, 'p2': None,
            'pretransform_pixels': np.array([pix_1, pix_2])
        }
    # if pretransform_pix_only:
    #     return {
    #         'valid_action': True,
    #         'pretransform_pixels': np.array([pix_1, pix_2])
    #     }
    # Note this order of x,y is not a bug
    x, y = pix_1
    p1 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)
    # Same here
    x, y = pix_2
    p2 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)

    return {
        'valid_action': p1 is not None and p2 is not None,
        'p1': p1,
        'p2': p2,
        'pretransform_pixels': np.array([pix_1, pix_2])
    }


def pixel_to_action_map(pixels, obs_dim, image_dim, rotations, scale_factors, pix_grasp_dist):
    rotation_matrixs = np.array([get_rotation_matrix(-r) for r in rotations])
    
    center = pixels.mean(axis=-2)*obs_dim/image_dim-obs_dim/2
    scale = obs_dim/image_dim*(np.linalg.norm(pixels[..., 0, :] - \
                                                        pixels[..., 1, :], axis=-1))/(2*pix_grasp_dist)
    rotation = np.degrees(np.arctan2(pixels[..., 0, 1] - pixels[..., 1, 1], \
                                        pixels[..., 0, 0] - pixels[..., 1, 0]))

    
    rotation_diff = np.abs(rotation - rotations[:, np.newaxis]).T
    rotation_indices = np.argmin(rotation_diff, axis=1).astype(int)

    min_scale = np.cos(np.min(rotation_diff,axis=1)*np.pi/180)*scale
    scale_diff = np.abs(min_scale - scale_factors[:, np.newaxis]).T

    scale_indices = np.argmin(scale_diff, axis=1).astype(int)
    center_indices = np.round(np.einsum('ijk,ik->ij', rotation_matrixs[rotation_indices], \
                                        center/scale_factors[scale_indices][:,None])+obs_dim/2).astype(int)
    
    return rotation_indices*len(scale_factors)+scale_indices, center_indices[..., 0], center_indices[..., 1]

