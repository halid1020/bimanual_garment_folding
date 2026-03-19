import numpy as np, torch, imutils, os, h5py, ray, random
from torchvision import transforms
from typing import Any, Dict, List, MutableMapping, Tuple
from time import time
from filelock import FileLock
from clothmate.core.network import MaximumValuePolicy
# from clothmate.core.simEnv import SimEnv

def get_dataset_size(path, pbar=None):
    if not os.path.exists(path):
        return 0
    num = 0
    with FileLock(path + ".lock"):
        with h5py.File(path, "r") as file:
            for category in file.values():
                for instance in category.values():
                    num += len(instance)
            return num
        
def step_env(all_envs, ready_envs, ready_actions, remaining_observations, deterministic):
    remaining_observations.extend([e.step_and_record.remote(a)
                                   for e, a in zip(ready_envs, ready_actions)])
    step_retval = []
    start = time()
    total_time = 0
    while True:

        if deterministic:
            ready = ray.get(
                remaining_observations)
        else:
            ready, remaining_observations = ray.wait(
                remaining_observations, num_returns=1)

        if len(ready) == 0:
            continue
        step_retval.extend(ready)
        total_time = time() - start
        if (total_time > 0.01 and len(step_retval) > 0)\
                or len(step_retval) == len(all_envs):
            break

    observations = []
    ready_envs = []

    for obs, env_id in ray.get(step_retval):
        observations.append(obs)
        ready_envs.append(env_id['val'])

    return ready_envs, observations, remaining_observations



def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, List) and isinstance(v[0], MutableMapping):
            for idx in range(len(v)):
                items.extend(flatten_dict(v[idx], f"{new_key}/{idx}", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def pixel_to_3d(depth_im, x, y,
                pose_matrix,
                fov=39.5978,
                depth_scale=1):
    intrinsics_matrix = compute_intrinsics(fov, depth_im.shape[0])
    click_z = depth_im[y, x]
    click_z *= depth_scale
    click_x = (x-intrinsics_matrix[0, 2]) * \
        click_z/intrinsics_matrix[0, 0]
    click_y = (y-intrinsics_matrix[1, 2]) * \
        click_z/intrinsics_matrix[1, 1]
    if click_z == 0:
        raise Exception('Invalid pick point')
    # 3d point in camera coordinates
    point_3d = np.asarray([click_x, click_y, click_z])
    point_3d = np.append(point_3d, 1.0).reshape(4, 1)
    # Convert camera coordinates to world coordinates
    target_position = np.dot(pose_matrix, point_3d)
    target_position = target_position[0:3, 0]
    target_position[0] = - target_position[0]
    return target_position

def get_transform_matrix(original_dim, resized_dim, rotation, scale):
    # resize
    resize_mat = scale2d(original_dim/resized_dim)
    # scale
    scale_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            scale2d(scale),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    # rotation
    rot_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            rot2d(rotation),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    return np.matmul(np.matmul(scale_mat, rot_mat), resize_mat)


def compute_intrinsics(fov, image_size):
    image_size = float(image_size)
    focal_length = (image_size / 2)\
        / np.tan((np.pi * fov / 180) / 2)
    return np.array([[focal_length, 0, image_size / 2],
                     [0, focal_length, image_size / 2],
                     [0, 0, 1]])

def translate2d(translation):
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1],
    ]).T

def scale2d(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ]).T

def rot2d(angle, degrees=True):
    if degrees:
        angle = np.pi*angle/180
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]).T


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

    t = -R @ centroid_A + centroid_B

    return R, t

def superimpose(current_verts, goal_verts, indices=None, symmetric_goal=False):

    current_verts = current_verts.copy()
    goal_verts = goal_verts.copy()
    # flipped_goal_verts = goal_verts.copy()
    # flipped_goal_verts[:, 0] = 2*np.mean(flipped_goal_verts[:, 0]) - flipped_goal_verts[:, 0]

    if indices is not None:
        R, t = rigid_transform_3D(current_verts[indices].T, goal_verts[indices].T)
    else:
        R, t = rigid_transform_3D(current_verts.T, goal_verts.T)

    icp_verts = (R @ current_verts.T + t).T

    return icp_verts

def generate_coordinate_map(dim, rotation, scale, normalize=False):

    MAX_SCALE=5
    scale = 1

    coordinate_dim = int(dim * (MAX_SCALE/scale))
    x, y = np.meshgrid(MAX_SCALE * np.linspace(-1 , 1, coordinate_dim), MAX_SCALE * np.linspace(-1 , 1, coordinate_dim), indexing="ij")
    x, y = x.reshape(coordinate_dim, coordinate_dim, 1), y.reshape(coordinate_dim, coordinate_dim, 1)
    xy = np.concatenate((x, y), axis=2)
    xy = imutils.rotate(xy, rotation)
    center = coordinate_dim/2

    new_dim = int(center + center * scale/MAX_SCALE) - int(center - center * scale/MAX_SCALE)
    offset = 0
    if int(new_dim) != dim:  
        offset = int(dim - new_dim)

    xy = xy[int(center - center * scale/MAX_SCALE) : int(center + center * scale/MAX_SCALE) + offset, \
         int(center - center * scale/MAX_SCALE):int(center + center * scale/MAX_SCALE) + offset, :]

    return xy
    

# @profile
def transform(img, rotation: float, scale: float, dim: int, constant_positional_encoding: bool = False):


    #to adjust code
    rotation *= -1 

    img = transforms.functional.resize(img, (dim, dim))
    img = transforms.functional.rotate(img, rotation, interpolation=transforms.InterpolationMode.BILINEAR)
    
    if scale < 1:
        img = transforms.functional.center_crop(img, (int(dim * scale), int(dim * scale)))
        img = transforms.functional.resize(img, (dim, dim))
    else:
        zeros = torch.zeros((img.shape[0], dim, dim), device=img.device)
        img = transforms.functional.resize(img, (int(dim/scale), int(dim/scale)))

        end = zeros.shape[-1]//2 + int(img.shape[-1])//2
        begin = zeros.shape[-1]//2 - int(img.shape[-1])//2
        if end - begin < img.shape[-1]:
            end += 1        
        
        zeros[..., begin:end, begin:end] = img
        img = zeros

    coordinate_map = torch.tensor(generate_coordinate_map(dim, rotation, scale, normalize=constant_positional_encoding))
    coordinate_map = coordinate_map.permute(2, 0, 1)
    
    img = torch.cat([img, coordinate_map], axis=0)

    return img

def seed_all(seed):
    print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_network(args, gpu=0):
    if(args.grid_search):
        with h5py.File(args.recreate_buffer, "r") as dataset:
            grid_search_params = {
                'vmap_idx':args.grid_search_vmap_idx,
                'primitive':args.grid_search_primitive,
            }
            print("[Policy] Grid search with params:", grid_search_params)
        policy = MaximumValuePolicy(**args, gpu=gpu, grid_search_params=grid_search_params)
    else:
        policy = MaximumValuePolicy(**args, gpu=gpu)

    optimizer = torch.optim.Adam(
        policy.value_net.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)

    dataset_path = args.dataset_path

    checkpoint_path = None

    if os.path.exists(f'{args.cont}/latest_ckpt.pth'):
        checkpoint_path = f'{args.cont}/latest_ckpt.pth'
        print("[Network Setup] Load checkpoint specified", checkpoint_path)

    if args.load is not None:
        print("[Network Setup] Load checkpoint specified", args.load)
        checkpoint_path = args.load

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=policy.device, weights_only=True)
        policy.load_state_dict(ckpt['net'], strict=False)
        if args.load is None:
            optimizer.load_state_dict(ckpt[f'optimizer'])
        else:
            for name, param in policy.named_parameters():
                if name in ckpt['net'].keys():
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    print(f'\t[Network Setup] Action Exploration Probability: {policy.action_expl_prob.item():.4e}')
    print(f'\t[Network Setup] Value Exploration Probability: {policy.value_expl_prob.item():.4e}')
    print(f'\t[Network Setup] Train Steps: {policy.train_steps.item()}')

    return policy, optimizer, dataset_path