import h5py, os, json
import numpy as np
from clothmate.utils.env_utils import generate_keypoints

TASK_PATH = "data/task/train.hdf5"
KEP_PATH = "keypoint/dataset/keypoints.hdf5"

cloth_mesh_path = 'data/cloth3d_pickle'
cloth_categorys = ['longsleeve', 'dress', 'pants', 'skirt', 'jumpsuit']
cloth_instance = {}

for cloth_category in cloth_categorys:
    instances_dict = json.load(open(os.path.join(cloth_mesh_path, cloth_category + '.json')))
    cloth_instance[cloth_category] = instances_dict['train']


temp = {key: {k: [] for k in ['top_right', 'top_left', 'bottom_right', 'bottom_left']} for key in cloth_instance.keys()}

with h5py.File(TASK_PATH, 'r') as task_file, h5py.File(KEP_PATH, 'a') as kep_file:
    for category, instances in cloth_instance.items():
        for instance in instances:
            for key, item in task_file.items():
                if key.split('_')[0] == category and item.attrs['cloth_instance'] == instance:
                    print(f'Processing {category} {instance}')
                    mesh_faces = item['mesh_faces'][:]
                    num_particles = item['mesh_verts'].shape[0] // 3
                    init_pos = item['init_particle_pos'][:].reshape(-1, 4)[:num_particles, :3]
                    init_rgb = item['init_rgb'][:]
                    keypoints = generate_keypoints(init_rgb, init_pos, mesh_faces)
                    
                    for k in keypoints:
                        temp[category][k].append(init_pos[keypoints[k][0]])

                    instance_id = instance.split('_')[0]
                    group = kep_file.create_group(f'{category}_{instance_id}')
                    for k, v in keypoints.items():
                        group.create_dataset(name=k, data=np.array(v), compression='gzip')
                    break
        for k in ['top_right', 'top_left', 'bottom_right', 'bottom_left']:
            kep_file.attrs[f'{category}_{k}_mean'] = np.mean(temp[category][k], axis=0)
            kep_file.attrs[f'{category}_{k}_max'] = np.max(temp[category][k], axis=0)
            kep_file.attrs[f'{category}_{k}_min'] = np.min(temp[category][k], axis=0)