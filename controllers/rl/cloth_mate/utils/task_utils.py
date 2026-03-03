import numpy as np 
import torch, h5py
from clothmate.utils.flex_utils import (
    set_scene,
    get_default_config,
    center_object,
    wait_until_stable,
    get_current_covered_area,
    get_rgb,
    PickerPickPlace
)
from tqdm import tqdm
import hashlib
from pathlib import Path
import pickle
from copy import deepcopy
import numpy as np
import pyflex
import random
import trimesh
import os
import json
from functools import partial
from argparse import Namespace
from argparse import ArgumentParser
import ray
from filelock import FileLock

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def release_point(pickpoint, mass):
    curr_pos = pyflex.get_positions()
    curr_pos[pickpoint * 4 + 3] = mass
    pyflex.set_positions(curr_pos)

def move(pickpoint, final_point, speed=0.05, cond=lambda : True, step_fn=None):
    curr_pos = pyflex.get_positions()
    init_point = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()
    for j in range(int(1/speed)):
        if not cond():
            break

        curr_pos = pyflex.get_positions()
        curr_vel = pyflex.get_velocities()
        pickpoint_pos = (final_point-init_point)*(j*speed) + init_point
        curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
        curr_pos[pickpoint * 4 + 3] = 0
        curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]

        pyflex.set_positions(curr_pos)
        pyflex.set_velocities(curr_vel)

        if step_fn is not None:
            step_fn()

def grasp_point(pickpoint):
    curr_pos = pyflex.get_positions()

    mass = curr_pos[pickpoint * 4 + 3]
    position = curr_pos[pickpoint * 4: pickpoint * 4 + 3]
    curr_pos[pickpoint * 4 + 3] = 0

    pyflex.set_positions(curr_pos)
    
    return mass, position

def pyflex_step_raw(data, info):
    if 'env_mesh_vertices' not in data:
        data['env_mesh_vertices'] = []
    data['env_mesh_vertices'].append(pyflex.get_positions().reshape((-1, 4))[:info['num_particles'], :3])
    pyflex.step()

def get_rotation_matrix(rotationVector, angle):
    angle = float(angle)
    axis = rotationVector/np.sqrt(np.dot(rotationVector , rotationVector))
    a = np.cos(angle/2)
    b,c,d = -axis*np.sin(angle/2.)
    return np.array( [ [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                       [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                       [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c] ]) 

def load_cloth(path):
    vertices, faces = [], []
    nocs = pickle.load(open(path,'rb'))
    vertices = nocs['verts']
    faces = nocs['faces']
    uvs = nocs['nocs']
 
    triangle_faces = list()
    for face in faces:
        triangle_faces.append(face[[0,1,2]])
        triangle_faces.append(face[[0,2,3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)
    
    vertices = np.array(vertices)
        
    vertices = vertices.dot(get_rotation_matrix(np.array([0, 1, 0]),np.pi))

    return vertices, \
        np.array(triangle_faces), \
        np.array(list(stretch_edges)), \
        np.array(list(bend_edges)), \
        np.array(list(shear_edges)),\
        np.array(uvs).astype(np.float32)

def generate_randomization(
        action_tool,
        cloth_mesh_path=None,
        mesh_category=None,
        task_difficulty='hard',
        task=None,
        scale=0.7,
        gui=False,
        randomize_direction=False,
        recreate_step=None,
        save_episode_data=False,
        cloth_mass=50,
        **kwargs):
    # args = kwargs
    args = Namespace(**kwargs)
    # assert task is not None

    config = deepcopy(get_default_config(scale=scale))

    mesh_stretch_edges = np.array([])
    mesh_bend_edges = np.array([])
    mesh_shear_edges = np.array([])
    mesh_faces = np.array([])
    mesh_nocs_verts = np.array([])

    assert cloth_mesh_path is not None

    json_path = os.path.join(cloth_mesh_path, mesh_category)

    instances_dict = json.load(open(json_path))
    train_instances = instances_dict['train']
    test_instances = instances_dict['test']
    cloth_category = train_instances[0].split('.')[0].split('_')[1]
        
    if args.task_id is not None:
        cloth_instance = args.task_id + f"_{cloth_category}.obj.pkl"
    else:

        if args.eval:
            pkl_files = test_instances
        else:
            pkl_files = train_instances
        cloth_instance = np.random.choice(pkl_files)

    pkl_path = os.path.join(cloth_mesh_path, cloth_category, cloth_instance)

    if recreate_step is not None:
        pkl_path = os.path.join(cloth_mesh_path, cloth_category, str(recreate_step['instance']))

    mesh_verts, mesh_faces, mesh_stretch_edges, \
        mesh_bend_edges, mesh_shear_edges, mesh_nocs_verts = load_cloth(pkl_path)
    
    #scale the cloth
    mesh_verts = mesh_verts * config['scale'] 

    num_particle = mesh_verts.shape[0]

    cloth_trimesh = trimesh.Trimesh(mesh_verts, mesh_faces)

    flattened_area = cloth_trimesh.area /2

    # Stretch, Bend and Shear Stiffness
    stiffness = (0.75, .02, .02)
    if mesh_category == 'Shirt':
        stiffness = (0.75, .02, .02)
    elif mesh_category == 'Trouser':
        stiffness = (0.9, 0.5, 1)

    config.update({
        'cloth_pos': [0, 1, 0],
        'cloth_stiff': stiffness,
        'cloth_mass': cloth_mass,
        'mesh_verts': mesh_verts.reshape(-1),
        'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
        'mesh_bend_edges': mesh_bend_edges.reshape(-1),
        'mesh_shear_edges': mesh_shear_edges.reshape(-1),
        'mesh_faces': mesh_faces.reshape(-1),
        'mesh_nocs_verts': mesh_nocs_verts,
    })

    mesh_save_data = {}
    info = {'num_particles': num_particle}
    pyflex_step = partial(pyflex_step_raw, mesh_save_data, info)

    config = set_scene(config)
    action_tool.reset([0., -1., 0.])
    particle_radius = config['scene_config']['radius']

    # Start with flattened cloth
    positions = pyflex.get_positions().reshape(-1, 4)
    pyflex.set_positions(positions)
    for _ in range(100):
        pyflex_step()

    center_object(num_particle)

    init_rgb = get_rgb()
    pre_cross_pos = pyflex.get_positions()

    curr_pos = pyflex.get_positions()
    xzy = curr_pos.reshape(-1, 4)[:num_particle, :3]
    x = xzy[:, 0]
    y = xzy[:, 2]

    cloth_height = float(np.max(y) - np.min(y))
    cloth_width = float(np.max(x) - np.min(x))
    
    # all_keypoint_groups = get_keypoint_groups(xzy)
    all_keypoint_groups = np.array([])

    init_positions = pyflex.get_positions()

    theta = 0
    if randomize_direction:
        theta = random.uniform(0, 2*np.pi)

    curr_verts = init_positions.copy().reshape(-1, 4)
    curr_verts[:, :3] = curr_verts[:, :3] @ get_rotation_matrix(np.array([0, 1, 0]),theta)

    pre_cross_pos = pre_cross_pos.reshape(-1, 4)
    pre_cross_pos[:, :3] = pre_cross_pos[:, :3] @ get_rotation_matrix(np.array([0, 1, 0]),theta)
    pre_cross_pos = pre_cross_pos.flatten()


    pyflex.set_positions(curr_verts.flatten())


    if recreate_step is not None:
        recreate_verts = recreate_step['vertices']
        v = pyflex.get_positions().reshape(-1, 4)
        v[:num_particle, :3] = recreate_verts
        pyflex.set_positions(v.flatten())
    

    center_object(num_particle)

    pickpoint = random.randint(0, num_particle - 1)

    if task_difficulty == 'hard':
        pyflex.set_positions(pre_cross_pos)
        mass, pickpoint_pos = grasp_point(pickpoint)
        rand_height = np.random.random(1) * 0.6 + 1
        target = pickpoint_pos + np.array([0.0, float(rand_height), 0.0])
        move(pickpoint, target, 0.005)
        wait_until_stable(gui=gui, step_sim_fn=pyflex_step)
        release_point(pickpoint, mass)

    elif task_difficulty == "easy":
        mass, pickpoint_pos = grasp_point(pickpoint)

        angle = np.random.random() * np.pi * 2
        magnitude = np.random.random() * 0.5
        height = np.random.random() * 0.5


        offset = np.array([np.cos(angle) * magnitude, height, np.sin(angle) * magnitude])
        target = pickpoint_pos + offset
        move(pickpoint, target, 0.01)
        release_point(pickpoint, mass)
            
    elif task_difficulty == 'none':
        pyflex_step()

    elif task_difficulty == 'flat':
        pyflex.set_positions(pre_cross_pos)
        pyflex_step()
    
    else:
        raise Exception("Not implemented")

    center_object(num_particle)
    random_translation_vector = np.array([random.uniform(-args.random_translation, args.random_translation), 0 ,random.uniform(-args.random_translation, args.random_translation)])
    positions = pyflex.get_positions()
    positions = positions.reshape(-1, 4)
    positions[:, :3] += random_translation_vector
    pyflex.set_positions(positions.flatten())
    pyflex_step()

    wait_until_stable(gui=gui, step_sim_fn=pyflex_step)

    task_rgb = get_rgb()

    heights = pyflex.get_positions().reshape(-1, 4)[:, 1]

    if heights.max() > 0.4:
        print("[TaskGenerator] Discarding task due to error due to height max")
        print(heights.max())
        return None

    if np.sum(task_rgb) == 0:
        print("TaskGenerator] Discarding task due to error due to empty scene")
        return None

    output = {
        'pickpoint':pickpoint,
        'particle_pos': pyflex.get_positions(),
        'init_particle_pos': init_positions,
        'particle_vel': pyflex.get_velocities(),
        'initial_coverage': get_current_covered_area(num_particle, particle_radius),
        'shape_pos': pyflex.get_shape_states(),
        'phase':  pyflex.get_phases(),
        'flatten_area': flattened_area,
        'flip_mesh': 0,
        'cloth_stiff': stiffness,
        'cloth_mass': cloth_mass,
        'task_difficulty': task_difficulty,
        'mesh_verts': mesh_verts.reshape(-1),
        'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
        'mesh_bend_edges': mesh_bend_edges.reshape(-1),
        'mesh_shear_edges': mesh_shear_edges.reshape(-1),
        'mesh_faces': mesh_faces.reshape(-1),
        'mesh_nocs_verts':mesh_nocs_verts,
        'cloth_instance':cloth_instance,
        'init_rgb':init_rgb,
        'task_rgb':task_rgb,
        'cloth_height':cloth_height,
        'cloth_width':cloth_width,
        'all_keypoint_groups':all_keypoint_groups,
        'pkl_path': pkl_path
    }

    if save_episode_data:
        output['episode_mesh_data'] = np.array(mesh_save_data['env_mesh_vertices'])
        # print("Episode shape", output['episode_mesh_data'].shape)

    # if os.path.exists(args.path):
    #     with FileLock(args.path + '.lock'):
    #         with h5py.File(args.path, 'r') as file:
    #             print(f"Progress:{len(file)}/{args.num_tasks}")

    # pyflex.clean()
    return output

def generate_tasks_helper(path: str,  gui: bool, recreate_steps=None, **kwargs):
    
    #for the pyflex init below
    msaaSamples = get_default_config()['scene_config']['msaaSamples']
    pyflex.init(
        not gui,  # headless: bool
        True,  # render: bool
        480, 
        480,
        0
        )  # camera dimensions: int x int
    action_tool = PickerPickPlace(
        num_picker=2,
        particle_radius=0.00625,
        picker_radius=0.05,
        picker_low=(-5, 0, -5),
        picker_high=(5, 5, 5))

    while True:

        if recreate_steps is not None:
            if len(recreate_steps) > 0:
                vertices, instance = recreate_steps.pop()
                print("Process has", len(recreate_steps), "tasks left")
            else:
                break

            task = generate_randomization(
                action_tool,
                gui=gui,
                recreate_step={"vertices":vertices, "instance":instance},
                **kwargs)
        else:
            task = generate_randomization(
                action_tool,
                gui=gui,
                **kwargs)
      
        if task is None:  
            continue
    
        if task['cloth_width'] * task['cloth_height'] == 0:
            continue

        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with FileLock(path + '.lock'):
            with h5py.File(path, 'a') as file:
                key = kwargs['mesh_category'].split('.')[0] + '_' + hashlib.sha1(f'{len(file)}'.encode()).hexdigest()
                group = file.create_group(key)
                for key, value in task.items():
                    if key == "cloth_instance":
                        value = str(value)
                    if type(value) == float or \
                            type(value) == int or\
                            type(value) == np.float64 or\
                            type(value) == str:
                        group.attrs[key] = value
                    else:
                        try:
                            group.create_dataset(
                                name=key,
                                data=value,
                                compression='gzip',
                                compression_opts=9)
                        except Exception as e:
                            print("Error writing to h5 file")
                            print(key, value)
                            print(type(value))
                            print(e)
                            exit(1)
                            # raise Exception("Error writing to h5 file")

#from flingbot
def get_default_config(
        particle_radius=0.0175,
        cloth_stiffness = (0.75, .02, .02),
        scale=0.8,
        ):
    config = {
        'scale':scale,
        'cloth_pos': [0.0, 1.0, 0.0],
        'cloth_size': [int(0.6 / particle_radius),
                       int(0.368 / particle_radius)],
        'cloth_stiff': cloth_stiffness,  # Stretch, Bend and Shear
        'camera_name': 'default_camera',
        'camera_params': {
            'default_camera':
                {
                    'render_type': ['cloth'],
                    'cam_position': [0, 2, 0],
                    'cam_angle': [np.pi/2, -np.pi / 2, 0],
                    'cam_size': [480, 480],
                    'cam_fov': 39.5978 / 180 * np.pi
                }
            },
        'scene_config': {
            'scene_id': 0,
            'radius': particle_radius * scale,
            'buoyancy': 0,
            'numExtraParticles': 20000,
            'collisionDistance': 0.0006,
            'msaaSamples': 0,
        },
        'flip_mesh': 0
    }

    return config

class Task:
    def __init__(self,
                 name: str,
                 flatten_area: float,
                 initial_coverage: float,
                 task_difficulty: str,
                 flip_mesh: int = 0,
                 particle_pos: np.array = [],
                 particle_vel: np.array = [],
                 shape_pos: np.array = [],
                 mesh_verts: np.array = [],
                 mesh_stretch_edges: np.array = [],
                 mesh_bend_edges: np.array = [],
                 mesh_shear_edges: np.array = [],
                 mesh_faces: np.array = [],
                 phase: np.array = [],
                 cloth_stiff: np.array = [],
                 cloth_mass: float = 0.5,
                 cloth_pos=[0, 2, 0],
                 pts_path= str,
                 pickpoint = int,
                 summary_path = str,
                 mesh_nocs_verts: np.array = [],
                 cloth_instance: int = 0,
                 init_particle_pos: np.array = [],
                 init_rgb: np.array = [],
                 task_rgb: np.array = [],
                 cloth_height: float = 0,
                 cloth_width: float = 0,
                 pkl_path: str = None,
                 all_keypoint_groups: np.array = [],
                ):
        self.name = name
        self.flatten_area = flatten_area
        self.initial_coverage = initial_coverage
        self.task_difficulty = task_difficulty
        self.cloth_mass = cloth_mass
        self.particle_pos = np.array(particle_pos)
        self.particle_vel = np.array(particle_vel)
        self.shape_pos = np.array(shape_pos)
        self.phase = np.array(phase)
        self.cloth_pos = np.array(cloth_pos)
        self.cloth_stiff = np.array(cloth_stiff)
        self.flip_mesh = flip_mesh
        self.mesh_verts = np.array(mesh_verts)
        self.mesh_stretch_edges = np.array(mesh_stretch_edges)
        self.mesh_bend_edges = np.array(mesh_bend_edges)
        self.mesh_shear_edges = np.array(mesh_shear_edges)
        self.mesh_faces = np.array(mesh_faces)
        self.mesh_nocs_verts = np.array(mesh_nocs_verts)
        self.init_rgb = np.array(init_rgb)
        self.task_rgb = np.array(task_rgb)
        self.all_keypoint_groups = np.array(all_keypoint_groups)

        config = get_default_config()
        self.camera_config = config['camera_params']
        self.scene_config = config['scene_config']
        self.scale = config['scale']

        self.pts_path = pts_path
        self.summary_path = summary_path
        self.pickpoint = pickpoint
        self.cloth_instance = cloth_instance
        self.cloth_category = name.split('_')[0]
        self.init_particle_pos = np.array(init_particle_pos)

        self.cloth_height = cloth_height
        self.cloth_width = cloth_width

        self.pkl_path = pkl_path

    def get_keypoint_data(self):
        keypoint_names = ['bottom_right', 'bottom_left', 'top_right', 'top_left', 'right_shoulder', 'left_shoulder']
        return {
            key: self.all_keypoint_groups[i] for i, key in enumerate(keypoint_names)
        }

    def get_config(self):
        return {
            'name': self.name,
            'cloth_pos': self.cloth_pos,
            'cloth_stiff': self.cloth_stiff,
            'cloth_mass': self.cloth_mass,
            'camera_name': 'default_camera',
            'camera_params': self.camera_config,
            'flip_mesh': self.flip_mesh,
            'flatten_area': self.flatten_area,
            'mesh_verts': self.mesh_verts,
            'mesh_stretch_edges': self.mesh_stretch_edges,
            'mesh_bend_edges': self.mesh_bend_edges,
            'mesh_shear_edges': self.mesh_shear_edges,
            'mesh_faces': self.mesh_faces,
            'mesh_nocs_verts': self.mesh_nocs_verts,
            'cloth_instance': self.cloth_instance,
            'cloth_category': self.cloth_category,
            'scene_config': self.scene_config,
            'scale': self.scale,
            'cloth_height': self.cloth_height,
            'cloth_width': self.cloth_width,
            'cloth_area': self.cloth_height * self.cloth_width,
            'task_difficulty': self.task_difficulty,
        }

    def get_images(self):
        return {
            'init_rgb': self.init_rgb,
            'task_rgb': self.task_rgb,
        }

    def get_state(self):
        return {
            'particle_pos': self.particle_pos,
            'particle_vel': self.particle_vel,
            'init_particle_pos': self.init_particle_pos,
            'shape_pos': self.shape_pos,
            'phase': self.phase,
            # 'camera_params': self.camera_params
        }

    def get_stats(self):
        return {
            'task_name': self.name,
            'cloth_mass': self.cloth_mass,
            'cloth_stiff': self.cloth_stiff,
            'max_coverage': self.flatten_area,
            'task_difficulty': self.task_difficulty,
            # 'init_coverage': self.initial_coverage
        }

    def get_garmentnets_data(self):
        return {
            'pts_path': self.pts_path,
            'summary_path': self.summary_path,
            'pickpoint': self.pickpoint
        }

    def __str__(self):
        output = f'[Task] {self.name}\n'
        output += f'\ttask_difficulty: {self.task_difficulty}\n'
        output += '\tinitial_coverage (%): ' +\
            f'{self.initial_coverage*100/self.flatten_area:.02f}\n'
        output += f'\tcloth_mass (kg): {self.cloth_mass:.04f}\n'
        output += f'\tcloth_stiff: {self.cloth_stiff}\n'
        output += f'\tflatten_area (m^2): {self.flatten_area:.04f}\n'
        output += f'\tcloth_instance: {self.cloth_instance}\n'
        return output


class TaskLoader:
    def __init__(self, hdf5_path: str, 
                    eval_hdf5_path: str, 
                     eval: bool = False, 
                     seed: int = 0,
                     recreate_task_query: float = None,
                     grid_search: bool = False,
                     get_init: str = None,
                     category: str = None,
                     task_id = None,
                     task_difficulty = ['hard']
                     ):
        
        seed_all(seed)
        self.task_id = task_id
        self.category = category
        self.hdf5_path = hdf5_path
        self.repeat = not eval
        self.keys = None
        self.recreate_task_query = recreate_task_query
        self.get_init = get_init

        if eval:
            print("[TaskLoader] Loading eval tasks")
            self.hdf5_path = eval_hdf5_path
            with h5py.File(self.hdf5_path, 'r') as tasks:
                if self.task_id is not None:
                    self.keys = self.task_id

                elif self.category is not None:
                    self.keys = []
                    for key in tasks:
                        if key.split('_')[0] == self.category and tasks[key].attrs['task_difficulty'] in task_difficulty:
                            self.keys.append(key)
                            
                else:
                    self.keys = [key for key in tasks if tasks[key].attrs['task_difficulty'] in task_difficulty]
        else:
            print("[TaskLoader] Loading train tasks")
            with h5py.File(self.hdf5_path, 'r') as tasks:
                if recreate_task_query is None:
                    if self.category is not None:
                        self.keys = []
                        for key in tasks:
                            if key.split('_')[0] == self.category:
                                self.keys.append(key)
                    else:
                        self.keys = [key for key in tasks if tasks[key].attrs['task_difficulty'] in task_difficulty]
                else:
                    print("[TaskLoader] Recreating task with query {}".format(recreate_task_query))
                    self.keys = [key for key in tasks if np.sum(get_init_verts_from_task(tasks[key])) == recreate_task_query]
                    if grid_search:
                        #ensure there are plenty of tasks for a grid search
                        MAX_GRID_SEARCH_TASKS = 100
                        self.keys = self.keys * MAX_GRID_SEARCH_TASKS

                print(f'[TaskLoader] Found {len(self.keys)} tasks from',
                    self.hdf5_path)
                if len(self.keys) == 0:
                    raise ValueError(f'[TaskLoader] No tasks found in {self.hdf5_path}')
        self.curr_task_idx = 0


    def get_next_task(self) -> Task:
        with h5py.File(self.hdf5_path, 'r') as tasks:
            
            if not self.repeat:
                if self.curr_task_idx >= len(self.keys):
                    print('[TaskLoader] Out of tasks')
                    exit(0)
                else:
                    key = self.keys[self.curr_task_idx]
                    group = tasks[key]
                    self.curr_task_idx += 1
                    print('[TaskLoader] {}/{}'.format(
                        self.curr_task_idx,
                        len(self.keys)))
                
            else:
                key = np.random.choice(self.keys)
                group = tasks[key]
                if self.get_init is not None:
                    if os.path.exists(self.get_init):
                        if self.exist_instance(self.get_init, tasks[key].attrs['cloth_instance']):
                            print('[TaskLoader] Skipping task with existing instance')
                            return self.get_next_task()
                if ('init_rgb' not in group) or (np.sum(group['init_rgb']) == 0):
                    print('[TaskLoader] Skipping task with no init_rgb')
                    return self.get_next_task()
            
            return Task(name=key, **group.attrs, **group)

    def exist_instance(self, h5py_path, cloth_instance):
        cloth_instance = cloth_instance.split('_')[0]
        with h5py.File(h5py_path, 'a') as f:
            if cloth_instance in f:
                return True
            else:
                return False
            
def get_init_verts_from_task(task_group):
    mesh_verts = task_group['mesh_verts']
    num_particles = mesh_verts.shape[0] // 3
    init_verts = np.array(task_group['init_particle_pos']).reshape(-1, 4)[:num_particles, :3]
    return init_verts
