
# Borrow from LaGarNet's modification on cloth-funnel's script

import h5py
import hashlib

from env.utils.trial_utils import *
from env.utils.flex_utils import (
    set_scene,
    get_default_config,
    center_object,
    wait_until_stable,
    get_current_covered_area,
    PickerPickPlace,
    get_rgb
)

from copy import deepcopy
import numpy as np
import pyflex
import torch
import time
import random
from time import sleep
from typing import List
from tqdm import tqdm
from argparse import ArgumentParser
from filelock import FileLock
from pathlib import Path
import trimesh
import ray
import os
import json
from functools import partial


def pyflex_step_raw(data, info):
    if 'env_mesh_vertices' not in data:
        data['env_mesh_vertices'] = []
    data['env_mesh_vertices'].append(pyflex.get_positions().reshape((-1, 4))[:info['num_particles'], :3])
    pyflex.step()

def seed_all(seed):
    # print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_randomization(
        action_tool,
        cloth_mesh_path=None,
        pkl_dir_name=None,
        mesh_category=None,
        trial_difficulty='hard',
        trial=None,
        scale=0.7,
        gui=False,
        dump_images=None,
        randomize_direction=False,
        randomize_instance=False,
        random_translation=np.array([0, 0, 0]),
        stiffness_factor=1,
        recreate_step=None,
        save_episode_data=False,
        cloth_mass=50,
        **kwargs):

    assert trial is not None

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
        
    if args.trial_id is not None:
        cloth_instance = args.trial_id + f"_{cloth_category}.obj.pkl"
    else:

        if args.eval:
            pkl_files = test_instances
        else:
            pkl_files = train_instances
        cloth_instance = np.random.choice(pkl_files)

    
    # cloth_instance = pkl_files[args.trial_id]
        # print(f"Picking {mesh_category} instance with random index:", args.trial_id)
    pkl_path = os.path.join(cloth_mesh_path, cloth_category, cloth_instance)
    print(pkl_path)

    if recreate_step is not None:
        pkl_path = os.path.join(cloth_mesh_path, cloth_category, str(recreate_step['instance']))

    
    #sample a number between -args.random_translation to args.random_translation
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
    print('cloth_height:', cloth_height, 'cloth_width:', cloth_width)
    
    all_keypoint_groups = get_keypoint_groups(xzy)

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

    if trial_difficulty == 'hard':
        pyflex.set_positions(pre_cross_pos)
        mass, pickpoint_pos = grasp_point(pickpoint)
        rand_height = np.random.random(1) * 0.6 + 1
        target = pickpoint_pos + np.array([0.0, float(rand_height), 0.0])
        move(pickpoint, target, 0.005)
        wait_until_stable(gui=gui, step_sim_fn=pyflex_step)
        release_point(pickpoint, mass)

    elif trial_difficulty == "easy":
        mass, pickpoint_pos = grasp_point(pickpoint)

        angle = np.random.random() * np.pi * 2
        magnitude = np.random.random() * 0.5
        height = np.random.random() * 0.5


        offset = np.array([np.cos(angle) * magnitude, height, np.sin(angle) * magnitude])
        target = pickpoint_pos + offset
        move(pickpoint, target, 0.01)
        release_point(pickpoint, mass)
            
    elif trial_difficulty == 'none':
        pyflex_step()

    elif trial_difficulty == 'flat':
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

    flattened_rgb = get_rgb()

    heights = pyflex.get_positions().reshape(-1, 4)[:, 1]

    if heights.max() > 0.4:
        #print("[TrialGenerator] Discarding trial due to error due to height max")
        print(heights.max())
        return None

    if np.sum(flattened_rgb) == 0:
        #print("TrialGenerator] Discarding trial due to error due to empty scene")
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
        'trial_difficulty': trial_difficulty,
        'mesh_verts': mesh_verts.reshape(-1),
        'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
        'mesh_bend_edges': mesh_bend_edges.reshape(-1),
        'mesh_shear_edges': mesh_shear_edges.reshape(-1),
        'mesh_faces': mesh_faces.reshape(-1),
        'mesh_nocs_verts':mesh_nocs_verts,
        'cloth_instance':cloth_instance,
        'init_rgb':init_rgb,
        'flattened_rgb':flattened_rgb,
        'cloth_height':cloth_height,
        'cloth_width':cloth_width,
        'all_keypoint_groups':all_keypoint_groups,
        'pkl_path': pkl_path
    }
    #print("[TrialGenerator] Trial generated")
    if save_episode_data:
        output['episode_mesh_data'] = np.array(mesh_save_data['env_mesh_vertices'])
        # print("Episode shape", output['episode_mesh_data'].shape)

    # if os.path.exists(args.path):
    #     with FileLock(args.path + '.lock'):
    #         with h5py.File(args.path, 'r') as file:
    #             print(f"Progress:{len(file)}/{args.num_trials}")

    # pyflex.clean()
    return output



def generate_trials_helper(path: str,  gui: bool, recreate_steps=None, **kwargs):
    
    #for the pyflex init below
    msaaSamples = get_default_config()['scene_config']['msaaSamples']
    pyflex.init(
        not gui,  # headless: bool
        True,  # render: bool
        480, 
        480)  # camera dimensions: int x int
    action_tool = PickerPickPlace(
        num_picker=2,
        particle_radius=0.00625, # TODO: we need to make this adaptive
        picker_radius=0.05,
        picker_low=(-5, 0, -5),
        picker_high=(5, 5, 5))

    while True:

        if recreate_steps is not None:
            if len(recreate_steps) > 0:
                vertices, instance = recreate_steps.pop()
                print("Process has", len(recreate_steps), "trials left")
            else:
                break

            trial = generate_randomization(
                action_tool,
                gui=gui,
                recreate_step={"vertices":vertices, "instance":instance},
                **kwargs)
        else:
            trial = generate_randomization(
                action_tool,
                gui=gui,
                **kwargs)

      
        if trial is None:  
            continue
    
        if trial['cloth_width'] * trial['cloth_height'] == 0:
            continue

        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with FileLock(path + '.lock'):
            with h5py.File(path, 'a') as file:
                if len(file) >= kwargs['num_trials']:
                    break
                #key = hashlib.sha1(f'{len(file)}'.encode()).hexdigest()
                key = hashlib.sha1(f'{time.time()}_{random.randint(0, 1000000)}'.encode()).hexdigest()
                #print(f"[TrialGenerator] Writing trial to {path} with key {key}")
                
                # creat group for the key if it doesn't exist
                group = file.create_group(key)
                
                for key, value in trial.items():
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

        
      

class Trial:
    def __init__(self,
                 name: str,
                 flatten_area: float,
                 initial_coverage: float,
                 trial_difficulty: str,
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
                 flattened_rgb: np.array = [],
                 cloth_height: float = 0,
                 cloth_width: float = 0,
                 pkl_path: str = None,
                 all_keypoint_groups: np.array = [],
                ):
        self.name = name
        self.flatten_area = flatten_area
        self.initial_coverage = initial_coverage
        self.trial_difficulty = trial_difficulty
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
        self.flattened_rgb = np.array(flattened_rgb)
        self.all_keypoint_groups = np.array(all_keypoint_groups)

        config = get_default_config()
        self.camera_config = config['camera_params']
        self.scene_config = config['scene_config']
        self.scale = config['scale']

        self.pts_path = pts_path
        self.summary_path = summary_path
        self.pickpoint = pickpoint
        self.cloth_instance = cloth_instance
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
            'scene_config': self.scene_config,
            'scale': self.scale,
            'cloth_height': self.cloth_height,
            'cloth_width': self.cloth_width,
            'cloth_area': self.cloth_height * self.cloth_width,
            'trial_difficulty': self.trial_difficulty,
        }

    def get_images(self):
        return {
            'init_rgb': self.init_rgb,
            'flattened_rgb': self.flattened_rgb,
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
            'trial_name': self.name,
            'cloth_mass': self.cloth_mass,
            'cloth_stiff': self.cloth_stiff,
            'max_coverage': self.flatten_area,
            'trial_difficulty': self.trial_difficulty,
            # 'init_coverage': self.initial_coverage
        }

    def get_garmentnets_data(self):
        return {
            'pts_path': self.pts_path,
            'summary_path': self.summary_path,
            'pickpoint': self.pickpoint
        }

    def __str__(self):
        output = f'[Trial] {self.name}\n'
        output += f'\ttrial_difficulty: {self.trial_difficulty}\n'
        output += '\tinitial_coverage (%): ' +\
            f'{self.initial_coverage*100/self.flatten_area:.02f}\n'
        output += f'\tcloth_mass (kg): {self.cloth_mass:.04f}\n'
        output += f'\tcloth_stiff: {self.cloth_stiff}\n'
        output += f'\tflatten_area (m^2): {self.flatten_area:.04f}\n'
        output += f'\tcloth_instance: {self.cloth_instance}\n'
        return output


def get_init_verts_from_trial(trial_group):
    mesh_verts = trial_group['mesh_verts']
    num_particles = mesh_verts.shape[0] // 3
    init_verts = np.array(trial_group['init_particle_pos']).reshape(-1, 4)[:num_particles, :3]
    return init_verts

class TrialLoader:
    def __init__(self, hdf5_path: str, 
                    eval_hdf5_path: str, 
                    repeat: bool = True,
                     eval: bool = False, 
                     seed: int = 0,
                     recreate_trial_query: float = None,
                     grid_search: bool = False
                     ):
        
        seed_all(seed)
        self.hdf5_path = hdf5_path
        self.repeat = not eval
        self.keys = None
        self.recreate_trial_query = recreate_trial_query

        if eval:
            print("[TrialLoader] Loading eval trials")
            self.hdf5_path = eval_hdf5_path
            with h5py.File(self.hdf5_path, 'r') as trials:
                self.keys = [key for key in trials if trials[key].attrs['trial_difficulty'] == 'hard']
        else:
            print("[TrialLoader] Loading train trials")
            with h5py.File(self.hdf5_path, 'r') as trials:
                if recreate_trial_query is None:
                    self.keys = [key for key in trials]
                else:
                    print("[TrialLoader] Recreating trial with query {}".format(recreate_trial_query))
                    self.keys = [key for key in trials if np.sum(get_init_verts_from_trial(trials[key])) == recreate_trial_query]
                    if grid_search:
                        #ensure there are plenty of trials for a grid search
                        MAX_GRID_SEARCH_TASKS = 100
                        self.keys = self.keys * MAX_GRID_SEARCH_TASKS

                print(f'[TrialLoader] Found {len(self.keys)} trials from',
                    self.hdf5_path)
                if len(self.keys) == 0:
                    raise ValueError(f'[TrialLoader] No trials found in {self.hdf5_path}')
        self.curr_trial_idx = 0


    def get_next_trial(self) -> Trial:
        with h5py.File(self.hdf5_path, 'r') as trials:
            
            if not self.repeat:
                key = self.keys[self.curr_trial_idx]
                group = trials[key]
                self.curr_trial_idx += 1
                print('[TrialLoader] {}/{}'.format(
                    self.curr_trial_idx,
                    len(self.keys)))
                if self.curr_trial_idx >= len(self.keys):
                    print('[TrialLoader] Out of trials')
                    while True:
                        sleep(5)
            else:
                key = np.random.choice(self.keys)
                group = trials[key]
                if ('init_rgb' not in group) or (np.sum(group['init_rgb']) == 0):
                    print('[TrialLoader] Skipping trial with no init_rgb')
                    return self.get_next_trial()
            
            return Trial(name=key, **group.attrs, **group)


if __name__ == "__main__":
    parser = ArgumentParser('Trial Generation')
    parser.add_argument("--path", type=str, required=True,
                        help="path to output HDF5 dataset")
    parser.add_argument("--cloth_mesh_path", type=str,
                        help="path to root dir containing the mesh data", default=f"{os.getcwd()}/cloth_funnels/cloth_data/cloth3d_pickle")                                       
    parser.add_argument("--cloth_type", type=str, default='mesh',
                        choices=['square', 'mesh'],
                        help="type of cloth trial to create")
    parser.add_argument("--mesh_category", type=str, default='Shirt',
        help="category of clothing")
    parser.add_argument("--num_trials", type=int, default=100,
                        help="number of trials to generate")

    parser.add_argument("--trial_id", type=str, default=None)
    parser.add_argument('--randomize_instance', action='store_true', default=True)
    parser.add_argument('--randomize_direction', action='store_true', default=False)
    parser.add_argument('--local_mode', action='store_true', default=False)

    parser.add_argument('--random_translation', type=float, default=0.0)
    parser.add_argument('--trial_difficulty', type=str, default="none", help="(hard, easy, none)", choices=["hard", "easy", "flat", "none", "pick"])
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--test_single', action='store_true', default=False)
    parser.add_argument('--stiffness_factor', type=int, default=1)

    parser.add_argument("--num_processes", type=int, default=8,
                        help="number of parallel environments")

    parser.add_argument("--num_perturbations", type=int, default=0)
    parser.add_argument("--save_perturbations", action='store_true', default=False)

    parser.add_argument("--from_replay_buffer", type=str, default=None)
    parser.add_argument("--recreate_gt", action='store_true', default=False)
    parser.add_argument("--recreate_step_idx", action='store_true', default=8)


    parser.add_argument("--gui", action='store_true',
                        help="Run headless or not")
    parser.add_argument("--dump_images", type=str, default=None, help="Path to dump images")
    parser.add_argument("--scale", type=float, default=0.8, help="Scale of the cloth")
    parser.add_argument("--trial", type=str, choices=["cross", "unfold"])
    args = parser.parse_args()


    mesh_category, cloth_mesh_path = \
        args.mesh_category, args.cloth_mesh_path

    print("Results will be recorded in:", args.path)

    #add the clothing_type to the path
    ray.init(local_mode=args.local_mode)
    if args.test_single:
        print("[Trial] Testing single trial")
        generate_trials_helper(**vars(args))


    elif args.from_replay_buffer is not None:


        json_path = os.path.join(cloth_mesh_path, mesh_category)
        instances_dict = json.load(open(json_path))
        test_instances = instances_dict['test']
        verts_to_instance = {}

        for instance in test_instances:
            pkl_path = os.path.join(cloth_mesh_path, cloth_mesh_path, instance.split("_")[1].split(".")[0], instance)   
            
            #sample a number between -args.random_translation to args.random_translation
            mesh_verts, mesh_faces, mesh_stretch_edges, \
                mesh_bend_edges, mesh_shear_edges, mesh_nocs_verts = load_cloth(pkl_path)

            verts_to_instance[mesh_verts.shape[0]] = instance


        print("[Trial] Reading from replay buffer")
        with h5py.File(args.from_replay_buffer, 'r') as dataset:
            keys = list(dataset.keys())
            print("[Trial] Done reading")
            recreate_keys = []

            episodes = {}
            for key in keys:
                #don't include the easy trials
                if dataset[key].attrs['trial_difficulty'] == 'easy':
                    continue
                episode = int(key.split("step")[0][:-1])
                if episode not in episodes:
                    episodes[episode] = []
                step = int(key.split("_")[1][4:])
                episodes[episode].append((step, key))

            recreate_keys = []
            for e, steps in episodes.items():
                chosen_steps = np.random.choice(np.arange(len(steps)), size=3, replace=False)
                # chosen_index = min(len(steps)-1, args.recreate_step_idx)
                for chosen_index in chosen_steps:
                    chosen_step = steps[chosen_index]
                    recreate_keys.append(chosen_step[1])

            num_trials = len(recreate_keys)
            
            split_keys = [[] for _ in range(args.num_processes)]

            for i, key in enumerate(recreate_keys):
                split_keys[i % len(split_keys)].append(key)

            helper_fn = ray.remote(generate_trials_helper).options(
                num_gpus=float(torch.cuda.device_count())/args.num_processes)


            handles = []
            for process_id, keys in enumerate(split_keys):

                init_verts = [np.array(dataset[key]['init_verts']) for key in keys]
                last_verts = [np.array(dataset[key]['postaction_verts']) for key in keys]
            
                try:
                    pkl_paths = [np.array(dataset[key]['trial_pkl_path']) for key in keys]
                except:
                    pkl_paths = [np.array(verts_to_instance[dataset[key]['init_verts'].shape[0]]) for key in keys]

                verts = init_verts if args.recreate_gt else last_verts
                recreate_steps = list(zip(verts, pkl_paths))
                handles.append(helper_fn.remote(**vars(args), recreate_steps=recreate_steps))

            with tqdm(total=num_trials,
                desc='Generating trials',
                dynamic_ncols=True) as pbar:
                while True:
                    ray.wait(handles, timeout=5)
                    if not os.path.exists(args.path):
                        continue
                    with FileLock(args.path + '.lock'):
                        with h5py.File(args.path, 'r') as file:
                            pbar.update(len(file) - pbar.n)
                            pbar.refresh()
                            if len(file) >= num_trials:
                                exit()

    else:
        with FileLock(args.path  + '.lock'):
            if os.path.exists(args.path):
                with h5py.File(args.path, 'r+') as file:
                    file_len = len(file)
                    print(f"Progress:{len(file)}/{args.num_trials}")
                ## remove the execess trials
                if file_len >= args.num_trials:
                    print("[Trial] Removing excess trials")
                
                    with h5py.File(args.path, 'a') as file:
                        for key in tqdm(list(file.keys())[:-args.num_trials], desc='Removing excess trials', dynamic_ncols=True):
                            del file[key]
                    exit(0)

        helper_fn = ray.remote(generate_trials_helper).options(
            num_gpus=float(torch.cuda.device_count())/args.num_processes)
        #added pkl dir, pts_dir, common_files, summary_path
        handles = [helper_fn.remote(**vars(args))
                for _ in range(args.num_processes)]
        
        with tqdm(total=args.num_trials,
                desc='Generating trials',
                dynamic_ncols=True) as pbar:
            while True:
                ray.wait(handles, timeout=5)
                if not os.path.exists(args.path):
                    continue
                with FileLock(args.path  + '.lock'):
                    with h5py.File(args.path, 'r') as file:
                        #print("Progress:{}/{}".format(len(file), args.num_trials))
                        len_file = len(file)
                        pbar.n = len(file)
                        #pbar.update(len(file) - pbar.n)  
                if len_file >= args.num_trials:
                    # stop all processes
                    ray.kill(handles)
                    ray.shutdown()
                    exit(0)
