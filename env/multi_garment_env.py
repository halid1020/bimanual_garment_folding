import os
import h5py
import numpy as np
import cv2
import json

from softgym.action_space.action_space import Picker
from softgym.utils.env_utils import get_coverage
import pyflex
from agent_arena import Arena
from tqdm import tqdm


from .action_primitives.hybrid_action_primitive import HybridActionPrimitive
from .garment_env_logger import GarmentEnvLogger
from .utils.env_utils import set_scene
from .utils.camera_utils import get_camera_matrix
from .garment_env import GarmentEnv

global ENV_NUM
ENV_NUM = 0

# @ray.remote
class MultiGarmentEnv(GarmentEnv):
    
    def __init__(self, config):
        config.name = f'multi-garment-{config.object}-env'
        self.num_eval_trials = 30
        self.num_train_trials = 100
        self.num_val_trials = 10
        super().__init__(config)

        
        #self.name =f'single-garment-fixed-init-env'

    ## TODO: if eid is out of range, we need to raise an error.   
    def reset(self, episode_config=None):
        print('episode_config', episode_config)
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        if 'eid' not in episode_config or episode_config['eid'] is None:

            # randomly select an episode whose 
            # eid equals to the number of episodes%CLOTH_FUNNEL_ENV_NUM = self.id
            if self.mode == 'train':
                episode_config['eid'] = np.random.randint(self.num_train_trials)
            elif self.mode == 'val':
                episode_config['eid'] = np.random.randint(self.num_val_trials)
            else:
                episode_config['eid'] = np.random.randint(self.num_eval_trials)
           
        init_state_params = self._get_init_state_params(episode_config['eid'])

        episode_config['eid'] = episode_config['eid']
        self.eid = episode_config['eid']

        self.sim_step = 0
        self.video_frames = []
        self.save_video = episode_config['save_video']

        self.episode_config = episode_config

        init_state_params['scene_config'] = self.scene_config
        init_state_params.update(self.default_config)
        set_scene(
            config=init_state_params, 
            state=init_state_params)
        self.num_mesh_particles = int(len(init_state_params['mesh_verts'])/3)
        #print('mesh particles', self.num_mesh_particles)
        self.init_state_params = init_state_params

        
        #print('set scene done')
        #print('pciker initial pos', self.picker_initial_pos)
        self.pickers.reset(self.picker_initial_pos)
        #print('picker reset done')

        self.init_coverae = self._get_coverage()
        self.flattened_obs = None
        self.get_flattened_obs()
        #self.flatten_coverage = init_state_params['flatten_area']
        
        self.info = {}
        self.last_info = None
        self.action_tool.reset(self) # get out of camera view, and open the gripper
        self._step_sim()
        
        self.last_flattened_step = -100
        self.task.reset(self)
        
        self.action_step = 0

        self.evaluate_result = None
        
        set_scene(
            config=init_state_params, 
            state=init_state_params)
        if self.init_mode == 'flattened':
            #print('init_mode')
            self.set_to_flatten()
        
        self.last_info = None
        self.sim_step = 0
        self.info = self._process_info({})
        self.clear_frames()

        
        return self.info
    
    
    def get_eval_configs(self):
        eval_configs = [
            {'eid': eid, 'tier': 0, 'save_video': True}
            for eid in range(self.num_eval_trials)
        ]
        
        return eval_configs

    def get_train_configs(self):
        train_configs = [
            {'eid': eid, 'tier': 0, 'save_video': True}
            for eid in range(self.num_train_trials)
        ]
        
        return train_configs

    
    def get_val_configs(self):
        val_configs = [
            {'eid': eid, 'tier': 0, 'save_video': True}
            for eid in range(self.num_val_trials)
        ]
        
        return val_configs

    def get_num_episodes(self) -> np.int:
        if self.mode == 'eval':
            return self.num_eval_trials
        elif self.mode == 'val':
            return self.num_val_trials
        elif self.mode == 'train':
            return self.num_train_trials
        else:
            raise NotImplementedError


    def _get_init_state_keys(self):
        
        eval_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-eval.hdf5')
        train_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-train.hdf5')

        eval_key_file = os.path.join(self.config.init_state_path, f'{self.name}-eval.json')
        train_key_file = os.path.join(self.config.init_state_path, f'{self.name}-train.json')

        self.eval_keys = self._get_init_keys_helper(eval_path, eval_key_file, difficulties=['hard'])
        self.train_keys = self._get_init_keys_helper(train_path, train_key_file)

        self.val_keys = self.eval_keys[:self.num_val_trials]
        self.eval_keys = self.eval_keys[self.num_val_trials:]

    def _get_init_state_params(self, eid):
        if self.mode == 'train':
            keys = self.train_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-train.hdf5')
        elif self.mode == 'eval':
            keys = self.eval_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-eval.hdf5')
        elif self.mode == 'val':
            keys = self.val_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{self.config.object}-eval.hdf5')

        key = keys[eid]
        with h5py.File(hdf5_path, 'r') as init_states:
            # print(hdf5_path, key)
            # Convert group to dict
            group = init_states[key]
            episode_params = dict(group.attrs)
            
            # If there are datasets in the group, add them to the dictionary
            #print('group keys', group.keys())
            for dataset_name in group.keys():
                episode_params[dataset_name] = group[dataset_name][()]

            self.episode_params = episode_params
            #print('episode_params', episode_params.keys())

        return episode_params