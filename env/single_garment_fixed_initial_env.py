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
class SingleGarmentFixedInitialEnv(GarmentEnv):
    
    def __init__(self, config):
        config.name = f'single-garment-fixed-init-env-{config.garment_type}'
        super().__init__(config)
        #self.name =f'single-garment-fixed-init-env'

    ## TODO: if eid is out of range, we need to raise an error.   
    def reset(self, episode_config=None):
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        episode_config['eid'] = 0
        self.eid = 0
        init_state_params = self._get_init_state_params(episode_config['eid'])



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
        
        
        if self.init_mode == 'flattened':
            self.set_to_flatten()
          
        
        self.overstretch = 0
        self.info = self._process_info({})
        self.clear_frames()

        
        return self.info
    
    
    def get_eval_configs(self):
        eval_configs = [
            {'eid': 0, 'tier': 0, 'save_video': True}
            for eid in range(30)
        ]
        
        return eval_configs

    
    def get_val_configs(self):
        return [
            {'eid': 0, 'tier': 0, 'save_video': True}
        ]




    def _get_init_state_keys(self):
        
        path = os.path.join(self.init_state_path, f'multi-{self.garment_type}-eval.hdf5')
        #train_path = os.path.join(self.init_state_path, f'multi-{self.garment_type}-train.hdf5')

        key_file = os.path.join(self.init_state_path, f'{self.name}-eval.json')
        #train_key_file = os.path.join(self.init_state_path, f'{self.name}-train.json')


        self.keys = self._get_init_keys_helper(path, key_file, difficulties=['hard'])
        
        # print len of keys
        self.num_trials = 1

    def _get_init_state_params(self, eid):
            
        hdf5_path = os.path.join(self.init_state_path, f'multi-{self.garment_type}-eval.hdf5')
        key = self.keys[eid]
        with h5py.File(hdf5_path, 'r') as init_states:

            group = init_states[key]
            
            episode_params = dict(group.attrs)
            #print('episode_params', episode_params)
            
            for dataset_name in group.keys():
                episode_params[dataset_name] = group[dataset_name][()]

            self.episode_params = episode_params

        return episode_params