import os
import h5py
import numpy as np
import uuid
import datetime

from itertools import zip_longest

from .utils.env_utils import set_scene
from .garment_env import GarmentEnv

global ENV_NUM
ENV_NUM = 0
# self.all_garment_types = ['longsleeve', 'trousers', 'skirt', 'dress']

# @ray.remote
class MultiGarmentEnv(GarmentEnv):
    
    def __init__(self, config):
        #config.name = f'multi-garment-{config.object}-env'
        
        self.num_eval_trials = 30
        self.num_train_trials = 100
        self.num_val_trials = 10


        config.name = f'multi-garment-{config.garment_type}-env'
        if config.garment_type == 'all':
            self.all_garment_types = config.all_garment_types
            self.num_eval_trials = len(self.all_garment_types)*8 # 32
            self.num_train_trials *= len(self.all_garment_types) # 400
            self.num_val_trials = len(self.all_garment_types)*3 # 12
        super().__init__(config)

        
        
        #print('num val', self.num_val_trials)

        
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

        #episode_config['eid'] = episode_config['eid']
        self.eid = episode_config['eid']

        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.uid = f'{timestamp}-{str(uuid.uuid4().hex)}-{self.aid}'

        self.sim_step = 0
        self.video_frames = []
        

        self.episode_config = episode_config

        init_state_params['scene_config'] = self.scene_config
        init_state_params.update(self.default_config)
        set_scene(
            config=init_state_params, 
            state=init_state_params)
        self.num_mesh_particles = int(len(init_state_params['mesh_verts'])/3)
        print('mesh particles', self.num_mesh_particles)
        self.init_state_params = init_state_params

        
        #print('set scene done')
        #print('pciker initial pos', self.picker_initial_pos)
        self.pickers.reset(self.picker_initial_pos)
        #print('picker reset done')

        

        self.init_coverae = self._get_coverage()
        self.flattened_obs = None
        self.save_video = False
        self.get_flattened_obs()
        #self.flatten_coverage = init_state_params['flatten_area']
        
        self.info = {}
        self.last_info = None
        self.action_tool.reset(self) # get out of camera view, and open the gripper
        self._step_sim()

        self.save_video = episode_config['save_video']


        self.last_flattened_step = -100
        self.task.reset(self)
        self.action_step = 0
        
       

        self.evaluate_result = None
        
        set_scene(
            config=init_state_params, 
            state=init_state_params)
        self.pickers.reset(self.picker_initial_pos)
        self.action_tool.reset(self) # get out of camera view, and open the gripper
        self._step_sim()
        
        if self.init_mode == 'flattened':
            #print('init_mode')
            self.set_to_flatten()
        
        self.last_info = None
        self.sim_step = 0
        self.overstretch = 0
        self.info = self._process_info({})
        self.clear_frames()

        self.info['observation']['is_first'] = True
        self.info['observation']['is_terminal'] = False

        
        return self.info
    
    
    def get_eval_configs(self):
        eval_configs = [
            {'eid': eid, 'tier': 0, 'save_video': True}
            for eid in range(self.num_eval_trials)
        ]
        
        return eval_configs

    def get_train_configs(self):
        train_configs = [
            {'eid': eid, 'tier': 0, 'save_video': self.config.get('save_video', False)}
            for eid in range(self.num_train_trials)
        ]
        
        return train_configs

    
    def get_val_configs(self):
        val_configs = [
            {'eid': eid, 'tier': 0, 'save_video': True}
            for eid in range(self.num_val_trials)
        ]
        print('len config', len(val_configs), 'num tiral', self.num_val_trials)
        return val_configs

    def get_num_episodes(self):
        if self.mode == 'eval':
            return self.num_eval_trials
        elif self.mode == 'val':
            return self.num_val_trials
        elif self.mode == 'train':
            return self.num_train_trials
        else:
            raise NotImplementedError


    def _get_init_state_keys(self):
        
        if self.config.garment_type == 'all':
            garment_types = self.all_garment_types  # fixed typo
            num_garments = len(garment_types)

            # Initialize empty lists
            self.eval_keys, self.val_keys, self.train_keys = [], [], []

            # Store per-garment key lists temporarily
            garment_eval_keys = []
            garment_val_keys = []
            garment_train_keys = []

            # Load and split per garment type
            for garment_type in garment_types:
                eval_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-eval.hdf5')
                train_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-train.hdf5')

                eval_key_file = os.path.join(self.config.init_state_path, f'{garment_type}-eval.json')
                train_key_file = os.path.join(self.config.init_state_path, f'{garment_type}-train.json')

                eval_keys = self._get_init_keys_helper(eval_path, eval_key_file, difficulties=['hard'])
                train_keys = self._get_init_keys_helper(train_path, train_key_file)

                # Split eval_keys into val and eval
                val_keys = eval_keys[: self.num_val_trials // num_garments]
                eval_keys = eval_keys[self.num_val_trials // num_garments : self.num_val_trials // num_garments + (self.num_eval_trials // num_garments)]

                # Trim train_keys to its share
                train_keys = train_keys[: self.num_train_trials // num_garments]

                garment_eval_keys.append(eval_keys)
                garment_val_keys.append(val_keys)
                garment_train_keys.append(train_keys)

            # --- Interleave keys across garments ---
            def interleave(lists):
                # zip(*lists) pairs first elements, second elements, etc.
                interleaved = []
                for group in zip(*lists):
                    interleaved.extend(group)
                return interleaved

            

            def interleave_flexible(lists):
                interleaved = []
                for group in zip_longest(*lists, fillvalue=None):
                    for item in group:
                        if item is not None:
                            interleaved.append(item)
                return interleaved

            self.eval_keys = interleave_flexible(garment_eval_keys)
            self.val_keys = interleave_flexible(garment_val_keys)
            self.train_keys = interleave_flexible(garment_train_keys)

            # print('self.train_keys', len(self.train_keys))
            # print('garment_train_keys', len(garment_train_keys))
        
        
        else: 
            eval_path = os.path.join(self.config.init_state_path, f'multi-{self.config.garment_type}-eval.hdf5')
            train_path = os.path.join(self.config.init_state_path, f'multi-{self.config.garment_type}-train.hdf5')

            eval_key_file = os.path.join(self.config.init_state_path, f'{self.name}-eval.json')
            train_key_file = os.path.join(self.config.init_state_path, f'{self.name}-train.json')

            self.eval_keys = self._get_init_keys_helper(eval_path, eval_key_file, difficulties=['hard'])
            self.train_keys = self._get_init_keys_helper(train_path, train_key_file)

            self.val_keys = self.eval_keys[:self.num_val_trials]
            self.eval_keys = self.eval_keys[self.num_val_trials:]

    def _get_init_state_params(self, eid):
        garment_type = self.config.garment_type
        if self.config.garment_type == 'all':
            garment_type = self.all_garment_types[eid%len(self.all_garment_types)]
            #eid //=  len(self.all_garment_types)
            # print('garment_type', garment_type)
            # print('here')

        if self.mode == 'train':
            print('get from train keys')
            keys = self.train_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-train.hdf5')
        elif self.mode == 'eval':
            keys = self.eval_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-eval.hdf5')
        elif self.mode == 'val':
            keys = self.val_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-eval.hdf5')
        #print('len(keys)', len(keys))
        while True:
            key = keys[eid]
            #print('key', key)
            with h5py.File(hdf5_path, 'r') as init_states:
                # print(hdf5_path, key)
                # Convert group to dict
                group = init_states[key]
                episode_params = dict(group.attrs)

                if not ('pkl_path' in episode_params.keys()):
                    eid += 1 if not self.garment_type == 'all' else len(self.all_garment_types)
                    print('eid', eid)
                    continue
                
                # If there are datasets in the group, add them to the dictionary
                #print('group keys', group.keys())
                for dataset_name in group.keys():
                    episode_params[dataset_name] = group[dataset_name][()]

                self.episode_params = episode_params#
            break
            #print('episode_params', episode_params.keys())

        return episode_params