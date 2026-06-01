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


class MultiGarmentEnv(GarmentEnv):
    
    def __init__(self, config):
        self.num_eval_trials = config.get('num_eval_trials', 30)
        self.num_train_trials = config.get('num_train_trials', 100)
        self.num_val_trials = config.get('num_val_trials', 10)

        config.name = f'multi-garment-{config.garment_type}-env'
        
        if config.garment_type == 'all':
            self.all_garment_types = config.all_garment_types
            self.num_eval_trials = len(self.all_garment_types) * 8 # e.g., 32
            self.num_train_trials *= len(self.all_garment_types)   # e.g., 400
            self.num_val_trials = len(self.all_garment_types) * 3  # e.g., 12
            
        super().__init__(config)

    # def reset(self, episode_config=None):
    #     if episode_config is None:
    #         episode_config = {}
            
    #     episode_config.setdefault('save_video', False)
        
    #     if episode_config.get('eid') is None:
    #         # Randomly select an episode ID based on the current mode's trial limits
    #         mode_trials = {
    #             'train': self.num_train_trials,
    #             'val': self.num_val_trials,
    #             'eval': self.num_eval_trials
    #         }
    #         # Default to eval limits if the mode isn't explicitly found
    #         max_trials = mode_trials.get(getattr(self, 'mode', 'eval'), self.num_eval_trials)
    #         episode_config['eid'] = np.random.randint(max_trials)
           
    #     self.eid = episode_config['eid']
    #     self.save_video = episode_config['save_video']
    #     self.episode_config = episode_config

    #     # Generate unique ID for the episode
    #     timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    #     self.uid = f'{timestamp}-{str(uuid.uuid4().hex)}-{self.aid}'

    #     # Fetch physical parameters for the chosen episode
    #     init_state_params = self._get_init_state_params(self.eid)
    #     init_state_params['scene_config'] = self.scene_config
    #     init_state_params.update(self.default_config)
        
    #     self.num_mesh_particles = int(len(init_state_params['mesh_verts']) / 3)
    #     print(f'[MultiGarmentEnv] mesh particles: {self.num_mesh_particles}')
    #     self.init_state_params = init_state_params

    #     print('[MultiGarmentEnv] Ready to set scene')
    #     set_scene(config=init_state_params, state=init_state_params)
        
    #     self.pickers.reset(self.picker_initial_pos)
    #     self.low_level_mesh_particles = []
    #     self.low_level_visible_pcs = []

    #     self.action_tool.reset(self) # Get out of camera view and open the gripper
    #     self._step_sim()

    #     # State tracking resets
    #     self.draw_fatten_contour = ('alignment' in self.task.name)
    #     self.sim_step = 0
    #     self.video_frames = []
    #     self.is_recording_low_level = False
       
    #     self.picker_poses = []
    #     self.last_flattened_step = -100
    #     self.action_step = 0
    #     self.overstretch = 0
    #     self.evaluate_result = None

    #     # Get environment coverage and flattened observation
    #     self.init_coverae = self._get_coverage() 
    #     self.flattened_obs = None
    #     self.get_flattened_obs()
       
    #     # Task and trajectory initialization
    #     self.task.reset(self)
    #     self._initialise_trajectory() 
        
    #     # Process initial info dictionary
    #     self.info = {}
    #     self.last_info = None
    #     self.all_infos = [self.info]
    #     self.info = self._process_info(self.info)
    #     self.clear_frames()

    #     # Final flag tracking
    #     self.info['observation']['is_first'] = True
    #     self.info['observation']['is_terminal'] = self.info.get('terminated', False)
        
    #     return self.info

    def reset(self, episode_config=None):
        if episode_config is None:
            episode_config = {}
            
        episode_config.setdefault('save_video', False)
        
        if episode_config.get('eid') is None:
            # Randomly select an episode ID based on the current mode's trial limits
            mode_trials = {
                'train': self.num_train_trials,
                'val': self.num_val_trials,
                'eval': self.num_eval_trials
            }
            # Default to eval limits if the mode isn't explicitly found
            max_trials = mode_trials.get(getattr(self, 'mode', 'eval'), self.num_eval_trials)
            episode_config['eid'] = np.random.randint(max_trials)
           
        self.eid = episode_config['eid']
        self.save_video = episode_config['save_video']
        self.episode_config = episode_config

        # --- NEW: Retry loop to ensure the garment starts inside the camera view ---
        max_retries = 50
        for attempt in range(max_retries):
            # Generate unique ID for the episode
            timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            self.uid = f'{timestamp}-{str(uuid.uuid4().hex)}-{self.aid}'

            # Fetch physical parameters for the chosen episode
            init_state_params = self._get_init_state_params(self.eid)
            init_state_params['scene_config'] = self.scene_config
            init_state_params.update(self.default_config)
            
            self.num_mesh_particles = int(len(init_state_params['mesh_verts']) / 3)
            self.init_state_params = init_state_params

            set_scene(config=init_state_params, state=init_state_params)
            
            self.pickers.reset(self.picker_initial_pos)
            self.low_level_mesh_particles = []
            self.low_level_visible_pcs = []

            self.action_tool.reset(self) # Get out of camera view and open the gripper
            self._step_sim()
            
            # --- Quick Visibility Check ---
            # We call _get_obs with flatten_obs=False so it safely crops the mask 
            # without triggering downstream flattened keypoint logic yet.
            temp_obs = self._get_obs(flatten_obs=False)
            if np.sum(temp_obs['mask']) >= 10:  # Matches the 'out_of_view' threshold
                break # It's visible, break out of the retry loop and continue
            
            print(f"[MultiGarmentEnv] Episode {self.eid} initialized out of view. Trying next episode...")
            self.eid += 1
            self.episode_config['eid'] = self.eid
            
        else:
            print("[MultiGarmentEnv] WARNING: Reached max retries looking for a visible garment!")

        # State tracking resets
        self.draw_fatten_contour = ('alignment' in self.task.name)
        self.sim_step = 0
        self.video_frames = []
        self.is_recording_low_level = False
       
        self.picker_poses = []
        self.last_flattened_step = -100
        self.action_step = 0
        self.overstretch = 0
        self.evaluate_result = None

        # Get environment coverage and flattened observation
        self.init_coverae = self._get_coverage() 
        self.flattened_obs = None
        self.get_flattened_obs()
       
        # Task and trajectory initialization
        self.task.reset(self)
        self._initialise_trajectory() 
        
        # Process initial info dictionary
        self.info = {}
        self.last_info = None
        self.all_infos = [self.info]
        self.info = self._process_info(self.info)
        self.clear_frames()

        # Final flag tracking
        self.info['observation']['is_first'] = True
        self.info['observation']['is_terminal'] = self.info.get('terminated', False)
        
        return self.info

    def _get_init_state_keys(self):
        """
        This method gathers episode keys (references to specific initial garment states) 
        from your HDF5 dataset. 
        
        If 'garment_type' is 'all', it loads keys for EVERY garment type, splits them 
        into Train/Val/Eval sets proportionally, and then "interleaves" them. 
        Interleaving ensures that during training/evaluation, the environment naturally 
        alternates between different garments (e.g., T-shirt -> Pants -> Dress) rather 
        than running 100 T-shirts in a row before seeing the next type.
        """
        if self.config.garment_type == 'all':
            garment_types = self.all_garment_types
            num_garments = len(garment_types)

            garment_eval_keys, garment_val_keys, garment_train_keys = [], [], []

            # Load and split keys per garment type
            for garment_type in garment_types:
                eval_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-eval.hdf5')
                train_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-train.hdf5')

                eval_key_file = os.path.join(self.config.init_state_path, f'{garment_type}-eval.json')
                train_key_file = os.path.join(self.config.init_state_path, f'{garment_type}-train.json')

                eval_keys = self._get_init_keys_helper(eval_path, eval_key_file, difficulties=['hard'])
                train_keys = self._get_init_keys_helper(train_path, train_key_file)

                # Split evaluation pool into Validation and Evaluation
                val_share = self.num_val_trials // num_garments
                eval_share = self.num_eval_trials // num_garments
                
                garment_val_keys.append(eval_keys[:val_share])
                garment_eval_keys.append(eval_keys[val_share : val_share + eval_share])
                
                # Trim training pool to its required share
                train_share = self.num_train_trials // num_garments
                garment_train_keys.append(train_keys[:train_share])

            # Helper to perfectly alternate items from multiple lists (e.g., zip_longest)
            def interleave_flexible(lists):
                return [item for group in zip_longest(*lists) for item in group if item is not None]

            self.eval_keys = interleave_flexible(garment_eval_keys)
            self.val_keys = interleave_flexible(garment_val_keys)
            self.train_keys = interleave_flexible(garment_train_keys)
        
        else: 
            # Standard single-garment loading
            eval_path = os.path.join(self.config.init_state_path, f'multi-{self.config.garment_type}-eval.hdf5')
            train_path = os.path.join(self.config.init_state_path, f'multi-{self.config.garment_type}-train.hdf5')

            eval_key_file = os.path.join(self.config.init_state_path, f'{self.name}-eval.json')
            train_key_file = os.path.join(self.config.init_state_path, f'{self.name}-train.json')

            eval_keys = self._get_init_keys_helper(eval_path, eval_key_file, difficulties=['hard'])
            self.train_keys = self._get_init_keys_helper(train_path, train_key_file)

            # Split Evaluation keys into Validation and Evaluation sets
            self.val_keys = eval_keys[:self.num_val_trials]
            self.eval_keys = eval_keys[self.num_val_trials:]
            
            self.val_uses_train_file = False
            
            # Fallback if there aren't enough eval keys: Prioritize Eval numbers, pull Val from Train
            if len(self.eval_keys) < self.num_eval_trials:
                self.eval_keys = eval_keys[:self.num_eval_trials]
                self.val_keys = self.train_keys[self.num_train_trials : self.num_train_trials + self.num_val_trials]
                self.val_uses_train_file = True  # Tell the params method to switch files

    def _get_init_state_params(self, eid):
        garment_type = self.config.garment_type
        if garment_type == 'all':
            # Determine which garment type this episode ID maps to
            garment_type = self.all_garment_types[eid % len(self.all_garment_types)]

        mode = getattr(self, 'mode', 'eval')
        
        # Select correct key pool and file path
        if mode == 'train':
            keys = self.train_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-train.hdf5')
        elif mode == 'eval':
            keys = self.eval_keys
            hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-eval.hdf5')
        else: # mode == 'val'
            keys = self.val_keys
            # If the fallback grabbed train keys for val, we must open the train HDF5
            if getattr(self, 'val_uses_train_file', False):
                hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-train.hdf5')
            else:
                hdf5_path = os.path.join(self.config.init_state_path, f'multi-{garment_type}-eval.hdf5')

        # Find the next valid configuration
        while True:
            # Added modulo safety to prevent IndexErrors if eid exceeds key length after skips
            safe_eid = eid % len(keys)
            key = keys[safe_eid]

            with h5py.File(hdf5_path, 'r') as init_states:
                if key not in init_states:
                    print(f'here!! {eid}, safe_eid {safe_eid}, len keys {len(keys)}')
                    eid += 1
                    continue
                group = init_states[key]
                episode_params = dict(group.attrs)

                # Validation check: Ensure the state has an associated pkl_path
                if 'pkl_path' not in episode_params:
                    eid += 1 if garment_type != 'all' else len(self.all_garment_types)
                    continue
                
                # Extract dataset arrays into memory
                for dataset_name in group.keys():
                    episode_params[dataset_name] = group[dataset_name][()]

                self.episode_params = episode_params
            break
            
        return episode_params