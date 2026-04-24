# The code is adopted from https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=VrX4VTl5pYNq

import os
from tqdm import tqdm
import torch
import numpy as np
from collections import deque
import torch
import cv2
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
from dotmap import DotMap
import torch.nn as nn
import time
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dotmap import DotMap
import torchvision.models as models

from actoris_harena import TrainableAgent
from actoris_harena.utilities.networks.utils import np_to_ts, ts_to_np
from actoris_harena.utilities.visual_utils import save_numpy_as_gif, save_video
from actoris_harena.utilities.save_utils import save_mask

from data_augmentation.register_augmeters import build_data_augmenter

from .utils \
    import get_resnet, replace_bn_with_gn, compute_classification_metrics
from .networks import ConditionalUnet1D, MLPClassifier, ResNetDecoder, ConditionalMLP1D
from .dataset import DiffusionDataset, normalize_data, unnormalize_data
from .constrain_action_functions import name2func

class DiffusionTransform():

    def __init__(self, config, stats):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.stats = stats
        #self.save_dir = config.save_dir

    def __call__(self, data, train=True):

        ret_data = {}
        #self.transform = DiffusionTransform(self.config)
        
        if not train:
            ret_data[self.config.input_obs] = data[self.config.input_obs].astype(np.float32)/255.0#
            # print('input obs shape', ret_data[self.config.input_obs].shape)
            
            if len(ret_data[self.config.input_obs].shape) == 3:
                ret_data[self.config.input_obs] = np.expand_dims(ret_data[self.config.input_obs], axis=0)
                ret_data[self.config.input_obs] = np.expand_dims(ret_data[self.config.input_obs], axis=0)

            ret_data[self.config.input_obs] = ret_data[self.config.input_obs].transpose(0, 1, 4, 2, 3)
            
            ret_data[self.config.input_obs] = np_to_ts(ret_data[self.config.input_obs], self.device)
            ret_data['vector_state'] = \
                normalize_data(data['vector_state'], 
                               self.stats[self.config.data_state])#
            ret_data['vector_state'] = np_to_ts(ret_data['vector_state'], self.device)
            if len(ret_data['vector_state'].shape) == 1:
                ret_data['vector_state'] = ret_data['vector_state'].unsqueeze(0)

            

        else:
            
            ret_data[self.config.input_obs] = data[self.config.data_obs]
            ret_data[self.config.input_obs] = np_to_ts(ret_data[self.config.input_obs][:, :self.config.obs_horizon], self.device)
            ret_data['vector_state'] = np_to_ts(data[self.config.data_state][:, :self.config.obs_horizon], self.device)
            ret_data['action'] = np_to_ts(data['action'], self.device)

        return ret_data

    def postprocess(self, data):
        # print('data keys', data.keys())
        ret_data = {}
        if 'action' in data.keys():
            data['action'] = unnormalize_data(data['action'], self.stats[self.config.data_action])
            ret_data['action'] = data['action']
        if self.config.input_obs in data.keys():
            #data[self.config.input_obs] = unnormalize_data(data[self.config.input_obs], self.stats[self.config.data_obs])
            ret_data[self.config.input_obs] = (ts_to_np(data[self.config.input_obs])*255.0).clip(0, 255).astype(np.uint8)
        return ret_data

class MultiPrimitiveDiffusionAdapter(TrainableAgent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'diffusion'
        self.config = config
        self.internal_states = {}
        self.buffer_actions = {}
        self.last_actions = {}
        self.obs_deque = {}
        self.collect_on_success = self.config.get('collect_on_success', True)
        self.measure_time = config.get('measure_time', False)
        self.debug = config.get('debug', False)
        self.constrain_action = name2func[config.get('constrain_action', 'identity')]
        self.val_interval = self.config.get('val_interval', 100)
        self.validate_training = self.config.get('validate_training', 100)

        self.primitive_integration = self.config.get('primitive_integration', 'none')
        if self.primitive_integration != 'none':
            
            self.primitives = config.primitives
            self.K = len(self.primitives)
            self.action_dims = [prim['dim'] if isinstance(prim, dict) else prim.dim for prim in self.primitives]
            
            self.prim_name2id = {item['name']: i for i, item in enumerate(self.primitives)}
            self.network_action_dim = max(self.action_dims)
            if self.primitive_integration == 'bin_as_output':
                self.network_action_dim += 1
            self.data_save_action_dim = self.network_action_dim
            if self.primitive_integration == 'one-hot-encoding':
                self.data_save_action_dim += 1
            
            self.primitive_action_masks = self._build_primitive_action_masks()
            self.mask_out_irrelavent_action_dim = self.config.get('mask_out_irrelavent_action_dim', False)
            
            
        else:
            self.network_action_dim = config.action_dim
            self.data_save_action_dim = config.action_dim

        self._init_networks()

        

        self._init_optimizer()
        self.loaded = False

        from .action_sampler import ActionSampler
        self.eval_action_sampler = ActionSampler[self.config.eval_action_sampler]()
        
        self.update_step = 0 #-1
        self.total_update_steps = self.config.total_update_steps
        self.dataset_inited = False


        self.data_augmenter = build_data_augmenter(config.data_augmenter)

        
        
    def _init_dataset(self):

        if self.config.dataset_mode == 'diffusion':
            train_dataset = DiffusionDataset(
                dataset_path=self.config.dataset_path,
                pred_horizon=self.config.pred_horizon,
                obs_horizon=self.config.obs_horizon,
                action_horizon=self.config.action_horizon
            )
            self.stats = train_dataset.stats
        elif self.config.dataset_mode == 'general':
            from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            config = self.config.dataset_config.toDict()
            train_dataset = TrajectoryDataset(**config, sample_mode='train')

            if self.validate_training:
                val_dataset = TrajectoryDataset(**config, sample_mode='val')

        else:
            raise ValueError('Invalid dataset mode')

        # ==========================================================
        # --- NEW FEATURE: Random Oversampling (ROS) via Config ---
        # ==========================================================
        use_ros = self.config.get('use_random_oversampling', False)
        sampler = None
        shuffle_data = True # Default to True unless using a sampler
        
        if use_ros and self.primitive_integration != 'none':
            print("[MultiPrimitiveDiffusionAdapter] Initializing WeightedRandomSampler for ROS...")
            from torch.utils.data import WeightedRandomSampler
            
            class_counts = np.zeros(self.K)
            sample_classes = []
            
            # Iterate through dataset to count primitive occurrences
            # Wrapped in tqdm as this might take a few seconds for large datasets
            for i in tqdm(range(len(train_dataset)), desc="Calculating ROS Weights"):
                action_data = train_dataset[i]['action']
                
                # Dynamically get the action key (usually 'default')
                act_key = list(action_data.keys())[0]
                prim_bin = action_data[act_key][0][0] 
                
                # Decode primitive ID
                prim_id = int(np.clip((((prim_bin + 1) / 2) * self.K), 0, self.K - 1))
                class_counts[prim_id] += 1
                sample_classes.append(prim_id)
                
            # Calculate weight per class (inverse frequency)
            # np.where prevents division by zero if a primitive is entirely missing
            class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
            
            # Assign a weight to every specific sample
            sample_weights = [class_weights[cls_id] for cls_id in sample_classes]
            
            # Create the PyTorch Sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            
            # PyTorch DataLoader throws an error if shuffle=True and sampler is provided
            shuffle_data = False 
            
            print(f"[MultiPrimitiveDiffusionAdapter] ROS Class Counts: {class_counts}")
        # ==========================================================
    
        torch.backends.cudnn.benchmark = True
        # Ensure persistent_workers is strictly False if num_workers is 0
        num_workers = self.config.get('num_workers', 0)
        persistent = self.config.get('persistent_workers', False) if num_workers > 0 else False
        prefetch_factor = self.config.get('prefetch_factor', None)

        self.dataloader = torch.utils.data.DataLoader(
            train_dataset, # <-- Use train_dataset
            batch_size=self.config.batch_size, 
            shuffle=shuffle_data,   
            sampler=sampler,        
            num_workers=num_workers,
            pin_memory=self.config.get('pin_memory', False),
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor
        )

        # Create Validation Dataloader
        if self.validate_training:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.batch_size, 
                shuffle=False, # No need to shuffle validation
                num_workers=max(1, num_workers // 4), # Fewer workers for val
                pin_memory=self.config.get('pin_memory', False),
                persistent_workers=persistent
            )
        
        self.dataset_inited = True
        
    def _init_demo_policy_dataset(self, arenas):
        
        
        arena = arenas[0] # assume only one arena
        org_horizon = arena.action_horizon
        arena.action_horizon = self.config.get('demo_horizon', org_horizon)
        from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            # convert dotmap to dict
        config = self.config.dataset_config #.toDict()
        config['io_mode'] = 'a'
        #print('config', config)
        train_dataset = TrajectoryDataset(**config, sample_mode='train')

        if self.validate_training:
            val_dataset = TrajectoryDataset(**config, sample_mode='val')

        import actoris_harena as ag_ar
        policy = ag_ar.build_agent(
            self.config.demo_policy, 
            self.config.get('demo_policy_config', DotMap({})),
            disable_wandb=True)

        qbar = tqdm(total=self.config.num_demos, 
                    desc='Collecting data from policy ...')

        qbar.update(train_dataset.num_trajectories())
        qbar.refresh()
        
        episode_id = train_dataset.num_trajectories()
        train_configs = arena.get_train_configs()
        while train_dataset.num_trajectories() < self.config.num_demos:
            observations = {obs_type: [] for obs_type in train_dataset.obs_types}
            actions = {act_type: [] for act_type in train_dataset.action_types}

            policy.reset([arena.id])
            print('[multi-primitive diffusion] reset episode id', episode_id)
            info = arena.reset(train_configs[episode_id])
            policy.init([info])
            info['reward'] = 0
            done = info['done']
            #print('done', done)
            while not done:
                action = policy.single_act(info)
                
                if action is None:
                    break
                
                for k, v in info['observation'].items():
                    if self.debug: print('[MultiPrimitiveDiffusionAdapter] key in info', k)
                    if k in observations.keys():
                        if k in ['rgb', 'depth', 'goal_rgb', 'goal_depth']:
                            #print('k ', k, 'v', v)
                            v_ = cv2.resize(v, (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                            observations[k].append(v_)
                        elif k in ['mask', 'goal_mask']:
                            
                            if self.debug:
                                step_idx = len(observations[k])
                                file_name = f"{k}_ep{episode_id}_step{step_idx}_before_resize"
                                save_mask(
                                    mask=v, 
                                    filename=file_name, 
                                    directory="tmp/debug_mluti_primitive_diffusion"
                                )

                            v_ = cv2.resize(v.astype(np.float32), (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                            v_ = v_ > 0.9
                            
                            if self.debug:
                                step_idx = len(observations[k])
                                file_name = f"{k}_ep{episode_id}_step{step_idx}"
                                save_mask(
                                    mask=v_, 
                                    filename=file_name, 
                                    directory="tmp/debug_mluti_primitive_diffusion"
                                )
                            
                            observations[k].append(v_)
                        else:
                            observations[k].append(v)
                            if self.debug:
                                print('[MultiPrimitiveDiffusionAdapter] k, v to save', k, v)
                
                add_action = action
                if self.config.primitive_integration in ['bin_as_output', 'one-hot-encoding']: 
                    # Unused dimenstions are zeros
                    action_name = list(action.keys())[0]
                    action_param = action[action_name]
                    prim_id = self.prim_name2id[action_name]
                    prim_act = (1.0*(prim_id+0.5)/self.K *2 - 1)
                    add_action = np.zeros(self.data_save_action_dim)
                    add_action[0] = prim_act
                    add_action[1:action_param.shape[0]+1] = action_param
                    #add_action = np.concatenate([prim_act, action_param])
                    
                elif self.config.primitive_integration == 'none':
                    add_action = action
                else:
                    raise NotImplementedError
                
                #print('add action', add_action)
                
                actions['default'].append(add_action)  
              
                info = arena.step(action)
                # print('[diffusion] demo reward', info['reward'])
                policy.update(info, add_action)
                info['reward'] = 0
                done = info['done']
                if (self.collect_on_success and info['success']):
                    break
            # print('[debug] keys', info['observation'].keys())
            for k, v in info['observation'].items():
                if k in observations.keys():
                    if k in ['rgb', 'depth', 'goal_rgb', 'goal_depth']:
                        v_ = cv2.resize(v, (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                        observations[k].append(v_)
                    elif k in ['mask', 'goal_mask']:
                        v_ = cv2.resize(v.astype(np.float32), (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                        v_ = v_ > 0.9
                        observations[k].append(v_)
                    else:
                        observations[k].append(v)
            #print('info eval', info['evaluation'])
            if self.config.debug:
                frames = arena.get_frames()
                if len(frames) > 0:
                    save_video(np.stack(arena.get_frames()), 'tmp', 'diffusion_demo')
                    save_numpy_as_gif(
                        np.stack(arena.get_frames()), 
                        path='tmp',
                        filename="diffusion_demo"
                    )
            if info['success'] or self.config.get('add_all_demos', False):
                #print('add to trajectory')
                for k, v in observations.items():
                    #print(f'[MultiPrimitiveDiffusionAdapter] k {k}')
                    print(f'[debug] k {k}')
                    observations[k] = np.stack(v)
                actions['default'] = np.stack(actions['default'])
                #print('actions default shape', actions['default'].shape)
                skip = False
                if not skip:
                    train_dataset.add_trajectory(observations, actions)
                    qbar.update(1)
                
            episode_id += 1
            print('[multi-primitive-diffusion] arena.get_num_episodes', arena.get_num_episodes() )
            episode_id %= arena.get_num_episodes()

        arena.action_horizon = org_horizon
        
        # ==========================================================
        # --- NEW FEATURE: Random Oversampling (ROS) via Config ---
        # ==========================================================
        use_ros = self.config.get('use_random_oversampling', False)
        sampler = None
        shuffle_data = True # Default to True unless using a sampler
        
        if use_ros and self.primitive_integration != 'none':
            print("[MultiPrimitiveDiffusionAdapter] Initializing WeightedRandomSampler for ROS...")
            from torch.utils.data import WeightedRandomSampler
            
            class_counts = np.zeros(self.K)
            sample_classes = []
            
            # Iterate through dataset to count primitive occurrences
            # Wrapped in tqdm as this might take a few seconds for large datasets
            for i in tqdm(range(len(train_dataset)), desc="Calculating ROS Weights"):
                action_data = train_dataset[i]['action']
                
                # Dynamically get the action key (usually 'default')
                act_key = list(action_data.keys())[0]
                prim_bin = action_data[act_key][0][0] 
                
                # Decode primitive ID
                prim_id = int(np.clip((((prim_bin + 1) / 2) * self.K), 0, self.K - 1))
                class_counts[prim_id] += 1
                sample_classes.append(prim_id)
                
            # Calculate weight per class (inverse frequency)
            # np.where prevents division by zero if a primitive is entirely missing
            class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
            
            # Assign a weight to every specific sample
            sample_weights = [class_weights[cls_id] for cls_id in sample_classes]
            
            # Create the PyTorch Sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            
            # PyTorch DataLoader throws an error if shuffle=True and sampler is provided
            shuffle_data = False 
            
            print(f"[MultiPrimitiveDiffusionAdapter] ROS Class Counts: {class_counts}")
        # ==========================================================

        torch.backends.cudnn.benchmark = True
        # Ensure persistent_workers is strictly False if num_workers is 0
        num_workers = self.config.get('num_workers', 0)
        persistent = self.config.get('persistent_workers', False) if num_workers > 0 else False
        prefetch_factor = self.config.get('prefetch_factor', None)
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset, # <-- Use train_dataset
            batch_size=self.config.batch_size, 
            shuffle=shuffle_data,   
            sampler=sampler,        
            num_workers=num_workers,
            pin_memory=self.config.get('pin_memory', False),
            persistent_workers=persistent,
            prefetch_factor=prefetch_factor
        )

        # Create Validation Dataloader
        if self.validate_training:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.batch_size, 
                shuffle=False, # No need to shuffle validation
                num_workers=max(1, num_workers // 4), # Fewer workers for val
                pin_memory=self.config.get('pin_memory', False),
                persistent_workers=persistent
            )



        self.dataset_inited = True

    
    def validate(self):
        if not hasattr(self, 'val_dataloader'):
            print("[Validation] No validation dataloader found. Skipping.")
            return

        self.nets.eval() # Switch to evaluation mode
        
        val_prim_losses = []
        val_state_mse = []
        val_state_l2 = []
        val_action_mse = []
        all_preds = []
        all_gts = []
        
        print(f"--- Running Validation at Step {self.update_step} ---")
        
        with torch.no_grad():
            for nbatch in self.val_dataloader:
                
                # Apply data augmenter for standardisation (train=False)
                if self.config.dataset_mode == 'general':
                    obs = nbatch['observation']
                    action = nbatch['action']['default']
                    nbatch = {v: k for v, k in obs.items()}
                    nbatch['action'] = action.reshape(*action.shape[:2], -1)
                    nbatch = self.data_augmenter(nbatch, train=False, device=self.device)

                # ==========================================================
                # --- Reconstruct composite observation tensors ---
                # ==========================================================
                if self.config.input_obs == 'rgbd':
                    nbatch['rgbd'] = torch.cat([nbatch['rgb'], nbatch['depth']], dim=2)
                if self.config.input_obs == 'rgb-workspace-mask':
                    nbatch['rgb-workspace-mask'] = torch.cat([nbatch['rgb'], nbatch['robot0_mask'], nbatch['robot1_mask']], dim=2)
                if self.config.input_obs == 'rgb-workspace-mask-goal':
                    nbatch['rgb-workspace-mask-goal'] = torch.cat([nbatch['rgb'], nbatch['robot0_mask'], nbatch['robot1_mask'], nbatch['goal_rgb']], dim=2)
                if self.config.input_obs == 'rgb+goal_rgb':
                    nbatch['rgb+goal_rgb'] = torch.cat([nbatch['rgb'], nbatch['goal_rgb']], dim=2)
                if self.config.input_obs == 'rgb+goal_mask':
                    nbatch['rgb+goal_mask'] = torch.cat([nbatch['rgb'], nbatch['goal_mask']], dim=2)

                B = nbatch[self.config.input_obs].shape[0]
                input_obs = nbatch[self.config.input_obs][:, :self.config.obs_horizon].flatten(end_dim=1).float().to(self.device)
                
                # 1. Extract Vision Features (Match training logic)
                if self.vision_encoder_type == 'original':
                    image_features = self.nets['vision_encoder'](input_obs)
                    obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                elif self.vision_encoder_type == 'gc_rssm_encoder':
                    rgb_part = input_obs[:, :3, :, :]
                    goal_rgb_part = input_obs[:, 3:6, :, :]
                    obs_feature = self.nets['vision_encoder'](rgb_part) 
                    goal_feature = self.nets['vision_encoder'](goal_rgb_part)
                    image_features = torch.cat([obs_feature, goal_feature], dim=-1)
                    obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                
                # ==========================================================
                # --- NEW: Apply MLP Projection ---
                # ==========================================================
                if getattr(self, 'use_projector', False):
                    obs_features = self.nets['obs_projector'](obs_features)

                if self.config.include_state:
                    vector_state = nbatch['vector_state'][:, :self.config.obs_horizon].to(self.device).float()
                    obs_features = torch.cat([obs_features, vector_state], dim=-1)

                obs_cond = obs_features.flatten(start_dim=1)

                # 2. Validate Keypoint Precision (L2 Distance) and State Loss
                if self.rep_learn == 'predict-state':
                    pred_combined = self.nets['state_predictor'](obs_features)
                    state_key = self.config.get('state_key', 'semkey_norm_pixel')
                    
                    gt_tensors_val = []
                    
                    # 1. Evaluate Current State
                    if state_key in nbatch:
                        gt_state = nbatch[state_key][:, :self.config.obs_horizon].reshape(B, self.config.obs_horizon, -1).float().to(self.device)
                        gt_tensors_val.append(gt_state)
                        
                        # Compute L2 norm across keypoint coordinates
                        state_dim = self.config.state_dim
                        num_kpts = state_dim // 2
                        pred_kpts = pred_combined[..., :state_dim].view(-1, num_kpts, 2)
                        gt_kpts = gt_state[..., :state_dim].view(-1, num_kpts, 2)
                        
                        l2_dist = torch.norm(pred_kpts - gt_kpts, dim=-1).mean()
                        val_state_l2.append(l2_dist.item())
                    
                    # 2. Evaluate Goal State (if active)
                    if getattr(self, 'predict_goal_state', False):
                        goal_key = self.config.get('goal_state_key', 'flattened_goal_semkey_norm_pixel')
                        if goal_key in nbatch:
                            gt_goal = nbatch[goal_key][:, :self.config.obs_horizon].reshape(B, self.config.obs_horizon, -1).float().to(self.device)
                            gt_tensors_val.append(gt_goal)
                            
                    # 3. Compute and store unified MSE loss
                    if gt_tensors_val:
                        gt_combined_val = torch.cat(gt_tensors_val, dim=-1)
                        mse_loss = nn.functional.mse_loss(pred_combined, gt_combined_val)
                        val_state_mse.append(mse_loss.item())

                # 3. Validate Primitive Classification Accuracy
                if self.primitive_integration == 'one-hot-encoding':
                    prim_logits = self.nets['prim_class_head'](obs_cond)
                    
                    prim_bin = nbatch['action'][:, 0, 0].to(self.device)
                    gt_prim_ids = (((prim_bin + 1) / 2) * self.K).long()
                    gt_prim_ids = torch.clamp(gt_prim_ids, 0, self.K - 1)
                    
                    loss = nn.functional.cross_entropy(prim_logits, gt_prim_ids)
                    val_prim_losses.append(loss.item())
                    
                    preds = torch.argmax(prim_logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_gts.extend(gt_prim_ids.cpu().numpy())
                    
                    # Create one-hot encoding for the diffusion condition
                    # (Using predicted primitives for a true inference evaluation)
                    prim_one_hot = nn.functional.one_hot(preds, num_classes=self.K).float()
                    obs_cond_diff = torch.cat([obs_cond, prim_one_hot], dim=-1)
                    
                    # The ground truth continuous action removes the first dim (primitive bin)
                    gt_action = nbatch['action'][:, :, 1:].to(self.device)
                else:
                    obs_cond_diff = obs_cond
                    gt_action = nbatch['action'].to(self.device)

                # ==========================================================
                # --- 4. Action Parameter Inference & Error Calculation ---
                # ==========================================================
                # Start from pure noise
                eval_naction = torch.randn((B, self.config.pred_horizon, self.diffusion_dim), device=self.device)
                loss_type = self.config.get('loss_type', 'diffusion')
                
                # Reverse Diffusion / Flow Matching Loop
                if loss_type == 'ot_flow_match':
                    num_steps = self.config.num_diffusion_iters
                    dt = 1.0 / num_steps
                    for i in range(num_steps):
                        t_val = i / num_steps
                        timestep_tensor = torch.full((B,), t_val * self.config.num_diffusion_iters, device=self.device, dtype=torch.float32)
                        v_pred = self.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond_diff)
                        eval_naction = eval_naction + v_pred * dt
                else:
                    self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
                    for k in self.noise_scheduler.timesteps:
                        timestep_tensor = torch.full((B,), k.item(), device=self.device, dtype=torch.long)
                        n_pred = self.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond_diff)
                        eval_naction = self.noise_scheduler.step(model_output=n_pred, timestep=k, sample=eval_naction).prev_sample
                
                # Extract the continuous action parameters (ignore the state prediction part if joint diffusion)
                pred_action = eval_naction[..., :self.network_action_dim]
                
                # Calculate Action Parameter MSE
                if self.primitive_integration != 'none' and getattr(self, 'mask_out_irrelavent_action_dim', False):
                    # Masking based on ground truth primitive to properly evaluate intended action parameters
                    mask = torch.zeros((B, self.config.pred_horizon, self.network_action_dim), device=self.device)
                    for b in range(B):
                        mask[b] = self.primitive_action_masks[gt_prim_ids[b].item()].to(self.device)
                    
                    diff = (pred_action - gt_action) * mask
                    valid_count = mask.sum().clamp(min=1.0)
                    mse = (diff ** 2).sum() / valid_count
                else:
                    mse = nn.functional.mse_loss(pred_action, gt_action)
                
                val_action_mse.append(mse.item())

        # 5. Aggregate and Log Metrics
        metrics = {}
        if val_state_l2:
            metrics['val/state_keypoint_l2_avg'] = np.mean(val_state_l2)
        
        if val_state_mse:
            metrics['val/state_pred_loss'] = np.mean(val_state_mse)

        if val_prim_losses:
            metrics['val/prim_loss'] = np.mean(val_prim_losses)
            acc = np.mean(np.array(all_preds) == np.array(all_gts))
            metrics['val/prim_accuracy'] = acc
            
            # Log Confusion Matrix to wandb
            import wandb
            if wandb.run is not None:
                confusion = {"val/prim_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_gts,
                    preds=all_preds,
                    class_names=[p['name'] if isinstance(p, dict) else p.name for p in self.primitives]
                )}
                self.logger.log(confusion, step=self.update_step)

        if val_action_mse:
            metrics['val/action_mse'] = np.mean(val_action_mse)

        if metrics:
            self.logger.log(metrics, step=self.update_step)
            print(f"Validation Results: {metrics}")

        self.nets.train() # Switch back to training mode
    
    def _init_optimizer(self):
        # Filter parameters that require gradients
        trainable_params = [p for p in self.nets.parameters() if p.requires_grad]

        self.ema = EMAModel(
            parameters=trainable_params,
            power=self.config.get('ema_power', 0.75))
        
        opt_params = self.config.get('optimiser_params', {})
        if hasattr(opt_params, 'toDict'):
            opt_params = opt_params.toDict()
            
        self.optimizer = torch.optim.AdamW(
            params=trainable_params,
            **opt_params
        )

        scheduler_name = self.config.get('lr_scheduler', 'cosine')
        warmup_steps = self.config.get('num_warmup_steps', 500)

        self.lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.config.total_update_steps
        )

        self.clip_norm = self.config.get('grad_clip_norm', -1)

    def _init_networks(self):
        self.rep_learn = self.config.get('rep_learn', 'none')
        self.input_channel = 3
        if self.config.input_obs == 'rgbd':
            self.input_channel = 4
        elif self.config.input_obs == 'depth':
            self.input_channel = 1
        elif self.config.input_obs == 'rgb-workspace-mask':
            self.input_channel = 5
        elif self.config.input_obs == 'rgb-workspace-mask-goal':
            self.input_channel = 8
        elif self.config.input_obs == 'rgb+goal_rgb':
            self.input_channel = 6
        elif self.config.input_obs == 'rgb+goal_mask':
            self.input_channel = 4

        self.vision_encoder_type = self.config.get('vision_encoder', 'original')
        
        if self.vision_encoder_type == 'original':
            self.vision_encoder = get_resnet('resnet18', input_channel=self.input_channel)
            self.vision_encoder = replace_bn_with_gn(self.vision_encoder)

        elif self.vision_encoder_type == 'vit':
            # Load pre-trained ViT. ViT-B/16 outputs a 768-dim embedding.
            self.vision_encoder = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            # Strip the classification head to output raw 768-dim features
            self.vision_encoder.heads = nn.Identity() 
            
            # Notice: No freezing logic! The encoder will train end-to-end.
            print("[MultiPrimitiveDiffusion] ViT encoder initialized for end-to-end fine-tuning.")
                        
        elif self.vision_encoder_type == 'gc_rssm_encoder':
            from ..rl.lagarnet.networks import ImageEncoder
            self.vision_encoder = ImageEncoder(
                image_dim=self.config.input_obs_dim,
                embedding_size=self.config.embedding_dim,
                activation_function=self.config.activation,
                batchnorm=self.config.encoder_batchnorm,
                residual=self.config.encoder_residual
            )
            
            
            # Load pretrained GC-RSSM encoder weights if specified
            pretrained_path = self.config.get('pretrained_encoder_path', None)
            if pretrained_path:
                if os.path.exists(pretrained_path):
                    checkpoint = torch.load(pretrained_path)
                    self.vision_encoder.load_state_dict(checkpoint['encoder'])
                    print(f"[MultiPrimitiveDiffusion] Loaded pretrained GC-RSSM encoder from {pretrained_path}")
                else:
                    print(f"[MultiPrimitiveDiffusion] Path {pretrained_path} does not exists. Cannot load the pretrained encoder.")
                
            # Freeze the encoder if specified
            if self.config.get('freeze_encoder', False):
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
                self.vision_encoder.eval()
                print("[MultiPrimitiveDiffusion] Vision encoder is frozen.")

        elif self.vision_encoder_type == 'gc_rssm_dynamic':
            # Import your transition model (adjust the import path based on your folder structure)
            from ..rl.lagarnet.networks import ImageEncoder
            from ..rl.lagarnet.gc_rssm import GoalConditionedTransitionModel

            self.vision_encoder = ImageEncoder(
                image_dim=self.config.input_obs_dim,
                embedding_size=self.config.embedding_dim,
                activation_function=self.config.activation,
                batchnorm=self.config.encoder_batchnorm,
                residual=self.config.encoder_residual
            )
            
            self.transition_model = GoalConditionedTransitionModel(
                belief_size=self.config.deterministic_latent_dim,
                state_size=self.config.stochastic_latent_dim,
                action_size=self.network_action_dim, 
                hidden_size=self.config.hidden_dim,
                embedding_size=self.config.embedding_dim,
                activation_function=self.config.activation,
                min_std_dev=self.config.get('min_std_dev', 0.1),
                embedding_layers=self.config.get('trans_layers', 1),
                state_layers=self.config.get('state_layers', 1)
            )

            pretrained_path = self.config.get('pretrained_encoder_path', None)
            if pretrained_path and os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path)
                
                # Load weights loosely to allow custom FC layers to train
                missing_e, _ = self.vision_encoder.load_state_dict(checkpoint['encoder'])
                missing_t, _ = self.transition_model.load_state_dict(checkpoint['transition_model'])
                print(f"[MultiPrimitiveDiffusion] Loaded pretrained GC-RSSM dynamic model from {pretrained_path}")

            if self.config.get('freeze_encoder', False):
                # Freeze encoder
                for name, param in self.vision_encoder.named_parameters():
                    if 'missing_e' in locals() and name in missing_e:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                # Freeze transition model
                for name, param in self.transition_model.named_parameters():
                    if 'missing_t' in locals() and name in missing_t:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print("[MultiPrimitiveDiffusion] Vision encoder and Transition model are frozen (except missing keys).")
        
        # ==========================================================
        # --- NEW: MLP Projection Block Setup ---
        # ==========================================================
        self.use_projector = self.config.get('use_projector', False)
        self.effective_obs_dim = self.config.obs_dim  # Default to original

        if self.use_projector:
            proj_hidden = self.config.get('projector_hidden_dims', [])
            proj_out = self.config.get('projector_out_dim', 128)
            
            layers = []
            in_dim = self.config.obs_dim
            for h in proj_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, proj_out))
            
            self.obs_projector = nn.Sequential(*layers)
            self.effective_obs_dim = proj_out
            print(f"[MultiPrimitiveDiffusion] MLP Projector initialized. Mapping {self.config.obs_dim} -> {self.effective_obs_dim}")
        else:
            self.obs_projector = nn.Identity()

        #self.obs_feature_dim = self.config.obs_dim * self.config.obs_horizon
        if self.primitive_integration == 'one-hot-encoding':

            cls_cfg = self.config.get("primitive_classifier", {}) # nn.Linear(self.config.obs_dim, self.K) by default

            self.prim_class_head = MLPClassifier(
                input_dim=self.effective_obs_dim,
                output_dim=self.K,
                hidden_dims=cls_cfg.get("hidden_dims", []),
                activation=cls_cfg.get("activation", "relu"),
                dropout=cls_cfg.get("dropout", 0.0),
                use_layernorm=cls_cfg.get("use_layernorm", False),
            )

            # Increase global_cond_dim to accommodate the one-hot vector
            global_cond_dim = (self.effective_obs_dim + self.K) * self.config.obs_horizon
            self.log_prim_metrics_every = self.config.get('log_prim_metrics_every', 200)
        else:
            global_cond_dim = self.effective_obs_dim * self.config.obs_horizon

        # Determine the total dimension for diffusion
        self.diffusion_dim = self.network_action_dim
        if self.rep_learn == 'predict-state-with-action':
            self.diffusion_dim += self.config.state_dim
            
            # --- NEW: Account for Goal State Dimension ---
            self.predict_goal_state = self.config.get('predict_goal_state', False)
            if self.predict_goal_state:
                self.goal_state_dim = self.config.get('goal_state_dim', 30)
                self.diffusion_dim += self.goal_state_dim
                
            print(f"[MultiPrimitiveDiffusion] Joint Action+State Diffusion Active. Total dim: {self.diffusion_dim}")

        # Check config to decide which backbone to use (defaults to unet)
        backbone_type = self.config.get('noise_pred_net', 'unet')

        if backbone_type == 'mlp':
            self.noise_pred_net = ConditionalMLP1D(
                input_dim=self.diffusion_dim,          # <--- Updated
                global_cond_dim=global_cond_dim,
                pred_horizon=self.config.pred_horizon,
                hidden_dims=self.config.get('mlp_hidden_dims', [512, 512, 512]),
                activation="relu",
                dropout=0.1
            )
        else:
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=self.diffusion_dim,          # <--- Updated
                global_cond_dim=global_cond_dim,
                diable_updown=(self.config.disable_updown if 'disable_updown' in self.config else False),
            )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.num_diffusion_iters, # default value 100
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        self.rep_learn = self.config.get('rep_learn', 'none')

        net_dict = {
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        }

        if self.use_projector:
            net_dict['obs_projector'] = self.obs_projector

        if self.vision_encoder_type == 'gc_rssm_dynamic':
            net_dict['transition_model'] = self.transition_model
        
        self.nets = nn.ModuleDict(net_dict)

        if self.primitive_integration == 'one-hot-encoding':
            self.nets['prim_class_head'] = self.prim_class_head

        if self.rep_learn == 'auto-encoder':
            self.nets['vision_decoder'] = ResNetDecoder(
                input_dim=512, 
                output_channel=self.input_channel
            )
        elif self.rep_learn == 'predict-state':
            # Check if we are extending the prediction to include the goal state
            self.predict_goal_state = self.config.get('predict_goal_state', False)
            self.goal_state_dim = self.config.get('goal_state_dim', 30) # Default assuming 15 points * 2 (x,y)
            
            out_dim = self.config.state_dim
            if self.predict_goal_state:
                out_dim += self.goal_state_dim
                print(f"[MultiPrimitiveDiffusion] state_predictor out_dim extended to {out_dim} (State + Goal)")

            # Unified MLP predicting both current and goal states
            self.nets['state_predictor'] = nn.Sequential(
                nn.Linear(self.effective_obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim)
            )

        self._test_network()

        self.device = self.config.get('device', 'cpu')
        self.nets.to(self.device)
        
    def _test_network(self):

        # --- Parameter Calculation Added Here ---
        total_params = sum(p.numel() for p in self.nets.parameters())
        trainable_params = sum(p.numel() for p in self.nets.parameters() if p.requires_grad)
        
        print("-" * 50)
        print(f"[MultiPrimitiveDiffusion Network Stats]")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Optional: Print parameters per sub-network for debugging
        for name, net in self.nets.items():
            net_params = sum(p.numel() for p in net.parameters())
            print(f"  - {name}: {net_params:,}")
        print("-" * 50)
        # ----------------------------------------

        with torch.no_grad():
            # example inputs

            if self.vision_encoder_type == 'original':
                image = torch.zeros(
                    (1, self.config.obs_horizon,
                    self.input_channel,96,96))
                # vision encoder
                image_features = self.nets['vision_encoder'](
                    image.flatten(end_dim=1)
                )
                # (2,512)
                obs = image_features.reshape(*image.shape[:2],-1)
            elif self.vision_encoder_type == 'vit':
                import torchvision.transforms.functional as TF
                
                # Dummy input: 5D (Batch=1, Time=obs_horizon, Channels=6, H=128, W=128)
                image = torch.zeros(
                    (1, self.config.obs_horizon, 6, 128, 128), 
                   
                )
                
                # FLATTEN FIRST: (B*T, C, H, W) -> (1, 6, 128, 128)
                flat_image = image.flatten(end_dim=1)
                
                # NOW SLICE CHANNELS:
                rgb_part = flat_image[:, :3, :, :]
                goal_rgb_part = flat_image[:, 3:6, :, :]

                # Resize to 224x224 for ViT
                rgb_resized = TF.resize(rgb_part, [224, 224], antialias=True)
                goal_resized = TF.resize(goal_rgb_part, [224, 224], antialias=True)

                # Forward pass
                obs_feature = self.nets['vision_encoder'](rgb_resized) 
                goal_feature = self.nets['vision_encoder'](goal_resized)

                # Concatenate features (768 + 768 = 1536) and reshape back to (B, T, Feature_Dim)
                obs = torch.cat([obs_feature, goal_feature], dim=-1)
                obs = obs.reshape(*image.shape[:2], -1)

            elif self.vision_encoder_type == 'gc_rssm_encoder':
                image = torch.zeros(
                    (1, self.config.obs_horizon,
                    3,64,64)) #.to(self.config.device)
                
                goal_image = torch.zeros(
                    (1, self.config.obs_horizon,
                    3,64,64)) #.to(self.config.device)

                # Flatten batch and time dimensions for the forward pass
                image_feat = self.nets['vision_encoder'](image.flatten(end_dim=1))
                goal_feat = self.nets['vision_encoder'](goal_image.flatten(end_dim=1))
                
                # Concatenate features and reshape back to (Batch, Obs_Horizon, Feature_Dim)
                obs = torch.cat([image_feat, goal_feat], dim=-1)
                obs = obs.reshape(*image.shape[:2], -1)
            elif self.vision_encoder_type == 'gc_rssm_dynamic':

                B, T = 1, self.config.obs_horizon
                image = torch.zeros(
                    (B, T,
                    3,64,64)) #.to(self.config.device)
                
                goal_image = torch.zeros(
                    (B, T,
                    3,64,64)) #.to(self.config.device)

                # Flatten batch and time dimensions for the forward pass
                obs_emb = self.nets['vision_encoder'](image.flatten(end_dim=1)).view(B, T, -1).transpose(0, 1)
                goal_emb = self.nets['vision_encoder'](goal_image.flatten(end_dim=1)).view(B, T, -1).transpose(0, 1)
                
                dummy_actions = torch.ones(T, B, self.network_action_dim)
                init_belief = torch.zeros(B, self.config.deterministic_latent_dim)
                init_state = torch.zeros(B, self.config.stochastic_latent_dim)
                nonterminals = torch.ones(T, B, 1)

                hidden = self.nets['transition_model'](
                    prev_state=init_state,
                    actions=dummy_actions,
                    prev_belief=init_belief,
                    goal_observations=goal_emb,
                    observations=obs_emb,
                    nonterminals=nonterminals
                )

                latents = torch.cat([hidden[0], hidden[4]], dim=-1) # Cat belief and posterior
                obs = latents.transpose(0, 1).reshape(B, T, -1) # Remove batch dim -> (T, deter+stoch)

            if getattr(self, 'use_projector', False):
                obs = self.nets['obs_projector'](obs)

            if self.config.include_state:
                vector_state = torch.zeros(
                    (1, self.config.obs_horizon, 
                    self.config.state_dim))
                obs = torch.cat([obs, vector_state],dim=-1)
            
            if self.rep_learn == 'predict-state':
                _ = self.nets['state_predictor'](obs)

            # print('[MultiPrimitiveDiffusion, _test_network] obs', obs.shape)
            
            test_dim = getattr(self, 'diffusion_dim', self.network_action_dim)
            noised_action = torch.randn(
                (1, self.config.pred_horizon, test_dim))
            diffusion_iter = torch.zeros((1,))
            # print('noised action', noised_action.shape)

            # the noise prediction network
            # takes noisy action, diffusion iteration and observation as input
            # predicts the noise added to action
            
            goal_cond = obs

            # 5. Handle One-Hot Encoding Integration
            if self.primitive_integration == 'one-hot-encoding':
                # Predict primitive logits from the flattened observation
                prim_logits = self.nets['prim_class_head'](goal_cond.squeeze(0))
                
                # For testing, we can just take the argmax or simulate a specific ID
                prim_id = torch.argmax(prim_logits, dim=-1) # Shape (Batch,)
                
                # Convert to one-hot: (Batch, K)
                prim_one_hot = nn.functional.one_hot(
                    prim_id, num_classes=self.K
                ).float().unsqueeze(0)

                #print(f'[MultiPrimitiveDiffusion, _test_network] goal_cond {goal_cond.shape}, prim_one_hot {prim_one_hot.shape}')

                # Concatenate one-hot vector to the global condition
                goal_cond = torch.cat([goal_cond, prim_one_hot], dim=-1)


            goal_cond = goal_cond.flatten(start_dim=1)

            noise = self.nets['noise_pred_net'](
                sample=noised_action,
                timestep=diffusion_iter,
                global_cond=goal_cond)

            # illustration of removing noise
            # the actual noise removal is performed by NoiseScheduler
            # and is dependent on the diffusion noise schedule
            # denoised_action = noised_action - noise


    def train(self, update_steps, arenas):
        if not self.dataset_inited:
            if self.config.train_mode == 'from_dataset':
                self._init_dataset()
            elif self.config.train_mode == 'from_policy':
                self._init_demo_policy_dataset(arenas)
            else:
                raise ValueError('Invalid train mode')
        
        update_steps = min(#
            self.config.total_update_steps - self.update_step,
            update_steps)
        
        if self.config.get('freeze_encoder', False):
            self.nets['vision_encoder'].eval()
            
        #print('train update steps', update_steps)
        pbar = tqdm(range(update_steps), desc="Training")

        for i in pbar:

            nbatch = next(iter(self.dataloader))
          

            if self.config.dataset_mode == 'diffusion':
                nbatch = self.data_augmenter(nbatch, train=True, device=self.device)
            else:
                obs = nbatch['observation']
                action = nbatch['action']['default']
                #print('[diffusion] action', action.shape, action[0])
                nbatch = {v: k for v, k in obs.items()}
                nbatch['action'] = action.reshape(*action.shape[:2], -1)
                #print('action after shape', nbatch['action'] .shape)
                nbatch = self.data_augmenter(nbatch, train=True, device=self.device)
                #print('[diffusion] action after augment', nbatch['action'].shape)
            
            if self.config.input_obs == 'rgbd':
                # concatenate rgb and depth
                nbatch['rgbd'] = torch.cat([
                    nbatch['rgb'], nbatch['depth']], dim=2)
            
            if self.config.input_obs == 'rgb-workspace-mask':
                nbatch['rgb-workspace-mask'] = torch.cat([
                    nbatch['rgb'], nbatch['robot0_mask'], nbatch['robot1_mask']], dim=2)
            
            if self.config.input_obs == 'rgb-workspace-mask-goal':
                nbatch['rgb-workspace-mask-goal'] = torch.cat([
                    nbatch['rgb'], nbatch['robot0_mask'], nbatch['robot1_mask'], nbatch['goal_rgb']], dim=2)
            
            if self.config.input_obs == 'rgb+goal_rgb':
                nbatch['rgb+goal_rgb'] = torch.cat([nbatch['rgb'], nbatch['goal_rgb']], dim=2)
            
            if self.config.input_obs == 'rgb+goal_mask':
                nbatch['rgb+goal_mask'] = torch.cat([nbatch['rgb'], nbatch['goal_mask']], dim=2)

            
            B = nbatch[self.config.input_obs].shape[0]
            input_obs = nbatch[self.config.input_obs][:, :self.config.obs_horizon]\
                .flatten(end_dim=1).float()


            if 'action' in nbatch:
                nbatch['action'] = nbatch['action'].float()

            
            if 'vector_state' in nbatch:
                 nbatch['vector_state'] = nbatch['vector_state'].float()
            
            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            use_amp = self.config.get('use_amp', False)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                

                if self.vision_encoder_type == 'original':
                    image_features = self.nets['vision_encoder'](input_obs)
                    obs_features = image_features.reshape(
                        B, self.config.obs_horizon, -1)
                
                elif self.vision_encoder_type == 'vit':
                    # input_obs is ALREADY 4D here: (B * T, 6, H, W)
                    #print('input_obs shape',  input_obs.shape)
                    rgb_part = input_obs[:, :3, :, :]
                    goal_rgb_part = input_obs[:, 3:6, :, :]

                    import torchvision.transforms.functional as TF
                    # REMOVED .flatten(end_dim=1) -> Keep it 4D!
                    rgb_resized = TF.resize(rgb_part, [224, 224], antialias=True)
                    goal_resized = TF.resize(goal_rgb_part, [224, 224], antialias=True)

                    obs_feature = self.nets['vision_encoder'](rgb_resized) 
                    goal_feature = self.nets['vision_encoder'](goal_resized)

                    # Concatenate -> Shape: (B * T, 1536)
                    image_features = torch.cat([obs_feature, goal_feature], dim=-1)
                    obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                    
                elif self.vision_encoder_type == 'gc_rssm_encoder':
                    # Slicing the 6-channel image into two 3-channel images (obs and goal)
                    # image shape is expected to be (obs_horizon, 6, H, W)
                    rgb_part = input_obs[:, :3, :, :]
                    goal_rgb_part = input_obs[:, 3:6, :, :]

                    obs_feature = self.nets['vision_encoder'](rgb_part) 
                    goal_feature = self.nets['vision_encoder'](goal_rgb_part)

                    # Concatenate the two feature vectors along the last dimension
                    image_features = torch.cat([obs_feature, goal_feature], dim=-1)

                    obs_features = image_features.reshape(
                        B, self.config.obs_horizon, -1)
                
                elif self.vision_encoder_type == 'gc_rssm_dynamic':
                    rgb_part = input_obs[:, :3, :, :]
                    goal_rgb_part = input_obs[:, 3:6, :, :]

                    # input_obs is ALREADY flattened to 4D: (B * T, C, H, W)
                    T = self.config.obs_horizon
                    # B is already defined safely earlier in the train method!
                    
                    # 1. Encode images directly without flattening again
                    obs_emb = self.nets['vision_encoder'](rgb_part).view(B, T, -1)
                    goal_emb = self.nets['vision_encoder'](goal_rgb_part).view(B, T, -1)

                    # 2. Swap to (Time, Batch, Dim) for RSSM
                    obs_emb = obs_emb.transpose(0, 1)
                    goal_emb = goal_emb.transpose(0, 1)

                    # 3. Setup initial states and dummy actions
                    dummy_actions = torch.zeros(T, B, self.network_action_dim, device=self.device)
                    init_belief = torch.zeros(B, self.config.deterministic_latent_dim, device=self.device)
                    init_state = torch.zeros(B, self.config.stochastic_latent_dim, device=self.device)
                    nonterminals = torch.ones(T, B, 1, device=self.device)

                    # 4. Unroll Transition Model
                    hidden = self.nets['transition_model'](
                        prev_state=init_state,
                        actions=dummy_actions,
                        prev_belief=init_belief,
                        goal_observations=goal_emb,
                        observations=obs_emb,
                        nonterminals=nonterminals
                    )

                    beliefs = hidden[0]          # (T, B, deter_dim)
                    posterior_states = hidden[4] # (T, B, stoch_dim)

                    # 5. Concatenate latents and swap back to (Batch, Time, Dim)
                    latents = torch.cat([beliefs, posterior_states], dim=-1)
                    obs_features = latents.transpose(0, 1) # (B, T, deter+stoch)
                
                # ==========================================================
                # --- NEW: Apply MLP Projection ---
                # ==========================================================
                if getattr(self, 'use_projector', False):
                    obs_features = self.nets['obs_projector'](obs_features)
                    
                #print(f'[diffusion] obs_features shape {obs_features.shape}, img_feature shape {image_features.shape}')

                rep_loss = torch.tensor(0.0, device=self.device)
                #state_pred_loss = torch.tensor(0.0, device=self.device) # Add initialization

                if self.rep_learn == 'auto-encoder':
                    # existing auto-encoder logic
                    reconstructed_obs = self.nets['vision_decoder'](image_features)
                    rep_loss = nn.functional.mse_loss(reconstructed_obs, input_obs)
                
                # Add this block for predict-state logic
                elif self.rep_learn == 'predict-state':
                    # Network outputs (B, obs_horizon, state_dim) OR (B, obs_horizon, state_dim + goal_state_dim)
                    pred_combined = self.nets['state_predictor'](obs_features) 
                    
                    state_key = self.config.get('state_key', 'semkey_norm_pixel')
                    gt_tensors = []
                    
                    # 1. Fetch current state ground truth
                    if state_key in nbatch:
                        gt_state = nbatch[state_key][:, :self.config.obs_horizon] 
                        gt_state = gt_state.reshape(B, self.config.obs_horizon, -1).float() 
                        gt_tensors.append(gt_state)
                    else:
                        print(f"Warning: {state_key} not found in batch for state prediction.")

                    # 2. Fetch goal state ground truth (if enabled)
                    if getattr(self, 'predict_goal_state', False):
                        goal_key = self.config.get('goal_state_key', 'goal_semkey_norm_pixel')
                        if goal_key in nbatch:
                            # Assuming goal keys have the same temporal structure or broadcast well
                            gt_goal = nbatch[goal_key][:, :self.config.obs_horizon]
                            gt_goal = gt_goal.reshape(B, self.config.obs_horizon, -1).float()
                            gt_tensors.append(gt_goal)
                        else:
                            print(f"Warning: {goal_key} not found in batch for goal prediction.")

                    # 3. Concatenate and compute unified loss
                    if gt_tensors:
                        gt_combined = torch.cat(gt_tensors, dim=-1) # Shape: (B, obs_horizon, out_dim)
                        rep_loss = nn.functional.mse_loss(pred_combined, gt_combined)

                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                if self.config.include_state:
                    vector_state = nbatch['vector_state'][:, :self.config.obs_horizon]
                    obs_features = torch.cat([obs_features, vector_state], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)

                prim_loss = torch.tensor(0)
                if self.primitive_integration == 'one-hot-encoding':
                    
                    # 1. Predict primitive ID from observation features
                    prim_logits = self.nets['prim_class_head'](obs_cond)
                    
                    # 2. Extract ground truth primitive ID from the action encoding
                    # Based on your _init_demo_policy_dataset, prim_id is encoded in action[0]
                    # We decode it back to the class index (0 to K-1)
                    prim_bin = nbatch['action'][:, 0, 0] 
                    gt_prim_ids = (((prim_bin + 1) / 2) * self.K).long()
                    gt_prim_ids = torch.clamp(gt_prim_ids, 0, self.K - 1)
                    
                    # 3. Calculate Cross Entropy Loss
                    weights_list = self.config.get('prim_class_weights', [1.0] * self.K)
                    class_weights = torch.tensor(weights_list, device=self.device, dtype=torch.float32)
                    prim_loss = nn.functional.cross_entropy(prim_logits, gt_prim_ids, weight=class_weights)

                    if self.update_step % self.log_prim_metrics_every == 0:

                        metrics = compute_classification_metrics(
                            prim_logits.detach(),
                            gt_prim_ids.detach(),
                            self.K
                        )

                        wandb_metrics = {
                            f"train/prim_{k}": v for k, v in metrics.items()
                        }

                        self.logger.log(wandb_metrics, step=self.update_step)
                        
                        import wandb
                        confusion = {"train/prim_confusion_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=gt_prim_ids.cpu().numpy(),
                            preds=torch.argmax(prim_logits, dim=-1).cpu().numpy(),
                            class_names=[p['name'] for p in self.primitives]
                        )}
                        self.logger.log(confusion, step=self.update_step)
                                    
                    # 4. Create one-hot encoding for conditioning the Diffusion net
                    # Use ground truth during training (Teacher Forcing)
                    prim_one_hot = nn.functional.one_hot(gt_prim_ids, num_classes=self.K).float()
                    
                    # 5. Concatenate to obs_cond
                    obs_cond = torch.cat([obs_cond, prim_one_hot], dim=-1)
                    nbatch['action'] = nbatch['action'][:, :, 1:]
                    # print(f'[MultiPrimitiveDiffusion, train] obs_cond shape {obs_cond.shape}')

                # ==========================================================
                # --- NEW LOGIC: Create Joint Target AFTER action is sliced
                # ==========================================================
                if self.rep_learn == 'predict-state-with-action':
                    state_key = self.config.get('state_key', 'semkey_norm_pixel')
                    gt_state = nbatch[state_key][:, :self.config.pred_horizon].float()
                    gt_state = gt_state.reshape(B, self.config.pred_horizon, -1)
                    
                    # --- NEW: Append Goal State if active ---
                    if getattr(self, 'predict_goal_state', False):
                        goal_key = self.config.get('goal_state_key', 'flattened_goal_semkey_norm_pixel')
                        if goal_key in nbatch:
                            gt_goal = nbatch[goal_key][:, :self.config.pred_horizon].float()
                            gt_goal = gt_goal.reshape(B, self.config.pred_horizon, -1)
                            diffusion_target = torch.cat([nbatch['action'], gt_state, gt_goal], dim=-1)
                        else:
                            print(f"Warning: {goal_key} not found in batch for joint diffusion target!")
                            diffusion_target = torch.cat([nbatch['action'], gt_state], dim=-1)
                    else:
                        diffusion_target = torch.cat([nbatch['action'], gt_state], dim=-1)
                else:
                    diffusion_target = nbatch['action']

                # --- APPLY DIFFUSION/OT TO JOINT TARGET ---
                loss_type = self.config.get('loss_type', 'diffusion')
                
                if loss_type == 'ot_flow_match':
                    noise = torch.randn_like(diffusion_target)
                    x_1 = diffusion_target
                    
                    t = torch.rand((B,), device=self.device, dtype=x_1.dtype)
                    t_expand = t.view(B, 1, 1)
                    noisy_actions = (1 - t_expand) * noise + t_expand * x_1
                    target = x_1 - noise
                    
                    timesteps = t * self.config.num_diffusion_iters
                    noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                else:
                    noise = torch.randn(diffusion_target.shape, device=self.device)
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps,
                        (B,), device=self.device
                    ).long()
                    
                    noisy_actions = self.noise_scheduler.add_noise(diffusion_target, noise, timesteps)
                    noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                    target = noise

                # --- MASKING LOGIC UPDATE ---
                if self.primitive_integration != 'none' and self.mask_out_irrelavent_action_dim:
                    actions = nbatch['action']
                    prim_bin = actions[:, 0, 0] 
                    prim_ids = (((prim_bin + 1) / 2) * self.K).long()
                    prim_ids = torch.clamp(prim_ids, 0, self.K - 1).cpu().detach().numpy()
                    B, T, _ = actions.shape
                    
                    # Create base action mask
                    mask = torch.zeros((B, T, self.network_action_dim), device=self.device)
                    for b in range(B):
                        mask[b] = self.primitive_action_masks[prim_ids[b]].clone().to(self.device)

                    # If predicting state with action, append 1.0s to the mask so state loss is never ignored
                    if self.rep_learn == 'predict-state-with-action':
                        # --- NEW: Expand mask size for Goal State ---
                        mask_extra_dim = self.config.state_dim
                        if getattr(self, 'predict_goal_state', False):
                            mask_extra_dim += getattr(self, 'goal_state_dim', 30)
                            
                        state_mask = torch.ones((B, T, mask_extra_dim), device=self.device)
                        mask = torch.cat([mask, state_mask], dim=-1)

                    diff = (noise_pred - target) * mask
                    valid_count = mask.sum().clamp(min=1.0)
                    actor_noise_loss = (diff ** 2).sum() / valid_count
                        
                else:
                    actor_noise_loss = nn.functional.mse_loss(noise_pred, target)

                # Fetch the weight from config, default to 1.0 if not set
                prim_weight = self.config.get('prim_weight', 1.0)
                
                # Apply the weight to the primitive loss
                total_loss = actor_noise_loss + (prim_loss * prim_weight) #co-update the encoder
                
                if self.rep_learn in ['auto-encoder', 'predict-state']:
                    total_loss += rep_loss * self.config.get('rep_weight', 0.1)
                    
            # optimize
            total_loss.backward()
            
            
            if self.clip_norm > 0:
                nn.utils.clip_grad_norm_(self.nets.parameters(), self.clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            self.lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            trainable_params = [p for p in self.nets.parameters() if p.requires_grad]
            self.ema.step(trainable_params)

            ## write loss value to tqdm progress bar
            # --- UNIFIED LOGGING DICTIONARY ---
            metrics_to_log = {
                'train/actor_noise_loss': actor_noise_loss.item(),
                'train/total_loss': total_loss.item()
            }
            
            # Log State Prediction / Auto-Encoder Loss
            if self.rep_learn in ['auto-encoder', 'predict-state']:
                # rep_loss is calculated earlier in the loop
                metrics_to_log['train/state_pred_loss'] = rep_loss.item()
                
            # Log Primitive Classification Losses
            if self.primitive_integration == 'one-hot-encoding':
                metrics_to_log['train/prim_loss_raw'] = prim_loss.item()
                metrics_to_log['train/prim_loss_weighted'] = (prim_loss * prim_weight).item()
            
            # Send everything to logger at once
            self.logger.log(metrics_to_log, step=self.update_step)
            
            # ==========================================================
            # --- PERIODIC FULL DENOISING METRIC EVALUATION ---
            # ==========================================================
            log_interval = self.config.get('log_state_eval_every', 500) # Configure how often this runs
            
            if self.rep_learn == 'predict-state-with-action' and self.update_step % log_interval == 0:
                with torch.no_grad():
                    self.nets.eval() # Switch to eval mode
                    
                    # Start from pure noise
                    eval_naction = torch.randn((B, self.config.pred_horizon, self.diffusion_dim), device=self.device)
                    
                    if loss_type == 'ot_flow_match':
                        num_steps = self.config.num_diffusion_iters
                        dt = 1.0 / num_steps
                        for i in range(num_steps):
                            t_val = i / num_steps
                            # Create a batch-sized timestep tensor
                            timestep_tensor = torch.full((B,), t_val * self.config.num_diffusion_iters, device=self.device, dtype=torch.float32)
                            v_pred = self.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                            eval_naction = eval_naction + v_pred * dt
                    else:
                        self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
                        for k in self.noise_scheduler.timesteps:
                            # Create a batch-sized timestep tensor
                            timestep_tensor = torch.full((B,), k.item(), device=self.device, dtype=torch.long)
                            n_pred = self.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                            eval_naction = self.noise_scheduler.step(model_output=n_pred, timestep=k, sample=eval_naction).prev_sample

                    # Calculate and Log Metrics against ground truth
                    act_dim = self.network_action_dim
                    state_dim = self.config.state_dim
                    num_kpts = state_dim // 2
                    
                    pred_curr_state = eval_naction[..., act_dim : act_dim + state_dim].view(B, -1, num_kpts, 2)
                    gt_curr_state = diffusion_target[..., act_dim : act_dim + state_dim].view(B, -1, num_kpts, 2)
                    
                    curr_l2_dist = torch.norm(pred_curr_state - gt_curr_state, dim=-1).mean()
                    self.logger.log({'train/curr_semkey_l2': curr_l2_dist.item()}, step=self.update_step)
                    
                    if getattr(self, 'predict_goal_state', False):
                        goal_dim = getattr(self, 'goal_state_dim', 30)
                        num_goal_kpts = goal_dim // 2
                        
                        pred_goal_state = eval_naction[..., act_dim + state_dim : act_dim + state_dim + goal_dim].view(B, -1, num_goal_kpts, 2)
                        gt_goal_state = diffusion_target[..., act_dim + state_dim : act_dim + state_dim + goal_dim].view(B, -1, num_goal_kpts, 2)
                        
                        goal_l2_dist = torch.norm(pred_goal_state - gt_goal_state, dim=-1).mean()
                        self.logger.log({'train/goal_semkey_l2': goal_l2_dist.item()}, step=self.update_step)

                    self.nets.train() # Safely switch back to train mode

           
            if self.validate_training and self.update_step > 0 \
                and self.update_step % self.val_interval == 0:
                self.validate()

            self.update_step += 1

    def _build_primitive_action_masks(self):
        """
        Returns a dict:
        prim_id -> mask (action_dim,)
        """
        masks = {}
        start = None
        if self.primitive_integration == 'one-hot-encoding':
            start = 0
        elif self.primitive_integration == 'bin_as_output':
            start = 1

        for pid, prim in enumerate(self.primitives):
            mask = np.zeros(self.network_action_dim, dtype=np.float32)

            # dimension 0 is the primitive selector → always valid
            if start == 1:
                mask[0] = 1.0

            if isinstance(prim, dict):
                dim = prim['dim']
            else:
                dim = prim.dim

            # parameters start from index 1
            mask[start:start + dim] = 1.0

            masks[pid] = torch.tensor(mask)
        
        #print('masks', masks)

        return masks

    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir

    
    def save(self):
        ckpt_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 1. Clean up old checkpoints to save space (ignore 'best' checkpoint)
        for filename in os.listdir(ckpt_dir):
            if filename.startswith('net_') and filename.endswith('.pt') and 'best' not in filename:
                file_path = os.path.join(ckpt_dir, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting old checkpoint {file_path}: {e}")

        # 2. Save the current checkpoint with the step number for the load() function
        ckpt_path = os.path.join(ckpt_dir, f'net_{self.update_step}.pt')
        torch.save(self.nets.state_dict(), ckpt_path)
    
    def save_best(self):
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_path = os.path.join(ckpt_path, f'net_best.pt')
        torch.save(self.nets.state_dict(), ckpt_path)
    
    def load_checkpoint(self, checkpoint):
        #print('loading checkpoint', checkpoint)
        ckpt_path = os.path.join(self.save_dir, 'checkpoints', f'net_{checkpoint}.pt')
        #print('ckpt path', ckpt_path)
        self.nets.load_state_dict(torch.load(ckpt_path))
        print(f'Loaded checkpoint: {checkpoint}')
        self.loaded = True


    def load(self):
        
        #print('loading checkpoint')
        ## find the latest checkpoint
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        #print('ckpt path', ckpt_path)
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_files = os.listdir(ckpt_path)
        ckpt_files = [ckpt for ckpt in ckpt_files if ckpt.endswith('.pt') and ('best' not in ckpt)]
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if len(ckpt_files) == 0:
            print('[MultiPrimitiveDiffusion, load] No checkpoint found')
            return 0
        ckpt_file = ckpt_files[-1]
        ckpt_path = os.path.join(ckpt_path, ckpt_file)
        self.nets.load_state_dict(torch.load(ckpt_path))

        print(f'Loaded checkpoint: {ckpt_file}')
        self.loaded = True
        self.update_step = int(ckpt_file.split('_')[1].split('.')[0])
        return self.update_step

    def load_best(self):
        # Construct the full path in one step
        ckpt_path = os.path.join(self.save_dir, 'checkpoints', 'net_best.pt')
        
        # Check if the file exists before trying to load
        if not os.path.exists(ckpt_path):
            print(f"[MultiPrimitiveDiffusion, load_best] Checkpoint not found at: {ckpt_path}")
            return 0 # Return None or a generic error code (like -1) to indicate failure
        
        # Load the state dictionary
        self.nets.load_state_dict(torch.load(ckpt_path))

        self.loaded = True
        print(f"[MultiPrimitiveDiffusion, load_best] Best checkpoint is loaded")
        return -2

    def single_act(self, info, update=False):
        start_time = time.time()
        if self.measure_time:
            arena_id = info['arena_id']

        if update == True:
            last_action = self.last_actions[info['arena_id']]
            
            if last_action is not None:
                self.update([info], [last_action])
            else:
                self.init([info])

        if len(self.buffer_actions[info['arena_id']]) == 0:
            image = torch.stack([x[self.config.input_obs] \
                                    for x in self.obs_deque[info['arena_id']]])
            sample_state = {'image': image}
            
            if self.config.use_mask:
                mask = torch.stack([x['mask'] for x in self.obs_deque[info['arena_id']]])
                sample_state['mask'] = mask

            if self.debug and self.config.input_obs == 'rgb-workspace-mask-goal':
                from .draw_utils import plot_rgb_workspace_mask_goal_features
                plot_rgb_workspace_mask_goal_features(image)

            if self.vision_encoder_type == 'original':
                obs_features = self.nets['vision_encoder'](image)

            elif self.vision_encoder_type == 'vit':
                import torchvision.transforms.functional as TF
                
                rgb_part = image[:, :3, :, :]
                goal_rgb_part = image[:, 3:6, :, :]

                rgb_resized = TF.resize(rgb_part, [224, 224], antialias=True)
                goal_resized = TF.resize(goal_rgb_part, [224, 224], antialias=True)

                obs_feature = self.nets['vision_encoder'](rgb_resized.to(self.device)) 
                goal_feature = self.nets['vision_encoder'](goal_resized.to(self.device))

                obs_features = torch.cat([obs_feature, goal_feature], dim=-1)
            
            elif self.vision_encoder_type == 'gc_rssm_encoder':
                rgb_part = image[:, :3, :, :]
                goal_rgb_part = image[:, 3:6, :, :]

                obs_feature = self.nets['vision_encoder'](rgb_part) 
                goal_feature = self.nets['vision_encoder'](goal_rgb_part)

                obs_features = torch.cat([obs_feature, goal_feature], dim=-1)

            elif self.vision_encoder_type == 'gc_rssm_dynamic':
                image_batched = image.unsqueeze(0)
                B, T = image_batched.shape[:2]

                rgb_part = image_batched[:, :, :3, :, :]
                goal_rgb_part = image_batched[:, :, 3:6, :, :]

                obs_emb = self.nets['vision_encoder'](rgb_part.flatten(end_dim=1)).view(B, T, -1).transpose(0, 1)
                goal_emb = self.nets['vision_encoder'](goal_rgb_part.flatten(end_dim=1)).view(B, T, -1).transpose(0, 1)

                dummy_actions = torch.ones(T, B, self.network_action_dim, device=self.device)
                init_belief = torch.zeros(B, self.config.deterministic_latent_dim, device=self.device)
                init_state = torch.zeros(B, self.config.stochastic_latent_dim, device=self.device)
                nonterminals = torch.ones(T, B, 1, device=self.device)

                hidden = self.nets['transition_model'](
                    prev_state=init_state,
                    actions=dummy_actions,
                    prev_belief=init_belief,
                    goal_observations=goal_emb,
                    observations=obs_emb,
                    nonterminals=nonterminals
                )

                latents = torch.cat([hidden[0], hidden[4]], dim=-1) 
                obs_features = latents.transpose(0, 1).squeeze(0)   

            if self.config.include_state:
                vector_state = torch.stack([x['vector_state'] \
                                            for x in self.obs_deque[info['arena_id']]])
                obs_features = torch.cat([obs_features, vector_state], dim=-1)

            # ==========================================================
            # --- NEW: Apply MLP Projection ---
            # ==========================================================
            if getattr(self, 'use_projector', False):
                obs_features = self.nets['obs_projector'](obs_features)
            
            # Initialize a variable to hold the probabilities for logging
            prim_probs_log = None 
            if self.primitive_integration == 'one-hot-encoding':
                prim_logits = self.nets['prim_class_head'](obs_features)
                
                # Calculate and capture probabilities
                prim_probs = torch.softmax(prim_logits, dim=-1)
                prim_probs_log = prim_probs.cpu().detach().numpy()
                
                prim_id = torch.argmax(prim_logits, dim=-1) # (1,)
                cur_prim_id = prim_id[-1].cpu().detach().item()
                
                # --- FIX: Restore the obs_cond creation for one-hot-encoding ---
                # Convert to one-hot encoding
                prim_enc = nn.functional.one_hot(prim_id, num_classes=self.K).float()
                
                # Condition is [Obs Features + One-Hot Primitive ID]
                obs_cond = torch.cat([obs_features, prim_enc], dim=-1)
                # ---------------------------------------------------------------
            else:
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                
            # --- START SAMPLING ---
            naction = self.eval_action_sampler.sample(
                state=sample_state, 
                horizon=self.config.pred_horizon, 
                action_dim=getattr(self, 'diffusion_dim', self.network_action_dim)
            ).to(self.device)
            
            if self.primitive_integration == 'one-hot-encoding':
                act_part = naction[..., :self.network_action_dim]
                state_part = naction[..., self.network_action_dim:]
                act_part = self.constrain_action(act_part, info, t=-1, debug=self.debug)
                naction = torch.cat([act_part, state_part], dim=-1)

            start = self.config.obs_horizon - 1
            end = start + self.config.action_horizon
            
            loss_type = self.config.get('loss_type', 'diffusion')
            
            if loss_type == 'ot_flow_match':
                num_steps = self.config.num_diffusion_iters
                dt = 1.0 / num_steps
                noise_actions = [ts_to_np(naction[:, start:end, :self.network_action_dim])]
                
                for i in range(num_steps):
                    t_val = i / num_steps
                    timestep_tensor = torch.tensor([t_val * self.config.num_diffusion_iters], device=self.device, dtype=torch.float32)
                    
                    v_pred = self.nets['noise_pred_net'](sample=naction, timestep=timestep_tensor, global_cond=obs_cond)
                    naction = naction + v_pred * dt
                    
                    if self.primitive_integration == 'one-hot-encoding':
                        act_part = naction[..., :self.network_action_dim]
                        state_part = naction[..., self.network_action_dim:]
                        act_part = self.constrain_action(act_part, info, t=i, debug=self.debug)
                        naction = torch.cat([act_part, state_part], dim=-1)
                        
                    noise_actions.append(ts_to_np(naction[:, start:end, :self.network_action_dim]))
                    
            else:
                self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
                noise_actions = [ts_to_np(naction[:, start:end, :self.network_action_dim])]
                for k in self.noise_scheduler.timesteps:
                    noise_pred = self.nets['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)
                    naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
                    
                    if self.primitive_integration == 'one-hot-encoding':
                        act_part = naction[..., :self.network_action_dim]
                        state_part = naction[..., self.network_action_dim:]
                        act_part = self.constrain_action(act_part, info, t=k, debug=self.debug)
                        naction = torch.cat([act_part, state_part], dim=-1)

                    noise_actions.append(ts_to_np(naction[:, start:end, :self.network_action_dim]))

            # Extract the final action portion from the joint tensor for execution
            final_action = naction[..., :self.network_action_dim]
            action_pred = self.data_augmenter.postprocess({'action': ts_to_np(final_action)})['action'][0]
                    
            if self.debug and self.primitive_integration == 'one-hot-encoding' and self.constrain_action == 'bimanual_mask':
                print('!!!!!! Constrain Debug!!!!!')
                from .constrain_action_functions import save_denoising_gif
                
                last_obs = self.obs_deque[info['arena_id']][-1]
                img_vis = last_obs[self.config.input_obs]
                
                if img_vis.shape[0] in [3, 4, 5, 6, 8]: 
                    img_vis = img_vis[:3] 
                
                masks = [None, None]
                if 'robot0_mask' in last_obs: masks[0] = last_obs['robot0_mask']
                if 'robot1_mask' in last_obs: masks[1] = last_obs['robot1_mask']
                
                step_idx = info.get('step', 0)
                
                save_denoising_gif(
                    image=img_vis, 
                    masks=masks, 
                    noise_actions_history=noise_actions, 
                    step_idx=step_idx
                )

            final_naction = naction[..., :self.network_action_dim]
            action_pred = self.data_augmenter.postprocess(
                {'action': ts_to_np(final_naction)})['action'][0]
            
            self.buffer_actions[info['arena_id']] = deque(
                action_pred[start:end,:], 
                maxlen=self.config.action_horizon)

        action = self.buffer_actions[info['arena_id']].popleft()
        action = action[:self.network_action_dim].reshape(self.network_action_dim)
        action = action.flatten()

        if self.config.primitive_integration == 'none':
            out_action = action
        elif self.config.primitive_integration == 'bin_as_output':
            prim_idx = int(((action[0] + 1)/2)*self.K - 1e-6)
            prim_name = self.primitives[prim_idx]['name'] if isinstance(self.primitives[prim_idx], dict) else self.primitives[prim_idx].name
            action = action[1:]
            out_action = {prim_name: action}
        elif self.primitive_integration == 'one-hot-encoding':
            prim_name = self.primitives[cur_prim_id]['name'] if isinstance(self.primitives[cur_prim_id], dict) else self.primitives[cur_prim_id].name
            out_action = {prim_name: action[:self.action_dims[cur_prim_id]]}
        else:
            raise NotImplementedError

        self.last_actions[info['arena_id']] = action

        # =======================================================================
        # --- NEW: Extract Keypoints and Update Internal States for Logging ---
        # =======================================================================
        
        # 1. Extract Predicted Keypoints based on representation learning route
        pred_keypoints = None
        
        # Route A: Joint prediction (state is appended to action tensor)
        if naction.shape[-1] > self.network_action_dim:
            pred_keypoints_ts = naction[..., self.network_action_dim:]
            pred_keypoints = ts_to_np(pred_keypoints_ts)
            
        # Route B: Dedicated state predictor network
        elif getattr(self, 'rep_learn', None) == 'predict-state' and 'state_predictor' in self.nets:
            with torch.no_grad():
                pred_ts = self.nets['state_predictor'](obs_features)
                pred_keypoints = ts_to_np(pred_ts)

        # 2. Extract Ground Truth Keypoints (safe fallback to None if not in observation)
        gt_keypoints = info['observation'].get('semkey_norm_pixel', 
                       info['observation'].get('vector_state', None))

        # 3. Update internal_states for the logger
        # Note: prim_probs_log naturally defaults to None if not populated earlier
        self.internal_states[info['arena_id']].update({
            'noise_actions_history': noise_actions,
            'primitive_probabilities': prim_probs_log,
            'predicted_keypoints': pred_keypoints,
            'gt_keypoints': gt_keypoints
        })
        # =======================================================================

        if self.measure_time:
            # We use a standard list notation to append correctly
            if 'inference_time' not in self.internal_states[info['arena_id']]:
                self.internal_states[info['arena_id']]['inference_time'] = []
            self.internal_states[info['arena_id']]['inference_time'].append(time.time() - start_time)
        
        duration = time.time() - start_time
        print(f"Arena {info.get('arena_id', 'Unknown')}: Action planned in {duration:.4f} seconds.")
        
        return out_action

    def act(self, infos, updates):
        
        ret_actions = []

        for info, upd in zip(infos, updates):
            
            #if upd:
            ret_action = self.single_act(info, upd)
            
            
            ret_actions.append(ret_action)
        
        #print('ret actions', ret_actions)
        return ret_actions
    
    def reset(self, arena_ids):
        
        if not self.loaded:
            self.load()
            
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}
            self.buffer_actions[arena_id] = deque(maxlen=self.config.action_horizon)
            self.last_actions[arena_id] = None


    def get_state(self):
        return self.internal_states

    def _process_info(self, info):
        #print('[Diffions, _process info]', info['observation'].keys())
        if 'depth' in info['observation'].keys():
            depth = info['observation']['depth'] #get the view from first camera.

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
                info['observation']['depth'] = depth

        if self.config.input_obs == 'rgbd':
            info['observation']['rgbd'] = np.concatenate(
                [info['observation']['rgb'][0].astype(np.float32), depth], axis=-1)
        
        if self.config.input_obs == 'rgb+goal_rgb':
            info['observation']['rgb+goal_rgb'] = np.concatenate(
                [info['observation']['rgb'].astype(np.float32), info['observation']['goal_rgb'].astype(np.float32)], axis=-1)

        def resize_mask_to_rgb(mask):
                H, W = rgb.shape[:2]

                # Ensure numpy array (already true, but safe)
                mask = np.asarray(mask)

                # Remove channel if present
                if mask.ndim == 3:
                    mask = mask[..., 0]

                # CRITICAL: cast dtype
                if mask.dtype != np.uint8 and mask.dtype != np.float32:
                    mask = mask.astype(np.float32)

                mask = cv2.resize(
                    mask,
                    (W, H),                      # (width, height)
                    interpolation=cv2.INTER_NEAREST
                )

                return mask[..., None]           # (H, W, 1)
        
        if self.config.input_obs == 'rgb-workspace-mask':
            rgb = info['observation']['rgb'].astype(np.float32)

            m0 = resize_mask_to_rgb(info['observation']['robot0_mask'])
            m1 = resize_mask_to_rgb(info['observation']['robot1_mask'])

            info['observation']['rgb-workspace-mask'] = np.concatenate(
                [rgb, m0, m1], axis=-1
            )
            #print('rgbd shape', info['observation']['rgbd'].shape)
        if self.config.input_obs == 'rgb-workspace-mask-goal':
            
            rgb = info['observation']['rgb'].astype(np.float32)
            goal = info['observation']['goal_rgb'].astype(np.float32)


            m0 = resize_mask_to_rgb(info['observation']['robot0_mask'])
            m1 = resize_mask_to_rgb(info['observation']['robot1_mask'])

            info['observation']['rgb-workspace-mask-goal'] = np.concatenate(
                [rgb, m0, m1, goal], axis=-1
            )
        
        if self.config.input_obs == 'rgb+goal_mask':
            rgb = info['observation']['rgb'].astype(np.float32)
            mask = resize_mask_to_rgb(info['observation']['mask'])

          

            info['observation']['rgb+goal_mask'] = np.concatenate(
                [rgb, mask], axis=-1
            )
            if self.debug: print('input shape', info['observation']['rgb+goal_mask'].shape)

        input_data = {
            self.config.input_obs: info['observation'][self.config.input_obs]\
                .reshape(1, 1, *info['observation'][self.config.input_obs].shape),
            
        }
        if 'use_mask' in self.config and self.config.use_mask:
            input_data['mask'] = info['observation']['mask']\
                .reshape(1, 1, *info['observation']['mask'].shape, 1)
            
        if self.config.include_state:
            input_data['vector_state'] = info['observation']['vector_state']\
                .reshape(1, 1, *info['observation']['vector_state'].shape)
        

        input_data = self.data_augmenter(input_data, train=False, device=self.device) 
                                    #sim2real=info['sim2real'] if 'sim2real' in info else False)
        
        vis = input_data[self.config.input_obs].squeeze(0).squeeze(0)

        obs = {
            self.config.input_obs: vis,  
        }


        if 'use_mask' in self.config and self.config.use_mask:
            mask = input_data['mask'].squeeze(0).squeeze(0)
            obs['mask'] = mask

        if self.config.include_state:
            vector_state = input_data['vector_state'].squeeze(0).squeeze(0)
            obs['vector_state'] = vector_state

        input_obs = self.data_augmenter.postprocess(obs)[self.config.input_obs]
        # print('self.internal_states', self.internal_states)
        # print('info[arena_id]', info['arena_id'])
        self.internal_states[info['arena_id']].update(
            {'input_obs': input_obs.transpose(1,2,0),
             'input_type': self.config.input_obs}
        )
        
        return obs

    def init(self, infos):
        #print('info keys', info.keys())
        for info in infos:
            obs = self._process_info(info)
            # for k, v in obs.items():
            #     print('k', k)
            #     print('v shape', v.shape)
            self.obs_deque[info['arena_id']] = deque([obs]*self.config.obs_horizon, 
                                maxlen=self.config.obs_horizon)

    def update(self, infos, actions):
        for info, action in zip(infos, actions):
            obs = self._process_info(info)
            self.obs_deque[info['arena_id']].append(obs)
    
    def set_eval(self):
        pass
    
    def set_train(self):
        pass