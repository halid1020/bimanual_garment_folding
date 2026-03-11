# The code is adopted from https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=VrX4VTl5pYNq

import os
from tqdm import tqdm
import torch
import numpy as np
from collections import deque
import torch
import cv2
from dotmap import DotMap
import torch.nn as nn
import time
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dotmap import DotMap

from actoris_harena import TrainableAgent
from actoris_harena.utilities.networks.utils import np_to_ts, ts_to_np
from actoris_harena.utilities.visual_utils import save_numpy_as_gif, save_video
from actoris_harena.utilities.save_utils import save_mask

from data_augmentation.register_augmeters import build_data_augmenter

from .utils \
    import get_resnet, replace_bn_with_gn, compute_classification_metrics
from .networks import ConditionalUnet1D, MLPClassifier, ResNetDecoder
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
            dataset = DiffusionDataset(
                dataset_path=self.config.dataset_path,
                pred_horizon=self.config.pred_horizon,
                obs_horizon=self.config.obs_horizon,
                action_horizon=self.config.action_horizon
            )
            self.stats = dataset.stats
        elif self.config.dataset_mode == 'general':
            from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            # convert dotmap to dict
            config = self.config.dataset_config.toDict()
            #print('config', config)
            dataset = TrajectoryDataset(**config)
            

            
        else:
            raise ValueError('Invalid dataset mode')

       
       
        torch.backends.cudnn.benchmark = True
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size, 
            shuffle=True,
        )
        self.dataset_inited = True
        #self.dataloader = None
    
    def _init_demo_policy_dataset(self, arenas):
        
        
        arena = arenas[0] # assume only one arena
        org_horizon = arena.action_horizon
        arena.action_horizon = self.config.get('demo_horizon', org_horizon)
        from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            # convert dotmap to dict
        config = self.config.dataset_config #.toDict()
        config['io_mode'] = 'a'
        #print('config', config)
        dataset = TrajectoryDataset(**config)

        import actoris_harena as ag_ar
        policy = ag_ar.build_agent(
            self.config.demo_policy, 
            self.config.get('demo_policy_config', DotMap({})),
            disable_wandb=True)

        qbar = tqdm(total=self.config.num_demos, 
                    desc='Collecting data from policy ...')

        qbar.update(dataset.num_trajectories())
        qbar.refresh()
        
        episode_id = dataset.num_trajectories()
        train_configs = arena.get_train_configs()
        while dataset.num_trajectories() < self.config.num_demos:
            observations = {obs_type: [] for obs_type in dataset.obs_types}
            actions = {act_type: [] for act_type in dataset.action_types}

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
                            v_ = cv2.resize(v, (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
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

                            v_ = cv2.resize(v.astype(np.float32), (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
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
                        v_ = cv2.resize(v, (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
                        observations[k].append(v_)
                    elif k in ['mask', 'goal_mask']:
                        v_ = cv2.resize(v.astype(np.float32), (dataset.obs_config[k]['shape'][0], dataset.obs_config[k]['shape'][1]))
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
                    dataset.add_trajectory(observations, actions)
                    qbar.update(1)
                
            episode_id += 1
            print('[multi-primitive-diffusion] arena.get_num_episodes', arena.get_num_episodes() )
            episode_id %= arena.get_num_episodes()

        arena.action_horizon = org_horizon
        torch.backends.cudnn.benchmark = True
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size, #64,
            #num_workers=2,
            shuffle=True,
            # accelerate cpu-gpu transfer
            #pin_memory=True,
            # don't kill worker process afte each epoch
            #persistent_workers=True
        )
        self.dataset_inited = True

    # def _init_optimizer(self):
    #     self.ema = EMAModel(
    #         parameters=self.nets.parameters(),
    #         power=0.75)
        
    #     self.optimizer = torch.optim.AdamW(
    #         params=self.nets.parameters(),
    #         lr=1e-4, weight_decay=1e-6)#

    #     self.lr_scheduler = get_scheduler(
    #         name='cosine',
    #         optimizer=self.optimizer,
    #         num_warmup_steps=500,
    #         num_training_steps=self.config.total_update_steps ## make it manual
    #     )
    
    def _init_optimizer(self):
        # Filter parameters that require gradients
        trainable_params = [p for p in self.nets.parameters() if p.requires_grad]

        self.ema = EMAModel(
            parameters=trainable_params,
            power=0.75)
        
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
        
        #self.obs_feature_dim = self.config.obs_dim * self.config.obs_horizon
        if self.primitive_integration == 'one-hot-encoding':
            self.prim_class_head = nn.Linear(self.config.obs_dim, self.K)

            cls_cfg = self.config.get("primitive_classifier", {}) # nn.Linear(self.config.obs_dim, self.K) by default

            self.prim_class_head = MLPClassifier(
                input_dim=self.config.obs_dim,
                output_dim=self.K,
                hidden_dims=cls_cfg.get("hidden_dims", []),
                activation=cls_cfg.get("activation", "relu"),
                dropout=cls_cfg.get("dropout", 0.0),
                use_layernorm=cls_cfg.get("use_layernorm", False),
            )

            # Increase global_cond_dim to accommodate the one-hot vector
            global_cond_dim = (self.config.obs_dim + self.K) * self.config.obs_horizon
            self.log_prim_metrics_every = self.config.get('log_prim_metrics_every', 200)
        else:
            global_cond_dim = self.config.obs_dim * self.config.obs_horizon

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.network_action_dim,
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
            # Create a simple MLP to project the visual embeddings to the state space
            self.nets['state_predictor'] = nn.Sequential(
                nn.Linear(self.config.obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.config.state_dim)
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
    
            if self.config.include_state:
                vector_state = torch.zeros(
                    (1, self.config.obs_horizon, 
                    self.config.state_dim))
                obs = torch.cat([obs, vector_state],dim=-1)
            
            if self.rep_learn == 'predict-state':
                _ = self.nets['state_predictor'](obs)

            # print('[MultiPrimitiveDiffusion, _test_network] obs', obs.shape)
            
            noised_action = torch.randn(
                (1, self.config.pred_horizon, self.network_action_dim))
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
          
            # encoder vision features
            #print('[diffusion] input obs shape', input_obs.shape)
            # image_features = self.nets['vision_encoder'](
            #     input_obs)
            # obs_features = image_features.reshape(
            #     B, self.config.obs_horizon, -1)

            if self.vision_encoder_type == 'original':
                image_features = self.nets['vision_encoder'](input_obs)
                obs_features = image_features.reshape(
                    B, self.config.obs_horizon, -1)
                
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
            
            #print(f'[diffusion] obs_features shape {obs_features.shape}, img_feature shape {image_features.shape}')

            rep_loss = torch.tensor(0.0, device=self.device)
            #state_pred_loss = torch.tensor(0.0, device=self.device) # Add initialization

            if self.rep_learn == 'auto-encoder':
                # existing auto-encoder logic
                reconstructed_obs = self.nets['vision_decoder'](image_features)
                rep_loss = nn.functional.mse_loss(reconstructed_obs, input_obs)
            
            # Add this block for predict-state logic
            elif self.rep_learn == 'predict-state':
                # obs_features shape is (B, obs_horizon, feature_dim)
                # Network outputs (B, obs_horizon, state_dim)
                pred_state = self.nets['state_predictor'](obs_features) 
                
                state_key = self.config.get('state_key', 'semkey_norm_pixel')
                if state_key in nbatch:
                    # Slice to the observation horizon
                    gt_state = nbatch[state_key][:, :self.config.obs_horizon] 
                    
                    # Flatten the state. Example: [15, 2] -> 30
                    gt_state = gt_state.reshape(B, self.config.obs_horizon, -1).float() 
                    
                    rep_loss = nn.functional.mse_loss(pred_state, gt_state)
                else:
                    print(f"Warning: {state_key} not found in batch for state prediction.")

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
                prim_loss = nn.functional.cross_entropy(prim_logits, gt_prim_ids)

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


            # sample noise to add to actions
            noise = torch.randn(nbatch['action'].shape, device=self.device)
            #print('noise shape', noise.shape)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            #print('action before adding noise',  nbatch['action'].shape)
            noisy_actions = self.noise_scheduler.add_noise(
                nbatch['action'], noise, timesteps)

            #print('noisy actino shape', noisy_actions.shape)

            # predict the noise residual
            noise_pred = self.noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)

            if self.primitive_integration != 'none' and self.mask_out_irrelavent_action_dim:
                # nbatch['action']: (B, T, action_dim)
                actions = nbatch['action'] # bin encoded prim acctions.

                # primitive bin is in action[..., 0] ∈ [-1, 1]
                prim_bin = actions[:, 0, 0]  # (B,)

                # same decoding logic you use in inference
                prim_ids = (((prim_bin + 1) / 2) * self.K).long()
                prim_ids = torch.clamp(prim_ids, 0, self.K - 1).cpu().detach().numpy()
                B, T, D = actions.shape
                device = actions.device

                mask = torch.zeros((B, T, D), device=device)

                for b in range(B):
                    mask[b] = self.primitive_action_masks[prim_ids[b]].clone().to(device)

                # apply mask
                diff = (noise_pred - noise) * mask

                # normalize by number of valid elements
                valid_count = mask.sum().clamp(min=1.0)

                actor_noise_loss = (diff ** 2).sum() / valid_count
                    
            else:
                # L2 loss
                actor_noise_loss = nn.functional.mse_loss(noise_pred, noise)

            total_loss = actor_noise_loss + prim_loss #co-update the encoder
            if self.rep_learn in ['auto-encoder', 'predict-state']:
                #print('loss!')
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
            pbar.set_description(f"Training (loss: {actor_noise_loss.item():.4f})")
            if self.rep_learn == 'auto-encoder':
                self.logger.log({'train/recon_loss': recon_loss.item()}, step=self.update_step)
            self.logger.log({'train/actor_noise_loss': actor_noise_loss.item()}, step=self.update_step)
            self.logger.log({'train/total_loss': total_loss.item()}, step=self.update_step)
            if self.primitive_integration == 'one-hot-encoding':
                self.logger.log({'train/prim_loss': prim_loss.item()}, step=self.update_step)
            
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
        
        ## save to the path self.save_dir/'checkpoints'/net_{update_step}.pt
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_path = os.path.join(ckpt_path, f'net_{self.update_step}.pt')
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
                #print('[Diffusion] info', info.keys())
                self.init([info])

        if len(self.buffer_actions[info['arena_id']]) == 0:
            image = torch.stack([x[self.config.input_obs] \
                                    for x in self.obs_deque[info['arena_id']]])
            sample_state = {'image': image}
            # from matplotlib import pyplot as plt
            # plt.imsave('tmp/input_obs.png', image[-1, 0].cpu().numpy())
            if self.config.use_mask:
                mask = torch.stack([x['mask'] for x in self.obs_deque[info['arena_id']]])
                sample_state['mask'] = mask

            if self.debug and self.config.input_obs == 'rgb-workspace-mask-goal':
                from .draw_utils import plot_rgb_workspace_mask_goal_features
                plot_rgb_workspace_mask_goal_features(image)

            if self.vision_encoder_type == 'original':
                obs_features = self.nets['vision_encoder'](image)
            
            elif self.vision_encoder_type == 'gc_rssm_encoder':
                # Slicing the 6-channel image into two 3-channel images (obs and goal)
                # image shape is expected to be (obs_horizon, 6, H, W)
                rgb_part = image[:, :3, :, :]
                goal_rgb_part = image[:, 3:6, :, :]

                obs_feature = self.nets['vision_encoder'](rgb_part) 
                goal_feature = self.nets['vision_encoder'](goal_rgb_part)

                # Concatenate the two feature vectors along the last dimension
                obs_features = torch.cat([obs_feature, goal_feature], dim=-1)

            elif self.vision_encoder_type == 'gc_rssm_dynamic':
                # Add a batch dimension -> (1, T, C, H, W)
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

                latents = torch.cat([hidden[0], hidden[4]], dim=-1) # Cat belief and posterior
                obs_features = latents.transpose(0, 1).squeeze(0)   # Remove batch dim -> (T, deter+stoch)

            # print('obs features shape', obs_features.shape)

            if self.config.include_state:
                vector_state = torch.stack([x['vector_state'] \
                                            for x in self.obs_deque[info['arena_id']]])
                # print('vector state shape', vector_state.shape)
                
                obs_features = torch.cat([obs_features, vector_state], dim=-1)
            
            if self.primitive_integration == 'one-hot-encoding':
                # Predict primitive ID from current observation features
                prim_logits = self.nets['prim_class_head'](obs_features)
                prim_id = torch.argmax(prim_logits, dim=-1) # (1,)
                cur_prim_id = prim_id[-1].cpu().detach().item()
                # Convert to one-hot encoding
                prim_enc = nn.functional.one_hot(prim_id, num_classes=self.K).float()
                
                # Condition is [Obs Features + One-Hot Primitive ID]
                obs_cond = torch.cat([obs_features, prim_enc], dim=-1)
            else:
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
            

            naction = self.eval_action_sampler.sample(
                state=sample_state, 
                horizon=self.config.pred_horizon, 
                action_dim=self.network_action_dim
            ).to(self.device)
            
            if self.primitive_integration == 'one-hot-encoding':
                naction = self.constrain_action(naction, info, t=-1, debug=self.debug)

            start = self.config.obs_horizon - 1
            end = start + self.config.action_horizon
            
            #torch.randn((1, self.config.pred_horizon, self.config.action_dim)).to(self.device)

            self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
            noise_actions = [ts_to_np(naction[:, start:end])]
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
                
                if self.primitive_integration == 'one-hot-encoding':
                    naction = self.constrain_action(naction, info, t=k, debug=self.debug)

                noise_actions.append(ts_to_np(naction[:, start:end]))
            
            # ---  Call Debug GIF Function ---
            if self.debug and self.primitive_integration == 'one-hot-encoding' and self.    constrain_action == 'bimanual_mask':
                print('!!!!!! Constrain Debug!!!!!')
                from .constrain_action_functions import save_denoising_gif
                
                # Extract image and masks from the deque/buffer
                # Taking the last observation (current state)
                last_obs = self.obs_deque[info['arena_id']][-1]
                
                # Get Image (Assuming 'rgb' is available and primary)
                # Adjust key if you use 'rgb-workspace-mask' etc.
                # Assuming input_obs usually stores the visual tensor
                img_vis = last_obs[self.config.input_obs]
                
                # If input_obs has channels first/stacked, take just the RGB part
                if img_vis.shape[0] in [3, 4, 5, 6, 8]: 
                    img_vis = img_vis[:3] # Take first 3 channels (RGB)
                
                masks = [None, None]
                if 'robot0_mask' in last_obs: masks[0] = last_obs['robot0_mask']
                if 'robot1_mask' in last_obs: masks[1] = last_obs['robot1_mask']
                
                step_idx = info.get('step', 0) # Or self.internal_states step count
                
                save_denoising_gif(
                    image=img_vis, 
                    masks=masks, 
                    noise_actions_history=noise_actions, 
                    step_idx=step_idx
                )

            action_pred = self.data_augmenter.postprocess(
                {'action': ts_to_np(naction)})['action'][0]
            
            self.buffer_actions[info['arena_id']] = deque(
                action_pred[start:end,:], 
                maxlen=self.config.action_horizon)

        action = self.buffer_actions[info['arena_id']]\
            .popleft().reshape(self.network_action_dim)
       
        action = action.flatten()

        ## recursively goes down the dictionary tree, when encounter list of integer number
        ## replace list with corresponding indexed values in `action`

        if self.config.primitive_integration == 'none':
            out_action = action
        elif self.config.primitive_integration == 'bin_as_output':
            prim_idx = int(((action[0] + 1)/2)*self.K - 1e-6)
            prim_name = self.primitives[prim_idx]['name'] if isinstance(self.primitives[prim_idx], dict) else self.primitives[prim_idx].name
            action = action[1:]
            out_action = {prim_name: action}
        elif self.primitive_integration == 'one-hot-encoding':
            # Use the ID predicted during the observation encoding step
            prim_name = self.primitives[cur_prim_id]['name'] if isinstance(self.primitives[cur_prim_id], dict) else self.primitives[cur_prim_id].name
            out_action = {prim_name: action[:self.action_dims[cur_prim_id]]}
        else:
            raise NotImplementedError

        self.last_actions[info['arena_id']] = action

        if self.measure_time:
            self.internal_states[[info['arena_id']]]['inference_time'].append(time.time() - start_time)
        
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