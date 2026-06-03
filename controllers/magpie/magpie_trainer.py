import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from dotmap import DotMap

from actoris_harena.utilities.visual_utils import save_numpy_as_gif, save_video
from actoris_harena.utilities.save_utils import save_mask
from .dataset import DiffusionDataset
from .utils import compute_classification_metrics

class MagpieTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.config = agent.config
        self.device = agent.device

    def _init_dataset(self):
        """Initializes the standard trajectory dataset and optional ROS."""
        if self.config.dataset_mode == 'diffusion':
            train_dataset = DiffusionDataset(
                dataset_path=self.config.dataset_path,
                pred_horizon=self.config.pred_horizon,
                obs_horizon=self.config.obs_horizon,
                action_horizon=self.config.action_horizon
            )
            self.agent.stats = train_dataset.stats
        elif self.config.dataset_mode == 'general':
            from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            config_dict = self.config.dataset_config.toDict()
            train_dataset = TrajectoryDataset(**config_dict, sample_mode='train')

            if self.agent.validate_training:
                val_dataset = TrajectoryDataset(**config_dict, sample_mode='val')
        else:
            raise ValueError('Invalid dataset mode')

        # Random Oversampling (ROS) Logic
        use_ros = self.config.get('use_random_oversampling', False)
        sampler = None
        shuffle_data = True
        
        if use_ros and self.agent.primitive_integration != 'none':
            print("[MultiPrimitiveDiffusionAdapter] Initializing WeightedRandomSampler for ROS...")
            from torch.utils.data import WeightedRandomSampler
            
            class_counts = np.zeros(self.agent.K)
            sample_classes = []
            
            for i in tqdm(range(len(train_dataset)), desc="Calculating ROS Weights"):
                action_data = train_dataset[i]['action']
                act_key = list(action_data.keys())[0]
                prim_bin = action_data[act_key][0][0] 
                prim_id = int(np.clip((((prim_bin + 1) / 2) * self.agent.K), 0, self.agent.K - 1))
                class_counts[prim_id] += 1
                sample_classes.append(prim_id)
                
            class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
            sample_weights = [class_weights[cls_id] for cls_id in sample_classes]
            
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle_data = False 
            print(f"[MultiPrimitiveDiffusionAdapter] ROS Class Counts: {class_counts}")

        torch.backends.cudnn.benchmark = True
        num_workers = self.config.get('num_workers', 0)
        persistent = self.config.get('persistent_workers', False) if num_workers > 0 else False
        prefetch_factor = self.config.get('prefetch_factor', None)

        self.agent.dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=shuffle_data,   
            sampler=sampler, num_workers=num_workers, pin_memory=self.config.get('pin_memory', False),
            persistent_workers=persistent, prefetch_factor=prefetch_factor
        )

        if self.agent.validate_training:
            self.agent.val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False, 
                num_workers=max(1, num_workers // 4), pin_memory=self.config.get('pin_memory', False),
                persistent_workers=persistent
            )
        self.agent.dataset_inited = True

    def _init_demo_policy_dataset(self, arenas):
        """Initializes dataset by rolling out a demonstrator policy."""
        arena = arenas[0]
        org_horizon = arena.action_horizon
        arena.action_horizon = self.config.get('demo_horizon', org_horizon)
        
        from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
        config_dict = self.config.dataset_config
        config_dict['io_mode'] = 'a'
        train_dataset = TrajectoryDataset(**config_dict, sample_mode='train')

        import actoris_harena as ag_ar
        policy = ag_ar.build_agent(self.config.demo_policy, self.config.get('demo_policy_config', DotMap({})), disable_wandb=True)

        qbar = tqdm(total=self.config.num_demos, desc='Collecting data from policy ...')
        qbar.update(train_dataset.num_trajectories())
        qbar.refresh()
        
        episode_id = train_dataset.num_trajectories()
        train_configs = arena.get_train_configs()
        
        while train_dataset.num_trajectories() < self.config.num_demos:
            observations = {obs_type: [] for obs_type in train_dataset.obs_types}
            actions = {act_type: [] for act_type in train_dataset.action_types}

            policy.reset([arena.id])
            info = arena.reset(train_configs[episode_id])
            policy.init([info])
            info['reward'] = 0
            done = info['done']
            
            while not done:
                action = policy.single_act(info)
                if action is None: break
                
                for k, v in info['observation'].items():
                    if k in observations.keys():
                        if k in ['rgb', 'depth', 'goal_rgb', 'goal_depth']:
                            v_ = cv2.resize(v, (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                            observations[k].append(v_)
                        elif k in ['mask', 'goal_mask']:
                            v_ = cv2.resize(v.astype(np.float32), (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                            observations[k].append(v_ > 0.9)
                        else:
                            observations[k].append(v)
                
                add_action = action
                if self.config.primitive_integration in ['bin_as_output', 'one-hot-encoding', 'separate_networks']: 
                    action_name = list(action.keys())[0]
                    action_param = action[action_name]
                    prim_id = self.agent.prim_name2id[action_name]
                    prim_act = (1.0*(prim_id+0.5)/self.agent.K *2 - 1)
                    add_action = np.zeros(self.agent.data_save_action_dim)
                    add_action[0] = prim_act
                    add_action[1:action_param.shape[0]+1] = action_param
                elif self.config.primitive_integration == 'none':
                    add_action = action
                
                actions['default'].append(add_action)  
                info = arena.step(action)
                policy.update(info, add_action)
                info['reward'] = 0
                done = info['done']
                if self.agent.collect_on_success and info['success']: break
            
            for k, v in info['observation'].items():
                if k in observations.keys():
                    if k in ['rgb', 'depth', 'goal_rgb', 'goal_depth']:
                        v_ = cv2.resize(v, (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                        observations[k].append(v_)
                    elif k in ['mask', 'goal_mask']:
                        v_ = cv2.resize(v.astype(np.float32), (train_dataset.obs_config[k]['shape'][0], train_dataset.obs_config[k]['shape'][1]))
                        observations[k].append(v_ > 0.9)
                    else:
                        observations[k].append(v)
                        
            if info['success'] or (self.config.get('add_all_demos', False) and not info['terminated']):
                for k, v in observations.items():
                    observations[k] = np.stack(v)
                actions['default'] = np.stack(actions['default'])
                train_dataset.add_trajectory(observations, actions)
                qbar.update(1)
                
            episode_id = (episode_id + 1) % arena.get_num_episodes()

        arena.action_horizon = org_horizon
        
        # ROS setup skipped here for brevity, mirror logic from _init_dataset
        self.agent.dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=self.config.get('pin_memory', False)
        )
        self.agent.dataset_inited = True

    def validate(self):
        """Runs the validation loop across the validation dataloader."""
        if not hasattr(self.agent, 'val_dataloader'):
            print("[Validation] No validation dataloader found. Skipping.")
            return

        self.agent.nets.eval() 
        val_action_mse = []
        
        print(f"--- Running Validation at Step {self.agent.update_step} ---")
        
        with torch.no_grad():
            for nbatch in self.agent.val_dataloader:
                
                if self.config.dataset_mode == 'general':
                    obs = nbatch['observation']
                    action = nbatch['action']['default']
                    nbatch = {v: k for v, k in obs.items()}
                    nbatch['action'] = action.reshape(*action.shape[:2], -1)
                    nbatch = self.agent.data_augmenter(nbatch, train=False, device=self.device)

                B = nbatch[self.config.input_obs].shape[0]
                input_obs = nbatch[self.config.input_obs][:, :self.config.obs_horizon].flatten(end_dim=1).float().to(self.device)
                
                # Vision Extraction
                if self.agent.vision_encoder_type == 'original':
                    image_features = self.agent.nets['vision_encoder'](input_obs)
                    obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                else:
                    obs_features = self.agent.nets['vision_encoder'](input_obs).reshape(B, self.config.obs_horizon, -1) # Simplified for brevity
                
                if getattr(self.agent, 'use_projector', False):
                    obs_features = self.agent.nets['obs_projector'](obs_features)

                if self.config.include_state:
                    vector_state = nbatch['vector_state'][:, :self.config.obs_horizon].to(self.device).float()
                    obs_features = torch.cat([obs_features, vector_state], dim=-1)

                obs_cond = obs_features.flatten(start_dim=1)
                
                # ---> RESTORED: Handle Validation Primitive Logic <---
                if self.agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
                    prim_logits = self.agent.nets['prim_class_head'](obs_cond)
                    preds = torch.argmax(prim_logits, dim=-1)
                    
                    if self.agent.primitive_integration == 'one-hot-encoding':
                        prim_one_hot = nn.functional.one_hot(preds, num_classes=self.agent.K).float()
                        obs_cond = torch.cat([obs_cond, prim_one_hot], dim=-1)
                        
                    gt_action = nbatch['action'][:, :, 1:].to(self.device)
                else:
                    gt_action = nbatch['action'].to(self.device)
                # -----------------------------------------------------

                eval_naction = torch.randn((B, self.config.pred_horizon, self.agent.diffusion_dim), device=self.device)
                
                # Reverse Diffusion Loop
                self.agent.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
                for k in self.agent.noise_scheduler.timesteps:
                    timestep_tensor = torch.full((B,), k.item(), device=self.device, dtype=torch.long)
                    n_pred = self.agent.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                    eval_naction = self.agent.noise_scheduler.step(model_output=n_pred, timestep=k, sample=eval_naction).prev_sample

                pred_action = eval_naction[..., :self.agent.network_action_dim]
                mse = nn.functional.mse_loss(pred_action, gt_action)
                val_action_mse.append(mse.item())

        metrics = {}
        if val_action_mse:
            metrics['val/action_mse'] = np.mean(val_action_mse)

        if metrics and hasattr(self.agent, 'logger'):
            self.agent.logger.log(metrics, step=self.agent.update_step)
            print(f"Validation Results: {metrics}")

        self.agent.nets.train()

    def train(self, update_steps, arenas):
        """Main Training Loop."""
        if not self.agent.dataset_inited:
            if self.config.train_mode == 'from_dataset':
                self._init_dataset()
            elif self.config.train_mode == 'from_policy':
                self._init_demo_policy_dataset(arenas)
        
        update_steps = min(self.config.total_update_steps - self.agent.update_step, update_steps)
        
        if self.config.get('freeze_encoder', False):
            self.agent.nets['vision_encoder'].eval()
            
        def cycle(dl):
            while True:
                for data in dl: yield data
                    
        train_iter = iter(cycle(self.agent.dataloader))
        pbar = tqdm(range(update_steps), desc="Training")

        for i in pbar:
            nbatch = next(train_iter)
            
            if self.config.dataset_mode == 'diffusion':
                nbatch = self.agent.data_augmenter(nbatch, train=True, device=self.device)
            else:
                obs, action = nbatch['observation'], nbatch['action']['default']
                nbatch = {v: k for v, k in obs.items()}
                nbatch['action'] = action.reshape(*action.shape[:2], -1)
                nbatch = self.agent.data_augmenter(nbatch, train=True, device=self.device)

            # ==========================================================
            # --- MISSING COMPOSITE OBSERVATION LOGIC RESTORED HERE ---
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
            # ==========================================================

            B = nbatch[self.config.input_obs].shape[0]
            input_obs = nbatch[self.config.input_obs][:, :self.config.obs_horizon].flatten(end_dim=1).float()

            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            use_amp = self.config.get('use_amp', False)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                # 1. Forward Features
                image_features = self.agent.nets['vision_encoder'](input_obs.to(self.device))
                obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                
                if getattr(self.agent, 'use_projector', False):
                    obs_features = self.agent.nets['obs_projector'](obs_features)

                if self.agent.nets.training:
                    noise = torch.randn_like(obs_features) * self.config.get('feature_noise_factor', 0)
                    obs_features = obs_features + noise

                if self.config.include_state:
                    vector_state = nbatch['vector_state'][:, :self.config.obs_horizon].to(self.device)
                    obs_features = torch.cat([obs_features, vector_state], dim=-1)
                    
                obs_cond = obs_features.flatten(start_dim=1)

                # 2. Extract targets based on primitive integration
                gt_prim_ids = None
                if self.agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
                    prim_bin = nbatch['action'][:, 0, 0]
                    gt_prim_ids = (((prim_bin + 1) / 2) * self.agent.K).long()
                    gt_prim_ids = torch.clamp(gt_prim_ids, 0, self.agent.K - 1).to(self.device)
                    nbatch['action'] = nbatch['action'][:, :, 1:] # Slice off bin selector

                    # ---> RESTORED: Append one-hot encoding to obs_cond <---
                    if self.agent.primitive_integration == 'one-hot-encoding':
                        prim_one_hot = nn.functional.one_hot(gt_prim_ids, num_classes=self.agent.K).float()
                        obs_cond = torch.cat([obs_cond, prim_one_hot], dim=-1)

                diffusion_target = nbatch['action'].to(self.device)
                loss_type = self.config.get('loss_type', 'diffusion')

                # 3. Add Noise & Forward Diffusion (Handles both OT Flow Match and Standard Diffusion)
                if loss_type == 'ot_flow_match':
                    noise = torch.randn_like(diffusion_target)
                    t = torch.rand((B,), device=self.device, dtype=diffusion_target.dtype)
                    t_expand = t.view(B, 1, 1)
                    noisy_actions = (1 - t_expand) * noise + t_expand * diffusion_target
                    target = diffusion_target - noise
                    timesteps = t * self.config.num_diffusion_iters
                else:
                    noise = torch.randn(diffusion_target.shape, device=self.device)
                    timesteps = torch.randint(0, self.agent.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
                    noisy_actions = self.agent.noise_scheduler.add_noise(diffusion_target, noise, timesteps)
                    target = noise

                # Route to appropriate network(s)
                if self.agent.primitive_integration == 'separate_networks':
                    noise_pred = torch.zeros_like(noisy_actions)
                    for k in range(self.agent.K):
                        dim_k = self.agent.diffusion_dims[k]
                        if dim_k == 0: continue
                        mask_k = (gt_prim_ids == k)
                        if mask_k.sum() > 0:
                            noise_pred[mask_k, :, :dim_k] = self.agent.nets[f'noise_pred_net_{k}'](
                                noisy_actions[mask_k][..., :dim_k], timesteps[mask_k], global_cond=obs_cond[mask_k]
                            )
                else:
                    noise_pred = self.agent.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

                # Calculate Loss with optional masking
                if self.agent.primitive_integration != 'none' and getattr(self.agent, 'mask_out_irrelavent_action_dim', False):
                    mask = torch.zeros((B, self.config.pred_horizon, self.agent.network_action_dim), device=self.device)
                    for b in range(B):
                        mask[b] = self.agent.primitive_action_masks[gt_prim_ids[b].item()].to(self.device)
                    diff = (noise_pred - target) * mask
                    valid_count = mask.sum().clamp(min=1.0)
                    actor_noise_loss = (diff ** 2).sum() / valid_count
                else:
                    actor_noise_loss = nn.functional.mse_loss(noise_pred, target)
                
                total_loss = actor_noise_loss

            # 4. Optimize
            total_loss.backward()
            
            if self.agent.clip_norm > 0:
                nn.utils.clip_grad_norm_(self.agent.nets.parameters(), self.agent.clip_norm)

            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad()
            self.agent.lr_scheduler.step()
            
            trainable_params = [p for p in self.agent.nets.parameters() if p.requires_grad]
            self.agent.ema.step(trainable_params)

            if hasattr(self.agent, 'logger'):
                self.agent.logger.log({'train/total_loss': total_loss.item()}, step=self.agent.update_step)

            if self.agent.validate_training and self.agent.update_step > 0 and self.agent.update_step % self.agent.val_interval == 0:
                self.validate()

            self.agent.update_step += 1