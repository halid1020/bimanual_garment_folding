"""
Magpie Trainer Module.

This module contains the MagpieTrainer class, which orchestrates the data loading, 
validation, and training loops for the Magpie diffusion-based robotics agent. It 
supports both standard Denoising Diffusion Probabilistic Models (DDPM) and 
Optimal Transport (OT) Flow Matching, along with primitive action integration.
"""

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
    """
    Coordinates the training and validation pipelines for the MagpieAgent.

    Attributes:
        agent: The core diffusion agent containing the networks and state.
        config (DotMap): Configuration dictionary containing hyperparameters.
        device (torch.device): Hardware device (CPU/GPU) for tensor operations.
    """

    def __init__(self, agent):
        """
        Initializes the MagpieTrainer with a reference to the main agent.

        Args:
            agent: The MagpieAgent instance containing neural networks and config.
        """
        self.agent = agent
        self.config = agent.config
        self.device = agent.device

    def _init_dataset(self):
        """
        Initializes the standard trajectory dataset and prepares PyTorch DataLoaders.

        Configures either a 'diffusion' or 'general' dataset mode. If primitive 
        integration is enabled, it sets up Random Oversampling (ROS) using a 
        WeightedRandomSampler to mitigate class imbalance among primitive actions.

        Raises:
            ValueError: If an unsupported `dataset_mode` is provided.
        """
        # 1. Dataset Instantiation
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
        
        elif self.config.dataset_mode == 'hindsight':
            from .hindsight_dataset import HindsightDataset
            config_dict = self.config.dataset_config.toDict()
            train_dataset = HindsightDataset(**config_dict, sample_mode='train')

            if self.agent.validate_training:
                val_dataset = HindsightDataset(**config_dict, sample_mode='val')

        else:
            raise ValueError('Invalid dataset mode. Expected "diffusion", "hindsignt", "general".')

        # 2. Random Oversampling (ROS) Logic for Imbalanced Primitives
        use_ros = self.config.get('use_random_oversampling', False)
        sampler = None
        shuffle_data = True
        
        if use_ros and self.agent.primitive_integration != 'none':
            print("[MultiPrimitiveDiffusionAdapter] Initializing WeightedRandomSampler for ROS...")
            from torch.utils.data import WeightedRandomSampler
            
            class_counts = np.zeros(self.agent.K)
            sample_classes = []
            
            # Tally occurrences of each primitive class
            for i in tqdm(range(len(train_dataset)), desc="Calculating ROS Weights"):
                action_data = train_dataset[i]['action']
                act_key = list(action_data.keys())[0]
                prim_bin = action_data[act_key][0][0] 
                
                # Extract and normalize the primitive ID
                prim_id = int(np.clip((((prim_bin + 1) / 2) * self.agent.K), 0, self.agent.K - 1))
                class_counts[prim_id] += 1
                sample_classes.append(prim_id)
                
            # Assign higher selection weights to underrepresented classes
            class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
            sample_weights = [class_weights[cls_id] for cls_id in sample_classes]
            
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle_data = False # Must disable native shuffling when using a custom sampler
            print(f"[MultiPrimitiveDiffusionAdapter] ROS Class Counts: {class_counts}")

        # 3. DataLoader Performance Optimization
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
        """
        Initializes the dataset by rolling out a demonstrator policy in the environment.

        Instead of loading pre-collected data, this method actively collects 
        trajectories from an expert policy until the desired number of demos is reached.

        Args:
            arenas (list): A list containing the simulation environments to roll out in.
        """
        arena = arenas[0]
        org_horizon = arena.action_horizon
        arena.action_horizon = self.config.get('demo_horizon', org_horizon)
        
        # Safely convert to a standard dict to prevent mutating the global config
        config_dict = self.config.dataset_config.toDict() if hasattr(self.config.dataset_config, 'toDict') else dict(self.config.dataset_config)
        config_dict['io_mode'] = 'a' # Append mode
        
        # Route dataset instantiation based on the config mode
        if self.config.dataset_mode == 'hindsight':
            from .hindsight_dataset import HindsightDataset
            train_dataset = HindsightDataset(**config_dict, sample_mode='train')
        else:
            from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
            # Strip the HER mapping if falling back to the general base class
            config_dict.pop('future_goal_mapping', None)
            train_dataset = TrajectoryDataset(**config_dict, sample_mode='train')

        import actoris_harena as ag_ar
        policy = ag_ar.build_agent(self.config.demo_policy, self.config.get('demo_policy_config', DotMap({})), disable_wandb=True)

        qbar = tqdm(total=self.config.num_demos, desc='Collecting data from policy ...')
        qbar.update(train_dataset.num_trajectories())
        qbar.refresh()
        
        episode_id = train_dataset.num_trajectories()
        train_configs = arena.get_train_configs()
        
        # 1. Trajectory Collection Loop
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
                
                # Store and format vision/mask observations
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
                
                # Handle primitive action mapping
                add_action = action
                if self.config.primitive_integration in ['bin_as_output', 'one-hot-encoding', 'separate_networks']: 
                    action_name = list(action.keys())[0]
                    action_param = action[action_name]
                    prim_id = self.agent.prim_name2id[action_name]
                    
                    # Convert primitive ID back to normalized [-1, 1] bin format
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
            
            # Capture final terminal observation
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
                        
            # Save successful trajectories
            if info['success'] or (self.config.get('add_all_demos', False) and not info['terminated']):
                for k, v in observations.items():
                    observations[k] = np.stack(v)
                actions['default'] = np.stack(actions['default'])
                train_dataset.add_trajectory(observations, actions)
                qbar.update(1)
                
            episode_id = (episode_id + 1) % arena.get_num_episodes()

        arena.action_horizon = org_horizon
        
        # ROS setup skipped here for brevity, mirrors logic from _init_dataset
        self.agent.dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=self.config.get('pin_memory', False)
        )
        self.agent.dataset_inited = True

    def validate(self):
        """
        Executes the validation loop across the validation dataloader.

        Computes action MSE, state prediction L2 distance, and primitive 
        classification accuracy (if applicable). Logs all metrics to wandb.
        """
        if not hasattr(self.agent, 'val_dataloader'):
            print("[Validation] No validation dataloader found. Skipping.")
            return

        self.agent.nets.eval() 
        
        # Metrics Accumulators
        val_prim_losses, val_state_mse, val_state_l2, val_action_mse = [], [], [], []
        all_preds, all_gts = [], []
        
        print(f"--- Running Validation at Step {self.agent.update_step} ---")
        
        with torch.no_grad():
            for nbatch in self.agent.val_dataloader:
                
                # 1. Observation Pre-processing
                if self.config.dataset_mode == 'general':
                    obs = nbatch['observation']
                    action = nbatch['action']['default']
                    nbatch = {v: k for v, k in obs.items()}
                    nbatch['action'] = action.reshape(*action.shape[:2], -1)
                    nbatch = self.agent.data_augmenter(nbatch, train=False, device=self.device)

                # Construct composite multi-modal observations
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
                
                # 2. Extract Vision and State Features
                if self.agent.vision_encoder_type == 'original':
                    image_features = self.agent.nets['vision_encoder'](input_obs)
                    obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                else:
                    obs_features = self.agent.nets['vision_encoder'](input_obs).reshape(B, self.config.obs_horizon, -1) 
                
                if getattr(self.agent, 'use_projector', False):
                    obs_features = self.agent.nets['obs_projector'](obs_features)

                if self.config.include_state:
                    vector_state = nbatch['vector_state'][:, :self.config.obs_horizon].to(self.device).float()
                    obs_features = torch.cat([obs_features, vector_state], dim=-1)

                obs_cond = obs_features.flatten(start_dim=1)
                
                # 3. Validate State Predictor Representation Learning
                if self.agent.rep_learn == 'predict-state':
                    pred_combined = self.agent.nets['state_predictor'](obs_features)
                    state_key = self.config.get('state_key', 'semkey_norm_pixel')
                    gt_tensors_val = []
                    
                    if state_key in nbatch:
                        gt_state = nbatch[state_key][:, :self.config.obs_horizon].reshape(B, self.config.obs_horizon, -1).float().to(self.device)
                        gt_tensors_val.append(gt_state)
                        
                        state_dim = self.config.state_dim
                        num_kpts = state_dim // 2
                        pred_kpts = pred_combined[..., :state_dim].view(-1, num_kpts, 2)
                        gt_kpts = gt_state[..., :state_dim].view(-1, num_kpts, 2)
                        
                        l2_dist = torch.norm(pred_kpts - gt_kpts, dim=-1).mean()
                        val_state_l2.append(l2_dist.item())
                    
                    if getattr(self.agent, 'predict_goal_state', False):
                        goal_key = self.config.get('goal_state_key', 'flattened_goal_semkey_norm_pixel')
                        if goal_key in nbatch:
                            gt_goal = nbatch[goal_key][:, :self.config.obs_horizon].reshape(B, self.config.obs_horizon, -1).float().to(self.device)
                            gt_tensors_val.append(gt_goal)
                            
                    if gt_tensors_val:
                        gt_combined_val = torch.cat(gt_tensors_val, dim=-1)
                        mse_loss = nn.functional.mse_loss(pred_combined, gt_combined_val)
                        val_state_mse.append(mse_loss.item())

                # 4. Validate Primitive Classification Head
                gt_prim_ids = None
                if self.agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
                    prim_logits = self.agent.nets['prim_class_head'](obs_cond)
                    preds = torch.argmax(prim_logits, dim=-1)
                    
                    prim_bin = nbatch['action'][:, 0, 0].to(self.device)
                    gt_prim_ids = (((prim_bin + 1) / 2) * self.agent.K).long()
                    gt_prim_ids = torch.clamp(gt_prim_ids, 0, self.agent.K - 1)
                    
                    loss = nn.functional.cross_entropy(prim_logits, gt_prim_ids)
                    val_prim_losses.append(loss.item())
                    all_preds.extend(preds.cpu().numpy())
                    all_gts.extend(gt_prim_ids.cpu().numpy())
                    
                    if self.agent.primitive_integration == 'one-hot-encoding':
                        prim_one_hot = nn.functional.one_hot(preds, num_classes=self.agent.K).float()
                        obs_cond = torch.cat([obs_cond, prim_one_hot], dim=-1)
                        
                    gt_action = nbatch['action'][:, :, 1:].to(self.device)
                else:
                    gt_action = nbatch['action'].to(self.device)

                # 5. Full Evaluation Denoising Pass
                eval_naction = torch.randn((B, self.config.pred_horizon, self.agent.diffusion_dim), device=self.device)
                loss_type = self.config.get('loss_type', 'diffusion')
                
                # Branch: Optimal Transport Flow Matching
                if loss_type == 'ot_flow_match':
                    num_steps = self.config.num_diffusion_iters
                    dt = 1.0 / num_steps
                    for i in range(num_steps):
                        t_val = i / num_steps
                        timestep_tensor = torch.full((B,), t_val * self.config.num_diffusion_iters, device=self.device, dtype=torch.float32)
                        
                        if self.agent.primitive_integration == 'separate_networks':
                            v_pred = torch.zeros_like(eval_naction)
                            for k in range(self.agent.K):
                                dim_k = self.agent.diffusion_dims[k]
                                if dim_k == 0: continue
                                mask_k = (preds == k) 
                                if mask_k.sum() > 0:
                                    net_out = self.agent.nets[f'noise_pred_net_{k}'](
                                        sample=eval_naction[mask_k][..., :dim_k], timestep=timestep_tensor[mask_k], global_cond=obs_cond[mask_k]
                                    )
                                    v_pred[mask_k, :, :dim_k] = net_out
                        else:
                            v_pred = self.agent.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                            
                        eval_naction = eval_naction + v_pred * dt
                
                # Branch: Standard DDPM Reverse Diffusion
                else:
                    self.agent.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
                    for k in self.agent.noise_scheduler.timesteps:
                        timestep_tensor = torch.full((B,), k.item(), device=self.device, dtype=torch.long)
                        
                        if self.agent.primitive_integration == 'separate_networks':
                            n_pred = torch.zeros_like(eval_naction)
                            for net_idx in range(self.agent.K):
                                dim_k = self.agent.diffusion_dims[net_idx]
                                if dim_k == 0: continue
                                mask_k = (preds == net_idx)
                                if mask_k.sum() > 0:
                                    net_out = self.agent.nets[f'noise_pred_net_{net_idx}'](
                                        sample=eval_naction[mask_k][..., :dim_k], timestep=timestep_tensor[mask_k], global_cond=obs_cond[mask_k]
                                    )
                                    n_pred[mask_k, :, :dim_k] = net_out
                        else:
                            n_pred = self.agent.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                            
                        eval_naction = self.agent.noise_scheduler.step(model_output=n_pred, timestep=k, sample=eval_naction).prev_sample

                # 6. Action MSE Calculation
                pred_action = eval_naction[..., :self.agent.network_action_dim]
                
                # Only calculate loss on the valid action dimensions corresponding to the chosen primitive
                if self.agent.primitive_integration != 'none' and getattr(self.agent, 'mask_out_irrelavent_action_dim', False):
                    mask = torch.zeros((B, self.config.pred_horizon, self.agent.network_action_dim), device=self.device)
                    for b in range(B):
                        mask[b] = self.agent.primitive_action_masks[gt_prim_ids[b].item()].to(self.device)
                    
                    diff = (pred_action - gt_action) * mask
                    valid_count = mask.sum().clamp(min=1.0) # Avoid div by zero
                    mse = (diff ** 2).sum() / valid_count
                else:
                    mse = nn.functional.mse_loss(pred_action, gt_action)
                
                val_action_mse.append(mse.item())

        # 7. Metric Aggregation and wandb Integration
        metrics = {}
        if val_state_l2: metrics['val/state_keypoint_l2_avg'] = np.mean(val_state_l2)
        if val_state_mse: metrics['val/state_pred_loss'] = np.mean(val_state_mse)
        if val_action_mse: metrics['val/action_mse'] = np.mean(val_action_mse)

        if val_prim_losses:
            metrics['val/prim_loss'] = np.mean(val_prim_losses)
            acc = np.mean(np.array(all_preds) == np.array(all_gts))
            metrics['val/prim_accuracy'] = acc
            
            import wandb
            if wandb.run is not None and hasattr(self.agent, 'logger'):
                confusion = {"val/prim_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_gts,
                    preds=all_preds,
                    class_names=[p['name'] if isinstance(p, dict) else p.name for p in self.agent.primitives]
                )}
                self.agent.logger.log(confusion, step=self.agent.update_step)

        if metrics and hasattr(self.agent, 'logger'):
            self.agent.logger.log(metrics, step=self.agent.update_step)
            print(f"Validation Results: {metrics}")

        self.agent.nets.train()

    def train(self, update_steps, arenas):
        """
        Executes the main training optimization loop.

        Args:
            update_steps (int): Total number of gradient steps to take in this call.
            arenas (list): Simulation environments (required if training from policy).
        """
        if not self.agent.dataset_inited:
            if self.config.train_mode == 'from_dataset':
                self._init_dataset()
            elif self.config.train_mode == 'from_policy':
                self._init_demo_policy_dataset(arenas)
        
        # Ensure we don't overshoot total configured steps
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
            
            # 1. Input Batch Setup
            if self.config.dataset_mode == 'diffusion':
                nbatch = self.agent.data_augmenter(nbatch, train=True, device=self.device)
            else:
                obs, action = nbatch['observation'], nbatch['action']['default']
                nbatch = {v: k for v, k in obs.items()}
                nbatch['action'] = action.reshape(*action.shape[:2], -1)
                nbatch = self.agent.data_augmenter(nbatch, train=True, device=self.device)

            if self.config.input_obs == 'rgbd': nbatch['rgbd'] = torch.cat([nbatch['rgb'], nbatch['depth']], dim=2)
            if self.config.input_obs == 'rgb-workspace-mask': nbatch['rgb-workspace-mask'] = torch.cat([nbatch['rgb'], nbatch['robot0_mask'], nbatch['robot1_mask']], dim=2)
            if self.config.input_obs == 'rgb-workspace-mask-goal': nbatch['rgb-workspace-mask-goal'] = torch.cat([nbatch['rgb'], nbatch['robot0_mask'], nbatch['robot1_mask'], nbatch['goal_rgb']], dim=2)
            if self.config.input_obs == 'rgb+goal_rgb': nbatch['rgb+goal_rgb'] = torch.cat([nbatch['rgb'], nbatch['goal_rgb']], dim=2)
            if self.config.input_obs == 'rgb+goal_mask': nbatch['rgb+goal_mask'] = torch.cat([nbatch['rgb'], nbatch['goal_mask']], dim=2)

            B = nbatch[self.config.input_obs].shape[0]
            input_obs = nbatch[self.config.input_obs][:, :self.config.obs_horizon].flatten(end_dim=1).float()

            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            use_amp = self.config.get('use_amp', False)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                # 2. Forward Feature Extraction
                image_features = self.agent.nets['vision_encoder'](input_obs.to(self.device))
                obs_features = image_features.reshape(B, self.config.obs_horizon, -1)
                
                if getattr(self.agent, 'use_projector', False):
                    obs_features = self.agent.nets['obs_projector'](obs_features)

                # Add noise to conditioning features for robustness
                if self.agent.nets.training:
                    noise = torch.randn_like(obs_features) * self.config.get('feature_noise_factor', 0)
                    obs_features = obs_features + noise

                # 3. Representation Learning Auxiliary Losses
                rep_loss = torch.tensor(0.0, device=self.device)
                
                if self.agent.rep_learn == 'auto-encoder':
                    reconstructed_obs = self.agent.nets['vision_decoder'](image_features)
                    rep_loss = nn.functional.mse_loss(reconstructed_obs, input_obs.to(self.device))
                elif self.agent.rep_learn == 'predict-state':
                    pred_combined = self.agent.nets['state_predictor'](obs_features) 
                    state_key = self.config.get('state_key', 'semkey_norm_pixel')
                    gt_tensors = []
                    
                    if state_key in nbatch:
                        gt_state = nbatch[state_key][:, :self.config.obs_horizon].reshape(B, self.config.obs_horizon, -1).float() 
                        gt_tensors.append(gt_state.to(self.device))
                    if getattr(self.agent, 'predict_goal_state', False):
                        goal_key = self.config.get('goal_state_key', 'goal_semkey_norm_pixel')
                        if goal_key in nbatch:
                            gt_goal = nbatch[goal_key][:, :self.config.obs_horizon].reshape(B, self.config.obs_horizon, -1).float()
                            gt_tensors.append(gt_goal.to(self.device))
                    if gt_tensors:
                        gt_combined = torch.cat(gt_tensors, dim=-1) 
                        rep_loss = nn.functional.mse_loss(pred_combined, gt_combined)

                if self.config.include_state:
                    vector_state = nbatch['vector_state'][:, :self.config.obs_horizon].to(self.device)
                    obs_features = torch.cat([obs_features, vector_state], dim=-1)
                    
                obs_cond = obs_features.flatten(start_dim=1)

                # 4. Primitive Classification Loss
                gt_prim_ids = None
                prim_loss = torch.tensor(0.0, device=self.device)
                
                if self.agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
                    prim_logits = self.agent.nets['prim_class_head'](obs_cond)
                    prim_bin = nbatch['action'][:, 0, 0]
                    gt_prim_ids = (((prim_bin + 1) / 2) * self.agent.K).long()
                    gt_prim_ids = torch.clamp(gt_prim_ids, 0, self.agent.K - 1).to(self.device)
                    
                    weights_list = self.config.get('prim_class_weights', [1.0] * self.agent.K)
                    class_weights = torch.tensor(weights_list, device=self.device, dtype=torch.float32)
                    prim_loss = nn.functional.cross_entropy(prim_logits, gt_prim_ids, weight=class_weights)

                    log_every = self.config.get('log_prim_metrics_every', 200)
                    
                    if hasattr(self.agent, 'logger') and self.agent.update_step % log_every == 0:
                        metrics = compute_classification_metrics(prim_logits.detach(), gt_prim_ids.detach(), self.agent.K)
                        wandb_metrics = {f"train/prim_{k}": v for k, v in metrics.items()}
                        self.agent.logger.log(wandb_metrics, step=self.agent.update_step)
                        
                        import wandb
                        if wandb.run is not None:
                            confusion = {"train/prim_confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None, y_true=gt_prim_ids.cpu().numpy(),
                                preds=torch.argmax(prim_logits, dim=-1).cpu().numpy(),
                                class_names=[p['name'] if isinstance(p, dict) else p.name for p in self.agent.primitives]
                            )}
                            self.agent.logger.log(confusion, step=self.agent.update_step)

                    # Remove primitive bin from action target
                    nbatch['action'] = nbatch['action'][:, :, 1:] 

                    if self.agent.primitive_integration == 'one-hot-encoding':
                        prim_one_hot = nn.functional.one_hot(gt_prim_ids, num_classes=self.agent.K).float()
                        obs_cond = torch.cat([obs_cond, prim_one_hot], dim=-1)

                # 5. Build Diffusion Target
                if self.agent.rep_learn == 'predict-state-with-action':
                    state_key = self.config.get('state_key', 'semkey_norm_pixel')
                    gt_state = nbatch[state_key][:, :self.config.pred_horizon].float().reshape(B, self.config.pred_horizon, -1).to(self.device)
                    
                    if getattr(self.agent, 'predict_goal_state', False):
                        goal_key = self.config.get('goal_state_key', 'flattened_goal_semkey_norm_pixel')
                        if goal_key in nbatch:
                            gt_goal = nbatch[goal_key][:, :self.config.pred_horizon].float().reshape(B, self.config.pred_horizon, -1).to(self.device)
                            diffusion_target = torch.cat([nbatch['action'].to(self.device), gt_state, gt_goal], dim=-1)
                        else:
                            diffusion_target = torch.cat([nbatch['action'].to(self.device), gt_state], dim=-1)
                    else:
                        diffusion_target = torch.cat([nbatch['action'].to(self.device), gt_state], dim=-1)
                else:
                    diffusion_target = nbatch['action'].to(self.device)
                    
                loss_type = self.config.get('loss_type', 'diffusion')

                # 6. Forward Forward Diffusion (Adding Noise)
                if loss_type == 'ot_flow_match':
                    noise = torch.randn_like(diffusion_target)
                    t = torch.rand((B,), device=self.device, dtype=diffusion_target.dtype)
                    t_expand = t.view(B, 1, 1)
                    # Interpolate linearly between pure noise (t=0) and data (t=1)
                    noisy_actions = (1 - t_expand) * noise + t_expand * diffusion_target
                    target = diffusion_target - noise
                    timesteps = t * self.config.num_diffusion_iters
                else:
                    noise = torch.randn(diffusion_target.shape, device=self.device)
                    timesteps = torch.randint(0, self.agent.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
                    noisy_actions = self.agent.noise_scheduler.add_noise(diffusion_target, noise, timesteps)
                    target = noise # DDPM trains to predict the added noise

                # 7. Model Predictions
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

                # 8. Loss Calculations
                if self.agent.primitive_integration != 'none' and getattr(self.agent, 'mask_out_irrelavent_action_dim', False):
                    mask = torch.zeros((B, self.config.pred_horizon, self.agent.network_action_dim), device=self.device)
                    for b in range(B):
                        mask[b] = self.agent.primitive_action_masks[gt_prim_ids[b].item()].to(self.device)
                        
                    # Also unmask state target dims if doing joint diffusion
                    if self.agent.rep_learn == 'predict-state-with-action':
                        mask_extra_dim = self.config.state_dim
                        if getattr(self.agent, 'predict_goal_state', False): mask_extra_dim += getattr(self.agent, 'goal_state_dim', 30)
                        state_mask = torch.ones((B, self.config.pred_horizon, mask_extra_dim), device=self.device)
                        mask = torch.cat([mask, state_mask], dim=-1)

                    diff = (noise_pred - target) * mask
                    valid_count = mask.sum().clamp(min=1.0)
                    actor_noise_loss = (diff ** 2).sum() / valid_count
                else:
                    actor_noise_loss = nn.functional.mse_loss(noise_pred, target)
                
                prim_weight = self.config.get('prim_weight', 1.0)
                total_loss = actor_noise_loss + (prim_loss * prim_weight) 
                
                if self.agent.rep_learn in ['auto-encoder', 'predict-state']:
                    total_loss += rep_loss * self.config.get('rep_weight', 0.1)

            # 9. Optimizer and Scheduler Updates
            total_loss.backward()
            
            if self.agent.clip_norm > 0:
                nn.utils.clip_grad_norm_(self.agent.nets.parameters(), self.agent.clip_norm)

            self.agent.optimizer.step()
            self.agent.optimizer.zero_grad()
            self.agent.lr_scheduler.step()
            
            trainable_params = [p for p in self.agent.nets.parameters() if p.requires_grad]
            self.agent.ema.step(trainable_params)

            # 10. Training Logging
            if hasattr(self.agent, 'logger'):
                metrics_to_log = {
                    'train/actor_noise_loss': actor_noise_loss.item(),
                    'train/total_loss': total_loss.item()
                }
                if self.agent.rep_learn in ['auto-encoder', 'predict-state']:
                    metrics_to_log['train/state_pred_loss'] = rep_loss.item()
                if self.agent.primitive_integration == 'one-hot-encoding':
                    metrics_to_log['train/prim_loss_raw'] = prim_loss.item()
                    metrics_to_log['train/prim_loss_weighted'] = (prim_loss * prim_weight).item()
                    
                self.agent.logger.log(metrics_to_log, step=self.agent.update_step)

            # 11. Periodic Full Denoising Metric Evaluation (for Joint Diffusion)
            log_interval = self.config.get('log_state_eval_every', 500)
            if self.agent.rep_learn == 'predict-state-with-action' and self.agent.update_step % log_interval == 0:
                with torch.no_grad():
                    self.agent.nets.eval()
                    eval_naction = torch.randn((B, self.config.pred_horizon, self.agent.diffusion_dim), device=self.device)
                    
                    if loss_type == 'ot_flow_match':
                        num_steps = self.config.num_diffusion_iters
                        dt = 1.0 / num_steps
                        for i in range(num_steps):
                            t_val = i / num_steps
                            timestep_tensor = torch.full((B,), t_val * self.config.num_diffusion_iters, device=self.device, dtype=torch.float32)
                            if self.agent.primitive_integration == 'separate_networks':
                                v_pred = torch.zeros_like(eval_naction)
                                for k in range(self.agent.K):
                                    dim_k = self.agent.diffusion_dims[k]
                                    if dim_k == 0: continue
                                    mask_k = (gt_prim_ids == k)
                                    if mask_k.sum() > 0:
                                        v_pred[mask_k, :, :dim_k] = self.agent.nets[f'noise_pred_net_{k}'](eval_naction[mask_k][..., :dim_k], timestep_tensor[mask_k], global_cond=obs_cond[mask_k])
                                eval_naction = eval_naction + v_pred * dt
                            else:
                                v_pred = self.agent.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                                eval_naction = eval_naction + v_pred * dt
                    else:
                        self.agent.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
                        for step_k in self.agent.noise_scheduler.timesteps:
                            timestep_tensor = torch.full((B,), step_k.item(), device=self.device, dtype=torch.long)
                            if self.agent.primitive_integration == 'separate_networks':
                                n_pred = torch.zeros_like(eval_naction)
                                for net_idx in range(self.agent.K):
                                    dim_k = self.agent.diffusion_dims[net_idx]
                                    if dim_k == 0: continue
                                    mask_k = (gt_prim_ids == net_idx)
                                    if mask_k.sum() > 0:
                                        n_pred[mask_k, :, :dim_k] = self.agent.nets[f'noise_pred_net_{net_idx}'](eval_naction[mask_k][..., :dim_k], timestep_tensor[mask_k], global_cond=obs_cond[mask_k])
                            else:
                                n_pred = self.agent.nets['noise_pred_net'](sample=eval_naction, timestep=timestep_tensor, global_cond=obs_cond)
                            eval_naction = self.agent.noise_scheduler.step(model_output=n_pred, timestep=step_k, sample=eval_naction).prev_sample

                    act_dim = self.agent.network_action_dim
                    state_dim = self.config.state_dim
                    num_kpts = state_dim // 2
                    pred_curr_state = eval_naction[..., act_dim : act_dim + state_dim].view(B, -1, num_kpts, 2)
                    gt_curr_state = diffusion_target[..., act_dim : act_dim + state_dim].view(B, -1, num_kpts, 2)
                    
                    curr_l2_dist = torch.norm(pred_curr_state - gt_curr_state, dim=-1).mean()
                    if hasattr(self.agent, 'logger'): self.agent.logger.log({'train/curr_semkey_l2': curr_l2_dist.item()}, step=self.agent.update_step)
                    
                    if getattr(self.agent, 'predict_goal_state', False):
                        goal_dim = getattr(self.agent, 'goal_state_dim', 30)
                        num_goal_kpts = goal_dim // 2
                        pred_goal_state = eval_naction[..., act_dim + state_dim : act_dim + state_dim + goal_dim].view(B, -1, num_goal_kpts, 2)
                        gt_goal_state = diffusion_target[..., act_dim + state_dim : act_dim + state_dim + goal_dim].view(B, -1, num_goal_kpts, 2)
                        goal_l2_dist = torch.norm(pred_goal_state - gt_goal_state, dim=-1).mean()
                        if hasattr(self.agent, 'logger'): self.agent.logger.log({'train/goal_semkey_l2': goal_l2_dist.item()}, step=self.agent.update_step)

                    self.agent.nets.train()

            if self.agent.validate_training and self.agent.update_step > 0 and self.agent.update_step % self.agent.val_interval == 0:
                self.validate()

            self.agent.update_step += 1