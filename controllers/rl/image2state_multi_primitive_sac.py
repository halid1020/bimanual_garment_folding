# image_primitive_sac.py
from __future__ import annotations

import math
import os
from typing import  Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
import cv2


from .vanilla_sac import VanillaSAC
from .networks import NatureCNNEncoderRegressor  # reuse the encoder
from .obs_state_replay_buffer import ObsStateReplayBuffer
from .obs_state_replay_buffer_zarr import ObsStateReplayBufferZarr
from .replay_buffer import ReplayBuffer
from .replay_buffer_zarr import ReplayBufferZarr
from .wandb_logger import WandbLogger
from .vanilla_sac import VanillaSAC, Actor, Critic

class Image2StateMultiPrimitiveSAC(VanillaSAC):
    """
    Combines Image->state representation learning (encoder + regressor) with
    discrete primitives (learned soft selection over K primitives).
    - Replay stores (obs_stack, state_stack, action_vector, reward, next_obs_stack, next_state_stack, done)
    - action_vector format: [prim_idx (float), param_0, param_1, ...]  (length = 1 + network_action_dim)
    Config expects union of fields used by Image2State_SAC and PrimitiveEncodingSAC:
    - each_image_shape, context_horizon, feature_dim, state_dim, action_dim, primitives (list of dicts with 'name' and 'dim')
    - actor_lr, critic_lr, encoder_lr, alpha_lr, replay_capacity, batch_size, gamma, target_update_interval, tau, action_range, ...
    """


    # ------------------------ network + buffer creation ------------------------
    def _make_actor_critic(self, cfg):
        # primitives
        self.primitives = cfg.primitives
        self.K = len(self.primitives)
        self.action_dims = [prim['dim'] if isinstance(prim, dict) else prim.dim for prim in self.primitives]
        self.obs_type = cfg.obs_type 
       

        if cfg.primitive_integration == 'expand_as_input':
            self.network_action_dim = max(self.action_dims)
            self.replay_action_dim = self.network_action_dim + 1  # prim id + params
            self.update_temperature = cfg.get('update_temperature', 0.01)
            self.sampling_temperature = cfg.get('sampling_temperature', 1.)
            #self.state_dim = cfg.feature_dim + (0 if self.disable_one_hot else self.K)

        elif cfg.primitive_integration == 'predict_bin_as_output':
            self.network_action_dim = max(self.action_dims) + 1
            self.replay_action_dim = self.network_action_dim
            #self.state_dim = cfg.feature_dim
        else:
            raise NotImplementedError

        # encoder & targets
        if cfg.obs_type == 'image':
            conv_layers = cfg.get("conv_layers", None)
            decoder_layers = cfg.get("decoder_layers", None)
            self.use_decoder = cfg.get("user_decoder", False)
            self.feature_dim = int(cfg.feature_dim)
            C, H, W = cfg.each_image_shape
            obs_shape = (C * self.context_horizon, H, W)
            self.encoder = NatureCNNEncoderRegressor(
                    obs_shape=obs_shape,
                    state_dim=cfg.state_dim,
                    feature_dim=cfg.feature_dim,
                    conv_layers=conv_layers,
                    use_decoder=self.use_decoder, 
                    decoder_layers=decoder_layers
            ).to(self.device)
            self.encoder_target = NatureCNNEncoderRegressor(
                    obs_shape=obs_shape,
                    state_dim=cfg.state_dim,
                    feature_dim=cfg.feature_dim,
                    conv_layers=conv_layers,
                    use_decoder=self.use_decoder, 
                    decoder_layers=decoder_layers
            ).to(self.device)
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)
            self.state_loss_coef = cfg.get('state_loss_coef', 1.0)
            self.recon_loss_coef = cfg.get('recon_loss_coef', 2.0)
            self.encoder_max_grad_norm = cfg.get('encoder_max_grad_norm', float('inf'))
            self.state_dim = cfg.feature_dim
            if cfg.primitive_integration == 'expand_as_input':
                self.state_dim = cfg.feature_dim + (0 if self.disable_one_hot else self.K)

        elif cfg.obs_type == 'state':
            self.state_dim = cfg.state_dim
            if cfg.primitive_integration == 'expand_as_input':
                self.state_dim = cfg.state + (0 if self.disable_one_hot else self.K)

        # primitive one-hot toggles
        self.critic_grad_clip_value = cfg.get('critic_grad_clip_value', float('inf'))
        self.disable_one_hot = cfg.get('disable_one_hot', False)
        
        self.detach_unused_action_params = cfg.get('detach_unused_action_params', False)
        self.preprocess_action_detach = False

        self.actor = Actor(self.state_dim, self.network_action_dim, cfg.hidden_dim).to(self.device)

        self.critic = Critic(self.state_dim, self.network_action_dim).to(self.device)
       

        self.critic_target = Critic(self.state_dim, self.network_action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())


        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        
        # alpha
        self.auto_alpha_learning = cfg.get('auto_alpha_learning', True)
        self.init_alpha = cfg.get("init_alpha", 1.0)
        if self.auto_alpha_learning:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor([math.log(self.init_alpha)], device=self.device, requires_grad=True)
            )
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = -float(self.network_action_dim)


    def _init_reply_buffer(self, config):
        
        if not self.init_reply:
            if config.obs_type == 'image':
                C, H, W = config.each_image_shape
                self.input_channel = C * self.context_horizon
                obs_shape = (self.input_channel, H, W)
                
                self.replay_device = config.get('replay_device', 'RAM')
                if self.replay_device == 'RAM':
                    self.replay = ObsStateReplayBuffer(config.replay_capacity, obs_shape, config.state_dim, self.replay_action_dim, self.device)
                elif self.replay_device == 'Disk':
                    self.replay = ObsStateReplayBufferZarr(
                        config.replay_capacity, obs_shape, config.state_dim, self.replay_action_dim, 
                        self.device, zarr_path=os.path.join(self.save_dir, 'replay_buffer.zarr'))
            elif config.obs_type == 'state':
                if self.replay_device == 'RAM':
                    self.replay = ReplayBuffer(config.replay_capacity, (config.state_dim, ), self.replay_action_dim, self.device)
                elif self.replay_device == 'Disk':
                    self.replay = ReplayBufferZarr(
                        config.replay_capacity, (config.state_dim, ), self.replay_action_dim, 
                        self.device, os.path.join(self.save_dir, 'replay_buffer.zarr'))
        
            self.init_reply = True

    # ------------------------ primitive helpers ------------------------
    def _one_hot(self, idxs: torch.LongTensor) -> torch.Tensor:
        assert torch.all((idxs >= 0) & (idxs < self.K)), \
            f"Primitive indices out of range! idxs={idxs}, valid range=[0,{self.K-1}]"
        B = idxs.shape[0]
        one_hot = torch.zeros((B, self.K), device=idxs.device, dtype=torch.float32)
        one_hot.scatter_(1, idxs.unsqueeze(1), 1.0)
        scalar = float(self.config.get("one_hot_scalar", 0.1))
        one_hot = one_hot * scalar
        return one_hot

    def _augment_emb_with_code(self, emb: torch.Tensor, prim_idx: torch.LongTensor) -> torch.Tensor:
        # emb: (B, feature_dim); prim_idx: (B,)
        if self.disable_one_hot:
            return emb
        codes = self._one_hot(prim_idx)
        return torch.cat([emb, codes], dim=-1)

    def _expand_emb_all_primitives(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        # emb: (B, feature_dim) -> returns (B*K, feature_aug_dim), prim_idxs_all (B*K,)
        B = emb.shape[0]
        emb_rep = emb.unsqueeze(1).repeat(1, self.K, 1).view(B * self.K, -1)
        prim_idxs = torch.arange(self.K, device=emb.device, dtype=torch.long).unsqueeze(0).repeat(B, 1).view(-1)
        emb_aug = self._augment_emb_with_code(emb_rep, prim_idxs)
        return emb_aug, prim_idxs

    def _split_actions_from_replay(self, actions: torch.Tensor):
        # actions: (B, action_param_dim + K) as stored in buffer
        if self.disable_one_hot:
            return actions, None
        prim_idx = actions[:, 0].long()
        action_params = actions[:, 1: self.network_action_dim+1]  # (B, param_dim)
        return action_params, prim_idx

    # ------------------------ I/O processing ------------------------
    def _process_obs_for_input(self, obs):
        """
        Preprocess observation (multi-image RGB stack + state vector).
        Keeps all channels (e.g. 6 for rgb+goal-rgb).
        """
        if self.obs_type == 'image':
            rgb, state = obs
            state = np.concatenate(state).flatten()
            rgb = np.concatenate(rgb, axis=-1)  # e.g. (480, 480, 6)
            #print('rgb shape before resize:', rgb.shape)

            if rgb.dtype != np.float32:
                rgb = rgb.astype(np.float32)

            # Target shape from config
            target_w = self.config.each_image_shape[2]
            target_h = self.config.each_image_shape[1]

            # Use the safe multi-channel resize
            rgb_resized = self._resize_multichannel(rgb, target_w, target_h)

            # Final shape: (C, H, W)
            return (rgb_resized.transpose(2, 0, 1), state)
        elif self.obs_type == 'state':
            return np.concatenate(obs).flatten()
    
    def _resize_multichannel(self, rgb, target_w, target_h):
        """
        Resize multi-RGB-channel image (supports 3, 6, 9, ... channels).
        Each group of 3 channels is resized separately and then concatenated back.
        """
        num_channels = rgb.shape[-1]
        resized_parts = []

        for i in range(0, num_channels, 3):
            # Slice 3-channel block
            rgb_part = rgb[..., i:i+3]

            # Safety check (in case num_channels isn't multiple of 3)
            if rgb_part.shape[-1] == 0:
                continue

            # Resize each 3-channel block independently
            resized = cv2.resize(rgb_part, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized_parts.append(resized)

        # Concatenate all resized triplets back
        return np.concatenate(resized_parts, axis=-1)

    def _process_context_for_replay(self, context):
        if self.obs_type == 'image':
            rgb = [c[0] for c in context]
            state = [c[1] for c in context]
            rgb_stack =  np.stack(rgb).reshape(
                self.config.context_horizon * self.config.each_image_shape[0], 
                *self.config.each_image_shape[1:])
            state_stack = np.stack(state).flatten()

            return rgb_stack, state_stack
        elif self.obs_type == 'state':
            context = np.stack(context).flatten()

            ## This is for integrating all garments.
            if context.shape[0] < self.config.state_dim:
                base = np.zeros((self.config.state_dim), dtype=np.float32)
                base[:context.shape[-1]] = context
                context = base
                
            return context

    def _process_context_for_input(self, context):
        if self.obs_type == 'image':
            rgb = [c[0] for c in context]
            state = [c[1] for c in context]
            rgb = torch.as_tensor(rgb, dtype=torch.float32, device=self.device)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return rgb, state
        elif self.obs_type == 'state':
            context = np.stack(context, axis=0)
        
            context = torch.as_tensor(context, dtype=torch.float32, device=self.device)

            ## This is for integrating all garments.
            if context.shape[-1] < self.config.state_dim:
                base = torch.zeros((context.shape[0], self.config.state_dim), dtype=torch.float32, device=self.device)
                base[:, :context.shape[-1]] = context
                context = base
                
            return context
        else:
            raise NotImplementedError

    # ------------------------ action selection / env interaction ------------------------
    def _select_action(self, info: dict, stochastic: bool = False):
        # build context (obs images and state)
        if self.obs_type == 'image':
            obs = [info['observation'][k] for k in self.obs_keys]
            state = [info['observation'][k] for k in self.config.state_keys]
            obs, state = self._process_obs_for_input((obs, state))
            aid = info['arena_id']
            if aid not in self.internal_states:
                self.reset([aid])
            self.internal_states[aid]['obs_que'].append((obs, state))
            while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
                self.internal_states[aid]['obs_que'].append((obs, state))
            obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        
            obs_stack, state_stack = self._process_context_for_input(obs_list)
            # encoder forward (embedding)
            e, _, _ = self.encoder(obs_stack)

            B = e.shape[0]  # usually 1 for acting
        elif self.obs_type == 'state':
            obs = [info['observation'][k] for k in self.obs_keys]
        
            obs = self._process_obs_for_input(obs)
            aid = info['arena_id']
            # maintain obs queue per arena
            if aid not in self.internal_states:
                self.reset([aid])
            self.internal_states[aid]['obs_que'].append(obs)
            while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
                self.internal_states[aid]['obs_que'].append(obs)
            obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
            e = self._process_context_for_input(obs_list)

        with torch.no_grad():
            if self.config.primitive_integration == 'expand_as_input':
                emb_all, _ = self._expand_emb_all_primitives(e)  # (B*K, feature_aug_dim)
                #print('emb_all shape', emb_all.shape)
                if stochastic:
                    a_all, logp_all = self.actor.sample(emb_all)
                else:
                    mean, _ = self.actor(emb_all)
                    a_all = torch.tanh(mean)
                    logp_all = None

                # critic Qs
                q1_all, q2_all = self.critic(emb_all, a_all.view(B * self.K, -1))
                q_all = torch.min(q1_all, q2_all).view(B, self.K)  # (B, K)
                probs = torch.softmax(q_all / self.sampling_temperature, dim=-1)

                if stochastic:
                    prim_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).detach().cpu().item()
                else:
                    prim_idx = torch.argmax(probs, dim=-1).detach().cpu().item()

                a_all = torch.clip(a_all, -self.config.action_range, self.config.action_range)

                best_action = a_all[prim_idx].detach().cpu().numpy()
                prim_name = self.primitives[prim_idx]['name'] if isinstance(self.primitives[prim_idx], dict) else self.primitives[prim_idx].name
                out_dict = {prim_name: best_action}
                vector_action = np.concatenate(([float(prim_idx)], best_action.flatten()))
                return out_dict, out_dict
    
            elif self.config.primitive_integration == 'predict_bin_as_output':
                if stochastic:
                    a_all, logp_all = self.actor.sample(e)
                else:
                    mean, _ = self.actor(e)
                    a_all = torch.tanh(mean)
                    logp_all = None
                
                a_all = a_all[0]
                prim_logit = a_all[0].detach().cpu().item()
                prim_idx = int(((prim_logit + 1)/2)*self.K - 1e-6)
                best_action = a_all[1:].detach().cpu().numpy()
                prim_name = self.primitives[prim_idx]['name'] if isinstance(self.primitives[prim_idx], dict) else self.primitives[prim_idx].name
                out_dict = {prim_name: best_action}
                vector_action = a_all.detach().cpu().numpy()
                return out_dict, vector_action
            else:
                raise NotImplementedError
            

    def _post_process_action_to_replay(self, action):
        # Find primitive id safely
        if self.config.primitive_integration == 'expand_as_input':
            prim_name = list(action.keys())[0]
            vector_action = list(action.values())[0]
            prim_id = next((i for i, prim in enumerate(self.primitives) if prim.name == prim_name), -1)
            if prim_id < 0:
                available = [p.name for p in self.primitives]
                raise ValueError(
                    f"Primitive name '{prim_name}' not found in primitives list! "
                    f"Available primitives: {available}"
                )
            accept_action = np.zeros(self.replay_action_dim, dtype=np.float32) 
            accept_action[1:len(vector_action)+1] = vector_action
            accept_action[0] = prim_id
            self.logger.log({
                f"train/primitive_id": prim_id,
            }, step=self.act_steps)
            
            return accept_action
        elif self.config.primitive_integration == 'predict_bin_as_output':
            # assume input is the action generated by the actor network
            prim_id = int((action[0] + 1)/2*self.K -1e-6)
            self.logger.log({
                f"train/primitive_id": prim_id,
            }, step=self.act_steps)
            return action.astype(np.float32)
        else:
            raise NotImplementedError

    # ------------------------ learning (update) ------------------------
    def _update_networks(self, batch: dict):
        cfg = self.config
        device = self.device
        # Expect batch.values() -> obs, state, action, reward, next_obs, next_state, done
        if self.obs_type == "image":
            obs, state, action, reward, next_obs, next_state, done = batch.values()
            B = obs.size(0)

            # Forward pass
            e, pred_state, recon = self.encoder(obs)
            # --- State prediction loss ---
            state_loss = F.mse_loss(pred_state, state)

            # --- Optional reconstruction loss ---
            if self.use_decoder:
                # Normalize once at the beginning — no double scaling
                obs_norm = obs.float() / 255.0
                recon_loss = F.mse_loss(recon, obs_norm)
            else:
                recon_loss = torch.tensor(0.0, device=obs.device)

            # Total encoder+regressor loss (weights optional)
            encoder_loss = (
                self.state_loss_coef * state_loss
                + self.recon_loss_coef * recon_loss
            )

            # Optimize
            self.encoder_optim.zero_grad()
            encoder_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.encoder_max_grad_norm)
            self.encoder_optim.step()

        elif self.obs_type == 'state':
            context, action, reward, next_context, done = batch.values()
            B = context.shape[0]
            e = context
       
        
        if self.config.primitive_integration == 'expand_as_input':
            action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)  # (B, param_dim), (B,)
            aug_state_taken = self._augment_emb_with_code(e.detach(), prim_idx_taken)
            pi, logp_taken = self.actor.sample(aug_state_taken)
            input_action = action_params_taken
        elif self.config.primitive_integration == 'predict_bin_as_output':
            input_action = action
            pi, logp_taken = self.actor.sample(e.detach())
        else:
            raise NotImplementedError
        
        alpha = self.log_alpha.exp().detach()
        alpha_loss = -(self.log_alpha * (logp_taken + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # --- compute target Q via enumeration/soft-weighting on next_obs ---
        with torch.no_grad():
            if self.obs_type == 'image':
                next_e_target, _, _ = self.encoder_target(next_obs)
            elif self.obs_type == 'state':
                next_e_target = next_context

            if self.config.primitive_integration == 'expand_as_input':
                next_aug_all, prim_idxs_all = self._expand_emb_all_primitives(next_e_target)  # (B*K, feat_aug)
                a_next_all, logp_next_all = self.actor.sample(next_aug_all)  # (B*K, param_dim), (B*K,1)
                q1_next_all, q2_next_all = self.critic_target(next_aug_all, a_next_all.view(B * self.K, -1))
                q_next_all = torch.min(q1_next_all, q2_next_all)  # (B*K,1)

                if self.K == 1:
                    weighted_q_minus_alpha_logp = q_next_all - alpha * logp_next_all
                else:
                    w_next = torch.softmax(q_next_all.view(B, self.K) / self.update_temperature, dim=-1).detach()
                    weighted_q_minus_alpha_logp = (w_next * (q_next_all.view(B, self.K) - alpha * logp_next_all.view(B, self.K)))
                    weighted_q_minus_alpha_logp = weighted_q_minus_alpha_logp.sum(dim=-1, keepdim=True)
                
                target_q = reward + (1 - done) * cfg.gamma * weighted_q_minus_alpha_logp  # (B,1)

            elif self.config.primitive_integration == 'predict_bin_as_output':
                a_next, logp_next = self.actor.sample(next_e_target)
                q1_next, q2_next = self.critic_target(next_e_target, a_next)
                q_next = torch.min(q1_next, q2_next)
                target_q = reward + (1 - done) * cfg.gamma * (q_next - alpha * logp_next)
            

        # --- critic update for taken actions ---
        input_state = aug_state_taken if self.config.primitive_integration == 'expand_as_input' else e.detach()
        q1_pred, q2_pred = self.critic(input_state, input_action)
        critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip_value)
        self.critic_optim.step()

       

        # --- actor update: compute per-primitive actions and weighted policy loss ---
        if self.config.primitive_integration == 'expand_as_input':
            aug_states_all, prim_idxs_all = self._expand_emb_all_primitives(e.detach())
            pi_all, logp_all = self.actor.sample(aug_states_all)

            # optionally detach unused padded action dims
            if self.detach_unused_action_params:
                if not self.preprocess_action_detach:
                    action_dims_tensor = torch.tensor(self.action_dims, device=pi_all.device, dtype=torch.long)
                    prim_dims_per_action = action_dims_tensor[prim_idxs_all]  # (B*K,)
                    action_dim_indices = torch.arange(self.network_action_dim, device=pi_all.device).unsqueeze(0)
                    self.is_padding_mask = action_dim_indices >= prim_dims_per_action.unsqueeze(1)
                    self.preprocess_action_detach = True
                pi_all_detached = pi_all.clone()
                pi_all_detached[self.is_padding_mask] = pi_all_detached[self.is_padding_mask].detach()
                pi_all = pi_all_detached

            q1_all, q2_all = self.critic(aug_states_all, pi_all)
            q_all = torch.min(q1_all, q2_all)  # (B*K,1)

            if self.K == 1:
                actor_loss = (alpha * logp_all - q_all).mean()
            else:
                w_pi = torch.softmax(q_all.view(B, self.K) / self.update_temperature, dim=-1).detach()  # (B,K)
                actor_loss_per = w_pi * (alpha * logp_all.view(B, self.K) - q_all.view(B, self.K))
                actor_loss = actor_loss_per.sum(dim=-1).mean()
        elif self.config.primitive_integration == 'predict_bin_as_output':
            pi, logp_all = self.actor.sample(e.detach())
            q1_pi, q2_pi = self.critic(e.detach(), pi)
            q_all = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha * logp_all - q_all).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha loss (can recompute from latest pi)
        # (we already updated alpha earlier; keep a consistent logging value)
        # soft updates
        if self.update_steps % cfg.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, cfg.tau)
            if self.obs_type == 'image':
                self._soft_update(self.encoder, self.encoder_target, cfg.tau)

        # logging
        with torch.no_grad():
            q_stats = {
                'q_mean': q_all.mean().item(),
                'q_max': q_all.max().item(),
                'q_min': q_all.min().item(),
                'logp_mean': logp_all.mean().item(),
                'logp_max': logp_all.max().item(),
                'logp_min': logp_all.min().item(),
                'critic_grad_norm': critic_grad_norm.item(),
            }
            self.logger.log({f"diag/{k}": v for k, v in q_stats.items()}, step=self.act_steps)

        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'alpha_loss': alpha_loss.item()
        }, step=self.act_steps)

        if self.obs_type == 'image':
            self.logger.log({
                'encoder_loss': encoder_loss.item(),
            }, step=self.act_steps)

    def _add_transition_replay(self, obs_for_replay, a, reward, next_obs_for_replay, done):

        if self.obs_type == 'image':
            obs_stack, state_stack = obs_for_replay
            next_obs_stack, next_state_stack = next_obs_for_replay
        
            self.replay.add(obs_stack, state_stack, a, reward, next_obs_stack, next_state_stack,  done)
        elif self.obs_type == 'state':
            self.replay.add(obs_for_replay, a, reward, next_obs_for_replay,  done)

    def _get_next_obs_for_process(self, next_info):
        if self.obs_type == 'image':
            next_obs = [next_info['observation'][k] for k in self.obs_keys]
            next_state = [next_info['observation'][k] for k in self.config.state_keys]
            return next_obs, next_state
        elif self.obs_type == 'state':
            next_obs = [next_info['observation'][k] for k in self.obs_keys]
            return next_obs
        else:
            raise NotImplementedError
    
    def _save_model(self, model_path):
        #os.makedirs(model_path, exist_ok=True)

        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            
            'critic_target': self.critic_target.state_dict(),
            
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'update_steps': self.update_steps,
            'act_steps': self.act_steps,
            'sim_steps': self.sim_steps,
            "wandb_run_id": self.logger.get_run_id(),
        }
        if self.obs_type == 'image':
            state.update({
                'encoder_optim': self.encoder_optim.state_dict(),
                'encoder': self.encoder.state_dict(),
                'encoder_target': self.encoder_target.state_dict(),
            })

        if self.auto_alpha_learning:
            state['log_alpha'] =  self.log_alpha.detach().cpu()
            state['alpha_optim'] = self.alpha_optim.state_dict()

        
        torch.save(state, model_path)

    def _save_replay_buffer(self, replay_path):
        """Save the replay buffer to disk, handling both RAM and Zarr cases."""
        if self.replay_device == 'RAM':
            # Standard in-memory version
            save_state = {
                'ptr': self.replay.ptr,
                'size': self.replay.size,
                'capacity': self.replay.capacity,
                'observation': torch.from_numpy(self.replay.observation),
               
                'actions': torch.from_numpy(self.replay.actions),
                'rewards': torch.from_numpy(self.replay.rewards),
                'next_observation': torch.from_numpy(self.replay.next_observation),
                
                'dones': torch.from_numpy(self.replay.dones),
            }
            if self.obs_type == 'image':
                save_state.update({
                    'state':  torch.from_numpy(self.replay.state),
                    'next_state':  torch.from_numpy(self.replay.next_state),
                })
            torch.save(save_state, replay_path)

        elif self.replay_device == 'Disk':
            # For Zarr version, we only save small metadata
            meta = {
                'ptr': self.replay.ptr,
                'size': self.replay.size,
                'capacity': self.replay.capacity,
                'zarr_path': str(self.replay.zarr_path)
            }
            torch.save(meta, replay_path)
            # Zarr arrays are already persisted automatically

        else:
            raise ValueError(f"Unknown replay device type: {self.replay_device}")
    

    def _load_model(self, model_path, resume=False):
        state = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        
        if self.obs_type == 'image':
            self.encoder.load_state_dict(state['encoder'])
            self.encoder_target.load_state_dict(state['encoder_target'])
            self.encoder_optim.load_state_dict(state['encoder_optim'])

        self.actor_optim.load_state_dict(state['actor_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])
        
        
        self.log_alpha = torch.nn.Parameter(state['log_alpha'].to(self.device).clone().requires_grad_(True))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.alpha_optim.load_state_dict(state['alpha_optim'])

        self.update_steps = state.get('update_steps', 0)
        self.act_steps = state.get('act_steps', 0)

        run_id = state.get("wandb_run_id", None)
        #print(f"[INFO] Resuming W&B run ID: {run_id}")

        if resume and (run_id is not None):
            self.logger = WandbLogger(
                project=self.config.project_name,
                name=self.config.exp_name,
                config=dict(self.config),
                run_id=run_id,
                resume=True
            )
        else:
            self.logger = WandbLogger(
                project=self.config.project_name,
                name=self.config.exp_name,
                config=dict(self.config),
                resume=False
            )
        
    
    def _load_replay_buffer(self, replay_file):
        """Load replay buffer metadata (and data if using RAM)."""
        self._init_reply_buffer(self.config)
        if not os.path.exists(replay_file):
            raise FileNotFoundError(f"Replay buffer file not found: {replay_file}")

        replay_state = torch.load(replay_file, map_location='cpu')

        if self.replay_device == 'RAM':
            # Restore full in-memory buffer
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']
            self.replay.observation = replay_state['observation'].cpu().numpy()
            if self.obs_type == 'image':
                self.replay.state = replay_state['state'].cpu().numpy()
                self.replay.next_state = replay_state['next_state'].cpu().numpy()

            self.replay.actions = replay_state['actions'].cpu().numpy()
            self.replay.rewards = replay_state['rewards'].cpu().numpy()
            self.replay.next_observation = replay_state['next_observation'].cpu().numpy()
            
            self.replay.dones = replay_state['dones'].cpu().numpy()

        elif self.replay_device == 'Disk':
            # Just reload Zarr arrays and metadata
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']

            # Reopen the Zarr store (in case a new session started)

            store = zarr.DirectoryStore(self.replay.zarr_path)
            self.replay.root = zarr.open_group(store=store, mode='a')

            # Rebind dataset references
            self.replay.observation = self.replay.root["observation"]
            self.replay.actions = self.replay.root["actions"]
            self.replay.rewards = self.replay.root["rewards"]
            self.replay.next_observation = self.replay.root["next_observation"]
            self.replay.dones = self.replay.root["dones"]
            if self.obs_type == 'image':
                self.replay.state = self.replay.root['state']
                self.replay.next_state = self.replay.root['next_state']

        else:
            raise ValueError(f"Unknown replay device type: {self.replay_device}")