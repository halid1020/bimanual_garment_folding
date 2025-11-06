# image_primitive_sac.py
from __future__ import annotations

import math
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import deque
from tqdm import tqdm

from agent_arena import TrainableAgent
from dotmap import DotMap

from .vanilla_sac import VanillaSAC
from .networks import NatureCNNEncoderRegressor  # reuse the encoder
from .obs_state_replay_buffer import ObsStateReplayBuffer
from .wandb_logger import WandbLogger
from .vanilla_sac import VanillaSAC, Actor, Critic

# class Critic(nn.Module):
#     def __init__(self, action_dim, feature_dim=512, hidden_dim=256):
#         super().__init__()
#         self.q1_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
#         self.q1_2 = nn.Linear(hidden_dim, hidden_dim)
#         self.q1_3 = nn.Linear(hidden_dim, 1)
       
#         self.q2_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
#         self.q2_2 = nn.Linear(hidden_dim, hidden_dim)
#         self.q2_3 = nn.Linear(hidden_dim, 1)


#     def forward(self, obs_emb, action):
#         x = torch.cat([obs_emb, action], dim=-1)
#         # Q1
#         q1 = F.relu(self.q1_1(x))
#         q1 = F.relu(self.q1_2(q1))
#         q1 = self.q1_3(q1)
#         # Q2
#         q2 = F.relu(self.q2_1(x))
#         q2 = F.relu(self.q2_2(q2))
#         q2 = self.q2_3(q2)
#         return q1, q2

# class Actor(nn.Module):
#     def __init__(self, action_dim, feature_dim=512, hidden_dim=256):
#         super().__init__()
#         self.fc1 = nn.Linear(feature_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.mean = nn.Linear(hidden_dim, action_dim)
#         self.log_std = nn.Linear(hidden_dim, action_dim)


#     def forward(self, obs_emb):
#         x = F.relu(self.fc1(obs_emb))
#         x = F.relu(self.fc2(x))
#         mean = self.mean(x)
#         log_std = self.log_std(x)
#         log_std = torch.clamp(log_std, -20, 2)
#         std = log_std.exp()
#         return mean, std


#     def sample(self, obs_emb):
#         #print('obs shape', obs.shape)
#         mean, std = self(obs_emb)
#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample() # reparameterization trick
#         y_t = torch.tanh(x_t)
#         action = y_t
#         log_prob = normal.log_prob(x_t)
#         log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)
#         return action, log_prob

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
        self.network_action_dim = max(self.action_dims)
        self.replay_action_dim = self.network_action_dim + 1  # prim id + params

        # encoder & targets
        C, H, W = cfg.each_image_shape
        obs_shape = (C * self.context_horizon, H, W)
        self.encoder = NatureCNNEncoderRegressor(obs_shape, cfg.state_dim, cfg.feature_dim).to(self.device)
        self.encoder_target = NatureCNNEncoderRegressor(obs_shape, cfg.state_dim, cfg.feature_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # primitive one-hot toggles
        self.disable_one_hot = cfg.get('disable_one_hot', False)
        self.one_hot_scalar = float(cfg.get('one_hot_scalar', 0.1))
        self.sampling_temperature = cfg.get('sampling_temperature', 1.0)
        self.update_temperature = cfg.get('update_temperature', 0.01)

        # augmented feature dim (embedding + one-hot K)
        self.feature_dim = int(cfg.feature_dim)
        self.feature_aug_dim = self.feature_dim + (0 if self.disable_one_hot else self.K)


        self.actor = Actor(self.feature_aug_dim, self.network_action_dim, cfg.hidden_dim).to(self.device)

        self.critic = Critic(self.feature_aug_dim, self.network_action_dim).to(self.device)
       

        self.critic_target = Critic(self.feature_aug_dim, self.network_action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.encoder_target = NatureCNNEncoderRegressor(obs_shape, cfg.state_dim, cfg.feature_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())


        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)

        # alpha
        self.auto_alpha_learning = cfg.get('auto_alpha_learning', True)
        self.init_alpha = cfg.get('init_alpha', 1.0)
        if self.auto_alpha_learning:
            self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(self.init_alpha), requires_grad=True, device=self.device))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = -float(self.network_action_dim)

        # detach unused action params logic
        self.detach_unused_action_params = cfg.get('detach_unused_action_params', False)
        self.preprocess_action_detach = False

    def _init_reply_buffer(self, cfg):
        C, H, W = cfg.each_image_shape
        obs_shape = (C * self.context_horizon, H, W)
        # use ObsStateReplayBuffer to store images + state
        self.replay = ObsStateReplayBuffer(cfg.replay_capacity, obs_shape, cfg.state_dim, self.replay_action_dim, self.device)

    # ------------------------ primitive helpers ------------------------
    def _one_hot(self, idxs: torch.LongTensor) -> torch.Tensor:
        B = idxs.shape[0]
        one_hot = torch.zeros((B, self.K), device=idxs.device, dtype=torch.float32)
        one_hot.scatter_(1, idxs.unsqueeze(1), 1.0)
        return one_hot * self.one_hot_scalar

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
        # actions: (B, 1 + network_action_dim) stored in buffer
        prim_idx = actions[:, 0].long()
        action_params = actions[:, 1: 1 + self.network_action_dim]
        return action_params, prim_idx

    # ------------------------ I/O processing ------------------------
    def _process_obs_for_input(self, obs_and_state):
        # same logic as Image2State_SAC._process_obs_for_input
        rgb, state = obs_and_state
        state = np.concatenate(state).flatten()
        rgb = np.concatenate(rgb, axis=-1)
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        rgb_resized = cv2.resize(rgb, (self.config.each_image_shape[2], self.config.each_image_shape[1]), interpolation=cv2.INTER_AREA)
        return (rgb_resized.transpose(2, 0, 1), state)

    def _process_context_for_replay(self, context):
        rgb = [c[0] for c in context]
        state = [c[1] for c in context]
        rgb_stack = np.stack(rgb).reshape(
            self.config.context_horizon * self.config.each_image_shape[0],
            *self.config.each_image_shape[1:])
        state_stack = np.stack(state).flatten()
        return rgb_stack, state_stack

    def _process_context_for_input(self, context):
        rgb = [c[0] for c in context]
        state = [c[1] for c in context]
        rgb = torch.as_tensor(rgb, dtype=torch.float32, device=self.device)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return rgb, state

    # ------------------------ action selection / env interaction ------------------------
    def _select_action(self, info: dict, stochastic: bool = False):
        # build context (obs images and state)
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
        e, _ = self.encoder(obs_stack)

        B = e.shape[0]  # usually 1 for acting
        with torch.no_grad():
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
        return out_dict, vector_action
    
    def _post_process_action_to_replay(self, action): # dictionary action e.g. {'push': [...]}
        prim_name = list(action.keys())[0]
        prim_id = -1 # Initialize with a default value
        for id_, prim in enumerate(self.primitives):
            if prim.name == prim_name:
                prim_id = id_
                break
        assert prim_id >= 0, "Primitive Id should be non-negative."
        self.logger.log({
            f"train/primitive_id": prim_id,
        }, step=self.act_steps)

        vector_action = list(action.values())[0] # Assume one-level of hierachy.
        accept_action = np.zeros(self.replay_action_dim, dtype=np.float32) 
        accept_action[1:len(vector_action)+1] = vector_action
        accept_action[0] = prim_id
        #print('accept_action', accept_action)
        return accept_action

    # ------------------------ learning (update) ------------------------
    def _update_networks(self, batch: dict):
        cfg = self.config
        device = self.device
        # Expect batch.values() -> obs, state, action, reward, next_obs, next_state, done
        obs, state, action, reward, next_obs, next_state, done = batch.values()
        B = obs.shape[0]

        # alpha update (we update alpha later, but compute pi/logp for current taken primitive embedding)
        action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)  # (B, param_dim), (B,)
        # forward current context through encoder
        e, pred_state = self.encoder(obs)
        aug_state_taken = self._augment_emb_with_code(e.detach(), prim_idx_taken)
        pi_taken, logp_taken = self.actor.sample(aug_state_taken)
        alpha = self.log_alpha.exp().detach()
        alpha_loss = -(self.log_alpha * (logp_taken + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # --- compute target Q via enumeration/soft-weighting on next_obs ---
        with torch.no_grad():
            next_e, _ = self.encoder(next_obs)
            next_e_target, _ = self.encoder_target(next_obs)
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

        # --- critic update for taken actions ---
        q1_pred, q2_pred = self.critic(aug_state_taken, action_params_taken.view(B, -1))
        critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=cfg.get('critic_grad_clip_value', float('inf')))
        self.critic_optim.step()

        # --- encoder regressor loss (state prediction) ---
        encoder_loss = F.mse_loss(pred_state, state)
        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        self.encoder_optim.step()

        # --- actor update: compute per-primitive actions and weighted policy loss ---
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

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha loss (can recompute from latest pi)
        # (we already updated alpha earlier; keep a consistent logging value)
        # soft updates
        if self.update_steps % cfg.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, cfg.tau)
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
            'encoder_loss': encoder_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'alpha_loss': alpha_loss.item()
        }, step=self.act_steps)

    def _add_transition_replay(self, obs_for_replay, a, reward, next_obs_for_replay, done):

        obs_stack, state_stack = obs_for_replay
        next_obs_stack, next_state_stack = next_obs_for_replay
       
        self.replay.add(obs_stack, state_stack, a, reward, next_obs_stack, next_state_stack,  done)

    def _get_next_obs_for_process(self, next_info):
        next_obs = [next_info['observation'][k] for k in self.obs_keys]
        next_state = [next_info['observation'][k] for k in self.config.state_keys]
        return next_obs, next_state