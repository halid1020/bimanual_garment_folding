from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import deque
from tqdm import tqdm
from .vanilla_sac import VanillaSAC

from agent_arena import TrainableAgent
from agent_arena.utilities.logger.logger_interface import Logger
from dotmap import DotMap
from .networks import ConvEncoder, MLPActor, Critic  # expects networks similar to your repo
from .replay_buffer import ReplayBuffer



class NatureCNNEncoder(nn.Module):
    """
    NatureCNN-style encoder used in Stable Baselines3 for image-based SAC.
    Input: (C, H, W), default (3, 84, 84)
    Output: feature vector of size `feature_dim` (default 512)
    """
    def __init__(self, obs_shape=(3, 84, 84), feature_dim=512):
        super().__init__()
        assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # 9x9 -> 7x7
            nn.ReLU()
        )

        # Compute flatten size
        with torch.no_grad():
            n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, feature_dim)

    def forward(self, obs):
        # Normalize image to [0,1]
        x = obs / 255.0
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return F.relu(x)



class Critic(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = NatureCNNEncoder(obs_shape, feature_dim)
        self.q1_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_3 = nn.Linear(hidden_dim, 1)
       
        self.q2_1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.q2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_3 = nn.Linear(hidden_dim, 1)


    def forward(self, obs, action):
        h = self.encoder(obs)
        x = torch.cat([h, action], dim=-1)
        # Q1
        q1 = F.relu(self.q1_1(x))
        q1 = F.relu(self.q1_2(q1))
        q1 = self.q1_3(q1)
        # Q2
        q2 = F.relu(self.q2_1(x))
        q2 = F.relu(self.q2_2(q2))
        q2 = self.q2_3(q2)
        return q1, q2

class Actor(nn.Module):
    def __init__(self, obs_shape, action_dim, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.encoder = NatureCNNEncoder(obs_shape, feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


    def forward(self, obs):
        h = self.encoder(obs)
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std


    def sample(self, obs):
        #print('obs shape', obs.shape)
        mean, std = self(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class VanillaImageSAC(VanillaSAC):
    """A vanilla Soft Actor-Critic for image observations.

    Assumptions:
    - Replay buffer stores contextual image observations as a stacked channel: (C * N, H, W)
    - The provided `ConvEncoder`, `MLPActor`, and `Critic` have interfaces compatible with this code:
        - encoder(input) -> z (batch x enc_dim)
        - actor.sample(z) -> (action, logp, _)
        - actor(z) -> (mean, log_std) optionally
        - critic(concat(z, action)) -> q
    """

    def __init__(self, config):
        super().__init__(config)
        #self.each_image_shape = cfg.each_image_shape
        

    def _make_actor_critic(self, cfg):
        C, H, W = cfg.each_image_shape
        self.input_channel = C * self.context_horizon
        obs_shape = (self.input_channel, H, W) 

        # actor and critics (two critics for twin-Q)
        self.action_dim = int(cfg.action_dim)
        self.actor = Actor(obs_shape, cfg.action_dim, cfg.feature_dim, cfg.hidden_dim).to(self.device)

        self.critic = Critic(obs_shape, cfg.action_dim, cfg.feature_dim).to(cfg.device)
       

        self.critic_target = Critic(obs_shape, cfg.action_dim, cfg.feature_dim).to(cfg.device)
       
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def _init_reply_buffer(self, cfg):
        
        C, H, W = cfg.each_image_shape
        self.input_channel = C * self.context_horizon
        obs_shape = (self.input_channel, H, W) 

        self.replay = ReplayBuffer(cfg.replay_capacity, obs_shape, self.action_dim, self.device)


    def _process_obs_for_input(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        rgb_resized = cv2.resize(rgb, (self.config.each_image_shape[2], self.config.each_image_shape[1]), interpolation=cv2.INTER_AREA)
        return rgb_resized.transpose(2, 0, 1)

    def _process_context_for_replay(self, context):
        return np.stack(context).reshape(
            self.config.context_horizon * self.config.each_image_shape[0], 
            *self.config.each_image_shape[1:])


    # # ---------------------- environment interaction ----------------------
    # def _collect_from_arena(self, arena):
    #     if self.last_done:
    #         if self.info is not None:
    #             evaluation = self.info['evaluation']
    #             success = int(self.info['success'])

    #             for k, v in evaluation.items():
    #                 self.logger.log({
    #                     f"train/eps_lst_step_eval_{k}": v,
    #                 }, step=self.act_steps) 
                
    #             self.logger.log({
    #                 "train/episode_return": self.episode_return,
    #                 "train/episode_length": self.episode_length,
    #                 'train/episode_success': success
    #             }, step=self.act_steps)

    #         self.info = arena.reset()
    #         self.set_train()
    #         self.reset([arena.id])
    #         self.episode_return = 0.0
    #         self.episode_length = 0

    #     # sample stochastic action for exploration
    #     a, _ = self._select_action(self.info, stochastic=True)
    #     # clip to action range
    #     a = np.clip(a, -self.config.action_range, self.config.action_range)
    #     #dict_action = {'continuous': a}  # user should adapt to their arena's expected action format
    #     next_info = arena.step(a)

    #     next_img_obs = next_info['observation'][self.obs_key]
    #     reward = next_info.get('reward', 0.0)[self.reward_key] if isinstance(next_info.get('reward', 0.0), dict) else next_info.get('reward', 0.0)
    #     self.logger.log(
    #         {'train/step_reward': reward}, step=self.act_steps
    #     )
    #     done = next_info.get('done', False)
    #     self.info = next_info
    #     a = next_info['applied_action']
    #     self.last_done = done

    #     aid = arena.id
    #     img_obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
    #     img_obs = np.stack(img_obs_list).reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

    #     # append next
    #     img_obs_list.append(self.pre_process(next_img_obs))
    #     next_img_obs_stack = np.stack(img_obs_list)[-self.context_horizon:].reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

    #     self.replay.add(img_obs, a.astype(np.float32), reward, next_img_obs_stack, done)
    #     self.act_steps += 1
    #     self.episode_return += reward
    #     self.episode_length += 1