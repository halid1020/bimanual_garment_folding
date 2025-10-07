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

from agent_arena import TrainableAgent

from dotmap import DotMap
from .networks import ConvEncoder, MLPActor, Critic  # expects networks similar to your repo
from .replay_buffer import ReplayBuffer

from .wandb_logger import WandbLogger


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.q1_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_3 = nn.Linear(hidden_dim, 1)
       
        self.q2_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_3 = nn.Linear(hidden_dim, 1)


    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
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
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)


    def forward(self, obs):
        x = F.relu(self.fc1(obs))
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

class VanillaSAC(TrainableAgent):
    """A vanilla Soft Actor-Critic for image observations.

    Assumptions:
    - Replay buffer stores contextual image observations as a stacked channel: (C * N, H, W)
    - The provided `ConvEncoder`, `MLPActor`, and `Critic` have interfaces compatible with this code:
        - encoder(input) -> z (batch x enc_dim)
        - actor.sample(z) -> (action, logp, _)
        - actor(z) -> (mean, log_std) optionally
        - critic(concat(z, action)) -> q
    """

    def __init__(self, config: Optional[DotMap] = None):
        cfg = default_config() if config is None else config
        super().__init__(cfg)
        self.name = cfg.name
        self.config = cfg
        self.device = torch.device(cfg.device)
        self.context_horizon = cfg.context_horizon
        # self.each_image_shape = cfg.each_image_shape
       

        self._make_actor_critic(cfg)

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # entropy temperature
        self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(0.1), requires_grad=True, device=self.device))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = -float(self.action_dim)

        # replay
        self._init_reply_buffer(cfg)
 
        # bookkeeping
        self.update_steps = 0
        self.loaded = False
        self.logger = WandbLogger(project="vanilla-sac", name="sac-agent", config=dict(cfg))
        self.obs_key = cfg.obs_key
        self.reward_key = cfg.reward_key
        self.last_done = True
        self.episode_return = 0.0
        self.episode_length = 0
        self.act_steps = 0
        self.initial_act_steps = cfg.initial_act_steps
        #self.act_steps_per_update = cfg.act_steps_per_update
        self.total_update_steps = cfg.total_update_steps
        self.info = None

    def _make_actor_critic(self, cfg):
        # actor and critics (two critics for twin-Q)
        self.action_dim = int(cfg.action_dim)
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)

        self.critic = Critic(cfg.state_dim,  cfg.action_dim).to(cfg.device)
       

        self.critic_target = Critic(cfg.state_dim,  cfg.action_dim).to(cfg.device)
       
        self.critic_target.load_state_dict(self.critic.state_dict())

    
    def _init_reply_buffer(self, cfg):
        
        self.replay = ReplayBuffer(cfg.replay_capacity, (cfg.state_dim, ), self.action_dim, self.device)


    # ---------------------- utils ----------------------
    def _pre_process(self, state) -> np.ndarray:
        return state

    def _select_action(self, info: dict, stochastic: bool = False):
        obs = info['observation'][self.obs_key]
        obs = self._pre_process(obs)
        aid = info['arena_id']
        # maintain obs queue per arena
        if aid not in self.internal_states:
            self.reset([aid])
        self.internal_states[aid]['obs_que'].append(obs)
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append(obs)
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_np = np.stack(obs_list, axis=0)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

        # encoder expects either (B, C*N, H, W) or (B, C, N, H, W) depending on your ConvEncoder
        # follow the same pattern as the framework: reshape to (B, C, N, H, W) then pass
        # B = obs_t.shape[0]
        # C = self.each_image_shape[0]
        # N = self.context_horizon
        # H, W = self.each_image_shape[1:]
        # z = self.encoder(obs_t.reshape(B, C, N, H, W))

        if stochastic:
            #print('obs_t', obs)
            a, logp = self.actor.sample(obs_t)
            a = a.detach().cpu().numpy().squeeze(0)
            return a, logp.detach().cpu().numpy().squeeze(0)
        else:
            mean, _ = self.actor(obs_t)
            action = torch.tanh(mean)
            return action.detach().cpu().numpy().squeeze(0), None

    def act(self, info_list, updates=None):
        self.set_eval()
        with torch.no_grad():
            return [self._select_action(info, stochastic=False)[0] for info in info_list]

    def explore_act(self, info_list):
        self.set_eval()
        with torch.no_grad():
            return [self._select_action(info, stochastic=True)[0] for info in info_list]

    def single_act(self, info, update=False):
        return self._select_action(info)[0]

    def set_eval(self):
        if getattr(self, 'mode', None) == 'eval':
            return
       
        self.actor.eval()
        self.critic.eval()
        self.mode = 'eval'

    def set_train(self):
        if getattr(self, 'mode', None) == 'train':
            return

        self.actor.train()
        self.critic.train()
        self.mode = 'train'

    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.copy_(tau * p_src.data + (1 - tau) * p_tgt.data)

    # ---------------------- learning ----------------------
    def _update_networks(self, batch: dict):
        cfg = self.config
        context, action, reward, next_context, done = batch.values()
        B = context.shape[0]
        # C = self.each_image_shape[0]
        # N = self.context_horizon
        # H, W = self.each_image_shape[1:]

        alpha = self.log_alpha.exp()

        # compute target Q
        with torch.no_grad():
            a_next, logp_next = self.actor.sample(next_context)
            q1_next, q2_next = self.critic_target(next_context, a_next)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * cfg.gamma * (q_next - alpha * logp_next)

        # current Q estimates
        a_curr = action.view(B, -1).to(self.device)
        q1_pred, q2_pred = self.critic(context, a_curr)

        # print('q1_pred', q1_pred.shape)
        # print('q2_pred', q1_pred.shape)
        # print('target_q', target_q.shape)

        critic_loss = 0.5*(F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        # optimize critics
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor loss
        pi, log_pi = self.actor.sample(context)
        q1_pi, q2_pi = self.critic(context, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_pi - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha loss
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # soft updates
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, cfg.tau)

        # logging
        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'alpha_loss': alpha_loss.item()
        }, step=self.act_steps)

    # ---------------------- environment interaction ----------------------
    def _collect_from_arena(self, arena):
        if self.last_done:
            if self.info is not None:
                evaluation = self.info['evaluation']
                success = int(self.info['success'])

                for k, v in evaluation.items():
                    self.logger.log({
                        f"train/eps_lst_step_eval_{k}": v,
                    }, step=self.act_steps) 
                
                self.logger.log({
                    "train/episode_return": self.episode_return,
                    "train/episode_length": self.episode_length,
                    'train/episode_success': success
                }, step=self.act_steps)

            self.info = arena.reset()
            self.set_train()
            self.reset([arena.id])
            self.episode_return = 0.0
            self.episode_length = 0

        # sample stochastic action for exploration
        a, _ = self._select_action(self.info, stochastic=True)
        # clip to action range
        a = np.clip(a, -self.config.action_range, self.config.action_range)
        #dict_action = {'continuous': a}  # user should adapt to their arena's expected action format
        next_info = arena.step(a)

        next_obs = next_info['observation'][self.obs_key]
        reward = next_info.get('reward', 0.0)[self.reward_key] if isinstance(next_info.get('reward', 0.0), dict) else next_info.get('reward', 0.0)
        self.logger.log(
            {'train/step_reward': reward}, step=self.act_steps
        )
        done = next_info.get('done', False)
        self.info = next_info
        a = next_info['applied_action']
        self.last_done = done

        aid = arena.id
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_stack = np.stack(obs_list).flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

        # append next
        obs_list.append(self._pre_process(next_obs))
        next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

        self.replay.add(obs_stack, a.astype(np.float32), reward, next_obs_stack, done)
        self.act_steps += 1
        self.episode_return += reward
        self.episode_length += 1

    def train(self, update_steps, arenas) -> bool:
        if arenas is None or len(arenas) == 0:
            raise ValueError("SAC.train requires at least one Arena.")
        arena = arenas[0]
        self.set_train()
        #print('here update!!')
        with tqdm(total=update_steps, desc=f"{self.name} Training", initial=0) as pbar:
            while self.replay.size < self.initial_act_steps:
                self._collect_from_arena(arena)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)

            for _ in range(update_steps):
                #print('here update')
                batch = self.replay.sample(self.config.batch_size)
                # optional data augmentation hook
                if hasattr(self, 'data_augmenter') and callable(self.data_augmenter):
                    self.data_augmenter(batch)
                self._update_networks(batch)
                self.update_steps += 1

                if self.update_steps % self.config.gradient_steps == 0:
                    for _ in range(self.config.train_freq):
                        self._collect_from_arena(arena)
                        pbar.set_postfix(
                            phase="training",
                            env_step=self.act_steps,
                            total_updates=self.update_steps,
                        )

                pbar.update(1)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)

        return True

    # ---------------------- save/load ----------------------
    def save(self, path: Optional[str] = None, checkpoint_id: Optional[int] = None) -> bool:
        path = path or self.save_dir
        path = os.path.join(path, 'checkpoints')
        os.makedirs(path, exist_ok=True)

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
        }

        if checkpoint_id is not None:
            torch.save(state, os.path.join(path, f'checkpoint_{checkpoint_id}.pt'))
        torch.save(state, os.path.join(path, 'last_model.pt'))

        # save replay
        replay_path = os.path.join(path, 'last_replay_buffer.pt')
        torch.save({
            'ptr': self.replay.ptr,
            'size': self.replay.size,
            'capacity': self.replay.capacity,
            'observation': torch.from_numpy(self.replay.observation),
            'actions': torch.from_numpy(self.replay.actions),
            'rewards': torch.from_numpy(self.replay.rewards),
            'next_observation': torch.from_numpy(self.replay.next_observation),
            'dones': torch.from_numpy(self.replay.dones),
        }, replay_path)

        return True

    def load(self, path: Optional[str] = None) -> int:
        path = path or self.save_dir
        path = os.path.join(path, 'checkpoints')
        model_file = os.path.join(path, 'last_model.pt')
        replay_file = os.path.join(path, 'last_replay_buffer.pt')

        if not os.path.exists(model_file):
            print(f"[WARN] Model file not found: {model_file}")
            return 0

        state = torch.load(model_file, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])

        self.actor_optim.load_state_dict(state['actor_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])
        self.alpha_optim.load_state_dict(state['alpha_optim'])

        self.log_alpha = torch.nn.Parameter(state['log_alpha'].to(self.device).clone().requires_grad_(True))

        self.update_steps = state.get('update_steps', 0)
        self.act_steps = state.get('act_steps', 0)

        if os.path.exists(replay_file):
            replay_state = torch.load(replay_file, map_location='cpu')
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']
            self.replay.observation = replay_state['observation'].cpu().numpy()
            self.replay.actions = replay_state['actions'].cpu().numpy()
            self.replay.rewards = replay_state['rewards'].cpu().numpy()
            self.replay.next_observation = replay_state['next_observation'].cpu().numpy()
            self.replay.dones = replay_state['dones'].cpu().numpy()

        self.loaded = True
        return self.update_steps

    def reset(self, arena_ids):
        for aid in arena_ids:
            self.internal_states[aid] = {}
            self.internal_states[aid]['obs_que'] = deque()

    def set_log_dir(self, log_dir):
        self.save_dir = log_dir
        self.logger.set_log_dir(log_dir)
