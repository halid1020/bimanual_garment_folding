from __future__ import annotations

import math
import os
from typing import Optional

import zarr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .vanilla_sac import VanillaSAC

from .networks import  Critic  # expects networks similar to your repo
from .obs_state_replay_buffer import ObsStateReplayBuffer
from .obs_state_replay_buffer_zarr import ObsStateReplayBufferZarr
# from .vanilla_image_sac import NatureCNNEncoder

from .wandb_logger import WandbLogger


class NatureCNNEncoder(nn.Module):
    def __init__(self, 
                 obs_shape=(3, 84, 84),
                 state_dim=45,
                 feature_dim=512,
                 conv_layers: Optional[list] = None):
        """
        Args:
            obs_shape: (C, H, W)
            state_dim: output state dimension
            feature_dim: intermediate feature dimension before regression
            conv_layers: list of dicts, e.g.
                [
                    {"out_channels": 32, "kernel_size": 8, "stride": 4},
                    {"out_channels": 64, "kernel_size": 4, "stride": 2},
                    {"out_channels": 64, "kernel_size": 3, "stride": 1}
                ]
        """
        super().__init__()
        assert len(obs_shape) == 3, "Input must be 3D (C,H,W)"
        C, H, W = obs_shape

        # Default to classic NatureCNN layout if not provided
        if conv_layers is None:
            conv_layers = [
                {"out_channels": 32, "kernel_size": 8, "stride": 4},
                {"out_channels": 64, "kernel_size": 4, "stride": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 1}
            ]

        conv_modules = []
        in_channels = C
        for layer_cfg in conv_layers:
            conv_modules += [
                nn.Conv2d(
                    in_channels,
                    layer_cfg["out_channels"],
                    kernel_size=layer_cfg["kernel_size"],
                    stride=layer_cfg["stride"]
                ),
                nn.ReLU()
            ]
            in_channels = layer_cfg["out_channels"]

        self.conv_net = nn.Sequential(*conv_modules)

        # compute flatten size
        with torch.no_grad():
            n_flatten = self.conv_net(torch.zeros(1, *obs_shape)).view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, feature_dim)
        self.regressor = nn.Linear(feature_dim, state_dim)

    def forward(self, obs):
        x = obs / 255.0
        x = self.conv_net(x)
        x = x.reshape(x.size(0), -1)
        e = F.relu(self.fc(x))
        x = self.regressor(e)
        return e, x

class Actor(nn.Module):
    def __init__(self, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        input_dim = feature_dim
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        self.mlp = nn.Sequential(*layers)
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

    def forward(self, obs_emb):
        x = self.mlp(obs_emb)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, obs_emb):
        mean, std = self(obs_emb)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, action_dim, feature_dim=512, hidden_dims=[256, 256]):
        super().__init__()
        input_dim = feature_dim + action_dim
        self.q1_net = self._build_q_net(input_dim, hidden_dims)
        self.q2_net = self._build_q_net(input_dim, hidden_dims)

    def _build_q_net(self, input_dim, hidden_dims):
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        return nn.Sequential(*layers)

    def forward(self, obs_emb, action):
        x = torch.cat([obs_emb, action], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2


class Image2State_SAC(VanillaSAC):

    def __init__(self, config):
        super().__init__(config)
        #self.each_image_shape = cfg.each_image_shape
        

    def _make_actor_critic(self, cfg):
        self.critic_grad_clip_value = cfg.get('critic_grad_clip_value', float('inf'))
        self.auto_alpha_learning = cfg.get('auto_alpha_learning', True)
        self.network_action_dim = int(cfg.action_dim)

        C, H, W = cfg.each_image_shape
        self.input_channel = C * self.context_horizon
        obs_shape = (self.input_channel, H, W)

        # --- NEW CONFIGURABLE NETWORK PARAMETERS ---
        conv_layers = cfg.get("conv_layers", None)
        actor_hidden_dims = cfg.get("actor_hidden_dims", [cfg.hidden_dim, cfg.hidden_dim])
        critic_hidden_dims = cfg.get("critic_hidden_dims", [cfg.hidden_dim, cfg.hidden_dim])

        # --- Create networks ---
        self.encoder = NatureCNNEncoder(
            obs_shape=obs_shape,
            state_dim=cfg.state_dim,
            feature_dim=cfg.feature_dim,
            conv_layers=conv_layers
        ).to(self.device)

        self.actor = Actor(
            action_dim=cfg.action_dim,
            feature_dim=cfg.feature_dim,
            hidden_dims=actor_hidden_dims
        ).to(self.device)

        self.critic = Critic(
            action_dim=cfg.action_dim,
            feature_dim=cfg.feature_dim,
            hidden_dims=critic_hidden_dims
        ).to(self.device)

        self.critic_target = Critic(cfg.action_dim, cfg.feature_dim, hidden_dims=critic_hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.encoder_target = NatureCNNEncoder(obs_shape, cfg.state_dim, cfg.feature_dim, conv_layers=conv_layers).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # --- Optimizers ---
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)

        self.init_alpha = cfg.get("init_alpha", 1.0)
        if self.auto_alpha_learning:
            self.log_alpha = torch.nn.Parameter(torch.tensor(math.log(self.init_alpha), device=self.device))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = -float(self.network_action_dim)

        self.replay_action_dim = self.network_action_dim


    def _init_reply_buffer(self, config):
        
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
        self.init_reply = True


    def _process_obs_for_input(self, obs_and_state):
        """
        Preprocess observation (multi-image RGB stack + state vector).
        Keeps all channels (e.g. 6 for rgb+goal-rgb).
        """
        rgb, state = obs_and_state
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

    # def _process_obs_for_input(self, obs_and_state):
    #     rgb, state = obs_and_state
    #     state = np.concatenate(state).flatten()
    #     rgb = np.concatenate(rgb, axis=-1)
    #     print('rgb shape', rgb.shape)
    #     if rgb.dtype != np.float32:
    #         rgb = rgb.astype(np.float32)
    #     rgb_resized = cv2.resize(rgb, (self.config.each_image_shape[2], self.config.each_image_shape[1]), interpolation=cv2.INTER_AREA)
    #     return (rgb_resized.transpose(2, 0, 1), state)

    def _process_context_for_replay(self, context):
        rgb = [c[0] for c in context]
        state = [c[1] for c in context]
        rgb_stack =  np.stack(rgb).reshape(
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

    def _update_networks(self, batch: dict):
        config = self.config
        #context, action, reward, next_context, done = batch.values()
        obs, state, action, reward, next_obs, next_state, done = batch.values()
        B = obs.shape[0]
        # C = self.each_image_shape[0]
        # N = self.context_horizon
        # H, W = self.each_image_shape[1:]

        alpha = self.log_alpha.exp()

        # compute target Q
        with torch.no_grad():
            next_e, _ = self.encoder(next_obs)
            target_next_e, _ = self.encoder_target(next_obs)
            a_next, logp_next = self.actor.sample(next_e)
            q1_next, q2_next = self.critic_target(target_next_e, a_next)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * config.gamma * (q_next - alpha * logp_next)

        # current Q estimates
        a_curr = action.view(B, -1).to(self.device)
        e, pred_state = self.encoder(obs)

        # does not allow encoder has the value information
        q1_pred, q2_pred = self.critic(e.detach(), a_curr) 
        critic_loss = 0.5*(F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        # optimize critics
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # optimise encoder
        encoder_loss = F.mse_loss(pred_state, state)

        self.encoder_optim.zero_grad()
        encoder_loss.backward()
        self.encoder_optim.step()

        # actor loss
        pi, log_pi = self.actor.sample(e.detach())
        q1_pi, q2_pi = self.critic(e.detach(), pi)
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
            self._soft_update(self.critic, self.critic_target, config.tau)
            self._soft_update(self.encoder, self.encoder_target, config.tau)

            with torch.no_grad():
                q_stats = {
                    'q_mean': min_q_pi.mean().item(),
                    'q_max': min_q_pi.max().item(),
                    'q_min': min_q_pi.min().item(),
                    'logp_mean': log_pi.mean().item(),
                    'logp_max': log_pi.max().item(),
                    'logp_min': log_pi.min().item(),
                }
                self.logger.log({f"diag/{k}": v for k,v in q_stats.items()}, step=self.act_steps)

        # logging
        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'encoder_loss': encoder_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'alpha_loss': alpha_loss.item()
        }, step=self.act_steps)

    
    def _select_action(self, info: dict, stochastic: bool = False):
        #print('info observation keys', info['observation'].keys())
        # print('observation keys', info['observation'].keys())
        obs = [info['observation'][k] for k in self.obs_keys]
        state = [info['observation'][k] for k in self.config.state_keys]
        
        obs, state = self._process_obs_for_input((obs, state))
        aid = info['arena_id']
        # maintain obs queue per arena
        if aid not in self.internal_states:
            self.reset([aid])
        self.internal_states[aid]['obs_que'].append((obs, state))
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append((obs, state))
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]

        obs_stack, state_stack = self._process_context_for_input(obs_list)
        e, _ = self.encoder(obs_stack)

        if stochastic:
            #print('obs_t', obs)
            
            a, logp = self.actor.sample(e)
            a = a.detach().cpu().numpy().squeeze(0)
            return a, logp.detach().cpu().numpy().squeeze(0)
        else:
            mean, _ = self.actor(e)
            action = torch.tanh(mean)
            return action.detach().cpu().numpy().squeeze(0), None

    
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

        next_obs = [next_info['observation'][k] for k in self.obs_keys]
        next_state = [next_info['observation'][k] for k in self.config.state_keys]
        reward = next_info.get('reward', 0.0)[self.reward_key] if isinstance(next_info.get('reward', 0.0), dict) else next_info.get('reward', 0.0)
        self.logger.log(
            {'train/step_reward': reward}, step=self.act_steps
        )
        evaluation = next_info.get('evaluation', {})
        for k, v in evaluation.items():
            self.logger.log({
                f"train/{k}": v,
            }, step=self.act_steps)
        done = next_info.get('done', False)
        self.info = next_info
        a = next_info['applied_action']
        self.last_done = done

        aid = arena.id
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_stack, state_stack = self._process_context_for_replay(obs_list)
        
        # append next
        obs_list.append(self._process_obs_for_input((next_obs, next_state)))
        next_obs_stack, next_state_stack = self._process_context_for_replay(obs_list[-self.context_horizon:])
        #next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])


        self.replay.add(obs_stack, state_stack, a.astype(np.float32), reward, next_obs_stack, next_state_stack,  done)
        self.act_steps += 1
        self.episode_return += reward
        self.episode_length += 1
    
    def set_eval(self):
        if getattr(self, 'mode', None) == 'eval':
            return
       
        self.actor.eval()
        self.critic.eval()
        self.encoder.eval()
        self.mode = 'eval'

    def set_train(self):
        if getattr(self, 'mode', None) == 'train':
            return

        self.actor.train()
        self.critic.train()
        self.encoder.train()
        self.mode = 'train'

    
    def _save_model(self, model_path):
        #os.makedirs(model_path, exist_ok=True)

        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'encoder': self.encoder.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'encoder_target': self.encoder_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'encoder_optim': self.encoder_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'update_steps': self.update_steps,
            'act_steps': self.act_steps,
            'sim_steps': self.sim_steps,
            "wandb_run_id": self.logger.get_run_id(),
        }

        if self.auto_alpha_learning:
            state['log_alpha'] =  self.log_alpha.detach().cpu()
            state['alpha_optim'] = self.alpha_optim.state_dict()

        
        torch.save(state, model_path)

    def _save_replay_buffer(self, replay_path):
        """Save the replay buffer to disk, handling both RAM and Zarr cases."""
        if self.replay_device == 'RAM':
            # Standard in-memory version
            torch.save({
                'ptr': self.replay.ptr,
                'size': self.replay.size,
                'capacity': self.replay.capacity,
                'observation': torch.from_numpy(self.replay.observation),
                'state':  torch.from_numpy(self.replay.state),
                'actions': torch.from_numpy(self.replay.actions),
                'rewards': torch.from_numpy(self.replay.rewards),
                'next_observation': torch.from_numpy(self.replay.next_observation),
                'next_state':  torch.from_numpy(self.replay.next_state),
                'dones': torch.from_numpy(self.replay.dones),
            }, replay_path)

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
        self.encoder.load_state_dict(state['encoder'])
        self.encoder_target.load_state_dict(state['encoder_target'])

        self.actor_optim.load_state_dict(state['actor_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])
        self.encoder_optim.load_state_dict(state['encoder_optim'])
        
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
        if not os.path.exists(replay_file):
            raise FileNotFoundError(f"Replay buffer file not found: {replay_file}")
        self._init_reply_buffer(self.config)

        replay_state = torch.load(replay_file, map_location='cpu')

        if self.replay_device == 'RAM':
            # Restore full in-memory buffer
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']
            self.replay.observation = replay_state['observation'].cpu().numpy()
            self.replay.state = replay_state['state'].cpu().numpy()
            self.replay.actions = replay_state['actions'].cpu().numpy()
            self.replay.rewards = replay_state['rewards'].cpu().numpy()
            self.replay.next_observation = replay_state['next_observation'].cpu().numpy()
            self.replay.next_state = replay_state['next_state'].cpu().numpy()
            self.replay.dones = replay_state['dones'].cpu().numpy()

        elif self.replay_device == 'Disk':
            # Just reload Zarr arrays and metadata
            self.replay.ptr = replay_state['ptr']
            self.replay.size = replay_state['size']
            self.replay.capacity = replay_state['capacity']

            # Reopen the Zarr store (in case a new session started)
            zarr_path = replay_state.get('zarr_path', getattr(self.replay, "zarr_path", None))
            if zarr_path is None:
                raise ValueError("Missing zarr_path in replay state for Disk mode.")

            store = zarr.DirectoryStore(zarr_path)
            self.replay.root = zarr.open_group(store=store, mode='a')

            # Rebind dataset references
            self.replay.observation = self.replay.root["observation"]
            self.replay.state = self.replay.root["state"]
            self.replay.actions = self.replay.root["actions"]
            self.replay.rewards = self.replay.root["rewards"]
            self.replay.next_observation = self.replay.root["next_observation"]
            self.replay.next_state = self.replay.root["next_state"]
            self.replay.dones = self.replay.root["dones"]

        else:
            raise ValueError(f"Unknown replay device type: {self.replay_device}")