"""
SAC-based RL agent implementing the project's TrainableAgent interface.

This version integrates both environment interaction (with a single Arena instance)
and network training. The agent collects transitions by rolling out in the arena,
adds them to the replay buffer, and then performs SAC gradient updates.

Design summary:
- Input: N context images from arena info dict (key: 'context_images').
- Encoder: CNN producing encoding vector z.
- Actors: K actor networks producing candidate actions.
- Critics: K critic networks evaluating (z, action) pairs.
- Action selection: choose action from actor whose critic gives max Q.
- Training: collect rollouts in `arena[0]`, push to replay buffer, update networks.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utilities.types import ActionType, InformationType
from .trainable_agent import TrainableAgent
from dotmap import DotMap

# ------------------------------ Utilities ----------------------------------

def default_config() -> DotMap:
    cfg = DotMap()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.encoder = DotMap()
    cfg.encoder.out_dim = 256
    cfg.encoder.cnn_channels = [32, 64, 128]
    cfg.encoder.kernel = 3
    cfg.encoder.pool = 2

    cfg.num_context = 4
    cfg.image_shape = (3, 64, 64)
    cfg.action_dim = 6
    cfg.num_primitives = 5
    cfg.hidden_dim = 256

    cfg.actor_lr = 3e-4
    cfg.critic_lr = 3e-4
    cfg.alpha_lr = 3e-4
    cfg.tau = 0.005
    cfg.gamma = 0.99
    cfg.batch_size = 256

    cfg.replay_capacity = int(1e6)
    cfg.target_entropy = -cfg.action_dim
    cfg.max_grad_norm = 10.0
    cfg.save_dir = None
    return cfg







# ------------------------------ SACAgent ----------------------------------

class ImageBasedMultiPrimitiveSAC(TrainableAgent):
    def __init__(self, config: Optional[DotMap] = None):
        cfg = default_config() if config is None else config
        super().__init__(cfg)
        self.config = cfg
        self.name = "sac-agent"
        self.device = torch.device(cfg.device)

        C, H, W = cfg.image_shape
        self.encoder = ConvEncoder(C, cfg.encoder.cnn_channels, cfg.encoder.out_dim).to(self.device)
        enc_dim = cfg.encoder.out_dim

        self.K = int(cfg.num_primitives)
        self.action_dim = int(cfg.action_dim)

        self.actors = nn.ModuleList([MLPActor(enc_dim, self.action_dim, cfg.hidden_dim) for _ in range(self.K)]).to(self.device)
        self.critics = nn.ModuleList([Critic(enc_dim + self.action_dim, cfg.hidden_dim) for _ in range(self.K)]).to(self.device)
        self.critics_target = nn.ModuleList([Critic(enc_dim + self.action_dim, cfg.hidden_dim) for _ in range(self.K)]).to(self.device)
        for tgt, src in zip(self.critics_target, self.critics):
            tgt.load_state_dict(src.state_dict())

        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in self.actors]
        self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.tensor(math.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(cfg.target_entropy)

        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.image_shape, cfg.num_context, self.action_dim, self.device)
        self.total_update_steps = 0
        self.loaded = False

    def _select_action(self, info: InformationType, stochastic: bool) -> np.ndarray:
        obs =  info["observation"]["image"]
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 4:
            obs_t = obs_t.unsqueeze(0)  # (1, N, C, H, W)
        z = self.encoder(obs_t)

        best_q, best_action = None, None
        for k in range(self.K):
            if stochastic:
                action, _, _ = self.actors[k].sample(z)
            else:
                mean, _ = self.actors[k](z)
                action = torch.tanh(mean)
            q = self.critics[k](torch.cat([z, action], dim=-1))
            if (best_q is None) or (q.item() > best_q):
                best_q = q.item()
                best_action = action
        return best_action.cpu().numpy().squeeze(0)

    def set_eval(self):
        if self.mode == 'eval':
            return
        self.encoder.eval()
        for a in self.actors: a.eval()
        for c in self.critics: c.eval()
        self.mode = 'eval'


    def act(self, info_list: List[InformationType], update: bool = False) -> List[ActionType]:
        
        self.set_eval()
        with torch.no_grad():
            return [self._select_action(info, stochastic=False) for info in info_list]


    def explore_act(self, info_list: List[InformationType]) -> List[ActionType]:
        self.encoder.eval()
        for a in self.actors: a.eval()
        for c in self.critics: c.eval()

        with torch.no_grad():
            return [self._select_action(info, stochastic=True) for info in info_list]


    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.copy_(tau * p_src.data + (1 - tau) * p_tgt.data)

    def _update_networks(self, batch):
        cfg = self.config
        context, action, reward, next_context, done = batch.values()
        z = self.encoder(context)
        z_next = self.encoder(next_context)
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_qs, next_logps = [], []
            for k in range(self.K):
                a_next, logp_next, _ = self.actors[k].sample(z_next)
                q_next = self.critics_target[k](torch.cat([z_next, a_next], dim=-1))
                next_qs.append(q_next)
                next_logps.append(logp_next)
            next_qs_stacked = torch.stack(next_qs, dim=0).squeeze(-1)
            next_logp_stacked = torch.stack(next_logps, dim=0).squeeze(-1)
            best_next_q, best_idx = next_qs_stacked.max(dim=0)
            best_next_logp = next_logp_stacked[best_idx, torch.arange(context.size(0))]
            target_q = reward + (1 - done) * cfg.gamma * (best_next_q.unsqueeze(-1) - alpha * best_next_logp.unsqueeze(-1))

        critic_inputs = torch.cat([z, action], dim=-1)
        q_preds = [self.critics[k](critic_inputs) for k in range(self.K)]
        q_preds_cat = torch.cat(q_preds, dim=1)
        target_q_rep = target_q.repeat(1, self.K)
        critic_loss = F.mse_loss(q_preds_cat, target_q_rep)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), cfg.max_grad_norm)
        self.critic_optim.step()

        for k in range(self.K):
            a_sample, logp, _ = self.actors[k].sample(z)
            q_val = self.critics[k](torch.cat([z, a_sample], dim=-1))
            actor_loss = (alpha * logp - q_val).mean()
            self.actor_optimizers[k].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[k].parameters(), cfg.max_grad_norm)
            self.actor_optimizers[k].step()

        logp_for_alpha = [self.actors[k].sample(z)[1] for k in range(self.K)]
        logp_mean = torch.cat(logp_for_alpha, dim=1).mean()
        alpha_loss = -(self.log_alpha * (logp_mean + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for k in range(self.K):
            self._soft_update(self.critics[k], self.critics_target[k], cfg.tau)

    def _collect_from_arena(self, arena):
        if self.last_done:
            info = arena.reset()
        
        img_obs = info['observation']['image']
        action = self.act([info])[0]
        next_info = arena.step(action)

        next_img_obs = next_info['observation']['image']
        reward = next_info.get("reward", 0.0)
        done = next_info.get("done", False)
        self.last_done = done

        self.replay.add(img_obs, action, reward, next_img_obs, done)
        self.total_act_steps += 1

        

    def train(self, update_steps: int, arenas: Optional[List[Any]] = None) -> bool:
        if arenas is None or len(arenas) == 0:
            raise ValueError("SACAgent.train requires at least one Arena.")
        arena = arenas[0]

        while self.replay.size < self.config.batch_size or self.replay.size < self.initial_act_steps:
            self._collect_from_arena(arena)

        for _ in range(update_steps): ## this is for updating network
            batch = self.replay.sample(self.config.batch_size)
            self._update_networks(batch)
            self.total_update_steps += 1
            
            for _ in range(self.act_steps_per_update):
                self._collect_from_arena(arena)
            
            # Save checkpoint periodically
            if self.total_update_steps % self.checkpoint_interval == 0:
                self.save(self.config.save_dir, checkpoint_id=self.total_update_steps)

        return True

    def save(self, path: Optional[str] = None, checkpoint_id: Optional[int] = None) -> bool:
        """
        Save SAC agent state, including model parameters and replay buffer.

        Args:
            path: Save directory.
            checkpoint_id: Optional ID for checkpoint (e.g. total update step).
        """
        path = path or self.config.save_dir
        if path is None:
            raise ValueError("No save path provided")
        os.makedirs(path, exist_ok=True)

        state = {
            'encoder': self.encoder.state_dict(),
            'actors': [a.state_dict() for a in self.actors],
            'critics': [c.state_dict() for c in self.critics],
            'critics_target': [c.state_dict() for c in self.critics_target],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optim': self.critic_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu().numpy(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'total_steps': self.total_update_steps,
            'total_update_steps': self.total_update_steps,
            'total_act_steps': self.total_act_steps,
        }

        # Save checkpoint with ID if provided
        if checkpoint_id is not None:
            ckpt_name = f"checkpoint_{checkpoint_id}.pt"
            torch.save(state, os.path.join(path, ckpt_name))

        # Always save "last_model.pt"
        torch.save(state, os.path.join(path, "last_model.pt"))

        # Save replay buffer
        replay_path = os.path.join(path, "last_replay_buffer.pkl")
        with open(replay_path, "wb") as f:
            pickle.dump(self.replay, f)

        return True


    def load(self, path: Optional[str] = None) -> int:
        """
        Load the latest SAC agent checkpoint (last_model + last_replay_buffer).

        Args:
            path: Directory to load from (default = self.config.save_dir).

        Returns:
            int: The total update step at which the model was saved, or -1 if loading failed.
        """
        path = path or self.config.save_dir
        if path is None:
            raise ValueError("No load path provided")

        model_file = os.path.join(path, "last_model.pt")
        replay_file = os.path.join(path, "last_replay_buffer.pkl")

        if not os.path.exists(model_file):
            print(f"[WARN] Model file not found: {model_file}")
            return -1

        # Load model + training variables
        state = torch.load(model_file, map_location=self.device)

        self.encoder.load_state_dict(state['encoder'])
        for a, s in zip(self.actors, state['actors']):
            a.load_state_dict(s)
        for c, s in zip(self.critics, state['critics']):
            c.load_state_dict(s)
        for ct, s in zip(self.critics_target, state['critics_target']):
            ct.load_state_dict(s)

        for opt, s in zip(self.actor_optimizers, state['actor_optimizers']):
            opt.load_state_dict(s)
        self.critic_optim.load_state_dict(state['critic_optim'])
        self.alpha_optim.load_state_dict(state['alpha_optim'])

        self.log_alpha = torch.tensor(state['log_alpha'], device=self.device)

        # Restore training variables
        self.total_update_steps = state.get('total_update_steps', 0)
        self.total_act_steps = state.get('total_act_steps', 0)
        self.last_done = state.get('last_done', True)

        # Load replay buffer if available
        if os.path.exists(replay_file):
            with open(replay_file, "rb") as f:
                self.replay = pickle.load(f)
        else:
            print(f"[WARN] Replay buffer file not found: {replay_file}")

        # Mark agent as loaded
        self.loaded = True

        return self.total_update_steps


    
    def load_checkpoint(self, checkpoint: int) -> bool:
        """
        Load SAC agent from a specific checkpoint.

        Args:
            checkpoint: The checkpoint ID (integer) to load.

        Returns:
            bool: True if loading succeeded, False otherwise.
        """
        if self.config.save_dir is None:
            raise ValueError("No save directory provided in config")

        model_file = os.path.join(self.config.save_dir, f"checkpoint_{checkpoint}.pt")
        replay_file = os.path.join(self.config.save_dir, f"checkpoint_{checkpoint}_replay.pkl")

        if not os.path.exists(model_file):
            print(f"[WARN] Checkpoint model file not found: {model_file}")
            return False

        # Load model + training variables
        state = torch.load(model_file, map_location=self.device)

        self.encoder.load_state_dict(state['encoder'])
        for a, s in zip(self.actors, state['actors']):
            a.load_state_dict(s)
        for c, s in zip(self.critics, state['critics']):
            c.load_state_dict(s)
        for ct, s in zip(self.critics_target, state['critics_target']):
            ct.load_state_dict(s)

        for opt, s in zip(self.actor_optimizers, state['actor_optimizers']):
            opt.load_state_dict(s)
        self.critic_optim.load_state_dict(state['critic_optim'])
        self.alpha_optim.load_state_dict(state['alpha_optim'])

        self.log_alpha = torch.tensor(state['log_alpha'], device=self.device)

        # Restore training variables
        self.total_update_steps = state.get('total_update_steps', 0)
        self.total_act_steps = state.get('total_act_steps', 0)
        self.last_done = state.get('last_done', True)

        # Load replay buffer if available
        if os.path.exists(replay_file):
            with open(replay_file, "rb") as f:
                self.replay = pickle.load(f)
        else:
            print(f"[WARN] Replay buffer file not found: {replay_file}")

        self.loaded = True
        return True

    def set_train(self):
        if self.mode == 'train':
            return
        self.encoder.train()
        for a in self.actors: 
            a.train()
        for c in self.critics: 
            c.train()
        self.mode = 'train'