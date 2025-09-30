from __future__ import annotations

import math
import os
from typing import Any, List, Optional
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from collections import deque
from tqdm import tqdm
import pickle


from agent_arena import TrainableAgent
from agent_arena.utilities.logger.logger_interface import Logger
from dotmap import DotMap
from .networks import *
from .replay_buffer import ReplayBuffer

# ------------------------------ Utilities ----------------------------------

def default_config() -> DotMap:
    cfg = DotMap()
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.encoder = DotMap()
    cfg.encoder.out_dim = 256
    cfg.encoder.cnn_channels = [32, 64, 128]
    cfg.encoder.kernel = 3
    cfg.encoder.pool = 2
    cfg.obs_key = 'rgb'

    cfg.context_horizon = 3
    cfg.each_image_shape = (3, 64, 64) # RGB
    cfg.num_primitives = 5
    cfg.hidden_dim = 256

    cfg.actor_lr = 3e-4
    cfg.critic_lr = 3e-4
    cfg.alpha_lr = 3e-4
    cfg.tau = 0.005
    cfg.gamma = 0.99
    cfg.batch_size = 16 #256

    cfg.replay_capacity = int(1e5)
    #cfg.target_entropy = -cfg.max_action_dim
    cfg.max_grad_norm = 10.0
    cfg.save_dir = None

    cfg.num_primitives = 4
    cfg.action_dims = [4, 8, 6, 8]
    cfg.primitive_param = [
        ("norm-pixel-pick-and-fling", [("pick_0", 2), ("pick_1", 2)]),
        ('norm-pixel-pick-and-place', [("pick_0", 2), ("pick_1", 2), ("place_0", 2), ("place_1", 2)]),
        ('norm-pixel-pick-and-drag', [("pick_0", 2), ("pick_1", 2), ("place_0", 2)]),
        ('norm-pixel-fold',  [("pick_0", 2), ("pick_1", 2), ("place_0", 2), ("place_1", 2)])
    ]

    cfg.reward_key = 'multi_stage_reward'

    return cfg

class WandbLogger(Logger):
    def __init__(self, project: str = "rl-project", name: Optional[str] = None, config: Optional[dict] = None):
        self.project = project
        self.name = name
        self.config = config or {}
        self.run = wandb.init(project=self.project, name=self.name, config=self.config)
        self.log_dir = None

    def set_log_dir(self, log_dir: str) -> None:
        """Set log directory (also saves W&B files there)."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        wandb.run.config.update({"logdir": log_dir}, allow_val_change=True)

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a dictionary of metrics to W&B."""
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()

class ImageBasedMultiPrimitiveSAC(TrainableAgent):
    def __init__(self, config: Optional[DotMap] = None):
        cfg = default_config() if config is None else config
        super().__init__(cfg)
        self.config = cfg
        self.name = "sac-agent"
        self.device = torch.device(cfg.device)
        self.context_horizon = cfg.context_horizon 
        self.each_image_shape = cfg.each_image_shape

        C, H, W = cfg.each_image_shape
        self.input_channel = cfg.context_horizon*C
        self.encoder = ConvEncoder(self.input_channel, cfg.encoder.cnn_channels, cfg.encoder.out_dim).to(self.device)
        enc_dim = cfg.encoder.out_dim
        self.action_dims = cfg.action_dims
        self.K = int(cfg.num_primitives)
        self.max_action_dim = max([cfg.action_dims[k] for k in range(self.K)]) + 1

        self.actors = nn.ModuleList([
            MLPActor(enc_dim, cfg.action_dims[k], cfg.hidden_dim) for k in range(self.K)
        ]).to(self.device)

        self.critics = nn.ModuleList([
            Critic(enc_dim + cfg.action_dims[k], cfg.hidden_dim) for k in range(self.K)
        ]).to(self.device)

        self.critics_target = nn.ModuleList([
            Critic(enc_dim + cfg.action_dims[k], cfg.hidden_dim) for k in range(self.K)
        ]).to(self.device)

        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=cfg.encoder_lr)
        self.actor_optim = torch.optim.Adam(self.actors.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critics.parameters(), lr=cfg.critic_lr)
       

        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(0.1), requires_grad=True, device=self.device, dtype=torch.float32))
            # math.log(0.1), device=self.device, dtype=torch.float32)
        #torch.tensor(math.log(0.1), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(-self.max_action_dim)

        self.replay = ReplayBuffer(cfg.replay_capacity, (self.input_channel, H, W), self.max_action_dim, self.device)
        self.update_steps = 0
        self.loaded = False
        self.logger = WandbLogger(
            project="multi-primitive-sac",
            name=self.name,
            config=dict(cfg)
        )
        self.obs_key = cfg.obs_key
        self.primitive_param = cfg.primitive_param
        self.last_done = True
        self.episode_return = 0
        self.episode_length = 0
        self.reward_key = cfg.reward_key
        self.act_steps = 0
        self.initial_act_steps = cfg.initial_act_steps
        self.act_steps_per_update = cfg.act_steps_per_update
        
        self.max_epsilon = self.config.get('max_epsilon', 1.0)
        self.min_epsilon = self.config.get('min_epsilon', 0.05)
        self.explore_mode = self.config.get('explore_mode', 'best')
        self.total_update_steps = self.config.get('total_update_steps', int(1e5))

    def pre_process(self, rgb):
        """
        Preprocess an RGB image.
        - Input: H x W x C (uint8 or float)
        - Resize to 64x64
        - Normalize to [-0.5, 0.5], C x H x W
        """
        # TODO: this process can be customised to each types of obs
        # ensure uint8 -> float32
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)

        # resize (cv2 expects W,H order)
        rgb_resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)

        # normalize to [-0.5, 0.5]
        rgb_norm = rgb_resized / 255.0 - 0.5

        return rgb_norm.transpose(2, 0, 1)

    def _dict_to_vector_action(self, dict_action):
        # extract primitive_name (should be a single key)
        primitive_name = next(iter(dict_action.keys()))
        #print('primitive name', primitive_name)
        params_dict = dict_action[primitive_name]

        # find primitive index
        best_k = None
        for k, (pname, params) in enumerate(self.primitive_param):
            if pname == primitive_name:
                best_k = k
                break
        if best_k is None:
            raise ValueError(f"Unknown primitive {primitive_name}")

        # flatten parameters in the same order as in primitive_param
        flat_params = []
        for param_name, dim in self.primitive_param[best_k][1]:
            val = np.array(params_dict[param_name]).reshape(-1)
            flat_params.extend(val.tolist())

        # prepend primitive index
        return np.array([best_k] + flat_params)


    def _select_action(self, info, stochastic=False):
        obs = info["observation"][self.obs_key]
        obs = self.pre_process(obs)
        
        aid = info['arena_id']
        self.internal_states[aid]['obs_que'].append(obs)
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append(obs)
        #print('len que', len(self.internal_states[aid]['obs_que']))
        
        # take last context_horizon frames
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]

        # stack into array (N, C, H, W) or (N, H, W, C) depending on your obs format
        obs_np = np.stack(obs_list, axis=0)   # (N, ...)

        # convert to torch
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0) # (1, N, C, H, W)

        z = self.encoder(obs_t)

        best_q, best_action, best_k = None, None, None
        sampled_actions = []
        sampled_Qs = []
        for k in range(self.K):
            if stochastic:
                action, _, _ = self.actors[k].sample(z)
                sampled_actions.append(action)
                
            else:
                mean, _ = self.actors[k](z)
                action = torch.tanh(mean)
            q = self.critics[k](torch.cat([z, action], dim=-1))
            sampled_Qs.append(q)
            if (best_q is None) or (q.item() > best_q):
                best_q = q.item()
                best_action = action
                best_k = k

        # Convert to numpy
        if not stochastic:
            best_action = best_action.detach().cpu().numpy().squeeze(0)
        elif self.explore_mode == 'e-greedy':
           
            frac = max(0, (self.total_update_steps - self.update_steps) / self.total_update_steps)
            epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * frac
            
            # print(f'epsilon {epsilon}')
            if np.random.rand() < epsilon:
                
                # pick a random primitive index
                best_k = np.random.randint(self.K)
                best_action = sampled_actions[best_k].detach().cpu().numpy().squeeze(0)
            else:
                # greedy: pick action with max Q
                best_action = best_action.detach().cpu().numpy().squeeze(0)
        elif self.explore_mode == 'best':
            # print('best')
            best_action = best_action.detach().cpu().numpy().squeeze(0)
        else:
            raise NotImplementedError

        # Build dictionary for chosen primitive
        primitive_name, params = self.primitive_param[best_k]
        out_dict = {primitive_name: {}}

        idx = 0
        for param_name, dim in params:
            out_dict[primitive_name][param_name] = best_action[idx: idx + dim]
            idx += dim

        best_action = np.array([best_k] + best_action.tolist())


        return out_dict, best_action


    def set_eval(self):
        if self.mode == 'eval':
            return
        self.encoder.eval()
        for a in self.actors: a.eval()
        for c in self.critics: c.eval()
        self.mode = 'eval'


    def act(self, info_list, updates):
        self.set_eval()
        with torch.no_grad():
            return [self._select_action(info, stochastic=False)[0] for info in info_list]

    def single_act(self, info, update=False):
        return self._select_action(info)[0]


    def explore_act(self, info_list):
        self.set_eval()
        with torch.no_grad():
            return [self._select_action(info, stochastic=True) for info in info_list]


    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        for p_src, p_tgt in zip(source.parameters(), target.parameters()):
            p_tgt.data.copy_(tau * p_src.data + (1 - tau) * p_tgt.data)

    def _update_networks(self, batch):
        cfg = self.config
        context, action, reward, next_context, done = batch.values()
        B = context.shape[0]
        C = self.each_image_shape[0]
        N = self.context_horizon
        H, W = self.each_image_shape[1:]
        #print('context shape', context.shape)
        z = self.encoder(context.reshape(B, C, N, H, W))
        z_next = self.encoder(next_context.reshape(B, C, N, H, W))
        alpha = self.log_alpha.exp()

        # -------- critic update --------
        with torch.no_grad():
            next_qs, next_logps = [], []
            for k in range(self.K):
                a_next, logp_next, _ = self.actors[k].sample(z_next)
                q_next = self.critics_target[k](torch.cat([z_next, a_next], dim=-1))
                next_qs.append(q_next)
                next_logps.append(logp_next)

            next_qs_stacked = torch.stack(next_qs, dim=0).squeeze(-1) # K * B
            #print('next_qs_stacked shape', next_qs_stacked.shape)
            next_logp_stacked = torch.stack(next_logps, dim=0).squeeze(-1)
            best_next_q, best_idx = next_qs_stacked.max(dim=0)
            best_next_logp = next_logp_stacked[best_idx, torch.arange(context.size(0))]
            target_q = reward + (1 - done) * cfg.gamma * (
                best_next_q.unsqueeze(-1) - alpha * best_next_logp.unsqueeze(-1)
            )
            #print('target_q shape', target_q.shape)

        ks = action[:, 0].long()  # primitive id (assumed stored in first dim)
        total_critic_loss, q_preds = 0, []

        for k in range(self.K):
            mask = (ks == k)
            if mask.sum() == 0:
                continue  # no samples for this primitive in batch

            z_k = z[mask]
            action_k = action[mask, 1 : 1 + self.action_dims[k]]  # skip primitive id
            critic_inputs_k = torch.cat([z_k, action_k], dim=-1)
            q_pred_k = self.critics[k](critic_inputs_k)

            target_q_k = target_q[mask]
            loss_k = F.mse_loss(q_pred_k, target_q_k)

            total_critic_loss += loss_k
            q_preds.append(q_pred_k)
            
            self.logger.log({f"critic_loss/primitive_{k}": loss_k.item()})


        # -------- actor update -------- #
        total_actor_loss = 0
        for k in range(self.K):
            a_sample, logp, _ = self.actors[k].sample(z.detach())
            q_val = self.critics[k](torch.cat([z.detach(), a_sample], dim=-1))
            actor_loss = (alpha * logp - q_val).mean()
            self.actor_optim.zero_grad()
            total_actor_loss += actor_loss
            # actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actors[k].parameters(), cfg.max_grad_norm)
            # self.actor_optim.step()

            # Log per-actor loss
            self.logger.log({"actor_loss/primitive_{}".format(k): actor_loss.item()})

        
        # Optimize each critic separately
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()
        total_critic_loss.backward()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actors.parameters(), cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self.encoder.parameters(), cfg.max_grad_norm)
        self.critic_optim.step()
        self.actor_optim.step()
        self.encoder_optim.step()

        # -------- alpha update --------
        logp_for_alpha = [self.actors[k].sample(z.detach())[1].detach() for k in range(self.K)]
        logp_mean = torch.cat(logp_for_alpha, dim=1).mean()
        alpha_loss = -(self.log_alpha * (logp_mean + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Log alpha
        self.logger.log({"alpha": self.log_alpha.exp().item()})

        # -------- soft update --------
        for k in range(self.K):
            self._soft_update(self.critics[k], self.critics_target[k], cfg.tau)

    def _collect_from_arena(self, arena):
        if self.last_done:
            self.info = arena.reset()
            self.set_train()
            self.reset([arena.id])
           
            self.logger.log({
                "train/episode_return": self.episode_return,
                "train/episode_length": self.episode_length,
            })

            self.episode_return = 0.0
            self.episode_length = 0

        #img_obs = self.pre_process(img_obs)
        dict_action, vector_action = self._select_action(self.info, stochastic=True)
        # print('\n\nvector action', vector_action)
        # print('dict_action', dict_action)
        #self.act([info], updates=[False])[0]
        next_info = arena.step(dict_action)

        dict_action_ = next_info['applied_action']
        vector_action_ = self._dict_to_vector_action(dict_action_)
        # print('\nreadjust vector action', vector_action_)
        # print('readjut diction action', dict_action_)

        next_img_obs = next_info['observation'][self.obs_key]
        reward = next_info.get("reward", 0.0)[self.reward_key]
        done = next_info.get("done", False)
        self.info = next_info
        self.last_done = done
        #self.episode_return = 0
        aid = arena.id
        img_obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        img_obs = np.stack(img_obs_list).reshape(self.context_horizon*self.each_image_shape[0], *self.each_image_shape[1:])
        
        img_obs_list.append(self.pre_process(next_img_obs))
        next_img_obs =  np.stack(img_obs_list)[-self.context_horizon:].reshape(self.context_horizon*self.each_image_shape[0], *self.each_image_shape[1:])

        
        accept_action = np.zeros(self.max_action_dim, dtype=np.float32)
        #print('vector_action', vector_action)
        accept_action[:len(vector_action_)] = vector_action_
        #print('action', action)
        self.replay.add(img_obs, accept_action, reward, next_img_obs, done)
        # if self.config.get('add_reject_actions', False) and not np.array_equal(vector_action_, vector_action):
        #     reject_action = np.zeros(self.max_action_dim, dtype=np.float32)
        #     reject_action[:len(vector_action)] = vector_action
        #     self.replay.add(img_obs, reject_action, self.config.get('reject_action_reward', -1), next_img_obs, done)
        
        self.act_steps += 1

        # Track episodic return
        self.episode_return += reward
        self.episode_length += 1

    def train(self, update_steps, arenas) -> bool:
        if arenas is None or len(arenas) == 0:
            raise ValueError("SACAgent.train requires at least one Arena.")
        arena = arenas[0]
        self.set_train()
        #self.save(self.config.save_dir, checkpoint_id='-1')
        with tqdm(total=update_steps, desc="Current Round of Training", initial=0) as pbar:
            pbar.set_postfix(
                    phase="start",
                    env_step=self.act_steps,
                    total_updates=self.update_steps
            )
            while self.replay.size < self.initial_act_steps:
                self._collect_from_arena(arena)
                pbar.set_postfix(
                    phase="pre-collecting",
                    env_step=self.act_steps,
                    total_updates=self.update_steps
                )

        
            for _ in range(update_steps):  # network updates
                batch = self.replay.sample(self.config.batch_size)
                self.data_augmenter(batch)
                self._update_networks(batch)
                self.update_steps += 1

                for _ in range(self.act_steps_per_update):
                    self._collect_from_arena(arena)
                    pbar.set_postfix(
                        phase="training",
                        env_step=self.act_steps,
                        total_updates=self.update_steps,
                    )

                pbar.update(1)

        return True

    def save(self, path: Optional[str] = None, checkpoint_id: Optional[int] = None) -> bool:
        """
        Save SAC agent state, including model parameters and replay buffer.

        Args:
            path: Save directory.
            checkpoint_id: Optional ID for checkpoint (e.g. total update step).
        """
        print(f'Saving Checkpoint {checkpoint_id} ....')
        path = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)
        dummy = torch.zeros(1, self.context_horizon, *self.each_image_shape).to(self.device)  # adjust (B, N, C, H, W) to your case
        self.encoder(dummy)  # this triggers _ensure_init and builds _project

        state = {
            'encoder': self.encoder.state_dict(),
            'actors': self.actors.state_dict(),
            'critics': self.critics.state_dict(),
            'critics_target': self.critics_target.state_dict(),
            'encoder_optim': self.encoder_optim.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'total_steps': self.update_steps,
            'update_steps': self.update_steps,
            'act_steps': self.act_steps,
        }

        # Save checkpoint with ID if provided
        if checkpoint_id is not None:
            ckpt_name = f"checkpoint_{checkpoint_id}.pt"
            torch.save(state, os.path.join(path, ckpt_name))

        # Always save "last_model.pt"
        torch.save(state, os.path.join(path, "last_model.pt"))

        # Save replay buffer
        # replay_path = os.path.join(path, "last_replay_buffer.pkl")
        # with open(replay_path, "wb") as f:
        #     pickle.dump(self.replay, f)

        # In your agent.save()
        replay_path = os.path.join(path, "last_replay_buffer.pt")
        replay_path = os.path.join(path, "last_replay_buffer.pt")
        torch.save({
            "ptr": self.replay.ptr,
            "size": self.replay.size,
            "capacity": self.replay.capacity,
            "observation": torch.from_numpy(self.replay.observation),
            "actions": torch.from_numpy(self.replay.actions),
            "rewards": torch.from_numpy(self.replay.rewards),
            "next_observation": torch.from_numpy(self.replay.next_observation),
            "dones": torch.from_numpy(self.replay.dones),
        }, replay_path)

        print(f'Finished saving checkpoint {checkpoint_id} !')

        return True


    def load(self, path: Optional[str] = None) -> int:
        """
        Load the latest SAC agent checkpoint (last_model + last_replay_buffer).

        Args:
            path: Directory to load from (default = self.config.save_dir).

        Returns:
            int: The total update step at which the model was saved, or -1 if loading failed.
        """
        print(f'Loading last checkpoint from {path} ...')
        path = path or self.save_dir
        # if path is None:
        #     raise ValueError("No load path provided")
        path = os.path.join(path, 'checkpoints')

        model_file = os.path.join(path, "last_model.pt")
        replay_file = os.path.join(path, "last_replay_buffer.pt")

        if not os.path.exists(model_file):
            print(f"[WARN] Model file not found: {model_file}")
            return 0

        # Load model + training variables
        state = torch.load(model_file, map_location=self.device)

        dummy = torch.zeros(1, self.context_horizon, *self.each_image_shape).to(self.device)  # adjust (B, N, C, H, W) to your case
        self.encoder(dummy)  # this triggers _ensure_init and builds _project

        self.encoder.load_state_dict(state['encoder'])
        self.actors.load_state_dict(state['actors'])
        self.critics.load_state_dict(state['critics'])
        self.critics_target.load_state_dict(state['critics_target'])

        self.actor_optim.load_state_dict(state['actor_optim'])
        self.encoder_optim.load_state_dict(state['encoder_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])
        self.alpha_optim.load_state_dict(state['alpha_optim'])

        self.log_alpha = self.log_alpha = torch.nn.Parameter(
            state['log_alpha'].to(self.device).clone().requires_grad_(True)
        )

        # Restore training variables
        self.update_steps = state.get('update_steps', 0)
        self.act_steps = state.get('act_steps', 0)
        #self.last_done = state.get('last_done', True)

        # Load replay buffer if available
        if os.path.exists(replay_file):
            replay_state = torch.load(replay_file, map_location="cpu")

            self.replay.ptr = replay_state["ptr"]
            self.replay.size = replay_state["size"]
            self.replay.capacity = replay_state["capacity"]

            self.replay.observation = replay_state["observation"].cpu().numpy()
            self.replay.actions = replay_state["actions"].cpu().numpy()
            self.replay.rewards = replay_state["rewards"].cpu().numpy()
            self.replay.next_observation = replay_state["next_observation"].cpu().numpy()
            self.replay.dones = replay_state["dones"].cpu().numpy()


        else:
            print(f"[WARN] Replay buffer file not found: {replay_file}")

        # Mark agent as loaded
        self.loaded = True

        print(f'Finished loading last checkpoint from {path}!')

        return self.update_steps


    
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
        #replay_file = os.path.join(self.config.save_dir, f"checkpoint_{checkpoint}_replay.pkl")

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

        for opt, s in zip(self.actor_optim, state['actor_optimizers']):
            opt.load_state_dict(s)
        self.critic_optim.load_state_dict(state['critic_optim'])
        self.alpha_optim.load_state_dict(state['alpha_optim'])

        self.log_alpha = torch.nn.Parameter(torch.tensor(state['log_alpha'], device=self.device, dtype=torch.float32))

        # Restore training variables
        self.update_steps = state.get('update_steps', 0)
        self.act_steps = state.get('act_steps', 0)
        self.last_done = state.get('last_done', True)

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

    def reset(self, arena_ids):
        for aid in arena_ids:
            self.internal_states[aid] = {}
            self.internal_states[aid]['obs_que'] = deque()

    def set_log_dir(self, log_dir):
        self.save_dir = log_dir
        self.logger.set_log_dir(log_dir)