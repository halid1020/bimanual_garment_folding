# maple_sac.py
from .vanilla_sac import VanillaSAC, Actor, Critic
from .replay_buffer import ReplayBuffer
from .wandb_logger import WandbLogger
from dotmap import DotMap


import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

EPS = 1e-6

"""
TODO:

1. All atomic actions
specifically interface with the Operational Space Control
(OSC) controller, which has 5 degrees of freedom: 3 degrees
to control the position of the end effector, 1 degree to control
the yaw angle, and (for all tasks but wiping) 1 degree to open
and close the gripper.

2. During this reaching phase, the reaching
primitive keeps its gripper closed (except for the non-tabletop
environments like door) and the grasping and pushing prim-
itives keep their grippers open  (done)

3. While we assume con-
tinuous primitive parameters we can also represent discrete
parameters and apply reparameterization with the Gumbel-
Softmax trick [25, 37]

4. Action Collection
- Sample primitive type at from task policy πtskφ (at|st)
- Sample primitive parameters xt from parameter policy πpψ (xt|st, at)
- Truncate sampled parameters to dimension of sampled primitive xt ← xt[: dat ]
- Execute at and xt in environment, obtain reward rt and next state st+1
- Add affordance score to reward rt ← rt + λsaff(st, xt; at)
- Add transition to replay buffer D ← D ∪ {st, at, xt, rt, st+1}
- Update timer t ← t + 1

5. For a consistent
comparison across baselines, our episode lengths are fixed
to 150 atomic timesteps, meaning that we execute a variable
number of primitives until we have exceeded the maximum
number of atomic actions for the episode. 

6. for the first
600k environment steps we set the target entropy for the task
policy and parameter policy to a high value to encourage
higher exploration during the initial stages of training.

7. Policy network output activation: tanh

7. Batch Size: 1024, Learning rate (all networks) 3e−5
Target network update rate τ 1e−3, replay buffer 1M, 

10 # Training steps per epoch 1000
# (Low-level) exploration actions per epoch 3000

--> train 1000 update steps, then do collection for 3000 steps

11. Reward scale 5.0
Affordance score scale λ 3.0
Automatic entropy tuning True
Target Task Policy Entropy 0.50 × log(k), k is number of primitives
Target Parameter Policy Entropy − maxa da

12. For exoloratinon primitive policy, let it sample from the probability distribution.

13. For the inference primtive policy, choose the most probable action primtive.
"""

class PrimitiveActor(nn.Module):
    """
    Discrete policy network with optional Gumbel-Softmax sampling.
    """
    def __init__(self, state_dim, hidden_dim, num_primitives):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_primitives)
        self.K = num_primitives

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)  # (B, K)
        return logits

    def sample(self, obs, temperature=1.0, hard=False):
        """
        Standard categorical sample (used for inference)
        """
        logits = self.forward(obs) / temperature
        dist = torch.distributions.Categorical(logits=logits)
        prim_idx = dist.sample()
        log_prob = dist.log_prob(prim_idx).unsqueeze(-1)
        all_log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return prim_idx, log_prob, all_log_probs, probs

    def sample_gumbel(self, obs, temperature=1.0, hard=True):
        """
        Differentiable Gumbel-Softmax sample
        Returns:
            y_hard: (B, K) one-hot or soft vector (if hard=False, soft)
            log_prob: (B,1) log prob of selected primitive (for entropy)
        """
        logits = self.forward(obs)
        y_soft = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)  # (B, K)
        # Compute log_prob for entropy regularization
        log_probs = F.log_softmax(logits, dim=-1)  # (B, K)
        # For entropy term, get log_prob of selected primitive
        if hard:
            prim_idx = y_soft.argmax(dim=-1)
            log_prob = log_probs.gather(1, prim_idx.unsqueeze(-1))  # (B,1)
        else:
            # soft vector: sum_i y_i * log p_i
            log_prob = (y_soft * log_probs).sum(dim=-1, keepdim=True)
        return y_soft, log_prob



class MAPLE(VanillaSAC):
    """
    MAPLE implementation with two explicit actors:
    1. self.primitive_actor: Discrete SAC policy pi_k(k | s)
    2. self.actor: Continuous SAC policy pi_theta(a_k | s, e_k)
    
    Config changes:
    - embedding_type: 'learnable' (default) or 'onehot'.
    - init_alpha_k: Initial alpha for discrete policy.
    - init_alpha_theta: Initial alpha for continuous policy.
    """

    def _make_actor_critic(self, config):
        # primitives
        self.primitives = config.primitives
        self.K = len(config.primitives)
        self.network_action_dim = max([prim.dim for prim in self.primitives])

        self.replay_action_dim = 1 + self.network_action_dim

        # --- Primitive Embeddings (MODIFIED) ---
        self.embedding_type = config.get('embedding_type', 'learnable')
        actor_params = [] # Parameters for the continuous actor optimizer
        
        if self.embedding_type == 'learnable':
            print("Using LEARNABLE embeddings.")
            self.code_dim = int(config.primitive_code_dim)
            emb = torch.randn(self.K, self.code_dim, device=self.device) * 0.1
            self.primitive_embeddings = nn.Parameter(emb, requires_grad=True)
            self.register_parameter("primitive_embeddings", self.primitive_embeddings)
            actor_params += [self.primitive_embeddings]
        elif self.embedding_type == 'onehot':
            print("Using ONE-HOT embeddings.")
            self.code_dim = self.K # One-hot dimension is num primitives
            self.primitive_embeddings = torch.eye(self.K, device=self.device)
            # Do not add to optimizer params
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}")

        self.aug_state_dim = int(config.state_dim) + self.code_dim

        # --- Create Networks ---
        self.primitive_actor = PrimitiveActor(config.state_dim, config.hidden_dim, self.K).to(self.device)
        self.actor = Actor(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic = Critic(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- Optimizers (MODIFIED) ---
        # Add actor (pi_theta) params to the list
        actor_params += list(self.actor.parameters())
            
        self.primitive_actor_optim = torch.optim.Adam(self.primitive_actor.parameters(), lr=config.actor_lr)
        self.actor_optim = torch.optim.Adam(actor_params, lr=config.actor_lr) # For pi_theta + embeddings
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # --- Entropy Alphas (MODIFIED) ---
        self.auto_alpha_learning = config.get('auto_alpha_learning', True)
        
        # Get separate init alphas from config
        self.init_alpha_theta = self.config.get("init_alpha_theta", 1.0)
        self.init_alpha_k = self.config.get("init_alpha_k", 1.0)
        
        if self.auto_alpha_learning:
            # Alpha for pi_theta (continuous)
            self.log_alpha_theta = torch.nn.Parameter(torch.tensor(math.log(self.init_alpha_theta), requires_grad=True, device=self.device))
            self.alpha_theta_optim = torch.optim.Adam([self.log_alpha_theta], lr=config.alpha_lr)
            self.target_entropy_theta = -float(self.network_action_dim) # -max(da)

            # Alpha for pi_k (discrete)
            self.log_alpha_k = torch.nn.Parameter(torch.tensor(math.log(self.init_alpha_k), requires_grad=True, device=self.device))
            self.alpha_k_optim = torch.optim.Adam([self.log_alpha_k], lr=config.alpha_lr)
            
            # MODIFIED: Target Task Policy Entropy: 0.50 * log(K)
            self.target_entropy_k = 0.50 * math.log(self.K)
        else:
            self.log_alpha_theta = torch.tensor(math.log(self.init_alpha_theta), device=self.device)
            self.log_alpha_k = torch.tensor(math.log(self.init_alpha_k), device=self.device)

        self.critic_grad_clip_value = config.get('critic_grad_clip_value', float('inf'))

    def _augment_state_with_embedding(self, state: torch.Tensor, prim_repr: torch.Tensor) -> torch.Tensor:
        """
        prim_repr: either LongTensor (B,) for hard index OR FloatTensor (B, K) for Gumbel-Softmax
        """
        B = state.shape[0]
        if prim_repr.dim() == 1:  # hard index
            emb = self.primitive_embeddings[prim_repr]  # (B, code_dim)
        else:  # soft one-hot vector
            # (B, K) x (K, code_dim) = (B, code_dim)
            emb = prim_repr @ self.primitive_embeddings
        return torch.cat([state, emb], dim=-1)  # (B, aug_state_dim)


    def _split_actions_from_replay(self, actions: torch.Tensor):
        # actions: (B, 1 + network_action_dim) stored in buffer
        prim_idx = actions[:, 0].long()
        action_params = actions[:, 1: 1 + self.network_action_dim]
        return action_params, prim_idx

    # ---------- selection / acting ----------
    def _select_action(self, info: dict, stochastic: bool = False):
        """
        Select action using MAPLE with Gumbel-Softmax for primitives.
        """
        # 1️⃣ Process observation/context
        obs = [info['observation'][k] for k in self.obs_keys]
        obs = self._process_obs_for_input(obs)
        aid = info['arena_id']
        if aid not in self.internal_states:
            self.reset([aid])
        self.internal_states[aid]['obs_que'].append(obs)
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append(obs)
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        ctx = self._process_context_for_input(obs_list)  # (1, state_dim)

        with torch.no_grad():
            # 2️⃣ Primitive selection
            if stochastic:
                # Sample using Gumbel-Softmax (soft) for exploration
                prim_onehot, _ = self.primitive_actor.sample_gumbel(ctx, temperature=1.0, hard=False)
                prim_idx_tensor = torch.argmax(prim_onehot, dim=-1)
            else:
                # Deterministic: argmax
                logits = self.primitive_actor(ctx)
                prim_idx_tensor = torch.argmax(logits, dim=-1)
                prim_onehot = F.one_hot(prim_idx_tensor, num_classes=self.K).float()

            # 3️⃣ Augment state with primitive embedding
            if self.embedding_type == 'learnable':
                # Use soft embedding for learnable case
                emb = prim_onehot @ self.primitive_embeddings  # (1, code_dim)
            else:  # one-hot
                emb = prim_onehot

            aug_state = torch.cat([ctx, emb], dim=-1)  # (1, aug_state_dim)

            # 4️⃣ Continuous action selection
            if stochastic:
                action, _ = self.actor.sample(aug_state)
                action = torch.clip(action, -self.config.action_range, self.config.action_range)
            else:
                mean, _ = self.actor(aug_state)
                action = torch.tanh(mean)

            # 5️⃣ Prepare output
            prim_idx = prim_idx_tensor.item()
            primitive_name = self.primitives[prim_idx]['name']
            dim = self.primitives[prim_idx]['dim']
            out_dict = {primitive_name: action.cpu().numpy().squeeze(0)}

            vector_action = np.concatenate(([float(prim_idx)], action.cpu().numpy().flatten()))

        return out_dict, vector_action

    def _update_networks(self, batch: dict):
        """
        Full MAPLE update with Gumbel-Softmax for discrete primitives.
        """
        config = self.config
        device = self.device
        context, action, reward, next_context, done = batch.values()
        B = context.shape[0]

        # Split actions
        action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)

        # Alphas
        alpha_k = self.log_alpha_k.exp().detach()
        alpha_theta = self.log_alpha_theta.exp().detach()

        # ====================
        # 1️⃣ Critic Target
        # ====================
        with torch.no_grad():
            # Sample next primitive with Gumbel-Softmax (soft or hard)
            y_next_soft, log_prob_k_next = self.primitive_actor.sample_gumbel(next_context, temperature=1.0, hard=True)

            # Augment next state with primitive
            next_aug_state = self._augment_state_with_embedding(next_context, y_next_soft)

            # Sample continuous action
            a_next, logp_next = self.actor.sample(next_aug_state)

            # Q targets
            q1_next, q2_next = self.critic_target(next_aug_state, a_next)
            q_next = torch.min(q1_next, q2_next)

            # Q_pi(s', k') = Q_target - alpha_theta * log pi_theta(a'|s', k')
            q_pi_next = q_next - alpha_theta * logp_next

            # Bellman target for critic
            target_q = reward + (1.0 - done) * config.gamma * (q_pi_next - alpha_k * log_prob_k_next)

        # ====================
        # 2️⃣ Critic Update
        # ====================
        # Use replayed primitive for critic input
        aug_state_taken = self._augment_state_with_embedding(context, prim_idx_taken)
        q1_pred, q2_pred = self.critic(aug_state_taken, action_params_taken.view(B, -1))
        critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip_value)
        self.critic_optim.step()

        # ====================
        # 3️⃣ Primitive Actor Update (Gumbel-Softmax)
        # ====================
        y_soft, log_prob_k = self.primitive_actor.sample_gumbel(context, temperature=1.0, hard=True)
        aug_state = self._augment_state_with_embedding(context, y_soft)

        # Evaluate continuous actor for Q
        pi_theta_action, logp_theta = self.actor.sample(aug_state)
        q1_pi, q2_pi = self.critic(aug_state, pi_theta_action)
        q_pi = torch.min(q1_pi, q2_pi)

        # Differentiable primitive actor loss
        primitive_actor_loss = (alpha_k * log_prob_k - q_pi).mean()

        self.primitive_actor_optim.zero_grad()
        primitive_actor_loss.backward()
        self.primitive_actor_optim.step()

        # ====================
        # 4️⃣ Continuous Actor Update
        # ====================
        # Reuse y_soft for continuous actor
        #aug_state = self._augment_state_with_embedding(context, y_soft)
        pi_theta_action, logp_theta = self.actor.sample(aug_state.detach())
        q1_pi, q2_pi = self.critic(aug_state.detach(), pi_theta_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha_theta * logp_theta - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ====================
        # 5️⃣ Alpha Updates
        # ====================
        alpha_k_loss = torch.tensor(0.0, device=device)
        alpha_theta_loss = torch.tensor(0.0, device=device)

        if self.auto_alpha_learning:
            alpha_theta_loss = -(self.log_alpha_theta * (logp_theta + self.target_entropy_theta).detach()).mean()
            self.alpha_theta_optim.zero_grad()
            alpha_theta_loss.backward()
            self.alpha_theta_optim.step()

            alpha_k_loss = -(self.log_alpha_k * (log_prob_k + self.target_entropy_k).detach()).mean()
            self.alpha_k_optim.zero_grad()
            alpha_k_loss.backward()
            self.alpha_k_optim.step()

        # ====================
        # 6️⃣ Soft target update
        # ====================
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, config.tau)

        # ====================
        # Logging
        # ====================
        with torch.no_grad():
            q_stats = {
                'q_mean': q_pi.mean().item(),
                'q_max': q_pi.max().item(),
                'q_min': q_pi.min().item(),
                'logp_k_mean': log_prob_k.mean().item(),
                'logp_k_max': log_prob_k.max().item(),
                'logp_k_min': log_prob_k.min().item(),
                'logp_theta_mean': logp_theta.mean().item(),
                'logp_theta_max': logp_theta.max().item(),
                'logp_theta_min': logp_theta.min().item(),
                'critic_grad_norm': critic_grad_norm.item(),
            }
            self.logger.log({f"diag/{k}": v for k,v in q_stats.items()}, step=self.act_steps)
            self.logger.log({
                'loss/critic': critic_loss.item(),
                'loss/actor_params': actor_loss.item(),
                'loss/actor_primitive': primitive_actor_loss.item(),
                'alpha/theta': alpha_theta,
                'alpha/k': alpha_k,
                'loss/alpha_theta': alpha_theta_loss.item(),
                'loss/alpha_k': alpha_k_loss.item(),
            }, step=self.act_steps)

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

    def _save_model(self, model_path):
        state = {
            'primitive_actor': self.primitive_actor.state_dict(), # ADDED
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            
            'primitive_actor_optim': self.primitive_actor_optim.state_dict(), # ADDED
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            
            'update_steps': self.update_steps,
            'act_steps': self.act_steps,
            'sim_steps': self.sim_steps,
            "wandb_run_id": self.logger.get_run_id(),
        }
        
        # Save embeddings ONLY if they are learnable
        if self.embedding_type == 'learnable':
            state['primitive_embeddings'] = self.primitive_embeddings.detach().cpu()

        if self.auto_alpha_learning:
            # Save both alphas
            state['log_alpha_theta'] =  self.log_alpha_theta.detach().cpu()
            state['alpha_theta_optim'] = self.alpha_theta_optim.state_dict()
            state['log_alpha_k'] =  self.log_alpha_k.detach().cpu()
            state['alpha_k_optim'] = self.alpha_k_optim.state_dict()

        torch.save(state, model_path)
    
    def _load_model(self, model_path, resume=False):
        state = torch.load(model_path, map_location=self.device)
        
        self.primitive_actor.load_state_dict(state['primitive_actor']) # ADDED
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])

        self.primitive_actor_optim.load_state_dict(state['primitive_actor_optim']) # ADDED
        self.actor_optim.load_state_dict(state['actor_optim'])
        self.critic_optim.load_state_dict(state['critic_optim'])
        
        # Load embeddings ONLY if they are learnable
        if self.embedding_type == 'learnable':
            if 'primitive_embeddings' in state:
                # Need to load into the nn.Parameter data
                emb_data = state['primitive_embeddings'].to(self.device)
                self.primitive_embeddings.data.copy_(emb_data)
            else:
                print(f"[WARN] Model checkpoint missing 'primitive_embeddings', using random init.")

        if self.auto_alpha_learning:
            # Load alpha_theta
            self.log_alpha_theta = torch.nn.Parameter(state['log_alpha_theta'].to(self.device).clone().requires_grad_(True))
            self.alpha_theta_optim = torch.optim.Adam([self.log_alpha_theta], lr=self.config.alpha_lr)
            self.alpha_theta_optim.load_state_dict(state['alpha_theta_optim'])
            
            # Load alpha_k
            self.log_alpha_k = torch.nn.Parameter(state['log_alpha_k'].to(self.device).clone().requires_grad_(True))
            self.alpha_k_optim = torch.optim.Adam([self.log_alpha_k], lr=self.config.alpha_lr)
            self.alpha_k_optim.load_state_dict(state['alpha_k_optim'])
        
        self.update_steps = state.get('update_steps', 0)
        self.act_steps = state.get('act_steps', 0)

        self.sim_steps = state.get('sim_steps', 0)
        self.last_sim_steps = self.sim_steps

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