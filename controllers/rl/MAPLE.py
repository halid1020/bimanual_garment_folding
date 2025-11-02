# maple_sac.py
from .vanilla_sac import VanillaSAC, Actor, Critic
# Import the new PrimitiveActor
from .networks import ConvEncoder, MLPActor, Critic  # Assuming Actor/Critic are from vanilla_sac
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
itives keep their grippers open

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

# ... (Place PrimitiveActor class definition here) ...
# (Assuming PrimitiveActor from the previous step is here)
class PrimitiveActor(nn.Module):
    """
    A simple discrete policy network.
    Outputs logits for K primitives.
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
        logits = self.logits(x)
        return logits

    def sample(self, obs, temperature=1.0):
        logits = self.forward(obs) / temperature
        dist = torch.distributions.Categorical(logits=logits)
        prim_idx = dist.sample()
        log_prob = dist.log_prob(prim_idx).unsqueeze(-1)
        # For actor loss, we need log_probs of *all* actions
        all_log_probs = F.log_softmax(logits, dim=-1) # (B, K)
        probs = F.softmax(logits, dim=-1) # (B, K)
        return prim_idx, log_prob, all_log_probs, probs


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
        self.K = int(config.num_primitives)
        
        if hasattr(config, 'action_dims'):
            self.network_action_dim = max([config.action_dims[k] for k in range(self.K)])
        else:
            self.network_action_dim = int(config.action_dim)

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

        self.sampling_temperature = config.get('sampling_temperature', 1.0)
        self.critic_grad_clip_value = config.get('critic_grad_clip_value', float('inf'))

    # ---------- helpers ----------
    def _augment_state_with_embedding(self, state: torch.Tensor, prim_idx: torch.LongTensor) -> torch.Tensor:
        # state: (B, state_dim), prim_idx: (B,)
        B = state.shape[0]
        emb = self.primitive_embeddings[prim_idx]  # (B, code_dim)
        return torch.cat([state, emb], dim=-1)  # (B, aug_state_dim)

    def _split_actions_from_replay(self, actions: torch.Tensor):
        # actions: (B, 1 + network_action_dim) stored in buffer
        prim_idx = actions[:, 0].long()
        action_params = actions[:, 1: 1 + self.network_action_dim]
        return action_params, prim_idx

    # ---------- selection / acting ----------
    def _select_action(self, info: dict, stochastic: bool = False):
        # ... (obs processing is the same as your draft) ...
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
            prim_idx, log_prob, all_log_probs, probs = self.primitive_actor(ctx)

            if stochastic:
                prim_idx_tensor = prim_idx
            else:
                # Inference: Choose the most probable action primitive (argmax)
                prim_idx_tensor = torch.argmax(probs, dim=-1)


            prim_idx = prim_idx_tensor.item()

            # 2. Augment state with chosen primitive's embedding
            aug_state = self._augment_state_with_embedding(ctx, prim_idx_tensor) # (1, aug_state_dim)


            # 3. Get continuous params from continuous policy
            if stochastic:
                best_action_tensor, _ = self.actor.sample(aug_state)
            else:
                mean, _ = self.actor(aug_state)
                best_action_tensor = torch.tanh(mean)

            
            best_action = best_action_tensor.detach().cpu().numpy().squeeze(0)
            
            # TODO: clip the best action

        try:
            primitive_name, params = self.primitive_param[prim_idx]['name'], self.primitive_param[prim_idx]['params']
            dims = self.primitive_param[prim_idx]['dims']
            out_dict = {primitive_name: {}}
            idx = 0
            for param_name, dim in zip(params, dims):
                out_dict[primitive_name][param_name] = best_action[idx: idx + dim]
                idx += dim
        except AttributeError:
            print("[WARN] `self.primitive_param` not set. Returning raw action.")
            out_dict = {"primitive": prim_idx, "params": best_action}


        vector_action = np.concatenate(([float(prim_idx)], best_action.flatten()))
        return out_dict, vector_action


    # ---------- learning ----------
    def _update_networks(self, batch: dict):
        config = self.config
        device = self.device
        context, action, reward, next_context, done = batch.values()
        B = context.shape[0]

        action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)

        # Get alphas
        alpha_k = self.log_alpha_k.exp().detach()
        alpha_theta = self.log_alpha_theta.exp().detach()

        # --- 1. Compute Target Q Value (MODIFIED) ---
        # We now compute a sampled target, not the expected value V(s')
        with torch.no_grad():
            # 1. Sample *one* next primitive k' ~ pi_k(s')
            next_prim_idx_k, next_log_prob_k, _, _ = self.primitive_actor.sample(next_context) # (B,), (B, 1)

            # 2. Get Q_pi(s', k') for *only* that sampled k'
            # Augment s' with the sampled k'
            next_aug_state = self._augment_state_with_embedding(next_context, next_prim_idx_k) # (B, aug_state_dim)
            
            # Sample a'_k' ~ pi_theta(s', k')
            a_next, logp_next = self.actor.sample(next_aug_state) # (B, param_dim), (B, 1)
            
            # Get Q_target(s', k', a'_k')
            q1_next, q2_next = self.critic_target(next_aug_state, a_next)
            q_next = torch.min(q1_next, q2_next) # (B, 1)
            
            # Q_pi(s', k') = Q_target(...) - alpha_theta * log pi_theta(...)
            q_pi_next = q_next - alpha_theta * logp_next # (B, 1)
            
            # 3. Compute the target value component from the sampled primitive
            # y_k = Q_pi(s', k') - alpha_k * log pi_k(k'|s')
            target_q_k_component = q_pi_next - alpha_k * next_log_prob_k # (B, 1)
            
            # 4. Compute the final Bellman target
            target_q = reward + (1.0 - done) * config.gamma * target_q_k_component  # (B,1)

        # --- 2. Critic Loss ---
        aug_state_taken = self._augment_state_with_embedding(context, prim_idx_taken)
        q1_pred, q2_pred = self.critic(aug_state_taken, action_params_taken.view(B, -1))
        critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip_value)
        self.critic_optim.step()

       
        
         # --- 3. Primitive Actor Loss (pi_k) (MODIFIED TO USE REPLAYED PRIMITIVE) ---
        # aug_state_taken already contains (s, k_replayed)
        prim_idx_k, log_prob_k, _, _ = self.primitive_actor.sample(context)
        aug_state = self._augment_state_with_embedding(context, prim_idx_k)

        
        pi_theta, logp_theta = self.actor.sample(aug_state.detach())
        q1_pi, q2_pi = self.critic(aug_state, pi_theta)
        q_pi = torch.min(q1_pi, q2_pi)

        primitive_actor_loss = (alpha_k * log_prob_k - q_pi.detach()).mean()

        self.primitive_actor_optim.zero_grad()
        primitive_actor_loss.backward()
        self.primitive_actor_optim.step()
        

        # --- 4. Parameter Actor Loss (pi_theta) ---
        actor_loss = (alpha_theta * logp_theta - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

       

        # --- 5. Alpha (Entropy) Updates ---
        # (This section remains unchanged)
        alpha_k_loss = torch.tensor(0.0, device=device)
        alpha_theta_loss = torch.tensor(0.0, device=device)
        
        if self.auto_alpha_learning:
            # Alpha_theta loss
            alpha_theta_loss = -(self.log_alpha_theta * (logp_theta + self.target_entropy_theta).detach()).mean()
            self.alpha_theta_optim.zero_grad()
            alpha_theta_loss.backward()
            self.alpha_theta_optim.step()

            # Alpha_k loss
            alpha_k_loss = -(self.log_alpha_k * (log_prob_k + self.target_entropy_k).detach()).mean()
            self.alpha_k_optim.zero_grad()
            alpha_k_loss.backward()
            self.alpha_k_optim.step()

        # --- Soft target update ---
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, config.tau)

        # --- Logging (Unchanged) ---
        with torch.no_grad():
            self.logger.log({
                'loss/critic': critic_loss.item(),
                'loss/actor_params': actor_loss.item(),
                'loss/actor_primitive': primitive_actor_loss.item(),
                'alpha/theta': alpha_theta,
                'alpha/k': alpha_k,
                'loss/alpha_theta': alpha_theta_loss.item(),
                'loss/alpha_k': alpha_k_loss.item(),
                # ... other stats
            }, step=self.act_steps)


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

        if checkpoint_id is not None:
            torch.save(state, os.path.join(path, f'checkpoint_{checkpoint_id}.pt'))
        torch.save(state, model_path)
    
    def _load_model(self, model_path):
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