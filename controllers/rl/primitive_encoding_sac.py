from .vanilla_sac import VanillaSAC, Actor, Critic

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
from .replay_buffer import ReplayBuffer

from .wandb_logger import WandbLogger


class PrimitiveEncodingSAC(VanillaSAC):
    """
    SAC variant that encodes a discrete set of K primitives as learnable codes.
    - config must contain:
        - num_primitives: K
        - primitive_code_dim: dim of embedding
        - action_param_dim: dimensionality of continuous action parameters (not including primitive one-hot)
        - action_range: same as in VanillaSAC (used to clip)
        - (other usual VanillaSAC config fields)
    Replay action format (to remain compatible with existing ReplayBuffer):
        [ action_params (action_param_dim), primitive_one_hot (num_primitives) ]
    """


    def _make_actor_critic(self, config):
        # number of discrete primitives
        self.primitive_param = config.primitive_param
        self.update_temperature = self.config.get('update_temperature', 0.01)
        self.sampling_temperature = self.config.get('sampling_temperature', 1.)
        self.K = int(config.num_primitives)
        self.network_action_dim = max([config.action_dims[k] for k in range(self.K)])
        self.replay_action_dim = self.network_action_dim  + 1

        # augmented state dimension for actor/critic: original state + primitive encoding
        self.aug_state_dim = int(config.state_dim) +  self.K

        # actor maps augmented state -> action_params (continuous)
        # reuse Actor class but give it aug_state_dim and action_param_dim
        self.actor = Actor(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)

        # critic consumes augmented state and action_params
        self.critic = Critic(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    
    def _init_reply_buffer(self, config):
        self.replay = ReplayBuffer(config.replay_capacity, (config.state_dim, ), self.replay_action_dim, self.device)



    # ---------- helpers ----------
    def _one_hot(self, idxs: torch.LongTensor) -> torch.Tensor:
        # idxs: (B,) long
        B = idxs.shape[0]
        one_hot = torch.zeros((B, self.K), device=idxs.device, dtype=torch.float32)
        one_hot.scatter_(1, idxs.unsqueeze(1), 1.0)
        # print('idx', idxs)
        # print('one hot', one_hot)
        return one_hot

    def _augment_state_with_code(self, state: torch.Tensor, prim_idx: torch.LongTensor) -> torch.Tensor:
        # state: (B, state_dim)
        # prim_idx: (B,) long
        # returns: (B, aug_state_dim)
        codes = self._one_hot(prim_idx)  # (B, code_dim)
        return torch.cat([state, codes], dim=-1)

    def _expand_state_all_primitives(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, state_dim)
        # returns: (B*K, aug_state_dim) where each state's primitive code is concatenated
        B = state.shape[0]
        # repeat states K times
        state_rep = state.unsqueeze(1).repeat(1, self.K, 1).view(B * self.K, -1)  # (B*K, state_dim)
        # create primitive indices 0..K-1 repeated for each batch
        prim_idxs = torch.arange(self.K, device=state.device, dtype=torch.long).unsqueeze(0).repeat(B, 1).view(-1)
        aug_state = self._augment_state_with_code(state_rep, prim_idxs)
        return aug_state, prim_idxs  # (B*K, aug_state_dim), (B*K,)

    def _split_actions_from_replay(self, actions: torch.Tensor):
        # actions: (B, action_param_dim + K) as stored in buffer
        prim_idx = actions[:, 0].long()
        action_params = actions[:, 1: self.network_action_dim+1]  # (B, param_dim)
        return action_params, prim_idx

    # ---------- selection / acting ----------
    def _select_action(self, info: dict, stochastic: bool = False):
        # override to produce (action_params + primitive_one_hot) packed action for env
        obs = [info['observation'][k] for k in self.obs_keys]
        obs = self._process_obs_for_input(obs)
        aid = info['arena_id']
        if aid not in self.internal_states:
            self.reset([aid])
        self.internal_states[aid]['obs_que'].append(obs)
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append(obs)
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        ctx = self._process_context_for_input(obs_list)  # tensor (1, state_dim) or (1, enc_dim)

        # compute per-primitive actions and Qs
        with torch.no_grad():
            B = ctx.shape[0]  # should be 1 usually for act
            aug_states_all, _ = self._expand_state_all_primitives(ctx)  # (B*K, aug_state_dim)
            # actor deterministic mean or stochastic sample:
            if stochastic:
                a_all, logp_all = self.actor.sample(aug_states_all)  # a_all: (B*K, param_dim), logp_all: (B*K,1)
            else:
                mean, _ = self.actor(aug_states_all)
                a_all = torch.tanh(mean)
                logp_all = None

            # critic per-primitive Q
            q1_all, q2_all = self.critic(aug_states_all, a_all.view(B * self.K, -1))
            q_all = torch.min(q1_all, q2_all).view(B, self.K)  # (B, K)

            # softmax over Qs -> probabilities
            probs = torch.softmax(q_all/self.sampling_temperature, dim=-1)  # (B, K)
            # choose primitive according to probs if stochastic, else argmax
            if stochastic:
                # sample primitive index per batch element
                prim_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().item()  # (B,)
            else:
                prim_idx = torch.argmax(probs, dim=-1).cpu().item()  # (B,)

            best_action = a_all[prim_idx].cpu().numpy()

            primitive_name, params = self.primitive_param[prim_idx]['name'],  self.primitive_param[prim_idx]['params']
            out_dict = {primitive_name: {}}

            idx = 0
            for param_name, dim in params:
                out_dict[primitive_name][param_name] = best_action[idx: idx + dim]
                idx += dim

            best_action = np.array([prim_idx] + best_action.tolist())

            # print('out dict', out_dict)
            # print('best_action', best_action)


            return out_dict, best_action

    # ---------- learning ----------
    def _update_networks(self, batch: dict):
        """
        batch expected to contain:
            context: torch.tensor (B, state_dim)
            action: torch.tensor (B, action_param_dim + K)  # params + one-hot
            reward: (B,1)
            next_context: (B, state_dim)
            done: (B,1)
        """
        config = self.config
        device = self.device
        context, action, reward, next_context, done = batch.values()
        context = context.to(device)
        next_context = next_context.to(device)
        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)
        B = context.shape[0]
        alpha = self.log_alpha.exp()

        # --- compute target Q using target critic and actor across all primitives ---
        # expand next_context across primitives
        with torch.no_grad():
            next_aug_states_all, prim_idxs_all = self._expand_state_all_primitives(next_context)  # (B*K, aug_state_dim)
            # sample actor for each next augmented state
            a_next_all, logp_next_all = self.actor.sample(next_aug_states_all)  # (B*K, param_dim), (B*K,1)
            q1_next_all, q2_next_all = self.critic_target(next_aug_states_all, a_next_all.view(B * self.K, -1))
            q_next_all = torch.min(q1_next_all, q2_next_all).view(B, self.K)  # (B, K)
            logp_next_all = logp_next_all.view(B, self.K)  # (B, K)

            # compute softmax weights over q_next_all (using raw q values)
            w_next = torch.softmax(q_next_all/self.update_temperature, dim=-1)  # (B, K) #check

            # compute weighted target Q per batch: note q_next_all already (B,K)
            weighted_q_minus_alpha_logp = (w_next * (q_next_all - alpha * logp_next_all)).sum(dim=-1, keepdim=True)  # (B,1)
            target_q = reward + (1.0 - done) * config.gamma * weighted_q_minus_alpha_logp  # (B,1)

        # --- critic update: compute Q for taken (primitive + params) from batch and MSE to target_q ---
        action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)  # (B,param_dim), (B,)
        aug_state_taken = self._augment_state_with_code(context, prim_idx_taken)  # (B, aug_state_dim)
        q1_pred, q2_pred = self.critic(aug_state_taken, action_params_taken.view(B, -1))
        critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # --- actor update: compute per-primitive actions & Qs for current context, softmax over Qs, weighted actor loss ---
        aug_states_all, _ = self._expand_state_all_primitives(context)  # (B*K, aug_state_dim)
        pi_all, logp_all = self.actor.sample(aug_states_all)  # (B*K, param_dim), (B*K,1)
        q1_all, q2_all = self.critic(aug_states_all, pi_all.view(B * self.K, -1))
        q_all = torch.min(q1_all, q2_all).view(B, self.K)  # (B,K)
        logp_all = logp_all.view(B, self.K)  # (B,K)

        # softmax weights over q_all
        w_pi = torch.softmax(q_all/self.update_temperature, dim=-1)  # (B,K) # check

        # actor loss per primitive: alpha * logp - Q; weighted sum
        actor_loss_per = (w_pi * (alpha * logp_all - q_all)).sum(dim=-1).mean()
        actor_loss = actor_loss_per

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # --- alpha (temperature) update ---
        alpha_loss = -(self.log_alpha * (logp_all + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # --- soft update target critic ---
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, config.tau)

        # logging
        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'alpha_loss': alpha_loss.item()
        }, step=self.act_steps)

    # ---------- replay pre/post-processing ----------
    def _process_context_for_replay(self, context):
        # same as Vanilla: flatten the stacked context into 1D vector (state_dim,)
        return np.stack(context).flatten()
    
    def _dict_to_vector_action(self, dict_action):
        # extract primitive_name (should be a single key)
        primitive_name = next(iter(dict_action.keys()))
        #print('primitive name', primitive_name)
        params_dict = dict_action[primitive_name]

        # find primitive index
        best_k = None
        for k, prim in enumerate(self.primitive_param):
            pname, params = prim['name'], prim['params']
            if pname == primitive_name:
                best_k = k
                break
        if best_k is None:
            raise ValueError(f"Unknown primitive {primitive_name}")

        # flatten parameters in the same order as in primitive_param
        flat_params = []
        for param_name, dim in self.primitive_param[best_k]['params']:
            val = np.array(params_dict[param_name]).reshape(-1)
            flat_params.extend(val.tolist())

        # prepend primitive index
        return np.array([best_k] + flat_params)

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
        
        dict_action, vector_action = self._select_action(self.info, stochastic=True)
        #print('dict action', dict_action)
        next_info = arena.step(dict_action)

        dict_action_ = next_info['applied_action']
        vector_action_ = self._dict_to_vector_action(dict_action_)


        next_obs = [next_info['observation'][k] for k in self.obs_keys]
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
        obs_stack = self._process_context_for_replay(obs_list)
        
        # append next
        obs_list.append(self._process_obs_for_input(next_obs))
        next_obs_stack = self._process_context_for_replay(obs_list[-self.context_horizon:])
        #next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])


        accept_action = np.zeros(self.replay_action_dim, dtype=np.float32)
        #print('vector_action', vector_action)
        accept_action[:len(vector_action_)] = vector_action_

        self.replay.add(obs_stack, accept_action, reward, next_obs_stack, done)
        self.act_steps += 1
        self.episode_return += reward
        self.episode_length += 1