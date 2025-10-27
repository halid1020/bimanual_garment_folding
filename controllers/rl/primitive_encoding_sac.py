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
        #self.critic_gradient_clip = config.get('critic_gradient_clip', False)
        self.critic_grad_clip_value = config.get('critic_grad_clip_value', float('inf'))
        self.primitive_param = config.primitive_param
        self.disable_one_hot = config.get('disable_one_hot', False)
        self.update_temperature = self.config.get('update_temperature', 0.01)
        self.sampling_temperature = self.config.get('sampling_temperature', 1.)
        self.K = int(config.num_primitives)
        self.network_action_dim = max([config.action_dims[k] for k in range(self.K)])
        #print('self.network_action_dim', self.network_action_dim)
        self.replay_action_dim = self.network_action_dim  + 1 if not self.disable_one_hot else self.network_action_dim

        # augmented state dimension for actor/critic: original state + primitive encoding
        self.aug_state_dim = int(config.state_dim) +  self.K if not self.disable_one_hot else int(config.state_dim) 

        # actor maps augmented state -> action_params (continuous)
        # reuse Actor class but give it aug_state_dim and action_param_dim
        self.actor = Actor(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)

        # critic consumes augmented state and action_params
        self.critic = Critic(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic_target = Critic(self.aug_state_dim, self.network_action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    
    def _init_reply_buffer(self, config):
        self.replay = ReplayBuffer(config.replay_capacity, (config.state_dim, ), self.replay_action_dim, self.device)



    # ---------- helpers ----------
    def _one_hot(self, idxs: torch.LongTensor) -> torch.Tensor:
        B = idxs.shape[0]
        one_hot = torch.zeros((B, self.K), device=idxs.device, dtype=torch.float32)
        one_hot.scatter_(1, idxs.unsqueeze(1), 1.0)
        scalar = float(self.config.get("one_hot_scalar", 0.1))
        one_hot = one_hot * scalar   # <- ensure multiplication is applied
        return one_hot

    def _augment_state_with_code(self, state: torch.Tensor, prim_idx: torch.LongTensor) -> torch.Tensor:
        # state: (B, state_dim)
        # prim_idx: (B,) long
        # returns: (B, aug_state_dim)
        if self.disable_one_hot:
            return state
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
        if self.disable_one_hot:
            return actions, None
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
        #print('ctx shape', ctx.shape, ctx)
        # compute per-primitive actions and Qs
        with torch.no_grad():
            B = ctx.shape[0]  # should be 1 usually for act
            aug_states_all, _ = self._expand_state_all_primitives(ctx)  # (B*K, aug_state_dim)
            #print('aug_stats_all shape', aug_states_all.shape, aug_states_all)
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
            #print('probs', probs)
            # choose primitive according to probs if stochastic, else argmax

            
            if stochastic:
                # sample primitive index per batch element
                prim_idx = torch.multinomial(probs, num_samples=1).squeeze(-1).detach().cpu().item()  # (B,)
                #print('chosen prim idx', prim_idx)
            else:
                prim_idx = torch.argmax(probs, dim=-1).detach().cpu().item()  # (B,)
            
            a_all = torch.clip(a_all, -self.config.action_range, self.config.action_range)

        
        best_action = a_all[prim_idx].detach().cpu().numpy()

        primitive_name, params = self.primitive_param[prim_idx]['name'],  self.primitive_param[prim_idx]['params']
            
        out_dict = {primitive_name: {}}

        idx = 0
        for param_name, dim in params:
            out_dict[primitive_name][param_name] = best_action[idx: idx + dim]
            idx += dim
        
        if self.disable_one_hot:
            return out_dict, best_action
        
        best_action = np.array([prim_idx] + best_action.tolist())
        
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
        #print('action', action[:2, :3])

        B = context.shape[0]

        
        # --- alpha (temperature) update --- ## Important Change
        action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)  # (B,param_dim), (B,)
        aug_state_taken = self._augment_state_with_code(context, prim_idx_taken)  # (B, aug_state_dim)
        pi, logp = self.actor.sample(aug_state_taken)
        alpha = self.log_alpha.exp().detach() # !!! Important Change
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()


        # --- compute target Q using target critic and actor across all primitives ---
        # expand next_context across primitives
        with torch.no_grad():
            #print('next_context', next_context[0, :3], next_context[0, -3:])
            next_aug_states_all, prim_idxs_all = self._expand_state_all_primitives(next_context)  # (B*K, aug_state_dim)
            #print('next_aug_states_all', next_aug_states_all[0, :3], next_aug_states_all[0, -3:])
            # sample actor for each next augmented state
            a_next_all, logp_next_all = self.actor.sample(next_aug_states_all)  # (B*K, param_dim), (B*K,1)
            q1_next_all, q2_next_all = self.critic_target(next_aug_states_all, a_next_all.view(B * self.K, -1))
            q_next_all = torch.min(q1_next_all, q2_next_all)  # (B*K, 1)
            #print('q_next_all shape and value', q_next_all.shape, q_next_all[0])

            # compute softmax weights over q_next_all (using raw q values)
            if self.K == 1:
                print('!here')
                weighted_q_minus_alpha_logp = q_next_all - alpha *  logp_next_all
            else:
                w_next = torch.softmax(q_next_all.view(B, self.K)/self.update_temperature, dim=-1).detach()  # (B, K) #check
                weighted_q_minus_alpha_logp = (w_next * (q_next_all.view(B, self.K) - alpha *  logp_next_all.view(B, self.K)))
                weighted_q_minus_alpha_logp = weighted_q_minus_alpha_logp.sum(dim=-1, keepdim=True)  # (B,1)
            #print('w_next', w_next)
            #print('weight shape', w_next.shape)

            # compute weighted target Q per batch: note q_next_all already (B,K)
            
            target_q = reward + (1 - done) * config.gamma * weighted_q_minus_alpha_logp  # (B,1)
            
            # print('done', done)
        
        # --- critic update: compute Q for taken (primitive + params) from batch and MSE to target_q ---
        #print('sampled action', action[0])
        action_params_taken, prim_idx_taken = self._split_actions_from_replay(action)  # (B,param_dim), (B,)
        #print('action_params_taken', action_params_taken[:2, :3])
        #print('splitted action', action_params_taken[0],  prim_idx_taken[0])
        #print('context', context[0, :3], context[0, -3:])
        aug_state_taken = self._augment_state_with_code(context, prim_idx_taken)  # (B, aug_state_dim)
        #print('aug_state_taken', aug_state_taken[0, :3], aug_state_taken[0, -3:])
        #print('aug_state_taken shape', aug_state_taken.shape)
        q1_pred, q2_pred = self.critic(aug_state_taken, action_params_taken.view(B, -1))
        critic_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_grad_clip_value)
        self.critic_optim.step()

        # --- actor update: compute per-primitive actions & Qs for current context, softmax over Qs, weighted actor loss ---
        aug_states_all, _ = self._expand_state_all_primitives(context)  # (B*K, aug_state_dim)
        #print('aug_states_all shape', aug_states_all.shape)
        pi_all, logp_all = self.actor.sample(aug_states_all)  # (B*K, param_dim), (B*K,1)
        # print('update action logp_all shape', logp_all.shape)
        # print('pi all shape', pi_all.shape)
        q1_all, q2_all = self.critic(aug_states_all, pi_all)
        #print('q1 all', q1_all[:5])
        #print('q1 all', q2_all[:5])
        q_all = torch.min(q1_all, q2_all)  # (B*K, 1)
        #print('q all', q_all[:5])
        #logp_all = logp_all.view(B, self.K)  # (B,K)
        #print('update action logp_all shape 2', logp_all.shape)

        # softmax weights over q_all
        if self.K == 1:
            print('here')
            w_pi = torch.ones((B, 1), device=device)
            actor_loss = (alpha * logp_all - q_all).mean()
        else:
            w_pi = torch.softmax(q_all.view(B, self.K)/self.update_temperature, dim=-1).detach()  # (B,K) # check
            actor_loss = w_pi * (alpha * logp_all.view(B, self.K)  - q_all.view(B, self.K))
            actor_loss = actor_loss.sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        
        # --- soft update target critic ---
        if self.update_steps % self.config.target_update_interval == 0:
            self._soft_update(self.critic, self.critic_target, config.tau)

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
            self.logger.log({f"diag/{k}": v for k,v in q_stats.items()}, step=self.act_steps)

        self.logger.log({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha,
            'alpha_loss': alpha_loss.item()
        }, step=self.act_steps)

    # ---------- replay pre/post-processing ----------
    
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
        if self.disable_one_hot:
            return flat_params
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
        
        # print('\ngenerated dict action', dict_action)
        # print('\ngenerated vector_action', vector_action)
        
        self.logger.log({
            f"train/primitive_id": vector_action[0],
        }, step=self.act_steps)

      
        #print('produced vector action', vector_action)
        next_info = arena.step(dict_action)

        dict_action_ = next_info['applied_action']
        #print('\napplied dict aciton', dict_action_)
        vector_action_ = self._dict_to_vector_action(dict_action_)
        #print('\napplied vector_action', vector_action_)


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
        # a = next_info['applied_action']
        self.last_done = done

        aid = arena.id
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_stack = self._process_context_for_replay(obs_list)
        
        # append next
        obs_list.append(self._process_obs_for_input(next_obs))
        next_obs_stack = self._process_context_for_replay(obs_list[-self.context_horizon:])
        #next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])


        accept_action = np.zeros(self.replay_action_dim, dtype=np.float32)
        accept_action[:len(vector_action_)] = vector_action_

        
        #print('accepted vector_action to replay', accept_action)

        self.replay.add(obs_stack, accept_action, reward, next_obs_stack, done)
        self.act_steps += 1
        self.episode_return += reward
        self.episode_length += 1