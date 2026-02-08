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

from actoris_harena import TrainableAgent

from dotmap import DotMap
from .replay_buffer import ReplayBuffer

from .wandb_logger import WandbLogger


class Primitive2VectorSAC(VanillaSAC):

    def __init__(self, config):
        self.primitive_param = []
        for prim in config.primitive_param:
            name, params, dims = prim['name'], prim['params'], prim['dims']
            prim_dict = {
                'name': name,
                'params': list(params),
                'dims': list(dims)
            }
            self.primitive_param.append(prim_dict)
        print(self.primitive_param)


        super().__init__(config)
    
    def _dict_to_vector_action(self, dict_action):
        # extract primitive_name (should be a single key)
        primitive_name = list(dict_action.keys())[0]

        #print('primitive name', primitive_name)
        params_dict = dict_action[primitive_name]

        # find primitive index
        chosen_primitive = None
        for k, prim in enumerate(self.primitive_param):
            pname = prim['name']
            if pname == primitive_name:
                chosen_primitive = k
                break
        if chosen_primitive is None:
            raise ValueError(f"Unknown primitive {primitive_name}")

        # flatten parameters in the same order as in primitive_param
        flat_params = []
        for i, param_name in enumerate(self.primitive_param[chosen_primitive]['params']):
            #val = np.array(params_dict[param_name]).reshape(-1)
            flat_params.append(params_dict[param_name])
        
        # prepend primitive index
        return np.stack(flat_params).flatten()

    def _vector_action_to_dict(self, vector_act):
        prim_idx = 0
        primitive_name, params = self.primitive_param[prim_idx]['name'],  self.primitive_param[prim_idx]['params']
        dims = self.primitive_param[prim_idx]['dims']
        # print('\n')
        # print(params)
        # print('\n')    
        out_dict = {primitive_name: {}}

        idx = 0
        for param_name, dim in zip(params, dims):
            out_dict[primitive_name][param_name] = vector_act[idx: idx + dim]
            idx += dim
        
        return out_dict

    def _select_action(self, info: dict, stochastic: bool = False):
        #print('info observation keys', info['observation'].keys())
        obs = [info['observation'][k] for k in self.obs_keys]
        obs = self._process_obs_for_input(obs)
        aid = info['arena_id']
        # maintain obs queue per arena
        if aid not in self.internal_states:
            self.reset([aid])
        self.internal_states[aid]['obs_que'].append(obs)
        while len(self.internal_states[aid]['obs_que']) < self.context_horizon:
            self.internal_states[aid]['obs_que'].append(obs)
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]

        if stochastic:
            #print('obs_t', obs)
            a, logp = self.actor.sample(self._process_context_for_input(obs_list))
            a = torch.clip(a, -self.config.action_range, self.config.action_range)
            a = a.detach().cpu().numpy().squeeze(0)
            #print('\ngeneraed action vectror', a)
            dict_act = self._vector_action_to_dict(a) ## Change!
            return dict_act, logp.detach().cpu().numpy().squeeze(0)
        else:
            mean, _ = self.actor(self._process_context_for_input(obs_list))
            action = torch.tanh(mean)
            action = torch.clip(action, -self.config.action_range, self.config.action_range)
            action = action.detach().cpu().numpy().squeeze(0)
            dict_act = self._vector_action_to_dict(action) ## Change!
            
            return dict_act, None

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


        action, _ = self._select_action(self.info, stochastic=True)
        next_info = arena.step(action)

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
        # action = next_info['applied_action'] Alert !!! Change!!!!
        action = self._dict_to_vector_action(action) ## Change!!!
        self.last_done = done

        aid = arena.id
        obs_list = list(self.internal_states[aid]['obs_que'])[-self.context_horizon:]
        obs_stack = self._process_context_for_replay(obs_list)
        
        # append next
        obs_list.append(self._process_obs_for_input(next_obs))
        next_obs_stack = self._process_context_for_replay(obs_list[-self.context_horizon:])
        #next_obs_stack = np.stack(obs_list)[-self.context_horizon:].flatten() #TODO: .reshape(self.context_horizon * self.each_image_shape[0], *self.each_image_shape[1:])

        action = action.astype(np.float32)
        self.replay.add(obs_stack, action, reward, next_obs_stack, done)
        self.act_steps += 1
        self.episode_return += reward
        self.episode_length += 1
