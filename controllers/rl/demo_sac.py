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

from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset


class DemoSAC(VanillaSAC):

    def __init__(self, config):
        self.demo_data_path = config.demo_data_path
        self.demo_data_dir = config.demo_data_dir
        self.demo_trial_num = config.demo_trial_num
        self.demo_obs_config = config.demo_obs_config
        self.demo_act_config = config.demo_act_config
        

        super().__init__(config)

        self._get_demo_act_steps()
    
    def _get_demo_act_steps(self):

        self.demo_dataset = TrajectoryDataset(
            data_path=self.demo_data_path,
            data_dir= self.demo_data_dir,
            io_mode='r',
            obs_config=self.demo_obs_config,
            act_config=self.demo_act_config,
            whole_trajectory=True
        )

        self.demo_act_steps = 0
        
        for i in range(self.demo_trial_num):
            demo_trj = self.demo_dataset.get_trajectory(i)
            self.demo_act_steps += demo_trj['action_steps']

    def _fill_relay_buffer_with_demo(self):
        print('Loading demonstration trajectories..')
        for i in range(self.demo_trial_num):
            demo_trj = self.demo_dataset.get_trajectory(i)
            
            obs = [demo_trj['observation'][k][0] for k in self.obs_keys]
            obs = self._process_obs_for_input(obs)
            self.reset([-1])
            self.internal_states[-1]['obs_que'].append(obs)

            self.episode_return = 0.0
            self.episode_length = 0

            while len(self.internal_states[-1]['obs_que']) < self.context_horizon:
                self.internal_states[-1]['obs_que'].append(obs)
            

            for j in range(demo_trj['action_steps']):

                next_obs = [demo_trj['observation'][k][j+1] for k in self.obs_keys]
                reward = demo_trj['observation']['reward'][j+1]
                self.logger.log(
                    {'train/step_reward': reward}, step=self.act_steps
                )
                action = demo_trj['action']['default'][j]

                obs_list = list(self.internal_states[-1]['obs_que'])[-self.context_horizon:]
                obs_stack = self._process_context_for_replay(obs_list)
                obs_list.append(self._process_obs_for_input(next_obs))
                next_obs_stack = self._process_context_for_replay(obs_list[-self.context_horizon:])
                done = False
                if j == demo_trj['action_steps'] - 1:
                    done = True
                self.replay.add(obs_stack,action, reward, next_obs_stack, done)
                self.act_steps += 1
                self.episode_return += reward
                self.episode_length += 1


            self.logger.log({
                "train/episode_return": self.episode_return,
                "train/episode_length": self.episode_length,
                'train/episode_success': 1 # TODO: make this more general
            }, step=self.act_steps)

        print('Finished.')

    def train(self, update_steps, arenas) -> bool:
        if arenas is None or len(arenas) == 0:
            raise ValueError("SAC.train requires at least one Arena.")
        arena = arenas[0]
        self.set_train()
        #print('here update!!')
        with tqdm(total=update_steps, desc=f"{self.name} Training", initial=0) as pbar:

            if self.replay.size < self.demo_act_steps:
                self._fill_relay_buffer_with_demo()

            while self.replay.size < self.initial_act_steps + self.demo_act_steps:
                self._collect_from_arena(arena)
                pbar.set_postfix(env_step=self.act_steps, updates=self.update_steps)



            for _ in range(update_steps):
                #print('here update')
                batch = self.replay.sample(self.config.batch_size)
                # optional data augmentation hook
                if hasattr(self, 'data_augmenter') and callable(self.data_augmenter):
                    #print('aguemtn!')
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