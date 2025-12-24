import os

import datetime
from ..video_logger import VideoLogger
from agent_arena import Arena

import numpy as np
import gym
import uuid


class DMC_Arena(Arena):
    def __init__(self, config):
        super().__init__(config)
        
        os.environ["MUJOCO_GL"] = "osmesa"
        from dm_control import suite

        #os.environ["MUJOCO_GL"] = "osmesa"
        self.num_eval_trials = 30
        self.num_train_trials = 1000
        self.num_val_trials = 5
        self.action_horizon = config.action_horizon

        self.action_repeat = config.action_repeat
        self.image_resolution = config.image_resolution
        camera = config.get('camera', None)
        if camera == None:
            camera = dict(quadruped=2).get(config.domain, 0)
        self.camera = camera
        self.logger = VideoLogger()
        self.eid = 0

        self._env = suite.load(
            config.domain,
            config.task,
            task_kwargs={"random": self.eid},
        )
        spec = self._env.action_spec()
        space = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

        self._mask = np.logical_and(
            np.isfinite(space.low), np.isfinite(space.high)
        )
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        #print(self.action_space)
        self.config = config
        self.frame_resolution = [256, 256]
        self.video_frames = []


    def reset(self, episode_config=None):
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        if 'eid' not in episode_config or episode_config['eid'] is None:

            # randomly select an episode whose 
            # eid equals to the number of episodes%CLOTH_FUNNEL_ENV_NUM = self.id
            if self.mode == 'train':
                episode_config['eid'] = np.random.randint(self.num_train_trials)
            elif self.mode == 'val':
                episode_config['eid'] = np.random.randint(self.num_val_trials)
            else:
                episode_config['eid'] = np.random.randint(self.num_eval_trials)
        
        self.eid = episode_config['eid']
        if self.mode == 'train':
            seed = episode_config['eid'] + self.num_val_trials + self.num_eval_trials
        elif self.mode == 'val':
            seed = episode_config['eid'] + self.num_eval_trials
        else:
            seed = episode_config['eid']

        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.uid = f'{timestamp}-{str(uuid.uuid4().hex)}-{self.aid}'
        
        self.save_video = episode_config['save_video']
        self.episode_config = episode_config

        self.info = {}
        self.last_info = None

        self._env = suite.load(
            self.config.domain,
            self.config.task,
            task_kwargs={"random": seed},
        )
        self.action_step = 0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render(resolution=self.image_resolution)
        obs["is_terminal"] = False
        obs["is_first"] = True
        self.total_reward = 0
        self.clear_frames()
        return {
            'observation': obs,
            'arena_id': self.aid,
            'evaluation': self.evaluate()
        }

    def step(self, action):
        #print('action', action)
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)

        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if self.save_video:
                rgb = self.render(resolution=self.frame_resolution)
                self.video_frames.append(rgb)
            if time_step.last():
                break
        self.total_reward += reward
        obs = dict(time_step.observation)
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render(resolution=self.image_resolution)
        # There is no terminal state in DMC
        obs["is_terminal"] = False
        obs["is_first"] = False
        self.action_step += 1
        done = False
        if self.action_step >= self.action_horizon:
            done = True
            obs["is_terminal"] = True
        self.info = {
            'observation': obs,
            'reward': {'default': reward},
            'done': done,
            'discount': np.array(time_step.discount, np.float32),
            'arena_id': self.aid,
            'success': self.success(),
            'applied_action': action,
            'evaluation': self.evaluate()
        }
        return self.info

    def render(self, mode='rgb_array', resolution=None):
        if mode != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*resolution, camera_id=0)
    
    def evaluate(self):
        return {
            'return': self.total_reward
        }

    def success(self):
        # Example threshold; adjust for your task
        return self.total_reward >= self.action_horizon * 0.5
    
    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            if len(value.shape) == 0:
                shape = (1,)
            else:
                shape = value.shape
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self.image_resolution + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)


    def get_action_space(self):
        return self.action_space
    
    
    def get_no_op(self):
        # returns an action of all zeros, slightly safer than minimum/maximum edge values
        spec = self.action_space
        return np.zeros_like(spec.minimum, dtype=np.float32)
    
    def sample_random_action(self):
        spec = self.action_space
        # sample uniformly in the continuous range [minimum, maximum]
        return np.random.uniform(spec.minimum, spec.maximum).astype(np.float32)