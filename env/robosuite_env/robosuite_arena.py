import numpy as np
import gym
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from agent_arena import Arena  # your abstract base
from omegaconf import OmegaConf
from ..video_logger import VideoLogger
from statistics import mean
# from .osc_controller import OperationalSpaceController

def to_dict(obj):
    """
    Safely convert an OmegaConf DictConfig, DotMap, or dict-like object
    into a standard Python dict (recursively).

    Works for:
      - OmegaConf.DictConfig (Hydra configs)
      - DotMap
      - regular dicts
    """
    # Try OmegaConf first
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass

    # Then DotMap
    try:
        from dotmap import DotMap
        if isinstance(obj, DotMap):
            return obj.toDict()
    except Exception:
        pass

    # Finally, assume plain dict or convertible
    if isinstance(obj, dict):
        return obj

    # Fallback: try constructor
    try:
        return dict(obj)
    except Exception:
        return obj


class RoboSuiteArena(Arena):
    def __init__(self, config):
        self.num_eval_trials = 30
        self.num_train_trials = 1000
        self.num_val_trials = 5

        super().__init__(config)
        self.config = config
        self.name = config.get("name", "robosuite_arena")
        self.sim_horizon = config.get("sim_horizon", 500)
        self.env_name = config.get("task_name", "Lift")  # Fix key
        self.use_camera_obs = config.get("use_camera_obs", False)
        self.has_renderer = config.get("disp", False)
        self.control_freq = config.get("control_freq", 20)
        self.resolution = config.get("resolution", (256, 256))
        self.obs_keys = config.get("obs_keys", None)

        # env_kwargs = to_dict(config.get("env_kwargs", {}))
        # env_kwargs['controller_configs'] = \
        #     suite.load_controller_config(default_controller=self.config["controller_name"])
        # Initialize robosuite environment

        # Initialize robosuite environment (no ObservationConfig in v1.4.1)
        self.env = GymWrapper(
            suite.make(
                env_name=self.env_name,
                has_renderer=self.has_renderer,
                has_offscreen_renderer=True,
                use_camera_obs=self.use_camera_obs,
                use_object_obs=True,   # object info
                control_freq=self.control_freq,
                reward_shaping=True,
                horizon=self.sim_horizon,
                robots=self.config.robots,
                controller_configs=suite.load_controller_config(
                    default_controller=self.config["controller_name"]
                )
            ),
        )

        # Store observation filtering preferences from YAML (optional)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.logger = VideoLogger()
    
    
    def _filter_observation(self, obs_dict):
        """
        Filter the observation dictionary according to self.obs_filter.
        Works in robosuite 1.4.1, where ObservationConfig isn't available.
        """
        if self.obs_keys is None:
            return None
        return np.concatenate([obs_dict[k] for k in self.obs_keys])


    # ------------------------
    # Required Abstract Methods
    # ------------------------


    def reset(self, episode_config= None):
        
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

        episode_config['eid'] = episode_config['eid']
        self.eid = episode_config['eid']
        self.save_frames = episode_config['save_video']

        obs_, _ = self.env.reset(seed=self.eid)
        obs_dict = self.env.env._get_observations(force_update=True)
        obs = self._filter_observation(obs_dict)
        #print('obs len', len(obs))
        if obs is None:
            obs = obs_


        self.clear_frames()
        self.sim_step = 0
        
        info = {
            'observation': {},
            "arena": self,
            "done": False,
            "reward": {'default': 0.0},
            "arena_id": self.id,
            "evaluation": {}
        }
        if self.use_camera_obs:
            info['observation']['rgb'] = obs
        else:
            info['observation']['state'] = obs.astype(np.float32)
        self.info = info
        return info

    def get_observation_space(self):
        return self.observation_space

    def step(self, action):
        #print('reward', self.env.reward())
        act_success = True
        try:
            obs_, reward, done, truncated, env_info = self.env.step(action)
            obs_dict = self.env.env._get_observations(force_update=True)
            obs = self._filter_observation(obs_dict)
            if obs is None:
                obs = obs_

            #obs = self._filter_observation(obs)
        except Exception:
            act_success = False
            reward = 0


        #print('obs', obs)
        info = {
            "observation": {},
            "reward": {"default": reward},
            "done": done or truncated,
            "arena": self,
            "arena_id": self.id,
            "applied_action": action,
            "evaluation": {},
            "success": self.success(),
            "fail_step": not act_success,
            'sim_steps': 1,
        }

        if self.use_camera_obs:
            info['observation']['rgb'] = obs
        else:
            info['observation']['state'] = obs.astype(np.float32)

        if self.save_frames:
            frame = self.env.sim.render(camera_name="frontview", width=self.resolution[0], height=self.resolution[1])
            frame = np.flipud(frame)
            self.video_frames.append(frame)

        self.sim_step += 1
        self.info = info
        return info

    def get_frames(self):
        return self.video_frames

    def clear_frames(self):
        self.video_frames = []

    def get_goal(self):
        # RoboSuite tasks have implicit goals (e.g., lifting cube to height)
        if hasattr(self.env, "goal"):
            return self.env.goal
        return {"description": f"Complete task {self.env_name}"}

    def get_action_space(self):
        return self.action_space

    def sample_random_action(self):
        return self.action_space.sample()

    def get_no_op(self):
        return np.zeros_like(self.action_space.sample())

    def get_action_horizon(self):
        return self.sim_horizon

    # Optional override
    def success(self):
        if hasattr(self.env, "check_success"):
            return self.env.check_success()
        # Fallback heuristic
        success_ = self.env._check_success() if hasattr(self.env, "_check_success") else False
        return success_
    
    def evaluate(self):
        # Standardized evaluation metric
        return {"episode_reward": self.env.reward()}

    def compare(self, results_1, results_2):
        # Compare based on reward or success rate
    
        avg_last_reward_1 = mean([sum(ep["episode_reward"]) for ep in results_1])
        avg_last_reward_2 = mean([sum(ep["episode_reward"]) for ep in results_2])

        return avg_last_reward_1 - avg_last_reward_2
