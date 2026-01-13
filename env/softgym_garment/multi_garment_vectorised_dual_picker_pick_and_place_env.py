import numpy as np
import gym
import random 

from .multi_garment_env import MultiGarmentEnv
from .multi_garment_env import MultiGarmentEnv
from .pixel_based_pick_and_place_env_logger import PixelBasedPickAndPlaceEnvLogger

global ENV_NUM
ENV_NUM = 0

# @ray.remote
class MultiGarmentVectorisedDualPickerPickAndPlaceEnv(MultiGarmentEnv):
    
    def __init__(self, config):
        super().__init__(config)
        self.action_space = gym.spaces.Box(-1, 1, (8, ), dtype=np.float32)
        self.logger = PixelBasedPickAndPlaceEnvLogger()
        

    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        self.overstretch = 0
        dict_action = {
            'norm-pixel-pick-and-place': {
                'pick_0': action[:2],
                'pick_1': action[2:4],
                'place_0': action[4:6],
                'place_1': action[6:],

            }
        }

        self.info = self.action_tool.step(self, dict_action)
        self.action_step += 1
        self.all_infos.append(self.info)
        self.info = self._process_info(self.info)
        applied_action = self.info['applied_action']['norm-pixel-pick-and-place']
        #vector_action = []
        if random.random() < 0.5:
            applied_action = applied_action.reshape(-1, 2)
            applied_action[[0, 1, 2, 3]] = applied_action[[1, 0, 3, 2]]
            applied_action = applied_action.flatten()

        #vector_action = np.stack(vector_action).flatten()

        self.info['applied_action'] = applied_action
        self.info['observation']['is_first'] = False
        self.info['observation']['is_terminal'] = self.info['done']
        return self.info