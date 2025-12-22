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
        dict_applied_action = self.info['applied_action']
        vector_action = []
        if random.random() < 0.5:
            for param_name in ['pick_0', 'pick_1', 'place_0', 'place_1']:
                vector_action.append(dict_applied_action['norm-pixel-pick-and-place'][param_name])
        else:
            for param_name in ['pick_1', 'pick_0', 'place_1', 'place_0']:
                vector_action.append(dict_applied_action['norm-pixel-pick-and-place'][param_name])
        #print('vector_action', vector_action)
        vector_action = np.stack(vector_action).flatten()

        self.info['applied_action'] = vector_action
        self.info['observation']['is_first'] = False
        self.info['observation']['is_terminal'] = self.info['done']
        return self.info