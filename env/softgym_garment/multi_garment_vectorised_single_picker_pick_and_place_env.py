import numpy as np
import gym
from .multi_garment_env import MultiGarmentEnv
from .multi_garment_env import MultiGarmentEnv
from .pixel_based_pick_and_place_env_logger import PixelBasedPickAndPlaceEnvLogger

global ENV_NUM
ENV_NUM = 0

class MultiGarmentVectorisedSinglePickerPickAndPlaceEnv(MultiGarmentEnv):
    
    def __init__(self, config):
        super().__init__(config)
        self.action_space = gym.spaces.Box(-1, 1, (4, ), dtype=np.float32)
        self.logger = PixelBasedPickAndPlaceEnvLogger()
        

    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        self.overstretch = 0
        dict_action = {
            'norm-pixel-pick-and-place': {
                'pick_0': action[:2],
                'place_0': action[2:4]
            }
        }

        self.info = self.action_tool.step(self, dict_action)
        self.action_step += 1
        self.all_infos.append(self.info)
        self.info = self._process_info(self.info)
        
        applied_action = self.info['applied_action']['norm-pixel-pick-and-place']

        self.info['applied_action'] = applied_action
        self.info['observation']['is_first'] = False
        self.info['observation']['is_terminal'] = self.info['done']
        return self.info
    
import os
import ray

@ray.remote(num_gpus=0.05)
class MultiGarmentVectorisedSinglePickerPickAndPlaceEnvRay(MultiGarmentVectorisedSinglePickerPickAndPlaceEnv):
    
    def __init__(self, config):
        super().__init__(config)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU ID
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Make CUDA calls synchronous