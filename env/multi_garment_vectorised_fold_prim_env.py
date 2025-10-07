import os
import h5py
import numpy as np
import cv2
import json

from softgym.action_space.action_space import Picker
from softgym.utils.env_utils import get_coverage
import pyflex
from agent_arena import Arena
from tqdm import tqdm


from .action_primitives.hybrid_action_primitive import HybridActionPrimitive
from .garment_env_logger import GarmentEnvLogger
from .utils.env_utils import set_scene
from .utils.camera_utils import get_camera_matrix
from .multi_garment_env import MultiGarmentEnv

global ENV_NUM
ENV_NUM = 0

# @ray.remote
class MultiGarmentVectorisedFoldPrimEnv(MultiGarmentEnv):
    
    def __init__(self, config):
        #config.name = f'single-garment-fixed-init-env'
        super().__init__(config)
        #self.name =f'single-garment-fixed-init-env'

    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None

        dict_action = {
            'norm-pixel-fold': {
                'pick_0': action[:2],
                'pick_1': action[2:4],
                'place_0': action[4:6],
                'place_1': action[6:8]
            }
        }

        info = self.action_tool.step(self, dict_action)
        self.action_step += 1
        self.info = self._process_info(info)
        dict_applied_action = self.info['applied_action']
        vector_action = []
        for param_name in ['pick_0', 'pick_1', 'place_0', 'place_1']:
            vector_action.append(dict_action['norm-pixel-fold'][param_name])
        #print('vector_action', vector_action)
        vector_action = np.stack(vector_action).flatten()

        self.info['applied_action'] = vector_action
        return self.info