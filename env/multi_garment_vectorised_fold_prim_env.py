import numpy as np

from .multi_garment_env import MultiGarmentEnv

global ENV_NUM
ENV_NUM = 0

# @ray.remote
class MultiGarmentVectorisedFoldPrimEnv(MultiGarmentEnv):
    
    def __init__(self, config):
        super().__init__(config)

    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        self.overstretch = 0
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