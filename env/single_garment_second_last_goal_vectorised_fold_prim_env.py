import numpy as np

from .single_garment_second_last_goal_init_env import SingleGarmentSecondLastGoaInitEnv

# @ray.remote
class SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv(SingleGarmentSecondLastGoaInitEnv):
    
    def __init__(self, config):
        #config.name = f'single-garment-fixed-init-env'
        super().__init__(config)
        #self.name =f'single-garment-fixed-init-env'

    def step(self, action): 
        self.last_info = self.info
        self.evaluate_result = None
        self.overstretch = 0

        if isinstance(action, dict):
            dict_action = action
        else:
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