import numpy as np
from actoris_harena.utilities.networks.utils import np_to_ts, ts_to_np
from .dataset import normalize_data, unnormalize_data

class DiffusionTransform:
    def __init__(self, config, stats):
        self.config = config
        self.device = self.config.get('device', 'cpu')
        self.stats = stats

    def __call__(self, data, train=True):
        ret_data = {}
        
        if not train:
            ret_data[self.config.input_obs] = data[self.config.input_obs].astype(np.float32) / 255.0
            
            if len(ret_data[self.config.input_obs].shape) == 3:
                ret_data[self.config.input_obs] = np.expand_dims(ret_data[self.config.input_obs], axis=0)
                ret_data[self.config.input_obs] = np.expand_dims(ret_data[self.config.input_obs], axis=0)

            ret_data[self.config.input_obs] = ret_data[self.config.input_obs].transpose(0, 1, 4, 2, 3)
            ret_data[self.config.input_obs] = np_to_ts(ret_data[self.config.input_obs], self.device)
            
            ret_data['vector_state'] = normalize_data(
                data['vector_state'], 
                self.stats[self.config.data_state]
            )
            ret_data['vector_state'] = np_to_ts(ret_data['vector_state'], self.device)
            
            if len(ret_data['vector_state'].shape) == 1:
                ret_data['vector_state'] = ret_data['vector_state'].unsqueeze(0)

        else:
            ret_data[self.config.input_obs] = data[self.config.data_obs]
            ret_data[self.config.input_obs] = np_to_ts(ret_data[self.config.input_obs][:, :self.config.obs_horizon], self.device)
            ret_data['vector_state'] = np_to_ts(data[self.config.data_state][:, :self.config.obs_horizon], self.device)
            ret_data['action'] = np_to_ts(data['action'], self.device)

        return ret_data

    def postprocess(self, data):
        ret_data = {}
        if 'action' in data.keys():
            data['action'] = unnormalize_data(data['action'], self.stats[self.config.data_action])
            ret_data['action'] = data['action']
            
        if self.config.input_obs in data.keys():
            ret_data[self.config.input_obs] = (ts_to_np(data[self.config.input_obs]) * 255.0).clip(0, 255).astype(np.uint8)
            
        return ret_data