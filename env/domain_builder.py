import os

from dotmap import DotMap
from ..envs.cloth_funnel_env import ClothFunnelEnv, ClothFunnelEnvRay
from ..tasks.garments.garment_flattening import GarmentFlatteningTask
## action:pixel-pick-and-place, initial:flatten, domain:mono-single-tshirt, task:flattening



class DomainBuilder():
    
        
    def build_from_config(domain, task, disp, horizon=8, rayid=0, ray=False):
        object = domain.split('-')[-1]
        realadapt = domain.split('-')[-2]
        

        #TODO: set the correct config
        config = {
            'object': object,
            'picker_radius': 0.03, #0.015,
            # 'particle_radius': 0.00625,
            'picker_threshold': 0.007, # 0.05,
            'picker_low': (-5, 0, -5),
            'picker_high': (5, 5, 5),
            'grasp_mode': {'closest': 1.0},
            "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
            'init_state_path': os.path.join(
                os.environ['AGENT_ARENA_PATH'], '..', 'data', 'cloth_funnel', 'init_states'),
            'task': task,
            'disp': True if disp == 'True' else False,
            'ray_id': rayid,
            'horizon': int(horizon),
        }
        
        if realadapt == 'realadapt':
            config['grasp_mode'] = {
                'around': 0.9,
                'miss': 0.1
            }

        config = DotMap(config)
        if task == 'flattening':
            task = GarmentFlatteningTask()
        elif task == 'center-sleeve-fold':
            assert 'tshirt' in 'domain'
            task = CentreSleeveFoldingTask()
        else:
            raise NotImplementedError
        
        if ray:
            env = ClothFunnelEnvRay.remote(config)
            env.set_task.remote(task)
        else:
            env = ClothFunnelEnv(config)
            env.set_task(task)
        return env