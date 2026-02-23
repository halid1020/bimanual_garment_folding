from actoris_harena.agent import MPC_CEM
from actoris_harena.agent.utilities.utils import *
from actoris_harena.utilities.save_utils import *
import numpy as np
from gym.spaces import Box

class ClothMaskWorkspacePickAndPlaceMPC(MPC_CEM):
    def __init__(self, config):

        super().__init__(config)
        #self.flatten_threshold =  kwargs['flatten_threshold']
        self.cloth_mask = config.cloth_mask
        self.debug = config.get('debug', False)
        #self.workspace_mask = config.workspace_mask
        #self.no_op = kwargs['no_op']
        if self.cloth_mask == 'from_model':
            self.cloth_mask_threshold = config.cloth_mask_threshold

        self.name = 'Cloth-Mask Workspace Pick-and-Place MPC'


    def get_name(self):
        return self.name + " on " + self.model.get_name()
    
    def get_phase(self):
        return 'flattening'


    def single_act(self, info, update=False):
        action_space = Box(low=-1, high=1, shape=(1, 4), dtype=np.float32)
        num_elites = int(0.1 * self.candidates)
        plan_hor = self.planning_horizon
        assert plan_hor == 1, "plan_hor should be 1"

        mean = np.tile(np.zeros([1, 4]).flatten(), [plan_hor]).reshape(plan_hor, -1)
        std = np.tile(np.ones([1, 4]).flatten(), [plan_hor]).reshape(plan_hor, -1)

        if self.cloth_mask == 'from_env':
            cloth_mask = info['observation']['cloth-workspace-mask']
            if self.debug:
                save_mask(cloth_mask, 'planning-cloth-workspace-mask',  './tmp')
                save_mask(info['observation']['mask'], 'planning-cloth-mask', './tmp')
                save_mask(info['observation']['robot0_mask'], 'planning-workspace-mask', './tmp')
        elif self.cloth_mask == 'from_model':
            cloth_mask = self.model.reconstruct_observation(self.model.cur_state)
            cloth_mask = cloth_mask.reshape(*cloth_mask.shape[-2:])
            cloth_mask = cloth_mask > self.cloth_mask_threshold
        workspace_mask = info['observation']['robot0_mask']
        #print('mean shape', mean.shape)
        iteration_means = []           
        for i in range(self.iterations):
            popsize = self.candidates
            samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)]).reshape(popsize, plan_hor, -1)
            
            H, W = cloth_mask.shape[:2] 
            assert H == W, "Cloth mask should be square"

            first_pick_actions = ((samples[:, 0, :2] + 1) * (H / 2)).astype(int)
            first_pick_actions = first_pick_actions.astype(int).clip(0, H-1).reshape(self.candidates, -1)
            place_actions = ((samples[:, :, 2:] + 1) * (H / 2)).astype(int)
            place_actions = place_actions.astype(int).clip(0, H-1).reshape(self.candidates, -1)

            if self.config.swap_action:
                #print('swap action triggered')
                valid_indices_for_pick = cloth_mask[first_pick_actions[:, 1], first_pick_actions[:, 0]] == 1
                valid_indices_for_place = workspace_mask[place_actions[:, 1], place_actions[:, 0]] == 1
            else:
                valid_indices_for_pick = cloth_mask[first_pick_actions[:, 0], first_pick_actions[:, 1]] == 1
                valid_indices_for_place = workspace_mask[place_actions[:, 0], place_actions[:, 1]] == 1

            valid_indices = valid_indices_for_pick & valid_indices_for_place
            samples = samples[valid_indices]
            popsize = samples.shape[0]

            if self.clip:
                samples = np.clip(samples, action_space.low[:1], action_space.high[:1])
                
            costs, _ = self._predict_and_eval(samples, info, 
                    goal=(info['goals'] if self.goal_condition else None))
            elites = samples[np.argsort(costs)][:num_elites]
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0)
            mean, std = new_mean, new_std
            #print('mean shape', mean.shape)
            iteration_means.append(mean.copy())
        
        

        ret_act = np.clip(mean.reshape(plan_hor, *(1, 4))[0], action_space.low[:1], action_space.high[:1])[0]
        ## evluate the cost of the action
        cost = self._predict_and_eval(np.array([ret_act]), info, goal=(info['goals'] if self.goal_condition else None))[0][0]
        
        self.internal_states[info['arena_id']] = {
            'action_cost': cost,
            'iteration_means': np.stack(iteration_means),
            'last_samples': samples,
            'last_costs': costs,
            'pick-mask': np.expand_dims(cloth_mask, axis=2),
            'place-mask': np.expand_dims(workspace_mask, axis=2)
        }
            #print('staet', state)
        
        return ret_act