from actoris_harena.agent import MPC_CEM
from actoris_harena.agent.utilities.utils import *
from actoris_harena.utilities.save_utils import *
import numpy as np
from gym.spaces import Box

class DualArmMaskPickAndPlaceMPC(MPC_CEM):
    def __init__(self, config):
        super().__init__(config)
        
        self.obj_mask = config.obj_mask
        self.debug = config.get('debug', False)
        self.place_orien = config.get('place_orien', False)
        self.pick_prien = config.get('pick_orien', False)
        self.apply_workspace = config.get('apply_workspace', False)
        
        if self.obj_mask == 'from_model':
            self.obj_mask_threshold = config.obj_mask_threshold

        self.name = 'Dual Arm Mask Pick-and-Place MPC'

    def get_name(self):
        return self.name + " on " + self.model.get_name()
    
    def get_phase(self):
        return 'flattening'

    def single_act(self, info, update=False):
        # Base actions: pick0(2), place0(2), pick1(2), place1(2) = 8
        self.A = 8 
        
        # Assuming orientation flags add 1 dimension per arm (2 total)
        if self.place_orien:
            self.A += 2
        if self.pick_prien:
            self.A += 2

        action_space = Box(low=-1, high=1, shape=(1, self.A), dtype=np.float32)
        num_elites = int(0.1 * self.candidates)
        plan_hor = self.planning_horizon
        assert plan_hor == 1, "plan_hor should be 1"

        mean = np.tile(np.zeros([1, self.A]).flatten(), [plan_hor]).reshape(plan_hor, -1)
        std = np.tile(np.ones([1, self.A]).flatten(), [plan_hor]).reshape(plan_hor, -1)

        if self.obj_mask == 'from_env':
            obj_mask = info['observation']['mask']
        elif self.obj_mask == 'from_model':
            obj_mask = self.model.reconstruct_observation(self.model.cur_state)
            obj_mask = obj_mask.reshape(*obj_mask.shape[-2:])
            obj_mask = obj_mask > self.obj_mask_threshold
        
        iteration_means = []           
        for i in range(self.iterations):
            popsize = self.candidates
            samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)]).reshape(popsize, plan_hor, -1)
            
            H, W = obj_mask.shape[:2] 
            assert H == W, "Obj mask should be square"

            # --- Extract Actions for Both Arms ---
            # pick0, place0
            pick0_actions = ((samples[:, 0, 0:2] + 1) * (H / 2)).astype(int).clip(0, H-1).reshape(self.candidates, -1)
            place0_actions = ((samples[:, 0, 2:4] + 1) * (H / 2)).astype(int).clip(0, H-1).reshape(self.candidates, -1)
            
            # pick1, place1
            pick1_actions = ((samples[:, 0, 4:6] + 1) * (H / 2)).astype(int).clip(0, H-1).reshape(self.candidates, -1)
            place1_actions = ((samples[:, 0, 6:8] + 1) * (H / 2)).astype(int).clip(0, H-1).reshape(self.candidates, -1)

            # --- Validate Indices Against Masks ---
            if self.config.swap_action:
                # (y, x) indexing
                valid_pick0 = obj_mask[pick0_actions[:, 1], pick0_actions[:, 0]] == 1
                valid_pick1 = obj_mask[pick1_actions[:, 1], pick1_actions[:, 0]] == 1
                
                if self.apply_workspace:
                    mask0 = info['observation']['robot0_mask']
                    # Assuming there is a robot1_mask. If not, fallback to robot0_mask
                    mask1 = info['observation'].get('robot1_mask', mask0) 
                    valid_place0 = mask0[place0_actions[:, 1], place0_actions[:, 0]] == 1
                    valid_place1 = mask1[place1_actions[:, 1], place1_actions[:, 0]] == 1
            else:
                # (x, y) indexing
                valid_pick0 = obj_mask[pick0_actions[:, 0], pick0_actions[:, 1]] == 1
                valid_pick1 = obj_mask[pick1_actions[:, 0], pick1_actions[:, 1]] == 1
                
                if self.apply_workspace:
                    mask0 = info['observation']['robot0_mask']
                    mask1 = info['observation'].get('robot1_mask', mask0)
                    valid_place0 = mask0[place0_actions[:, 0], place0_actions[:, 1]] == 1
                    valid_place1 = mask1[place1_actions[:, 0], place1_actions[:, 1]] == 1

            # Both arms must make a valid pick
            valid_indices = valid_pick0 & valid_pick1
            
            # Both arms must make a valid place (if applicable)
            if self.apply_workspace:
                valid_indices &= valid_place0 & valid_place1

            # --- Filter and Evaluate ---
            samples = samples[valid_indices]
            popsize = samples.shape[0]

            # Failsafe in case all samples are invalid to prevent arg-sort crashing
            if popsize == 0:
                break 

            if self.clip:
                samples = np.clip(samples, action_space.low[:1], action_space.high[:1])
            
            costs, _ = self._predict_and_eval(samples, info, 
                    goal=(info['goals'] if self.goal_condition else None))
            
            elites = samples[np.argsort(costs)][:num_elites]
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0)
            mean, std = new_mean, new_std
            
            iteration_means.append(mean.copy())
        
        ret_act = np.clip(mean.reshape(plan_hor, *(1, self.A))[0], action_space.low[:1], action_space.high[:1])[0]

        cost = self._predict_and_eval(np.array([ret_act]), info, goal=(info['goals'] if self.goal_condition else None))[0][0]
        
        self.internal_states[info['arena_id']] = {
            'action_cost': cost,
            'iteration_means': np.stack(iteration_means) if iteration_means else np.array([]),
            'last_samples': samples,
            'last_costs': costs if 'costs' in locals() else np.array([]),
            'pick-mask': np.expand_dims(obj_mask, axis=2),
        }
        
        return ret_act