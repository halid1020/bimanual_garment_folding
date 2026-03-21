from actoris_harena.agent import MPC_CEM
from actoris_harena.agent.utilities.utils import *
from actoris_harena.utilities.save_utils import *
import numpy as np
from gym.spaces import Box
import os      # <-- Added for directory creation
import time    # <-- Added for unique filenames
import cv2     # <-- Added for saving the image (you can also use PIL or matplotlib)
from env.softgym_garment.draw_utils import draw_pick_and_place

class SingleArmMaskPickAndPlaceMPC(MPC_CEM):
    def __init__(self, config):
        super().__init__(config)
        
        self.obj_mask = config.obj_mask
        self.debug = config.get('debug', False)
        self.place_orien = config.get('place_orien', False)
        self.pick_prien = config.get('pick_orien', False)
        self.apply_workspace = config.get('apply_workspace', False)
        
        if self.obj_mask == 'from_model':
            self.obj_mask_threshold = config.obj_mask_threshold

        self.name = 'Mask Pick-and-Place MPC'

    def get_name(self):
        return self.name + " on " + self.model.get_name()
    
    def get_phase(self):
        return 'flattening'

    def single_act(self, info, update=False):
        self.A = 4
        if self.place_orien:
            self.A += 1
        if self.pick_prien:
            self.A += 1

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
        
        if self.debug:
            os.makedirs('tmp/planning', exist_ok=True)
            # Create a unique filename using a timestamp
            filename = f"tmp/planning/obj_mask.png"
            
            # Assuming the mask is binary (0 and 1), multiply by 255 to make it visible
            mask_image = (obj_mask * 255).astype(np.uint8)
            cv2.imwrite(filename, mask_image)
        
        # ---------------------------------------------------------------------
        # DEBUGGING: Save the workspace mask here, before the iterations loop
        # ---------------------------------------------------------------------
        if self.apply_workspace:
            workspace_mask = info['observation']['robot0_mask']
            obj_mask &= workspace_mask
            
            # Only save if debug mode is on (optional, but good practice)
            if self.debug:
                os.makedirs('tmp/planning', exist_ok=True)
                
                # Create a unique filename using a timestamp
                filename = f"tmp/planning/robot0_mask.png"
                
                # Assuming the mask is binary (0 and 1), multiply by 255 to make it visible
                mask_image = (workspace_mask * 255).astype(np.uint8)
                cv2.imwrite(filename, mask_image)


                 
                
                # Alternative: If you want to save the raw NumPy array instead of an image
                # np.save(f"tmp/planning/robot0_mask_{int(time.time() * 1000)}.npy", workspace_mask)
        # ---------------------------------------------------------------------

        iteration_means = []           
        for i in range(self.iterations):
            popsize = self.candidates
            samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)]).reshape(popsize, plan_hor, -1)
            H, W = obj_mask.shape[:2] 
            assert H == W, "Obj mask should be square"

            first_pick_actions = ((samples[:, 0, :2] + 1) * (H / 2)).astype(int)
            first_pick_actions = first_pick_actions.astype(int).clip(0, H-1).reshape(self.candidates, -1)
            place_actions = ((samples[:, :, 2:4] + 1) * (H / 2)).astype(int)
            place_actions = place_actions.astype(int).clip(0, H-1).reshape(self.candidates, -1)

            if self.config.swap_action:
                valid_indices_for_pick = obj_mask[first_pick_actions[:, 1], first_pick_actions[:, 0]] == 1
                if self.apply_workspace:
                    valid_indices_for_place = workspace_mask[place_actions[:, 1], place_actions[:, 0]] == 1
            else:
                valid_indices_for_pick = obj_mask[first_pick_actions[:, 0], first_pick_actions[:, 1]] == 1
                if self.apply_workspace:
                    valid_indices_for_place = workspace_mask[place_actions[:, 0], place_actions[:, 1]] == 1

            valid_indices = valid_indices_for_pick
            
            if self.apply_workspace:
                valid_indices &= valid_indices_for_place

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
            
            iteration_means.append(mean.copy())
        
        ret_act = np.clip(mean.reshape(plan_hor, *(1, self.A))[0], action_space.low[:1], action_space.high[:1])[0]

        if self.debug:
            rgb = info['observation']['rgb']
            act_image = draw_pick_and_place(rgb, ret_act)
            os.makedirs('tmp/planning', exist_ok=True)
                
                # Create a unique filename using a timestamp
            filename = f"tmp/planning/planned_action.png"
                
            cv2.imwrite(filename, act_image)

        cost = self._predict_and_eval(np.array([ret_act]), info, goal=(info['goals'] if self.goal_condition else None))[0][0]
        
        self.internal_states[info['arena_id']] = {
            'action_cost': cost,
            'iteration_means': np.stack(iteration_means),
            'last_samples': samples,
            'last_costs': costs,
            'pick-mask': np.expand_dims(obj_mask, axis=2),
            # 'place-mask': np.expand_dims(workspace_mask, axis=2)
        }
        
        return ret_act