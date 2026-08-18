from actoris_harena.agent import MPC_CEM
from actoris_harena.agent.utilities.utils import *
from actoris_harena.utilities.save_utils import *
import numpy as np
from gym.spaces import Box
import os      # <-- Added for directory creation
import time    # <-- Added for unique filenames
import cv2     # <-- Added for saving the image (you can also use PIL or matplotlib)
from scipy.ndimage import distance_transform_edt
from env.softgym_garment.draw_utils import draw_pick_and_place

class SingleArmMaskPickAndPlaceMPC(MPC_CEM):
    def __init__(self, config):
        super().__init__(config)
        
        self.obj_mask = config.obj_mask
        self.debug = config.get('debug', False)
        self.place_orien = config.get('place_orien', False)
        self.pick_prien = config.get('pick_orien', False)
        self.apply_workspace = config.get('apply_workspace', False)
        self.constrain_actions = config.get('constrain_actions', True)
        # Mask rejection normally applies to the first planned action only. With
        # constrain_all_steps, later actions of an H > 1 plan are additionally
        # constrained to the cloth mask the model itself predicts for that step.
        self.constrain_all_steps = config.get('constrain_all_steps', False)
        self.pred_mask_threshold = config.get('pred_mask_threshold', 0.5)

        if self.obj_mask == 'from_model':
            self.obj_mask_threshold = config.obj_mask_threshold

        self.name = 'Mask Pick-and-Place MPC'

    def get_name(self):
        return self.name + " on " + self.model.get_name()
    
    def get_phase(self):
        return 'flattening'

    def _predicted_step_masks(self, mean, info, workspace_mask=None):
        """Cloth masks the model predicts along the current CEM mean plan.

        Returns a list indexed by planned action k >= 1; entry k - 1 is the mask
        of the state reached after actions 0..k-1, i.e. the state at which
        action k would be picked. Decoding along the mean plan rather than per
        candidate keeps this to one extra decoder call per CEM iteration.
        """
        plan_hor = self.planning_horizon
        if plan_hor < 2:
            return []
        traj = self.model.unroll_action_from_cur_state(
            mean.reshape(1, plan_hor, self.A), info)
        recon = self.model.reconstruct_observation(traj)   # horizon x 1 x 1 x H x W
        recon = recon.reshape(plan_hor, *recon.shape[-2:])

        masks = []
        for k in range(plan_hor - 1):
            mask = recon[k] > self.pred_mask_threshold
            if workspace_mask is not None:
                mask = mask & workspace_mask
            masks.append(mask if mask.any() else None)
        return masks

    def _project_to_mask(self, pixels, mask):
        """Snap out-of-mask pixels to the nearest valid pixel of `mask`."""
        valid = mask[pixels[:, 0], pixels[:, 1]] == 1
        if valid.all():
            return pixels
        # Nearest valid pixel for every pixel of the grid, computed once.
        _, indices = distance_transform_edt(~mask, return_indices=True)
        bad = ~valid
        rows, cols = pixels[bad, 0], pixels[bad, 1]
        pixels = pixels.copy()
        pixels[bad, 0] = indices[0][rows, cols]
        pixels[bad, 1] = indices[1][rows, cols]
        return pixels

    def _constrain_later_steps(self, samples, mean, info, workspace_mask=None):
        """Apply the predicted-mask constraint to planned actions k >= 1.

        Rejection is not usable here: surviving a mask test at every step of an
        H-step plan leaves roughly p^H of the population, which empties well
        before H = 5. We project instead, which enforces the constraint whilst
        preserving the population size. Step 0 keeps plain rejection, so H = 1
        is unaffected by this path.
        """
        masks = self._predicted_step_masks(mean, info, workspace_mask)
        for k in range(1, self.planning_horizon):
            mask = masks[k - 1]
            if mask is None:
                continue
            side = mask.shape[0]
            pick = ((samples[:, k, :2] + 1) * (side / 2)).astype(int)
            pick = pick.clip(0, side - 1)
            if self.config.swap_action:
                pick = self._project_to_mask(pick[:, ::-1], mask)[:, ::-1]
            else:
                pick = self._project_to_mask(pick, mask)
            samples[:, k, :2] = (pick + 0.5) * (2.0 / side) - 1.0
        return samples

    def single_act(self, info, update=False):
        self.A = 4
        if self.place_orien:
            self.A += 1
        if self.pick_prien:
            self.A += 1

        action_space = Box(low=-1, high=1, shape=(1, self.A), dtype=np.float32)
        num_elites = int(0.1 * self.candidates)
        plan_hor = self.planning_horizon

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
        iteration_means = []
        costs = []           
        for i in range(self.iterations):
            popsize = self.candidates
            samples = np.stack([np.random.normal(mean, std) for _ in range(popsize)]).reshape(popsize, plan_hor, -1)

            if self.constrain_actions:
                H, W = obj_mask.shape[:2]
                assert H == W, "Obj mask should be square"

                # Mask constraints apply to the first planned action only; the cloth
                # state after step 1 is unknown, so later actions are left unconstrained.
                first_pick_actions = ((samples[:, 0, :2] + 1) * (H / 2)).astype(int)
                first_pick_actions = first_pick_actions.clip(0, H-1).reshape(popsize, -1)
                place_actions = ((samples[:, 0, 2:4] + 1) * (H / 2)).astype(int)
                place_actions = place_actions.clip(0, H-1).reshape(popsize, -1)

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

                if popsize == 0:
                    break

            if self.constrain_all_steps and plan_hor > 1 and samples.shape[0] > 0:
                samples = self._constrain_later_steps(
                    samples, mean, info,
                    workspace_mask if self.apply_workspace else None)

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

        final_plan = np.clip(mean.reshape(1, plan_hor, self.A), action_space.low[:1], action_space.high[:1])
        cost = self._predict_and_eval(final_plan, info, goal=(info['goals'] if self.goal_condition else None))[0][0]
        
        self.internal_states[info['arena_id']] = {
            'action_cost': cost,
            'iteration_means': np.stack(iteration_means) if len(iteration_means) > 0 else [],
            'last_samples': samples,
            'last_costs': costs,
            'pick-mask': np.expand_dims(obj_mask, axis=2),
            # 'place-mask': np.expand_dims(workspace_mask, axis=2)
        }
        
        return ret_act