import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from real_robot.utils.mask_utils import get_max_IoU

from .utils import *

SUCCESS_TRESHOLD = 0.05
IOU_TRESHOLDS = [0.8, 0.8, 0.82]


class RealWorldGarmentFoldingTask():
    def __init__(self, config):
        self.num_goals = config.num_goals
        self.name = config.fold_name
        self.goal_steps = config.goal_steps

        self.config = config
        self.demonstrator = config.demonstrator
        self.goals = [] # This needs to be loaded  or generated
        self.has_succeeded = False

        self.asset_dir = f"{os.environ['MP_FOLD_PATH']}/assets"
        
        

    def reset(self, arena):
        """Reset environment and generate goals if necessary."""    
        # Generate goals (10 small variations)
        self.goals = []
        self.goals = self._load_or_generate_goals(arena, self.num_goals)

        self.aligned_pairs = []
        self.has_succeeded = False

        return {"goals": self.goals}

    def _generate_a_goal(self, arena):
        """Generates trajectory and actions using the demonstrator."""
        goal_obs = []
        actions = [] # List to store step-wise actions
        
        print(f"\n*** ACTION REQUIRED ***")
        print(f"Please manually flatten the garment in the simulation window.")
        print(f"Ensure it is centered/aligned as desired.")
        input(f"Press [Enter] when the garment is ready to start demonstration...")
        print(f"[Goal Manager] Starting demonstrator collection...\n")
        
        measure_time = arena.measure_time
        arena.measure_time = False
        info = arena._process_info({})
        arena.info = info
        self.demonstrator.reset([arena.id])
        
        goal_obs.append(info)
        #print('Here!!!')
        step = 0
        while step < self.goal_steps: 
            # Capture action from demonstrator
            #print('ask action!!!')
            print(f'[Goal Manager] Current step {step}, goal_steps {self.goal_steps}')
            action = self.demonstrator.single_act(info) 
            actions.append(action)
            
            # Step environment
            info = arena.step(action)
            goal_obs.append(info)
            
            step += 1
        
        arena.measure_time = measure_time
        # Return both observations and actions
        return goal_obs, actions


    
    def _load_or_generate_goals(self, arena, num_goals):
        goals = []
        
        # --- 1. Ask User for Garment ID ---
        
        self.garment_id = arena.garment_id
        if not self.garment_id:
            self.garment_id = "default_garment"

        self.goal_dir = os.path.join(self.asset_dir, arena.name, self.name, self.garment_id, 'goals')


        # --- 4. Loop over goals ---
        for i in range(num_goals):
            goal_path = os.path.join(self.goal_dir, f"goal_{i}")
        

            if not os.path.exists(goal_path):
                print(f'Generating goal {i}...')

                print(f"\n[Goal Manager] Cached goals NOT found for '{self.garment_id}'.")
                print(f"[Goal Manager] Directory: {self.goal_dir}")
                
                
                # Pass manual_init=True for the first goal (since user just flattened it)
                # For subsequent goals (i > 0), if you want them to auto-reset or 
                # be variations, handle logic here. Assuming user only sets up once 
                # or the demonstrator loop handles reset. 
                # *If* you need the user to reset for *every* goal, move the input() inside this loop.
                # Here we assume user sets up once, and subsequent goals might be variations or same.
                
                # Using manual_init=True ensures we don't auto-reset the user's work
                goal_obs, actions = self._generate_a_goal(arena)
                
                os.makedirs(goal_path, exist_ok=True)

                for step_idx, subgoal in enumerate(goal_obs):
                    # Save RGB
                    plt.imsave(os.path.join(goal_path, f"rgb_step_{step_idx}.png"), 
                               subgoal['observation']['rgb']/255.0)
                    
                    # Save Action (step-wise)
                    if step_idx < len(actions):
                        np.save(os.path.join(goal_path, f"action_step_{step_idx}.npy"), actions[step_idx])

                # Construct return object
                current_goal = []
                for step_idx, obs in enumerate(goal_obs):
                    step_data = obs.copy()
                    if step_idx < len(actions):
                        step_data['action'] = actions[step_idx]
                    else:
                        step_data['action'] = None
                    current_goal.append(step_data)
                
                goals.append(current_goal)

            else:
                # Load existing
                if i == 0: print(f'[Goal Manager] Loading cached goals from {self.goal_dir}...')
                
                goal = []
                step_idx = 0
                while True:
                    rgb_p = os.path.join(goal_path, f"rgb_step_{step_idx}.png")
                    act_p = os.path.join(goal_path, f"action_step_{step_idx}.npy")

                    if not os.path.exists(rgb_p):
                        break

                    # Load Data
                    rgb = (plt.imread(rgb_p)*255).astype(np.uint8)
                    
                    action = None
                    if os.path.exists(act_p):
                        action = np.load(act_p, allow_pickle=True)

                    subgoal = {
                        'observation': {
                            'rgb': rgb[:, :, :3],
                        },
                        'action': action
                    }
                    goal.append(subgoal)
                    step_idx += 1
                    
                    if step_idx > self.config.goal_steps + 10: break # Safety break

                goals.append(goal)

        return goals
    
    def evaluate(self, arena):
        """Evaluate folding quality using particle alignment and semantic keypoints."""
        if len(self.goals) == 0:
            return {}
        
        return {
            'max_IoU': self._get_max_IoU(arena),
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
        }


    def reward(self, last_info, action, info): 
        

        
        #Multi stage reward
        if last_info == None:
            last_info = info
  
        if info['success']:
            if self.config.get('big_success_bonus', True):
                multi_stage_reward += self.config.goal_steps*(info['arena'].action_horizon - info['observation']['action_step'])
                pdr_ += (info['arena'].action_horizon - info['observation']['action_step'])
            else:
                multi_stage_reward = self.config.goal_steps
        else:

            # -------------------------------
            # IoU-based multi-stage reward
            # Sequential goal matching
            # -------------------------------

            arena = info['arena']
            trj_infos = arena.get_trajectory_infos()
            N = len(trj_infos)
            K = self.config.goal_steps

            # Always give some shaping based on current IoU to goal[0]
            multi_stage_reward = self._max_iou_for_goal_step(arena, trj_infos[-1], 0)

            K = min(K, N)
            # Need at least K trajectory steps to attempt full alignment
            
            traj_window = trj_infos[N - K : N]

            best_reward = multi_stage_reward

            # Try all possible start positions for goal[0]
            for start in range(K):
                matched_steps = 0
                last_iou = 0.0

                for g in range(K):
                    t = start + g
                    if t >= K:
                        break

                    iou = self._max_iou_for_goal_step(
                        arena,
                        traj_window[t],
                        g
                    )

                    #print(f"[reward] K={K} start={start}, step={t}, goal={g}, iou={iou:.3f}")

                    if iou >= IOU_TRESHOLDS[g]:
                        matched_steps += 1
                        last_iou = 0.0
                    else:
                        last_iou = iou
                        break

                    if t == K - 1:
                        reward = matched_steps + last_iou
                        best_reward = max(best_reward, reward)
                
            if self.config.get('base_reward', None) == 'aug-converage-alignment' and multi_stage_reward <= 1.0:
                multi_stage_reward = coverage_alignment_reward(last_info, action, info) # combination of delta NC and delta IOU
                if info['evaluation']['normalised_coverage'] > NC_FLATTENING_TRESHOLD and info['evaluation']['max_IoU_to_flattened'] > IOU_FLATTENING_TRESHOLD:
                    multi_stage_reward = 1

            multi_stage_reward = best_reward

        return {
            'multi_stage_reward': multi_stage_reward,
        }
    
    def _max_iou_for_goal_step(self, arena, traj_info, goal_step):
        """
        Compute max IoU between current trajectory step mask
        and all goal masks at a specific goal_step.
        """
        cur_mask = traj_info['observation']['mask']
        max_IoU = 0.0

        for goal in self.goals:
            goal_mask = goal[goal_step]['observation']['rgb'].sum(axis=2) > 0
            IoU, _ = get_max_IoU(cur_mask, goal_mask, debug=self.config.debug)
            max_IoU = max(max_IoU, IoU)

        return max_IoU

    def get_goals(self):
        return self.goals

    def get_goal(self):
        return self.goals[0]
    
    def success(self, arena):
        return False
        # trj_infos = arena.get_trajectory_infos()
        # N = len(trj_infos)
        # K = self.config.goal_steps
        # #print('goal steps', K)
        # if len(trj_infos) < K:
        #     return False
        
        # # if has succeed, check the current cloth is messed up or not
        # # if mess up, reset success and return False
        # # else return True
        # if self.has_succeeded:
        #     mask = trj_infos[-1]['observation']['mask']
        #     max_IoU = 0
        #     for goal in self.goals:
        #         goal_mask = goal[-1]['observation']["rgb"].sum(axis=2) > 0
        #         IoU, _ = get_max_IoU(mask, goal_mask, debug=self.config.debug)
        #         max_IoU = max(IoU, max_IoU)
        #     if max_IoU < IOU_TRESHOLDS[-1]:
        #         self.has_succeeded = False
        #         print('[folding task] Success is messed up!')
        #         return False
        #     else:
        #         print('[folding task] Successful Step!')
        #         return True
        
        # # if has not succeeded before, check consquent sub goals matches the trajecotry operation/
        # for k in range(K):
        #     mask  = trj_infos[N - K + k]['observation']['mask']
        #     max_IoU = 0
        #     for goal in self.goals:
        #         goal_mask = goal[k]['observation']["rgb"].sum(axis=2) > 0
        #         IoU, _ = get_max_IoU(mask, goal_mask, debug=self.config.debug)
        #         max_IoU = max(IoU, max_IoU)
        #     #print(f'goal step {k}, current step {N - K + k}: max_IoU: {max_IoU}')
        #     if max_IoU < IOU_TRESHOLDS[k]:
        #         return False
        # print('[folding task] Successful Step!')
        # self.has_succeeded = True
        # return True
        
    def _get_max_IoU(self, arena):
        cur_mask = arena.cloth_mask
        max_IoU = 0
        for goal in self.goals[:1]:
            goal_mask = goal[-1]['observation']["rgb"].sum(axis=2) > 0 ## only useful for background is black
            
            IoU, matched_IoU = get_max_IoU(cur_mask, goal_mask, debug=self.config.debug)
            if IoU > max_IoU:
                max_IoU = IoU
        
        return IoU
    
    def _get_max_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        IoU, matched_IoU = get_max_IoU(cur_mask, arena.get_flattened_obs()['observation']['mask'], debug=self.config.debug)
        
        return IoU
    
    def _get_normalised_coverage(self, arena):
        res = arena.coverage / arena.flatten_coverage
        
        # clip between 0 and 1
        return np.clip(res, 0, 1)

    def compare(self, results_1, results_2):
        threshold=0.95

        # --- Compute averages for results_1 ---
        score_1 = mean([ep["max_IoU"][-1] for ep in results_1])
        # --- Compute averages for results_2 ---
        score_2 = mean([ep["max_IoU"][-1] for ep in results_2])

        # --- Otherwise prefer higher score ---
        if score_1 > score_2:
            return 1
        elif score_1 < score_2:
            return -1
        else:
            return 0