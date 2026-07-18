import os
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from statistics import mean

from real_robot.utils.mask_utils import get_max_IoU, calculate_iou
from termcolor import colored

from .utils import *
from real_robot.utils.save_utils import *

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
        
        print(colored(f"\n*** [RealWorldGarmentFoldingTask] ACTION REQUIRED ***", "green"))
        print(colored(f"[RealWorldGarmentFoldingTask] Please manually flatten the garment in the simulation window.", "green"))
        print(colored(f"[RealWorldGarmentFoldingTask] Ensure it is centered/aligned as desired.", "green"))
        input(colored(f"[RealWorldGarmentFoldingTask] Press [Enter] when the garment is ready to start demonstration...", "green"))
        print(colored(f"[RealWorldGarmentFoldingTask] Starting demonstrator collection...\n", "green"))
        
        measure_time = arena.measure_time
        arena.measure_time = False
        info = arena._process_info({})
        arena.info = info
        self.demonstrator.reset([arena.id])
        
        goal_obs.append(info)
        step = 0
        while step < self.goal_steps: 
            # Capture action from demonstrator
            #print('ask action!!!')
            print(f'[RealWorldGarmentFoldingTask] Current step {step}, goal_steps {self.goal_steps}')
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
        
        self.garment_id = arena.garment_id
        if not self.garment_id:
            self.garment_id = "default_garment"

        self.goal_dir = os.path.join(self.asset_dir, arena.name, self.name, self.garment_id, 'goals')


        # --- 4. Loop over goals ---
        for i in range(num_goals):
            goal_path = os.path.join(self.goal_dir, f"goal_{i}")
        

            if not os.path.exists(goal_path):
                print(f'Generating goal {i}...')

                print(f"\n[RealWorldGarmentFoldingTask] Cached goals NOT found for '{self.garment_id}'.")
                print(f"[RealWorldGarmentFoldingTask] Directory: {self.goal_dir}")
                
                
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
                    # plt.imsave(os.path.join(goal_path, f"rgb_step_{step_idx}.png"), 
                    #            subgoal['observation']['rgb']/255.0)
                    
                    save_colour(subgoal['observation']['rgb'], 
                                f"rgb_step_{step_idx}",
                                goal_path)
                    save_depth(subgoal['observation']['depth'], 
                                f"depth_step_{step_idx}",
                                goal_path)
                    
                    save_mask(subgoal['observation']['mask'],
                                f"mask_step_{step_idx}",
                                goal_path)
                    
                    if step_idx < len(actions):
                        # Save as JSON (preferred per TODO)
                        save_action_json(actions[step_idx], f"action_step_{step_idx}", goal_path)

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
                if i == 0: print(f'[RealWorldGarmentFoldingTask] Loading cached goals from {self.goal_dir}...')
                
                goal = []
                step_idx = 0
                while True:
                    rgb_filename = f"rgb_step_{step_idx}"
                    if not os.path.exists(os.path.join(goal_path, rgb_filename + ".png")):
                        break

                    # Load Images
                    rgb = load_colour(rgb_filename, goal_path)
                    depth = load_depth(f"depth_step_{step_idx}", goal_path)
                    # New naming is `mask_step_{j}`; fall back to the legacy
                    # `masks_step_{j}` for backward compatibility with old caches.
                    mask = load_mask(f"mask_step_{step_idx}", goal_path)
                    if mask is None:
                        mask = load_mask(f"masks_step_{step_idx}", goal_path)

                    # --- RESOLVED: Load Action (JSON with fallback to NPY) ---
                    action = None
                    action_filename = f"action_step_{step_idx}"
                    
                    # Try loading JSON first
                    action = load_action_json(action_filename, goal_path)

                    subgoal = {
                        'observation': {
                            'rgb': rgb[:, :, :3],
                            'depth': depth,
                            'mask': mask
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

        # Initialise accumulators before any branch to avoid UnboundLocalError.
        multi_stage_reward = 0.0
        pdr_ = 0.0

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

    def _get_max_IoU(self, arena):
        cur_mask = arena.cloth_mask
        max_IoU = 0
        for goal in self.goals[:1]:
            goal_mask = goal[-1]['observation']["rgb"].sum(axis=2) > 0 ## only useful for background is black
            
            IoU, matched_IoU = get_max_IoU(cur_mask, goal_mask, debug=self.config.debug)
            if IoU > max_IoU:
                max_IoU = IoU

        return max_IoU
    
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


class RealWorldGarmentCanonAlignFoldingTask(RealWorldGarmentFoldingTask):
    """Goal-directed real-world canonicalisation-alignment-then-folding task.

    Mirrors the simulation `GarmentFoldingTask` (canonicalisation_alignment_centre_
    sleeve_folding): subgoal 0 is a user-defined canonical/aligned target, followed
    by `goal_steps` demonstrated fold steps. Evaluation/success use strict mask IoU
    per subgoal with a contiguous temporal suffix, exactly as in sim.
    """

    def __init__(self, config):
        super().__init__(config)
        # Per-subgoal IoU thresholds (length K = goal_steps + 1).
        self.iou_thresholds = list(config.get('iou_thresholds', [0.8, 0.8, 0.82]))
        # NB: self.name = config.fold_name (set by super) must contain 'alignment'
        # so the arena enables the flatten-contour drawing; the config supplies
        # fold_name: 'canonicalisation_alignment_centre_sleeve_folding'.

    # ------------------------------------------------------------------ goals ---
    def _generate_a_goal(self, arena):
        """Generate a canonical-alignment target (subgoal 0) then demonstrate folds.

        Returns (goal_obs, actions) where goal_obs[0] is the warped canonical target
        and goal_obs[1:] are the captured post-fold infos. Structure (dicts with an
        'observation' key) matches the parent so saving/loading is unchanged.
        """
        # --- 1. Take the flattened obs and deep-copy its observation ---
        flattened = arena.get_flattened_obs()
        canonical = {'observation': copy.deepcopy(flattened['observation'])}

        # --- 2. Let the user place the canonical/aligned target via the shared UI ---
        instructions = [
            "Rotate/drag so the BOTTOM (hem) of the garment",
            "faces the TOP of the image, and the SHOULDERS",
            "lie on the horizontal mid-line.",
        ]
        adjust_goal_mask_ui(canonical['observation'],
                            extra_instructions=instructions,
                            draw_mid_line=True)
        target_mask = canonical['observation']['mask']

        # --- 3. Physical placement confirm loop ---
        print(colored("\n*** [RealWorldGarmentCanonAlignFoldingTask] ACTION REQUIRED ***", "green"))
        print(colored("[RealWorldGarmentCanonAlignFoldingTask] Physically place the garment to match the canonical target (yellow contour).", "green"))
        win = 'Match canonical target (yellow contour)'
        cv2.namedWindow(win)
        target_uint8 = (np.array(target_mask) > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(target_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        while True:
            live = arena._process_info({}, task_related=False, flattened_obs=False)
            cur_rgb = live['observation']['rgb']
            cur_mask = live['observation']['mask']
            iou = calculate_iou(cur_mask, target_mask)

            disp = cur_rgb.copy()
            cv2.drawContours(disp, contours, -1, (0, 255, 255), 2)
            cv2.putText(disp, f"IoU to canonical target: {iou:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, disp)
            cv2.waitKey(300)

            print(colored(f"[RealWorldGarmentCanonAlignFoldingTask] Current IoU to canonical target: {iou:.3f}", "green"))
            resp = input(colored("[RealWorldGarmentCanonAlignFoldingTask] Press [Enter] to re-capture, or type 'c' to confirm placement: ", "green")).strip().lower()
            if resp == 'c':
                break
        cv2.destroyAllWindows()

        # --- 4. Demonstrator fold loop (subgoal 0 = warped canonical target) ---
        measure_time = arena.measure_time
        arena.measure_time = False
        info = arena._process_info({})
        arena.info = info
        self.demonstrator.reset([arena.id])

        goal_obs = [canonical]
        actions = []
        step = 0
        while step < self.goal_steps:
            print(f'[RealWorldGarmentCanonAlignFoldingTask] Current step {step}, goal_steps {self.goal_steps}')
            action = self.demonstrator.single_act(info)
            actions.append(action)

            info = arena.step(action)
            goal_obs.append(info)

            step += 1

        arena.measure_time = measure_time
        return goal_obs, actions

    # ------------------------------------------------------------- evaluation ---
    def _iou_for_goal_step(self, traj_info, goal_step):
        """Strict IoU of a trajectory step mask against subgoal `goal_step`.

        Max over all goal variations. Falls back to the RGB footprint if a legacy
        cache lacks the mask file (loaded mask is None).
        """
        cur_mask = traj_info['observation']['mask']
        best = 0.0
        for goal in self.goals:
            goal_mask = goal[goal_step]['observation'].get('mask', None)
            if goal_mask is None:
                goal_mask = goal[goal_step]['observation']['rgb'].sum(axis=2) > 0
            iou = calculate_iou(cur_mask, goal_mask)
            best = max(best, iou)
        return best

    def evaluate(self, arena):
        """Port of the sim folding evaluate: per-subgoal IoUs + active subgoal."""
        if len(self.goals) == 0:
            return {}

        K = self.goal_steps + 1
        trj = arena.get_trajectory_infos()
        N = len(trj)
        current_info = trj[-1]

        # IoU of the current state against every subgoal.
        subgoal_ious = [self._iou_for_goal_step(current_info, g) for g in range(K)]

        # Furthest subgoal reached as a CONTIGUOUS temporal suffix ending now: the
        # last (g+1) trajectory steps must match subgoals 0..g in order. Mirrors
        # success()'s last-K window, so the overlay stays in lock-step with success.
        achieved = -1
        for g in range(min(K, N)):
            if all(
                self._iou_for_goal_step(trj[N - 1 - g + j], j)
                >= self.iou_thresholds[j]
                for j in range(g + 1)
            ):
                achieved = g

        # Hold the final subgoal whenever the current frame still matches it (latch),
        # mirroring success()'s latched branch. Independent of `has_succeeded` to
        # avoid the one-step lag (see CLAUDE.md task notes).
        if subgoal_ious[K - 1] >= self.iou_thresholds[K - 1]:
            achieved = K - 1
        active_idx = min(achieved + 1, K - 1)

        return {
            'max_IoU': self._get_max_IoU(arena),
            'max_IoU_to_flattened': self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena),
            'algn_IoU': subgoal_ious[K - 1],
            'active_subgoal_idx': active_idx,
            'active_subgoal_iou': subgoal_ious[active_idx],
            'iou_thresholds': self.iou_thresholds,
        }

    # ---------------------------------------------------------------- success ---
    def success(self, arena):
        """Port of the sim folding success: last-K window match + latched re-check."""
        trj = arena.get_trajectory_infos()
        N = len(trj)
        K = self.goal_steps + 1

        # Latched: once succeeded, re-verify the CURRENT frame still matches the
        # final subgoal (un-latch if a corrective step broke it, like sim).
        if self.has_succeeded:
            if self._iou_for_goal_step(trj[-1], K - 1) < self.iou_thresholds[K - 1]:
                self.has_succeeded = False
                print(colored('[RealWorldGarmentCanonAlignFoldingTask] Success is messed up!', "yellow"))
                return False
            print(colored('[RealWorldGarmentCanonAlignFoldingTask] Successful Step!', "green"))
            return True

        # Not enough steps yet to assess the full flat->fold sequence.
        if N < K:
            return False

        # Main check: last-K window matches subgoals 0..K-1 in order.
        for j in range(K):
            if self._iou_for_goal_step(trj[N - K + j], j) < self.iou_thresholds[j]:
                return False

        self.has_succeeded = True
        print(colored('[RealWorldGarmentCanonAlignFoldingTask] Successful Step!', "green"))
        return True