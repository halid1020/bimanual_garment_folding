import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import csv

from real_robot.robot.video_logger import VideoLogger
from real_robot.utils.save_utils import (
    save_colour, save_depth, save_mask, save_action_json, NumpyEncoder
)
from real_robot.utils.draw_utils import (
    draw_pick_and_place, 
    draw_text_with_bg, 
    apply_workspace_shade,
    PRIMITIVE_COLORS,
    TEXT_Y_STEP
)

class SingleArmPixelLogger(VideoLogger):
    """
    Logger specifically for Single Arm Pick and Place episodes.
    Assumes all actions are 'Pick & Place'.
    """

    def __call__(self, episode_config, result, filename=None):
        super().__call__(episode_config, result, filename=filename)
        eid = episode_config.get('eid', 0)
        
        # Standard Visualisation Setup
        frames = [info["observation"]["rgb"] for info in result["information"]]
        
        # Defensive check for masks
        robot0_masks = None
        if result["information"] and 'robot0_mask' in result["information"][0]["observation"]:
            robot0_masks = [info["observation"]["robot0_mask"] for info in result["information"]]
        
        actions = result["actions"]
        
        if filename is None:
            filename = 'single_arm_manipulation'
            
        # 1. Setup Directories
        viz_dir = os.path.join(self.log_dir, filename, 'performance_visualisation')
        os.makedirs(viz_dir, exist_ok=True)
        
        episode_data_dir = os.path.join(self.log_dir, filename, f'episode_{eid}')
        os.makedirs(episode_data_dir, exist_ok=True)

        images = []
        H, W = 512, 512 # Target resolution for visualization
        
        # Iterate through steps
        # Iterate through ALL steps in information (handles the N+1 final state)
        for i, info in enumerate(result["information"]):
            # --- Save Step-wise Data ---
            step_dir = os.path.join(episode_data_dir, f'step_{i}')
            os.makedirs(step_dir, exist_ok=True)
            
            obs = info["observation"]
            
            # A. Save Images
            save_colour(obs["rgb"], filename='rgb', directory=step_dir, rgb2bgr=True)
            
            if "depth" in obs:
                save_depth(obs["depth"], filename='depth', directory=step_dir)
            if "mask" in obs:
                save_mask(obs["mask"], filename='mask', directory=step_dir)
            if "robot0_mask" in obs:
                save_mask(obs["robot0_mask"], filename='robot0_mask', directory=step_dir)
            if "roi_rgb" in obs:
                save_colour(obs["roi_rgb"], filename='roi_rgb', directory=step_dir, rgb2bgr=True)
            if "roi_workspace_mask" in obs:
                save_mask(obs["roi_workspace_mask"], filename='roi_workspace_mask', directory=step_dir)
            

            # B. Save Action (The final state has no subsequent action)
            act = actions[i] if i < len(actions) else None
            
            if act is not None:
                act_serializable = act.tolist() if isinstance(act, np.ndarray) else act
                with open(os.path.join(step_dir, 'action.json'), 'w') as f:
                    json.dump(act_serializable, f, indent=4, cls=NumpyEncoder)
            
            # C. Save Info (Added garment_id here)
            step_info = {
                "evaluation": info.get("evaluation", {}),
                "success": info.get("success", False),
                "reward": info.get("reward", 0.0),
                "done": info.get("done", False),
                "eid": eid,
                "garment_id": info.get("garment_id"), # Saves the garment ID!
                "activate_transfer_workspace_hueristic": info.get('activate_transfer_workspace_hueristic', False)
            }
            with open(os.path.join(step_dir, 'info.json'), 'w') as f:
                json.dump(step_info, f, indent=4, cls=NumpyEncoder)
            
            # D. Save Internal States
            step_internal_state = info.get('internal_states')

            # Fallback if it's stored in the main result dict
            if step_internal_state is None and 'internal_states' in result and i < len(result['internal_states']):
                step_internal_state = result['internal_states'][i]

            if step_internal_state is not None:
                # 1. Make a shallow copy to safely mutate
                state_to_save = step_internal_state.copy()
                
                # 2. Extract and format the MPC pick-mask (Standard Image)
                if 'pick-mask' in state_to_save:
                    pick_mask = state_to_save.pop('pick-mask')
                    if isinstance(pick_mask, np.ndarray):
                        if pick_mask.ndim == 3 and pick_mask.shape[-1] == 1:
                            pick_mask = pick_mask.squeeze(-1)
                        if pick_mask.dtype == bool:
                            pick_mask = (pick_mask.astype(np.uint8)) * 255
                    save_mask(pick_mask, filename='internal_pick_mask', directory=step_dir)

                # 3. Extract all heavy arrays/tensors and save as .npy
                npy_keys = [
                    'raw_input_obs', 'input_obs', 'recon_obs',
                    'stoch_state', 'deter_state', 
                    'iteration_means', 'last_samples', 'last_costs'
                ]
                
                for key in npy_keys:
                    if key in state_to_save:
                        data_arr = state_to_save.pop(key)
                        
                        # If it's a PyTorch Tensor, convert to numpy safely
                        if hasattr(data_arr, 'cpu'):
                            data_arr = data_arr.cpu().detach().numpy()
                        # If it happens to be a raw Python list, cast to numpy
                        elif not isinstance(data_arr, np.ndarray):
                            data_arr = np.array(data_arr)
                            
                        # Save directly as a binary numpy file
                        np.save(os.path.join(step_dir, f'{key}.npy'), data_arr)

                # 4. Remove 'latent_state' dict to prevent JSON crashing
                if 'latent_state' in state_to_save:
                    state_to_save.pop('latent_state')

                # 5. Save the remaining scalar and metadata to JSON
                # This will now be a nice, lightweight file!
                with open(os.path.join(step_dir, 'internal_states.json'), 'w') as f:
                    json.dump(state_to_save, f, indent=4, cls=NumpyEncoder)
            # ---------------------------------------------------------
            # Visualization Logic
            # ---------------------------------------------------------
            
            # Ensure we safely grab the frame even if arrays are misaligned
            if i < len(frames):
                img = frames[i].copy()
            else:
                img = obs["rgb"].copy() 
                
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

            # 1. Draw Workspace Mask
            if robot0_masks is not None and i < len(robot0_masks):
                mask0 = robot0_masks[i].astype(bool)
                if mask0.shape[:2] != (W, H):
                    mask0 = cv2.resize(mask0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # Apply Blue/Cool tint
                img = apply_workspace_shade(img, mask0, color=(0, 0, 255), alpha=0.2)

            # 2. Determine Action to Draw
            action_to_draw = act

            # Priority 1: Use 'applied_action' from info_next (fixed slight bug in your original)
            if i + 1 < len(result["information"]):
                info_next = result["information"][i+1]
                if 'applied_action' in info_next:
                    action_to_draw = info_next['applied_action']
    
            # 3. Draw Text
            if act is not None:
                step_text = f"Step {i+1}: Pick & Place"
            else:
                step_text = f"Step {i+1}: Final State" # Label for the last image
                
            color_text = PRIMITIVE_COLORS.get("norm-pixel-pick-and-place", (255, 255, 255))
            draw_text_with_bg(img, step_text, (10, TEXT_Y_STEP), color_text)

            # 4. Draw Action Arrow
            if action_to_draw is not None:
                action_np = np.array(action_to_draw).flatten()
                # draw_pick_and_place handles denormalization and drawing
                img = draw_pick_and_place(img, action_np, rgb2bgr=True)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            images.append(img)

        # Remove the old `if len(frames) > len(images): images.append(frames[-1])` 
        # because the loop now explicitly handles the final state.

        if len(frames) > len(images):
             img = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
             images.append(img)

        # --- Matplotlib Grid Saving ---
        MAX_COLS = 6
        num_images = len(images)
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS
        if num_rows == 0: num_rows = 1

        fig, axes = plt.subplots(num_rows, MAX_COLS, figsize=(3 * MAX_COLS, 3 * num_rows))

        # Handle squeeze
        if num_rows == 1: axes = np.expand_dims(axes, axis=0)
        if MAX_COLS == 1: axes = np.expand_dims(axes, axis=1)

        idx = 0
        for r in range(num_rows):
            for c in range(MAX_COLS):
                ax = axes[r][c]
                if idx < num_images:
                    img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.axis("off")
                    idx += 1
                else:
                    ax.axis("off")

        plt.tight_layout()
        save_path = os.path.join(viz_dir, f"episode_{eid}_trajectory.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        # Fix: Extract arena (avoiding the arena = info = ... overwrite just in case)
        arena = result["information"][0]['arena']

        if getattr(arena, 'measure_time', False):
            perception_time = getattr(arena, 'perception_time', [])
            process_action_time = getattr(arena, 'process_action_time', [])
            primitive_time = getattr(arena, 'primitive_time', [])
            inference_time = result.get('internal_states', [{}])[-1].get('inference_time', [])

            # Helper to ensure data is iterable, even if it's a single float
            def to_list(data):
                if data is None: return []
                if isinstance(data, (int, float)): return [data]
                return list(data)

            inf_t = to_list(inference_time)
            perc_t = to_list(perception_time)
            proc_t = to_list(process_action_time)
            prim_t = to_list(primitive_time)

            # Determine total rows based on the longest timing list
            max_steps = max(len(inf_t), len(perc_t), len(proc_t), len(prim_t), 0)

            csv_path = os.path.join(episode_data_dir, 'timing_report.csv')
            with open(csv_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write the column headers
                writer.writerow([
                    "Step", 
                    "Policy Inference (s)", 
                    "Perception (s)", 
                    "Process Action (s)", 
                    "Primitives Exec (s)"
                ])
                
                # Write step-wise rows
                for step in range(max_steps):
                    # Fetch value if it exists for this step, format to 3 decimals, else "N/A"
                    val_inf  = f"{inf_t[step]:.3f}"  if step < len(inf_t)  else "N/A"
                    val_perc = f"{perc_t[step]:.3f}" if step < len(perc_t) else "N/A"
                    val_proc = f"{proc_t[step]:.3f}" if step < len(proc_t) else "N/A"
                    val_prim = f"{prim_t[step]:.3f}" if step < len(prim_t) else "N/A"
                    
                    writer.writerow([step, val_inf, val_perc, val_proc, val_prim])