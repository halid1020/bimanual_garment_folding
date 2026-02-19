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
        for i, act in enumerate(actions):
            # --- Save Step-wise Data ---
            step_dir = os.path.join(episode_data_dir, f'step_{i}')
            os.makedirs(step_dir, exist_ok=True)
            
                
            info = result["information"][i]
            obs = info["observation"]
            
            # A. Save Images
            save_colour(obs["rgb"], filename='rgb', directory=step_dir, rgb2bgr=True)
            
            if "depth" in obs:
                save_depth(obs["depth"], filename='depth', directory=step_dir)
            if "mask" in obs:
                save_mask(obs["mask"], filename='mask', directory=step_dir)
            if "robot0_mask" in obs:
                save_mask(obs["robot0_mask"], filename='robot0_mask', directory=step_dir)

            # B. Save Action
            act_serializable = act.tolist() if isinstance(act, np.ndarray) else act
            with open(os.path.join(step_dir, 'action.json'), 'w') as f:
                json.dump(act_serializable, f, indent=4, cls=NumpyEncoder)
            
            # C. Save Info
            step_info = {
                "evaluation": info.get("evaluation", {}),
                "success": info.get("success", False),
                "reward": info.get("reward", 0.0),
                "done": info.get("done", False),
                "eid": eid
            }
            with open(os.path.join(step_dir, 'info.json'), 'w') as f:
                json.dump(step_info, f, indent=4, cls=NumpyEncoder)

            # ---------------------------------------------------------
            # Visualization Logic
            # ---------------------------------------------------------
            img = frames[i].copy()
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

            # 1. Draw Workspace Mask
            if robot0_masks is not None:
                mask0 = robot0_masks[i].astype(bool)
                if mask0.shape[:2] != (W, H):
                    mask0 = cv2.resize(mask0.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # Apply Blue/Cool tint
                img = apply_workspace_shade(img, mask0, color=(255, 0, 0), alpha=0.2)

            # 2. Determine Action to Draw
            action_to_draw = act

            # Priority 1: Use 'applied_action' from info (exact executed coordinates)
            if i + 1 < len(result["information"]):
                info_next = result["information"][i+1]
                if 'applied_action' in info:
                    action_to_draw = info_next['applied_action']
    

            # 3. Draw Text
            step_text = f"Step {i+1}: Pick & Place"
            color_text = PRIMITIVE_COLORS.get("norm-pixel-pick-and-place", (255, 255, 255))
            draw_text_with_bg(img, step_text, (10, TEXT_Y_STEP), color_text)

            # 4. Draw Action Arrow
            if action_to_draw is not None:
                action_np = np.array(action_to_draw).flatten()
                # draw_pick_and_place handles denormalization and drawing
                img = draw_pick_and_place(img, action_np)

            images.append(img)

        if len(frames) > len(images):
             images.append(frames[-1])

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

        arena = info = result["information"][0]['arena']

        if arena.measure_time:
            perception_time =  getattr(arena, 'perception_time', [])
            process_action_time = getattr(arena, 'process_action_time', [])
            primitive_time = getattr(arena, 'primitive_time', [])
            inference_time = result['internal_states'][-1].get('inference_time', [])
            csv_path = os.path.join(episode_data_dir, 'timing_report.csv')
            with open(csv_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Metric", "Mean (s)", "Std (s)", "Samples (N)"])
                def write_stat(name, data):
                    if not data:
                        writer.writerow([name, "N/A", "N/A", 0])
                    elif isinstance(data, (int, float)):
                        writer.writerow([name, f"{data:.4f}", "0.0000", 1])
                    else:
                        writer.writerow([name, f"{np.mean(data):.4f}", f"{np.std(data):.4f}", len(data)])

                write_stat("Policy Inference / Step", inference_time)
                write_stat("Perception / Step", perception_time)
                write_stat("Process Action / Step", process_action_time)
                write_stat("Primitives Exec / Step", primitive_time)