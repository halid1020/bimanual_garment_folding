import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from real_robot.utils.draw_utils import *
from real_robot.robot.video_logger import VideoLogger

# IMPORTANT: Update this import to match where you saved the utility functions provided
from real_robot.utils.save_utils import (
    save_colour, save_depth, save_mask, save_action_json, NumpyEncoder
)

class PixelBasedPrimitiveImpEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None):
        super().__call__(episode_config, result, filename=filename)
        eid = episode_config['eid']
        
        # Standard Visualisation Setup
        frames = [info["observation"]["rgb"] for info in result["information"]]
        robot0_masks = None
        robot1_masks = None
        
        # Defensive check for masks
        if result["information"] and 'robot0_mask' in result["information"][0]["observation"]:
            robot0_masks = [info["observation"]["robot0_mask"] for info in result["information"]]
        if result["information"] and 'robot1_mask' in result["information"][0]["observation"]:
            robot1_masks = [info["observation"]["robot1_mask"] for info in result["information"]]
        
        actions = result["actions"]
        H, W = 512, 512

        if filename is None:
            filename = 'manupilation'
            
        # 1. Setup Directories
        # Visualisation directory
        viz_dir = os.path.join(self.log_dir, filename, 'performance_visualisation')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Step-wise data directory: logs/filename/episode_{eid}/
        episode_data_dir = os.path.join(self.log_dir, filename, f'episode_{eid}')
        os.makedirs(episode_data_dir, exist_ok=True)

        # --- Helper: Project 3D points to 2D image coordinates ---
        def project_trajectory(points_3d, T_base_cam, intr, arena_meta):
            if not points_3d: return []
            
            if hasattr(intr, 'fx'): 
                fx, fy = intr.fx, intr.fy
                cx, cy = intr.ppx, intr.ppy
            else: 
                fx, fy = intr[0,0], intr[1,1]
                cx, cy = intr[0,2], intr[1,2]

            T_cam_base = np.linalg.inv(T_base_cam)
            points_2d = []
            
            crop_x1 = arena_meta.x1
            crop_y1 = arena_meta.y1
            crop_size = arena_meta.crop_size
            target_res = 512.0 
            scale = target_res / crop_size

            for p in points_3d:
                p_h = np.append(p, 1.0) 
                p_cam = T_cam_base @ p_h
                z = p_cam[2]
                if z <= 0: continue 

                u_raw = (p_cam[0] * fx / z) + cx
                v_raw = (p_cam[1] * fy / z) + cy

                u_final = (u_raw - crop_x1) * scale
                v_final = (v_raw - crop_y1) * scale
                
                points_2d.append((int(u_final), int(v_final)))
            return points_2d

        images = []
        
        # Iterate through steps
        for i, act in enumerate(actions):
            # --- NEW: Save Step-wise Data ---
            step_dir = os.path.join(episode_data_dir, f'step_{i}')
            os.makedirs(step_dir, exist_ok=True)
            
            info = result["information"][i]
            obs = info["observation"]
            
            # A. Save Images (RGB, Depth, Masks)
            # Note: rgb2bgr=True because observation['rgb'] is usually RGB, but cv2 writes BGR
            save_colour(obs["rgb"], filename='rgb', directory=step_dir, rgb2bgr=True)
            
            if "depth" in obs:
                save_depth(obs["depth"], filename='depth', directory=step_dir)
                
            if "mask" in obs:
                save_mask(obs["mask"], filename='mask', directory=step_dir)
                
            if "robot0_mask" in obs:
                save_mask(obs["robot0_mask"], filename='robot0_mask', directory=step_dir)
            if "robot1_mask" in obs:
                save_mask(obs["robot1_mask"], filename='robot1_mask', directory=step_dir)

            # B. Save Action
            save_action_json(act, filename='action', directory=step_dir)
            
            # C. Save Evaluation & Success Info
            step_info = {
                "evaluation": info.get("evaluation", {}),
                "success": info.get("success", False),
                "reward": info.get("reward", 0.0),
                "done": info.get("done", False),
                "eid": eid
            }
            
            with open(os.path.join(step_dir, 'info.json'), 'w') as f:
                json.dump(step_info, f, indent=4, cls=NumpyEncoder)
                
            # D. Save Garment Name
            garment_name = "unknown"
            # Try getting from Arena object first
            if "arena" in info and hasattr(info["arena"], "garment_id"):
                garment_name = getattr(info["arena"], "garment_id", "unknown")
            # Fallback to episode config
            elif "garment_id" in episode_config:
                garment_name = episode_config["garment_id"]
            
            with open(os.path.join(step_dir, 'garment_name.txt'), 'w') as f:
                f.write(str(garment_name))

            # ---------------------------------------------------------
            # Existing Visualization Logic (Drawing on images)
            # ---------------------------------------------------------
            key = list(act.keys())[0]
            val = act[key]

            img = frames[i].copy()
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            if info.get('draw_flatten_contour', False):
                # Locate the goal mask (checking standard locations based on your Arena)
                goal_mask = None
                if 'goal_mask' in obs:
                    goal_mask = obs['goal_mask']
                elif 'goal' in info and 'mask' in info['goal']:
                    goal_mask = info['goal']['mask']
                elif 'flattened-mask' in obs:
                    goal_mask = obs['flattened-mask']
                
                if goal_mask is not None:
                    # Convert to uint8 (0-255) for OpenCV contour detection
                    mask_uint8 = (goal_mask > 0).astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    # Find contours (EXTERNAL only to get the outer boundary)
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw contours in Green (0, 255, 0) with a thickness of 2
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

            if robot0_masks is not None:
                mask0 = cv2.resize(robot0_masks[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                img = apply_workspace_shade(img, mask0, color=(255, 0, 0), alpha=0.2)
                
            if robot1_masks is not None:
                mask1 = cv2.resize(robot1_masks[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                img = apply_workspace_shade(img, mask1, color=(0, 0, 255), alpha=0.2)

            primitive_color = PRIMITIVE_COLORS.get(key, PRIMITIVE_COLORS["default"])

            step_text = f"Step {i+1}: "
            if key == "norm-pixel-pick-and-place":
                step_text += "Pick and Place"
            elif key == "norm-pixel-pick-and-fling":
                step_text += "Pick and Fling"
            elif key == "no-operation":
                step_text += "No Operation"
            else:
                step_text += key

            draw_text_with_bg(img, step_text, (10, TEXT_Y_STEP), primitive_color)

            if key == "norm-pixel-pick-and-place":
                if i + 1 < len(result["information"]):
                    info_next = result["information"][i+1]
                    if 'applied_action' in info_next:
                        applied_action = info_next['applied_action']
                        applied_action = applied_action.get('norm-pixel-pick-and-place')
                        if applied_action is not None:
                            img = draw_pick_and_place(img, applied_action)

            elif key == "norm-pixel-pick-and-fling":
                traj_px_0, traj_px_1 = None, None
                if i + 1 < len(result["information"]):
                    info_next = result["information"][i+1]
                    traj_data = info_next.get("debug_trajectory")
                    arena_instance = info_next.get("arena")

                    if traj_data and arena_instance:
                        scene = arena_instance.dual_arm
                        traj_px_0 = project_trajectory(traj_data['ur5e'], scene.T_ur5e_cam, scene.intr, arena_instance)
                        traj_px_1 = project_trajectory(traj_data['ur16e'], scene.T_ur16e_cam, scene.intr, arena_instance)

                img = draw_pick_and_fling(img, val, traj_0=traj_px_0, traj_1=traj_px_1)

            images.append(img)

        if len(frames) > len(images):
             images.append(frames[-1])

        # --- Matplotlib Grid Saving ---
        MAX_COLS = 6
        num_images = len(images)
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS
        if num_rows == 0: num_rows = 1

        fig, axes = plt.subplots(num_rows, MAX_COLS, figsize=(3 * MAX_COLS, 3 * num_rows))

        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if MAX_COLS == 1:
            axes = np.expand_dims(axes, axis=1)

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