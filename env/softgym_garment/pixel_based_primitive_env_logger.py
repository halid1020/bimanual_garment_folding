import os
import cv2
import numpy as np
import json
from ..video_logger import VideoLogger
import matplotlib.pyplot as plt

from .draw_utils import *
# Make sure to import your NumpyEncoder from wherever it lives in your project
from real_robot.utils.save_utils import NumpyEncoder 

class PixelBasedPrimitiveEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None, wandb_logger=None):
        super().__call__(episode_config, result, filename=filename)
        eid = episode_config['eid']
        frames = [info["observation"]["rgb"] for info in result["information"]]
        robot0_masks = None
        robot1_masks = None
        if 'robot0_mask' in result["information"][0]["observation"]:
            robot0_masks = [info["observation"]["robot0_mask"] for info in result["information"]]
        if 'robot1_mask' in result["information"][0]["observation"]:
            robot1_masks = [info["observation"]["robot1_mask"] for info in result["information"]]
        
        actions = result["actions"]
        picker_traj = [info["observation"]["picker_norm_pixel_pos"]
                       for info in result["information"]]  # [T][2,2]

        H, W = 512, 512

        if filename is None:
            filename = 'manupilation'
        out_dir = os.path.join(self.log_dir, filename, 'performance_visualisation')
        os.makedirs(out_dir, exist_ok=True)

        # =======================================================================
        # --- NEW: Setup Directory for Step Data and Save Internal States ---
        # =======================================================================
        episode_data_dir = os.path.join(self.log_dir, filename, f'episode_{eid}')
        os.makedirs(episode_data_dir, exist_ok=True)

        # Iterate through ALL steps to save internal states (including the final state)
        for i, info in enumerate(result["information"]):
            step_dir = os.path.join(episode_data_dir, f'step_{i}')
            os.makedirs(step_dir, exist_ok=True)
            
            step_internal_state = info.get('internal_states')
            
            # Fallback if it's stored in the main result dict
            if step_internal_state is None and 'internal_states' in result and i < len(result['internal_states']):
                step_internal_state = result['internal_states'][i]

            if step_internal_state is not None:
                # 1. Make a shallow copy to safely mutate
                state_to_save = step_internal_state.copy()
                
                # 2. Extract heavy arrays/tensors to save as .npy
                npy_keys = [
                    'noise_actions_history', 'primitive_probabilities',
                    'predicted_keypoints', 'gt_keypoints',
                    # Add any other big arrays you might have here like 'recon_obs', etc.
                ]
                
                for key in npy_keys:
                    if key in state_to_save:
                        data_arr = state_to_save.pop(key)
                        
                        # Safely convert PyTorch tensors or Lists to numpy arrays
                        if hasattr(data_arr, 'cpu'):
                            data_arr = data_arr.cpu().detach().numpy()
                        elif not isinstance(data_arr, np.ndarray):
                            if data_arr is not None:
                                data_arr = np.array(data_arr)
                                
                        if data_arr is not None:
                            np.save(os.path.join(step_dir, f'{key}.npy'), data_arr)

        # =======================================================================
                    
        images = []
        for i, act in enumerate(actions):
            info = result["information"][i]
            obs = info["observation"]
            key = list(act.keys())[0]
            val = act[key]

            img = frames[i].copy()
            # --- Resize to 512×512 BEFORE drawing text ---
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
                mask0 = cv2.resize(
                    robot0_masks[i].astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                img = apply_workspace_shade(
                    img,
                    mask0,
                    color=(255, 0, 0),  # Blue in BGR
                    alpha=0.2
                )
                
            if robot1_masks is not None:
                mask1 = cv2.resize(
                    robot1_masks[i].astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                img = apply_workspace_shade(
                    img,
                    mask1,
                    color=(0, 0, 255),  # Red in BGR
                    alpha=0.2
                )
            

            primitive_color = PRIMITIVE_COLORS.get(key, PRIMITIVE_COLORS["default"])

            step_text = f"Step {i+1}: "
            if key == "norm-pixel-dual-pick-and-place":
                # If there are 4 coordinates (pick0_x, pick0_y, pick1_x, pick1_y), it's dual.
                step_text += "Dual Pick and Place"
            elif key == "norm-pixel-single-pick-and-place":
                step_text += "Single Pick and Place"
            elif key == "norm-pixel-pick-and-fling":
                step_text += "Pick and Fling"
            elif key == "no-operation":
                step_text += "No operation"
            else:
                step_text += key

            draw_text_with_bg(
                img,
                step_text,
                (10, TEXT_Y_STEP),
                primitive_color,
                scale=1,     # <--- ADDED: Lowers text size (default is usually ~1.0)
                thickness=2    # <--- ADDED: Makes the thinner text look clean
            )

            # ================================
            #          FOLD / PICK-PLACE
            # ================================
            if key in ["norm-pixel-single-pick-and-place", "norm-pixel-dual-pick-and-place"]:
                # ADDED BOUNDS CHECK
                if i + 1 < len(result["information"]):
                    info_next = result["information"][i+1]
                    if 'applied_action' in info_next:
                        # Get the action using the current key
                        applied_action = info_next['applied_action'].get(key)
                        
                        # Fallback just in case the environment still saves it under the old generic name
                        if applied_action is None:
                            applied_action = info_next['applied_action'].get('norm-pixel-pick-and-place')
                            
                        if applied_action is not None:
                            img = draw_pick_and_place(img, applied_action)

            # ================================
            #        PICK & FLING
            # ================================
            elif key == "norm-pixel-pick-and-fling":
                # pick_0 = norm_to_px(val[:2], W, H)
                # pick_1 = norm_to_px(val[2:4], W, H)

                # ADDED BOUNDS CHECK
                traj = None
                if i + 1 < len(picker_traj):
                    traj = picker_traj[i + 1]

                ## TODO: do not overlap with the step-wise primitive information text
                if traj is None or len(traj) == 0:
                    draw_text_with_bg(
                        img,
                        "Rejected",
                        (10, TEXT_Y_STATUS),
                        MILD_RED,
                        scale=1.2
                    )
                    
                else:
                    traj = traj[10:-10]

                    traj_px_0 = [norm_to_px(step[0], W, H) for step in traj]
                    traj_px_1 = [norm_to_px(step[1], W, H) for step in traj]

                    T = len(traj)
                    num_samples = 20
                    idx = np.linspace(0, T - 1, min(T, num_samples)).astype(int)

                    sampled_0 = [traj_px_0[j] for j in idx]
                    sampled_1 = [traj_px_1[j] for j in idx]

                    # Colormaps
                    cmap0 = cv2.COLORMAP_COOL
                    cmap1 = cv2.COLORMAP_AUTUMN

                    BLUE = cv2.applyColorMap(
                        np.uint8([[[0]]]), cmap0
                    )[0, 0].tolist()
                    
                    RED = cv2.applyColorMap(
                        np.uint8([[[0]]]), cmap1
                    )[0, 0].tolist()

                    for s in range(1, len(idx)):
                        alpha = s / (len(idx) - 1)

                        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
                        color0 = cv2.applyColorMap(value, cmap0)[0, 0].tolist()
                        color1 = cv2.applyColorMap(value, cmap1)[0, 0].tolist()

                        cv2.line(img, swap(sampled_0[s-1]), swap(sampled_0[s]), color0, 8)
                        cv2.line(img, swap(sampled_1[s-1]), swap(sampled_1[s]), color1, 8)

                    def draw_hollow_circle(img, p, color, radius=10, thickness=3):
                        cv2.circle(img, swap(p), radius, color, thickness)

                    draw_hollow_circle(img, sampled_0[0], BLUE)
                    draw_hollow_circle(img, sampled_1[0], RED)
                    # ------------------------------------------------------------------

                    # Final triangle marker
                    def draw_triangle_down(img, center, color, size=15):
                        cx, cy = center  # (x, y)

                        pts = np.array([
                            (cy - size, cx - size),  # top-left
                            (cy + size, cx - size),  # top-right
                            (cy,        cx + size),  # bottom (apex)
                        ], np.int32)

                        cv2.fillPoly(img, [pts], color)

                    draw_triangle_down(img, traj_px_0[-1], BLUE)
                    draw_triangle_down(img, traj_px_1[-1], RED)

            images.append(img)

        # Add final frame safely
        if len(frames) > len(images):
             images.append(frames[-1])

        # ===========================================
        #  CONCAT images in a matplotlib grid
        # ===========================================

        MAX_COLS = 6
        num_images = len(images)

        # Compute number of rows
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS
        if num_rows == 0: num_rows = 1

        # Create the figure
        fig, axes = plt.subplots(
            num_rows, MAX_COLS,
            figsize=(3 * MAX_COLS, 3 * num_rows),  # adjust tile size here
        )

        # If only one row/col, axes may not be 2D — fix it:
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if MAX_COLS == 1:
            axes = np.expand_dims(axes, axis=1)

        # Fill the grid
        idx = 0
        for r in range(num_rows):
            for c in range(MAX_COLS):
                ax = axes[r][c]

                if idx < num_images:
                    # RGB images in OpenCV are BGR — convert before plotting
                    img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.axis("off")
                    idx += 1
                else:
                    # Hide empty cells
                    ax.axis("off")

        # Tight layout avoids big gaps
        plt.tight_layout()

        # Save final image
        save_path = os.path.join(out_dir, f"episode_{eid}_trajectory.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        if wandb_logger is not None:
            wandb_logger.log(
                {f"trajectory/episode_{eid}": save_path}
            )