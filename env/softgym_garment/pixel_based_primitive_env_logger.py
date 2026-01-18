import os
import cv2
import numpy as np
from ..video_logger import VideoLogger
import matplotlib.pyplot as plt

from .draw_utils import *




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

        images = []
        for i, act in enumerate(actions):
            # TODO: at the top-left corner, use white text to indicidate step id and applied primitive type "Step 1: Pick-and-Place" or "Step 2: Pick-and-Fling"
            key = list(act.keys())[0]
            val = act[key]

            img = frames[i].copy()
            # --- Resize to 512×512 BEFORE drawing text ---
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

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
            if key == "norm-pixel-pick-and-place":
                step_text += "Pick and Place"
            elif key == "norm-pixel-pick-and-fling":
                step_text += "Pick and Fling"
            elif key == "no-operation":
                step_text += "No Operation"
            else:
                step_text += key

            draw_text_with_bg(
                img,
                step_text,
                (10, TEXT_Y_STEP),
                primitive_color
            )



            # RED   = (50, 50, 200)     # softer red
            # BLUE  = (200, 50, 50)     # softer blue
            # print('result length', len(result["information"]))
            # print('action lenght', len(actions))
            # ================================
            #          FOLD / PICK-PLACE
            # ================================
            if key == "norm-pixel-pick-and-place":
                applied_action = result["information"][i+1]['applied_action']
                # print('applied action before', applied_action)
                applied_action = applied_action['norm-pixel-pick-and-place']
                # print('applied action', applied_action)
                img = draw_pick_and_place(img, applied_action)

            # ================================
            #        PICK & FLING
            # ================================
            elif key == "norm-pixel-pick-and-fling":
                pick_0 = norm_to_px(val[:2], W, H)
                pick_1 = norm_to_px(val[2:4], W, H)

                # Retrieve trajectory of NEXT environment step
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

        # Add final frame
        images.append(frames[-1])

        # ===========================================
        #  CONCAT images in a matplotlib grid
        # ===========================================

        MAX_COLS = 6
        num_images = len(images)

        # Compute number of rows
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS

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
                {f"trajectory/episode_{eid}": save_path}, 
                step=episode_config.get('step') # Optional: if you have a step count
            )