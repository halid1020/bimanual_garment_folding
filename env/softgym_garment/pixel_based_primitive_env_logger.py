import os
import cv2
import numpy as np
from ..video_logger import VideoLogger
import matplotlib.pyplot as plt

class PixelBasedPrimitiveEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None):
        super().__call__(episode_config, result, filename=filename)
        eid = episode_config['eid']
        frames = [info["observation"]["rgb"] for info in result["information"]]
        actions = result["actions"]
        picker_traj = [info["observation"]["picker_norm_pixel_pos"]
                       for info in result["information"]]  # [T][2,2]

        H, W = 512, 512

        if filename is None:
            filename = 'manupilation'
        out_dir = os.path.join(self.log_dir, filename, 'performance_visualisation')
        os.makedirs(out_dir, exist_ok=True)

        # Convert normalized coords → pixel coords
        def norm_to_px(v):
            x = int((v[0] + 1) * 0.5 * W)
            y = int((v[1] + 1) * 0.5 * H)
            return x, y

        # Swap (x,y) → (row,col) for OpenCV drawing
        def swap(p):
            return (p[1], p[0])

        images = []
        for i, act in enumerate(actions):
            # TODO: at the top-left corner, use white text to indicidate step id and applied primitive type "Step 1: Pick-and-Place" or "Step 2: Pick-and-Fling"
            key = list(act.keys())[0]
            val = act[key]

            img = frames[i].copy()
            # --- Resize to 512×512 BEFORE drawing text ---
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            

            step_text = f"Step {i+1}: "
            if key == "norm-pixel-fold":
                step_text += "Pick-and-Place"
            elif key == "norm-pixel-pick-and-fling":
                step_text += "Pick-and-Fling"
            else:
                step_text += key

            cv2.putText(
                img,
                step_text,
                (10, 30),                 # top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),          # white text
                2,
                cv2.LINE_AA
            )

            RED   = (50, 50, 200)     # softer red
            BLUE  = (200, 50, 50)     # softer blue

            # ================================
            #          FOLD / PICK-PLACE
            # ================================
            if key == "norm-pixel-fold":
                pick_0  = norm_to_px(val[:2])
                pick_1  = norm_to_px(val[2:4])
                place_0 = norm_to_px(val[4:6])
                place_1 = norm_to_px(val[6:8])

                # ----- Ensure BLUE pick is always the LEFT one -----
                picks = [(pick_0, place_0), (pick_1, place_1)]
                picks_sorted = sorted(picks, key=lambda p: p[0][0])  # sort by x

                (left_pick, left_place), (right_pick, right_place) = picks_sorted
                # BLUE = left, RED = right

                # ----- Draw arrows with SMALLER arrowheads -----
                small_tip = 0.08    # << smaller arrowhead tip size

                cv2.arrowedLine(img, swap(left_pick),  swap(left_place),  BLUE, 5, tipLength=small_tip)
                cv2.arrowedLine(img, swap(right_pick), swap(right_place), RED,  5, tipLength=small_tip)

                # ----- Hollow circles -----
                cv2.circle(img, swap(left_pick),  8, BLUE, 2)
                cv2.circle(img, swap(right_pick), 8, RED,  2)

            # ================================
            #        PICK & FLING
            # ================================
            elif key == "norm-pixel-pick-and-fling":
                pick_0 = norm_to_px(val[:2])
                pick_1 = norm_to_px(val[2:4])

                # Retrieve trajectory of NEXT environment step
                traj = picker_traj[i + 1]
                if traj is None or len(traj) == 0:
                    cv2.putText(
                        img,
                        'Rejected',
                        (20, 30),                 # top-left corner
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),          # white text
                        2,
                        cv2.LINE_AA
                    )
                    
                else:
                    traj = traj[10:-10]

                    traj_px_0 = [norm_to_px(step[0]) for step in traj]
                    traj_px_1 = [norm_to_px(step[1]) for step in traj]

                    T = len(traj)
                    num_samples = 20
                    idx = np.linspace(0, T - 1, min(T, num_samples)).astype(int)

                    sampled_0 = [traj_px_0[j] for j in idx]
                    sampled_1 = [traj_px_1[j] for j in idx]

                    # Colormaps
                    cmap0 = cv2.COLORMAP_AUTUMN
                    cmap1 = cv2.COLORMAP_WINTER

                    for s in range(1, len(idx)):
                        alpha = s / (len(idx) - 1)

                        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
                        color0 = cv2.applyColorMap(value, cmap0)[0, 0].tolist()
                        color1 = cv2.applyColorMap(value, cmap1)[0, 0].tolist()

                        cv2.line(img, swap(sampled_0[s-1]), swap(sampled_0[s]), color0, 5)
                        cv2.line(img, swap(sampled_1[s-1]), swap(sampled_1[s]), color1, 5)

                    # -------- TODO FIX: replace X markers with hollow circles ----------
                    def draw_hollow_circle(img, p, color, radius=8, thickness=3):
                        cv2.circle(img, swap(p), radius, color, thickness)

                    draw_hollow_circle(img, sampled_0[0], RED)
                    draw_hollow_circle(img, sampled_1[0], BLUE)
                    # ------------------------------------------------------------------

                    # Final triangle marker
                    def draw_triangle(img, center, color, size=12):
                        cx, cy = center
                        pts = np.array([
                            (cy, cx - size),
                            (cy - size, cx + size),
                            (cy + size, cx + size)
                        ], np.int32)
                        cv2.fillPoly(img, [pts], color)

                    draw_triangle(img, traj_px_0[-1], RED)
                    draw_triangle(img, traj_px_1[-1], BLUE)


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

