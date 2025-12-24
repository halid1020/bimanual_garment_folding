import os
import cv2
import numpy as np
from ..video_logger import VideoLogger
import matplotlib.pyplot as plt

PRIMITIVE_COLORS = {
    "norm-pixel-pick-and-place": (255, 180, 80),          # orange
    "norm-pixel-pick-and-fling": (80, 200, 255),# cyan
    "no-operation": (255, 255, 255),             # gray
    "default": (255, 255, 255),                  # white
}

MILD_RED = (120, 120, 220)  # soft red (B, G, R)

TEXT_Y_STEP = 35
TEXT_Y_STATUS = 80
TEXT_BG_ALPHA = 0.6

def draw_big_arrowhead(img, p_from, p_to, color, size=28):
    """
    p_from, p_to are (x, y)
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6

    ux, uy = dx / norm, dy / norm      # direction
    px, py = -uy, ux                   # perpendicular

    tip = np.array([p_to[0], p_to[1]])

    left = np.array([
        p_to[0] - size * ux + size * 0.6 * px,
        p_to[1] - size * uy + size * 0.6 * py
    ])

    right = np.array([
        p_to[0] - size * ux - size * 0.6 * px,
        p_to[1] - size * uy - size * 0.6 * py
    ])

    # Convert to OpenCV (col, row) only here
    pts = np.array([
        (int(tip[0]),   int(tip[1])),
        (int(left[0]),  int(left[1])),
        (int(right[0]), int(right[1]))
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts], color)

def draw_colored_line(img, p_start, p_end, cmap, thickness=8, num_samples=20):
    """
    Draw a straight line from p_start → p_end with color gradient.
    p_start, p_end: (x, y) in pixel coords
    """
    xs = np.linspace(p_start[0], p_end[0], num_samples).astype(int)
    ys = np.linspace(p_start[1], p_end[1], num_samples).astype(int)

    for i in range(1, num_samples):
        alpha = i / (num_samples - 1)

        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
        color = cv2.applyColorMap(value, cmap)[0, 0].tolist()

        cv2.line(
            img,
            (xs[i - 1], ys[i - 1]),
            (xs[i], ys[i]),
            color,
            thickness
        )

    draw_big_arrowhead(
        img,
        (xs[-2], ys[-2]),
        (xs[-1], ys[-1]),
        color,
        size=30
    )

def draw_text_with_bg(img, text, org, color, scale=1.2, thickness=4):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - h - 5),
        (x + w + 5, y + 5),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, TEXT_BG_ALPHA, img, 1 - TEXT_BG_ALPHA, 0, img)
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

class PixelBasedPrimitiveEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None):
        super().__call__(episode_config, result, filename=filename)
        eid = episode_config['eid']
        frames = [info["observation"]["rgb"] for info in result["information"]]
        actions = result["actions"]
        picker_traj = [info["observation"]["picker_norm_pixel_pos"]
                       for info in result["information"]]  # [T][2,2]

        print('result["information"] keys', result["information"][0].keys())

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



            RED   = (50, 50, 200)     # softer red
            BLUE  = (200, 50, 50)     # softer blue
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
                pick_0  = norm_to_px(applied_action[:2])
                pick_1  = norm_to_px(applied_action[2:4])
                place_0 = norm_to_px(applied_action[4:6])
                place_1 = norm_to_px(applied_action[6:8])

                # ----- Ensure BLUE pick is always the LEFT one -----
                picks = [(pick_0, place_0), (pick_1, place_1)]
                picks_sorted = sorted(picks, key=lambda p: p[0][1])  # sort by x

                (left_pick, left_place), (right_pick, right_place) = picks_sorted
                # BLUE = left, RED = right

                # ----- Draw arrows with SMALLER arrowheads -----
                small_tip = 0.08    # << smaller arrowhead tip size

                # Colormaps (same convention as fling)
                cmap_left  = cv2.COLORMAP_WINTER   # BLUE-ish
                cmap_right = cv2.COLORMAP_AUTUMN   # RED-ish

                draw_colored_line(
                    img,
                    swap(left_pick),
                    swap(left_place),
                    cmap_left,
                    thickness=8,
                    num_samples=20
                )

                draw_colored_line(
                    img,
                    swap(right_pick),
                    swap(right_place),
                    cmap_right,
                    thickness=8,
                    num_samples=20
                )


                # ----- Hollow circles -----
                cv2.circle(img, swap(left_pick),  10, BLUE, 3)
                cv2.circle(img, swap(right_pick), 10, RED,  3)

            # ================================
            #        PICK & FLING
            # ================================
            elif key == "norm-pixel-pick-and-fling":
                pick_0 = norm_to_px(val[:2])
                pick_1 = norm_to_px(val[2:4])

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

                        cv2.line(img, swap(sampled_0[s-1]), swap(sampled_0[s]), color0, 8)
                        cv2.line(img, swap(sampled_1[s-1]), swap(sampled_1[s]), color1, 8)

                    def draw_hollow_circle(img, p, color, radius=10, thickness=3):
                        cv2.circle(img, swap(p), radius, color, thickness)

                    draw_hollow_circle(img, sampled_0[0], RED)
                    draw_hollow_circle(img, sampled_1[0], BLUE)
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

                    draw_triangle_down(img, traj_px_0[-1], RED)
                    draw_triangle_down(img, traj_px_1[-1], BLUE)

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