import os
import cv2
import numpy as np
from ..video_logger import VideoLogger
import matplotlib.pyplot as plt

TEXT_Y_STEP = 30
TEXT_BG_ALPHA = 0.6

BLUE = (200, 50, 50)   # left picker
RED  = (50, 50, 200)   # right picker
MILD_RED = (120, 120, 220)  # soft red (B, G, R)

def draw_text_with_bg(img, text, org, color=(255, 255, 255), scale=1.0, thickness=2):
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


class PixelBasedPickAndPlaceEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None):
        super().__call__(episode_config, result, filename=filename)

        eid = episode_config["eid"]
        frames = [info["observation"]["rgb"] for info in result["information"]]
        #actions = result["actions"]

        H, W = 512, 512

        if filename is None:
            filename = "manipulation"

        out_dir = os.path.join(self.log_dir, filename, "performance_visualisation")
        os.makedirs(out_dir, exist_ok=True)

        # -------------------------------
        # Normalized → pixel coordinates
        # -------------------------------
        def norm_to_px(v):
            x = int((v[0] + 1) * 0.5 * W)
            y = int((v[1] + 1) * 0.5 * H)
            return x, y

        # OpenCV uses (row, col)
        def swap(p):
            return (p[1], p[0])

        images = []

        for i in range(len(frames)-1):
            img = frames[i].copy()
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

            # -------------------------------
            # Step text
            # -------------------------------
            step_text = f"Step {i + 1}: Pick and Place"
            draw_text_with_bg(img, step_text, (10, TEXT_Y_STEP))

            # -------------------------------
            # Extract action
            # -------------------------------
            # act must contain exactly:
            # pick_0, pick_1, place_0, place_1
            #print('results keys', result.keys())
            applied_action = result["information"][i+1]['applied_action']

            
            small_tip = 0.08
            if len(applied_action) > 4:
                pick_0  = norm_to_px(applied_action[:2])
                pick_1  = norm_to_px(applied_action[2:4])
                place_0 = norm_to_px(applied_action[4:6])
                place_1 = norm_to_px(applied_action[6:8])
                # -------------------------------
                # Ensure LEFT pick is BLUE
                # -------------------------------
                picks = [(pick_0, place_0), (pick_1, place_1)]
                picks_sorted = sorted(picks, key=lambda p: p[0][1])  # sort by x

                (left_pick, left_place), (right_pick, right_place) = picks_sorted
                # -------------------------------
                # Draw arrows + hollow circles
                # -------------------------------
               

                cv2.arrowedLine(
                    img, swap(left_pick), swap(left_place),
                    BLUE, 5, tipLength=small_tip
                )
                cv2.arrowedLine(
                    img, swap(right_pick), swap(right_place),
                    RED, 5, tipLength=small_tip
                )

                cv2.circle(img, swap(left_pick),  8, BLUE, 2)
                cv2.circle(img, swap(right_pick), 8, RED,  2)
                
            else:
                
                left_pick  = norm_to_px(applied_action[:2])
                left_place  = norm_to_px(applied_action[2:4])
                
                cv2.arrowedLine(
                    img, swap(left_pick), swap(left_place),
                    BLUE, 5, tipLength=small_tip
                )
                

                cv2.circle(img, swap(left_pick),  8, BLUE, 2)
              

            

            images.append(img)

        # Add final frame (no action)
        images.append(cv2.resize(frames[-1], (W, H)))

        # ===========================================
        #  Matplotlib grid visualization
        # ===========================================
        MAX_COLS = 6
        num_images = len(images)
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS

        fig, axes = plt.subplots(
            num_rows, MAX_COLS,
            figsize=(3 * MAX_COLS, 3 * num_rows),
        )

        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        idx = 0
        for r in range(num_rows):
            for c in range(MAX_COLS):
                ax = axes[r][c]
                if idx < num_images:
                    img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                ax.axis("off")
                idx += 1

        plt.tight_layout()

        save_path = os.path.join(out_dir, f"episode_{eid}_trajectory.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
