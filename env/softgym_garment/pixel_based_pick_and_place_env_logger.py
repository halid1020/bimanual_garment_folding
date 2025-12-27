import os
import cv2
import numpy as np
from ..video_logger import VideoLogger
import matplotlib.pyplot as plt
from .draw_utils import *


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

        

        images = []

        for i in range(len(frames)-1):
            img = frames[i].copy()
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

            # -------------------------------
            # Step text
            # -------------------------------
            step_text = f"Step {i + 1}: Pick and Place"
            primitive_color = PRIMITIVE_COLORS.get("norm-pixel-pick-and-place", PRIMITIVE_COLORS["default"])
            draw_text_with_bg(
                img,
                step_text,
                (10, TEXT_Y_STEP),
                primitive_color
            )
            # -------------------------------
            # Extract action
            # -------------------------------
            # act must contain exactly:
            # pick_0, pick_1, place_0, place_1
            #print('results keys', result.keys())
            applied_action = result["information"][i+1]['applied_action']

            img = draw_pick_and_place(img, applied_action)
            

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
