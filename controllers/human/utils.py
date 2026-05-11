import cv2
import numpy as np
import os

REMOTE = True
ACTUAL_DISPLAY = os.environ.get("DISPLAY", "localhost:10.0")
if REMOTE:
    CV2_DISPLAY =  "localhost:10.0" 
    SIM_DISPLAY =  ""
else:
    CV2_DISPLAY =  ":0"
    SIM_DISPLAY =  ":0"


import cv2
import numpy as np

def overlay_workspaces(rgb, state):
    """
    Overlays the workspace masks for robot0 and robot1 onto the provided RGB image.
    """
    H, W = rgb.shape[:2]
    
    if 'robot0_mask' in state['observation']:
        mask0 = state['observation']['robot0_mask'].astype(bool)

        if mask0.shape[:2] != (H, W):
            mask0 = cv2.resize(
                mask0.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        rgb = apply_workspace_shade(
            rgb,
            mask0,
            color=(255, 0, 0),  # Blue in BGR
            alpha=0.2
        )

    if 'robot1_mask' in state['observation']:
        mask1 = state['observation']['robot1_mask'].astype(bool)

        if mask1.shape[:2] != (H, W):
            mask1 = cv2.resize(
                mask1.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        rgb = apply_workspace_shade(
            rgb,
            mask1,
            color=(0, 0, 255),  # Red in BGR
            alpha=0.2
        )
        
    return rgb

def apply_workspace_shade(rgb, mask, color, alpha=0.35):
    """
    Shade pixels where mask == True with given BGR color.
    """
    shaded = rgb.copy()
    overlay = np.zeros_like(rgb, dtype=np.uint8)
    overlay[:] = color

    shaded[mask] = cv2.addWeighted(
        rgb[mask], 1 - alpha,
        overlay[mask], alpha,
        0
    )
    return shaded

def draw_text_top_right(
    img,
    lines,
    margin=10,
    font_scale=0.7,
    thickness=2
):
    """
    Draw multiple text lines at the top-right corner of img.
    """
    h, w = img.shape[:2]
    y = margin + 20

    for text, color in lines:
        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )
        x = w - tw - margin

        # background box
        cv2.rectangle(
            img,
            (x - 5, y - th - 5),
            (x + tw + 5, y + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        y += th + 15