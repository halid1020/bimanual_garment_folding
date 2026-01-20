import cv2
import numpy as np

CV2_DISPLAY = ":1" #":0" # "localhost:10.0"
SIM_DISPLAY =  ":1" #:0" # ""


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