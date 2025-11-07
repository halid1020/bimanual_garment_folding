import cv2
import numpy as np


def click_points_pick_and_place(window_name, img, mask=None):
    """
    Displays the RGB image (and optional mask overlay) for user to click four points:
    pick0, place0, pick1, place1.

    Args:
        window_name (str): Window title.
        img (np.ndarray): RGB image (H, W, 3).
        mask (np.ndarray, optional): Binary or grayscale mask. If provided, it's overlaid on the RGB image.

    Returns:
        tuple: (pick0, place0, pick1, place1)
    """
    clicks = []
    clone = img.copy()

    # If a mask is provided, overlay it transparently on the RGB
    if mask is not None:
        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        mask_color = np.zeros_like(clone)
        mask_color[mask_gray > 0] = (0, 255, 0)  # green overlay
        alpha = 0.4
        overlay = cv2.addWeighted(clone, 1 - alpha, mask_color, alpha, 0)
    else:
        overlay = clone

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, overlay)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, overlay)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Please click 4 points in order: PICK0 → PLACE0 → PICK1 → PLACE1")
    print("Press 'q' to cancel early.")

    while len(clicks) < 4:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)

    if len(clicks) < 4:
        raise RuntimeError(f"Expected 4 points, got {len(clicks)}")

    return clicks[0], clicks[1], clicks[2], clicks[3]

def click_points_pick_and_fling(window_name, img):
    clicks = []
    clone = img.copy()

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))
            cv2.circle(clone, (int(x), int(y)), 5, (0,255,0), -1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) 
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("Please click PICK point then PLACE point on the image window.")
    while len(clicks) < 2:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)
    if len(clicks) < 2:
        raise RuntimeError("2 points not selected")
    return clicks[0], clicks[1]
