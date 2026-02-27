import cv2
import numpy as np

MILD_RED = (120, 120, 220)  # soft red (B, G, R)

TEXT_Y_STEP = 35
TEXT_Y_STATUS = 80
TEXT_BG_ALPHA = 0.6

BLUE = (200, 50, 50)   # left picker
RED  = (50, 50, 200)   # right picker
MILD_RED = (120, 120, 220)  # soft red (B, G, R)

PRIMITIVE_COLORS = {
    "norm-pixel-pick-and-place": (255, 180, 80),          # orange
    "norm-pixel-pick-and-fling": (80, 200, 255),# cyan
    "no-operation": (255, 255, 255),             # gray
    "default": (255, 255, 255),                  # white
}

# -------------------------------
# Normalized → pixel coordinates
# -------------------------------
def norm_to_px(v, W, H, swap=False):
    xid = 1
    yid = 0
    if swap:
        xid = 0
        yid = 1
    x = int((v[xid] + 1) * 0.5 * W)
    y = int((v[yid] + 1) * 0.5 * H)
    return x, y

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


# OpenCV uses (row, col)
def swap(p):
    return (p[1], p[0])

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

def draw_pick_and_place(img, action):
    H, W = img.shape[:2]
    if len(action) > 4:
        pick_0  = norm_to_px(action[:2], W, H)
        pick_1  = norm_to_px(action[2:4], W, H)
        place_0 = norm_to_px(action[4:6], W, H)
        place_1 = norm_to_px(action[6:8], W, H)

        # ----- Ensure BLUE pick is always the LEFT one -----
        picks = [(pick_0, place_0), (pick_1, place_1)]
        picks_sorted = sorted(picks, key=lambda p: p[0][1])  # sort by x

        (left_pick, left_place), (right_pick, right_place) = picks_sorted
        
    else:

        left_pick  = norm_to_px(action[:2], W, H)
        left_place  = norm_to_px(action[2:4], W, H)

   
    # BLUE = left, RED = right

    # ----- Draw arrows with SMALLER arrowheads -----
    small_tip = 0.08    # << smaller arrowhead tip size

    # Colormaps (same convention as fling)
    cmap_left  = cv2.COLORMAP_COOL   # BLUE-ish
    cmap_right = cv2.COLORMAP_AUTUMN   # RED-ish

    draw_colored_line(
        img,
        left_pick,
        left_place,
        cmap_left,
        thickness=8,
        num_samples=20
    )

    if len(action) > 4:
        draw_colored_line(
            img,
            right_pick,
            right_place,
            cmap_right,
            thickness=8,
            num_samples=20
        )


    BLUE = cv2.applyColorMap(
        np.uint8([[[0]]]), cmap_left
    )[0, 0].tolist()
                    
    RED = cv2.applyColorMap(
        np.uint8([[[0]]]), cmap_right
    )[0, 0].tolist()
    
    # ----- Hollow circles -----
    cv2.circle(img, left_pick,  10, BLUE, 3)

    if len(action) > 4:
        cv2.circle(img, right_pick, 10, RED,  3)

    return img


# ------------------------------------------------------------------------------
# Add/Replace these in real_robot/utils/draw_utils.py
# ------------------------------------------------------------------------------

def draw_gradient_path(img, points, cmap, thickness=6):
    """
    Draws a polyline from a list of (x,y) points using a color gradient,
    ending with a triangular arrowhead.
    NOTE: Assumes input 'points' are (x,y) and swaps them to (y,x) for drawing calls.
    """
    if points is None or len(points) < 2:
        return

    num_pts = len(points)
    
    # 1. Draw the line segments with gradient
    for i in range(1, num_pts):
        # alpha goes from >0.0 to 1.0
        alpha = i / (num_pts - 1)
        
        # Map alpha to colormap value (255 -> 0 gives transition)
        value = np.uint8([[[int((1.0 - alpha) * 255)]]])
        color = cv2.applyColorMap(value, cmap)[0, 0].tolist()

        # Note: Calling swap() to convert (x,y) -> (y,x) for cv2 convention in this file
        cv2.line(
            img, 
            points[i - 1], 
            points[i], 
            color, 
            thickness
        )
    
    # 2. Draw the Arrowhead at the end
    # Use the last two points to determine orientation
    p_from = points[-2]
    p_to = points[-1]

    # Get the final color (alpha=1.0 -> value=0 corresponds to end of colormap)
    value_end = np.uint8([[[0]]])
    color_end = cv2.applyColorMap(value_end, cmap)[0, 0].tolist()

    # Draw arrowhead using swapped coordinates (y,x)
    draw_big_arrowhead(
        img, 
        p_from, 
        p_to, 
        color_end, 
        size=30 # Same size as used in draw_colored_line
    )

def draw_pick_and_fling(img, action, traj_0=None, traj_1=None):
    """
    Draws pick markers and optional fling trajectories with arrowheads.
    
    Args:
        img: The image to draw on.
        action: The normalized action vector [pick0_x, pick0_y, pick1_x, pick1_y, ...].
        traj_0: (Optional) List of (x,y) pixels for the first robot's trajectory.
        traj_1: (Optional) List of (x,y) pixels for the second robot's trajectory.
    """
    H, W = img.shape[:2]
    
    # 1. Parse Pick Points (norm_to_px returns x,y)
    pick_0 = norm_to_px(action[:2], W, H, swap=False)
    pick_1 = norm_to_px(action[2:4], W, H, swap=False)

    # 2. Sort Left/Right by X-coordinate to ensure consistent coloring (Blue=Left, Red=Right)
    picks_data = [(pick_0, traj_0), (pick_1, traj_1)]
    picks_sorted = sorted(picks_data, key=lambda p: p[0][0]) 

    (left_pick, left_traj), (right_pick, right_traj) = picks_sorted

    # 3. Define Colors
    cmap_left = cv2.COLORMAP_COOL    # Blue-ish
    cmap_right = cv2.COLORMAP_AUTUMN # Red-ish

    # Extract solid colors for circles (start of colormap)
    # Using 255 here gets the start color, matching the logic in draw_gradient_path where 1.0 alpha maps to 0 value.
    BLUE = cv2.applyColorMap(np.uint8([[[255]]]), cmap_left)[0, 0].tolist()
    RED  = cv2.applyColorMap(np.uint8([[[255]]]), cmap_right)[0, 0].tolist()

    # 4. Draw Trajectories (if provided)
    # Increase thickness slightly for better visibility against backgrounds
    if left_traj is not None:
        draw_gradient_path(img, left_traj, cmap_right, thickness=6)
    
    if right_traj is not None:
        draw_gradient_path(img, right_traj, cmap_left, thickness=6)

    # 5. Draw Pick Circles (Hollow)
    # Calling swap() to convert (x,y) -> (y,x)
    cv2.circle(img, left_pick, 10, BLUE, 3)
    cv2.circle(img, right_pick, 10, RED, 3)

    return img