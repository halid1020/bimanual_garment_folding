"""
Utility functions for environment rendering, interactive UI, and image processing.
Centralizes OpenCV rendering logic to prevent redundancy across human policy agents.
"""

import os
import cv2
import numpy as np

# Display Configuration
REMOTE = False
# Automatically grabs the active display (e.g., :1) from the OS, defaults to :0 if empty
ACTUAL_DISPLAY = os.environ.get("DISPLAY", ":0")

# Uses the remote address if REMOTE=True, otherwise uses the automatically detected display
CV2_DISPLAY = "localhost:10.0" if REMOTE else ACTUAL_DISPLAY

# Runs headlessly if REMOTE=True, otherwise binds to the automatically detected display
SIM_DISPLAY = "" if REMOTE else ACTUAL_DISPLAY


def overlay_workspaces(rgb: np.ndarray, state: dict) -> np.ndarray:
    H, W = rgb.shape[:2]
    
    for robot_key, color in [('robot0_mask', (255, 0, 0)), ('robot1_mask', (0, 0, 255))]:
        if robot_key in state['observation']:
            mask = state['observation'][robot_key].astype(bool)
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            rgb = apply_workspace_shade(rgb, mask, color=color, alpha=0.2)
            
    return rgb

def apply_workspace_shade(rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.35) -> np.ndarray:
    shaded = rgb.copy()
    overlay = np.zeros_like(rgb, dtype=np.uint8)
    overlay[:] = color
    shaded[mask] = cv2.addWeighted(rgb[mask], 1 - alpha, overlay[mask], alpha, 0)
    return shaded

def draw_text_bottom_right(img: np.ndarray, lines: list, margin: int = 10, font_scale: float = 0.5, thickness: int = 1):
    """
    Draws multiple text lines with black background boxes at the bottom-right corner.
    """
    h, w = img.shape[:2]
    
    # Calculate the total height required for the text block
    total_height = 0
    rendered_lines = []
    for text, color in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        rendered_lines.append((text, color, tw, th))
        total_height += th + 15
        
    # Start Y from the bottom
    current_y = h - total_height - margin
    
    for text, color, tw, th in rendered_lines:
        x = w - tw - margin
        # Draw background box
        cv2.rectangle(img, (x - 5, current_y - 5), (x + tw + 5, current_y + th + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(img, text, (x, current_y + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        current_y += th + 15

def draw_evaluation_metrics(img: np.ndarray, state: dict):
    """
    Extracts the environment metrics and neatly overlays them on the bottom right of the whole UI.
    """
    if 'evaluation' in state and state['evaluation']:
        metrics = state['evaluation']
        
        text_lines = [
            (f"Success: {state.get('success', False)}", (0, 255, 0) if state.get('success', False) else (0, 0, 255)),
            (f"NC: {metrics.get('normalised_coverage', 0):.3f}", (255, 255, 255)),
            (f"Max IoU(flat): {metrics.get('max_IoU_to_flattened', 0):.3f}", (255, 255, 255)),
            (f"Align IoU(flat): {metrics.get('algn_IoU_to_flattened', 0):.3f}", (255, 255, 255))
        ]
        
        # --- ADDED: Max and Align IoU for the fold state ---
        if 'max_IoU' in metrics:
            text_lines.append((f"Max IoU(fold): {metrics['max_IoU']:.3f}", (255, 255, 255)))
        if 'algn_IoU' in metrics:
            text_lines.append((f"Align IoU(fold): {metrics['algn_IoU']:.3f}", (255, 255, 255)))
        
        draw_text_bottom_right(img, text_lines, font_scale=0.7, thickness=2)

def overlay_active_goal_contour(rgb: np.ndarray, state: dict) -> np.ndarray:
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    if 'goals' in state and isinstance(state['goals'], list) and len(state['goals']) > 0:
        evaluation = state.get('evaluation', {})
        active_idx = evaluation.get('active_subgoal_idx', 0)
        
        TARGET_THRESHOLDS = evaluation.get('iou_thresholds', [0.80] * len(state['goals']))
        active_goal_state = state['goals'][active_idx]
        
        if 'observation' in active_goal_state and 'mask' in active_goal_state['observation']:
            goal_mask = active_goal_state['observation']['mask']
            current_mask = state['observation']['mask']
            
            _draw_contour(rgb, goal_mask)
            
            current_iou = calculate_strict_iou(current_mask, goal_mask)
            target_iou = TARGET_THRESHOLDS[active_idx] if active_idx < len(TARGET_THRESHOLDS) else 0.80
            
            progress_color = (0, 255, 0) if current_iou >= target_iou else (0, 255, 255)
            
            h, w = rgb.shape[:2]
            # --- ADJUSTED: Increased spacing between the two lines (h - 35 and h - 10) ---
            cv2.putText(rgb, f"Target: Subgoal {active_idx + 1}/{len(state['goals'])}", 
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
            cv2.putText(rgb, f"IoU: {current_iou:.3f} / Target: {target_iou:.3f}", 
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, progress_color, FONT_THICKNESS)
                        
    elif 'goal' in state and 'mask' in state['goal']:
        goal_mask = state['goal']['mask']
        current_mask = state['observation']['mask']
        
        _draw_contour(rgb, goal_mask)
        
        current_iou = calculate_strict_iou(current_mask, goal_mask)
        target_iou = 0.80
        progress_color = (0, 255, 0) if current_iou >= target_iou else (0, 255, 255)
        
        h, w = rgb.shape[:2]
        # --- ADJUSTED: Increased spacing here as well ---
        cv2.putText(rgb, f"Target: Final Subgoal", 
                    (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        cv2.putText(rgb, f"IoU: {current_iou:.3f} / Target: {target_iou:.3f}", 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, progress_color, FONT_THICKNESS)
        
    return rgb

def _draw_contour(img: np.ndarray, mask: np.ndarray):
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
    mask_uint8 = np.array(mask * 255, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

def append_goal_grid(img: np.ndarray, state: dict) -> np.ndarray:
    if 'goals' not in state:
        if 'goal' in state and 'rgb' in state['goal']:
            goal_rgb = cv2.cvtColor(state['goal']['rgb'], cv2.COLOR_BGR2RGB)
            goal_rgb = cv2.resize(goal_rgb, (img.shape[0], img.shape[0]))
            cv2.putText(goal_rgb, "Final Subgoal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- NEW: Draw metrics strictly on the goal image ---
            draw_evaluation_metrics(goal_rgb, state)
            
            return np.concatenate([img, goal_rgb], axis=1)
        return img

    rgbs = []
    for i, goal in enumerate(state['goals']):
        g = goal['observation']['rgb']
        if g.shape[-1] == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        g = cv2.resize(g, (256, 256))
        
        cv2.putText(g, f"Subgoal {i+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rgbs.append(g)

    while len(rgbs) < 4:
        rgbs.append(np.zeros((256, 256, 3), dtype=np.uint8))

    top_row = np.concatenate([rgbs[0], rgbs[1]], axis=1)
    bottom_row = np.concatenate([rgbs[2], rgbs[3]], axis=1)
    goal_rgb = np.concatenate([top_row, bottom_row], axis=0)
    
    # --- NEW: Draw metrics strictly on the 2x2 goal grid ---
    draw_evaluation_metrics(goal_rgb, state)
    
    combined_img = np.concatenate([img, goal_rgb], axis=1)
    cv2.line(combined_img, (img.shape[1], 0), (img.shape[1], combined_img.shape[0]), (255, 255, 255), 2)
    
    return combined_img

def get_user_primitive_selection(img: np.ndarray, primitive_names: list, window_name: str) -> int:
    """
    Spawns an interactive OpenCV window with a side panel of clickable buttons 
    to select an action primitive. Also supports keyboard shortcuts (1, 2, 3...).
    """
    os.environ["DISPLAY"] = CV2_DISPLAY
    panel_width = 280
    button_height = 50
    margin = 15
    
    # Draw dark panel
    panel_img = np.ones((img.shape[0], panel_width, 3), dtype=np.uint8) * 40
    
    cv2.putText(panel_img, "Select Primitive:", (margin, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    buttons = []
    start_y = 50
    for i, name in enumerate(primitive_names):
        # Clean up the raw string names for a prettier UI
        display_name = name.replace("norm-pixel-", "").replace("-", " ").title()
        
        btn_top_left = (margin, start_y + i * (button_height + margin))
        btn_bottom_right = (panel_width - margin, start_y + i * (button_height + margin) + button_height)
        buttons.append((btn_top_left, btn_bottom_right))
        
        # Draw Button Box
        cv2.rectangle(panel_img, btn_top_left, btn_bottom_right, (80, 80, 80), -1)
        # Draw Button Text
        cv2.putText(panel_img, f"{i+1}. {display_name}", 
                    (btn_top_left[0] + 10, btn_top_left[1] + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    display_img = np.concatenate([img, panel_img], axis=1)
    
    selected_idx = [-1] # Wrapped in list to allow mutation inside callback
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= img.shape[1]:  # Clicked in the side panel
                panel_x = x - img.shape[1]
                for i, (tl, br) in enumerate(buttons):
                    if tl[0] <= panel_x <= br[0] and tl[1] <= y <= br[1]:
                        selected_idx[0] = i
                        break
    
    cv2.imshow(window_name, display_img)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while selected_idx[0] == -1:
        key = cv2.waitKey(10) & 0xFF
        # Keyboard shortcuts 1 through N
        if ord('1') <= key <= ord(str(len(primitive_names))):
            selected_idx[0] = key - ord('1')
    
    cv2.destroyAllWindows()
    os.environ["DISPLAY"] = SIM_DISPLAY
    
    return selected_idx[0]

def get_user_clicks_with_undo(img: np.ndarray, num_clicks: int, window_name: str) -> list:
    os.environ["DISPLAY"] = CV2_DISPLAY
    clicks = []
    
    panel_width = 80
    button_top_left = (10, 10)
    button_bottom_right = (panel_width - 10, 50)

    def redraw_image():
        temp_img = img.copy()
        for i, (x, y) in enumerate(clicks):
            color = (0, 255, 0) if i < 2 else (0, 0, 255)
            if i % 2 == 0: 
                cv2.circle(temp_img, (x, y), 5, color, -1)
            else:          
                cv2.drawMarker(temp_img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        panel_img = np.ones((img.shape[0], panel_width, 3), dtype=np.uint8) * 50
        cv2.rectangle(panel_img, button_top_left, button_bottom_right, (0, 0, 0), -1)
        cv2.putText(panel_img, 'UNDO', (button_top_left[0] + 5, button_top_left[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display_img = np.concatenate([temp_img, panel_img], axis=1)
        cv2.imshow(window_name, display_img)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= img.shape[1]:  
                panel_x = x - img.shape[1]
                if (button_top_left[0] <= panel_x <= button_bottom_right[0] and 
                    button_top_left[1] <= y <= button_bottom_right[1]):
                    if clicks:
                        clicks.pop()
                        redraw_image()
            else:  
                if len(clicks) < num_clicks:
                    clicks.append((x, y))
                    redraw_image()

    redraw_image()
    cv2.setMouseCallback(window_name, mouse_callback)

    while len(clicks) < num_clicks:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    os.environ["DISPLAY"] = SIM_DISPLAY
    return clicks

def calculate_strict_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the strict, absolute pixel-wise IoU between two masks (no translation)."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0

def normalize_clicks(clicks: list, width: int, height: int, header_offset: int = 0) -> list:
    normalized = []
    for x, y in clicks:
        adj_y = y - header_offset
        normalized.extend([
            (adj_y / width) * 2 - 1,
            (x / height) * 2 - 1
        ])
    return normalized