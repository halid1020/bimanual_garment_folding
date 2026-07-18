import numpy as np
import cv2
from termcolor import colored

IOU_FLATTENING_TRESHOLD = 0.82
NC_FLATTENING_TRESHOLD = 0.95


def adjust_goal_mask_ui(obs, extra_instructions=None, draw_mid_line=False):
    """Interactive OpenCV UI to translate/rotate a cloth goal mask.

    Renders the workspace masks as a red/blue canvas (Robot 0 in Red, Robot 1 in
    Blue) with the draggable cloth mask overlaid in Green. Mouse-drag translates,
    'A'/'D' rotate by 1 deg, ENTER/ESC confirm. The final transform is applied
    (in-place) to obs['mask'] (INTER_NEAREST), obs['rgb'] (INTER_LINEAR) and
    obs['depth'] (INTER_NEAREST).

    Args:
        obs: observation dict with 'mask', 'robot0_mask', 'robot1_mask' (and
             optionally 'rgb', 'depth'). Mutated in place.
        extra_instructions: optional list of strings, printed in green to the
             terminal and rendered as extra on-screen lines.
        draw_mid_line: if True, draw a horizontal reference line across the image
             middle on the display each frame (NOT baked into the saved obs).

    Returns:
        (obs, (offset_x, offset_y, angle)) -- obs is the same (mutated) dict.
    """
    # --- 1. Extract masks to build the UI canvas ---
    cloth_mask = obs['mask'].astype(np.uint8) * 255
    h, w = cloth_mask.shape

    # FIX: Convert boolean masks to 0-255 scale so they are visible on the image
    r0_mask_vis = obs['robot0_mask'].astype(np.uint8) * 255
    r1_mask_vis = obs['robot1_mask'].astype(np.uint8) * 255

    # Create background canvas: Robot 0 in Red, Robot 1 in Blue
    canvas_base = np.zeros((h, w, 3), dtype=np.uint8)
    canvas_base[:, :, 2] = r0_mask_vis  # Red channel
    canvas_base[:, :, 0] = r1_mask_vis  # Blue channel

    # --- 2. Set up interactive variables ---
    dragging = False
    ix, iy = -1, -1
    offset_x, offset_y = 0, 0
    current_offset_x, current_offset_y = 0, 0
    current_angle = 0  # Track rotation in degrees

    def drag_mouse_callback(event, x, y, flags, param):
        nonlocal dragging, ix, iy, offset_x, offset_y, current_offset_x, current_offset_y

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                current_offset_x = offset_x + (x - ix)
                current_offset_y = offset_y + (y - iy)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            offset_x = current_offset_x
            offset_y = current_offset_y

    window_name = 'Drag Goal Mask (A/D to Rotate, ENTER to confirm)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, drag_mouse_callback)

    print(colored("\n[adjust_goal_mask_ui] Please drag the GREEN cloth mask to the desired goal location.", "green"))
    print(colored("[adjust_goal_mask_ui] Use 'A' and 'D' keys to rotate.", "green"))
    print(colored("[adjust_goal_mask_ui] Press ENTER to confirm.\n", "green"))
    if extra_instructions:
        for line in extra_instructions:
            print(colored(f"[adjust_goal_mask_ui] {line}", "green"))

    # Define center of rotation (center of the image)
    center = (w // 2, h // 2)

    # --- 3. Render Loop ---
    while True:
        # Create transformation matrix: Rotate first, then Translate
        M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
        M[0, 2] += current_offset_x
        M[1, 2] += current_offset_y

        shifted_mask = cv2.warpAffine(cloth_mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # Overlay cloth mask in Green channel
        display = canvas_base.copy()
        display[:, :, 1] = np.maximum(display[:, :, 1], shifted_mask)

        # Optional horizontal reference line across the image middle (display-only)
        if draw_mid_line:
            cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 255), 1)

        # Add on-screen instructions
        cv2.putText(display, "Drag mouse to move.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "'A' / 'D' to rotate.", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "ENTER to confirm.", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Extra instruction lines rendered under the existing ones
        if extra_instructions:
            for j, line in enumerate(extra_instructions):
                cv2.putText(display, line, (10, 120 + j * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window_name, display)

        # Break loop on Enter (13) or ESC (27)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 27:
            break
        elif key == ord('a') or key == ord('A'):
            current_angle += 1  # Rotate counter-clockwise 1 degree
        elif key == ord('d') or key == ord('D'):
            current_angle -= 1  # Rotate clockwise 1 degree

    cv2.destroyAllWindows()

    # --- 4. Apply the final transformation ---
    if offset_x != 0 or offset_y != 0 or current_angle != 0:
        M_final = cv2.getRotationMatrix2D(center, current_angle, 1.0)
        M_final[0, 2] += offset_x
        M_final[1, 2] += offset_y

        # Warp the boolean mask using INTER_NEAREST to keep sharp 0/1 edges
        new_mask = cv2.warpAffine(obs['mask'].astype(np.uint8), M_final, (w, h), flags=cv2.INTER_NEAREST)
        obs['mask'] = new_mask > 0

        # Warp RGB and Depth
        if 'rgb' in obs:
            # INTER_LINEAR is fine for RGB
            obs['rgb'] = cv2.warpAffine(obs['rgb'], M_final, (w, h), flags=cv2.INTER_LINEAR)
        if 'depth' in obs:
            # INTER_NEAREST is strictly needed for depth to prevent floating point interpolation artifacts
            obs['depth'] = cv2.warpAffine(obs['depth'], M_final, (w, h), flags=cv2.INTER_NEAREST)

        print(f"[adjust_goal_mask_ui] Goal updated. Offset X:{offset_x}, Y:{offset_y}, Angle: {current_angle} deg")

    return obs, (offset_x, offset_y, current_angle)

def speedFolding_approx_reward(last_info, action, info):
    """
        In the original paper, it used a pretrained smoothness classifier to calculate the smoothness of the folding.
        Here, we use the max IoU to approximate the smoothness.
    """
    if last_info is None:
        last_info = info
    delta_coverage = info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage'] # -1 to 1

    smoothness = info['evaluation']['max_IoU_to_flattened'] - last_info['evaluation']['max_IoU_to_flattened'] # -1 to 1

    alpha = 2
    beta = 1

    return max(np.tanh(alpha*delta_coverage + beta*smoothness), 0) # 0-1

def coverage_alignment_reward(last_info, action, info):

    if last_info is None:
        last_info = info
    r_ca = speedFolding_approx_reward(last_info, action, info)
    dc = info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage']
    ds = info['evaluation']['max_IoU_to_flattened'] - last_info['evaluation']['max_IoU_to_flattened']
    nc = info['evaluation']['normalised_coverage']
    iou = info['evaluation']['max_IoU_to_flattened']
    epsilon_c = 1e-4
    epsilon_s = 1e-4
    max_c = 0.99
    max_iou = 0.85
    b = 0.7
    
    if nc - dc > 0.9 and nc < 0.9:
        return 0
    
    if nc >= 0.95:
        return b
    
    return r_ca


def coverage_alignment_bonus_and_penalty(last_info, action, info, config=None):
    if config is None:
        config = {}
        
    if last_info is None:
        last_info = info

    # Extract current and previous metrics
    nc_curr = info['evaluation']['normalised_coverage']
    nc_prev = last_info['evaluation']['normalised_coverage']
    iou_curr = info['evaluation']['max_IoU_to_flattened']
    iou_prev = last_info['evaluation']['max_IoU_to_flattened']

    # Calculate differences
    dNC = nc_curr - nc_prev
    dIoU = iou_curr - iou_prev

    # Apply base reward alignment
    alpha = config.get('alpha', 0.5)
    reward = alpha * dNC + (1 - alpha) * dIoU

    # Thresholds for success
    NC_success_thresh = config.get('NC_success_threshold', 0.9)
    IoU_success_thresh = config.get('IoU_success_threshold', 0.8)

    # Calculate boolean masks for current and previous steps
    is_success_curr = (nc_curr > NC_success_thresh) and (iou_curr > IoU_success_thresh)
    is_success_prev = (nc_prev > NC_success_thresh) and (iou_prev > IoU_success_thresh)

    # 1. Apply Success Bonus
    if config.get('apply_success_bonus', False):
        if is_success_curr:
            reward = 1.0

    # 2. Apply "Mess Up" Penalty (Additive)
    if config.get('apply_mess_up_penalty', False):
        if is_success_prev and not is_success_curr:
            reward -= config.get('mess_up_penalty_value', 1.0)

    # 3. Apply Large Action Penalty in Success States (Additive)
    if config.get('apply_large_action_penalty_when_success', False):
        # Parse the action to get pick and place positions
        if action is None:
            action_dist = 0.0
        else:
            # Assuming the dict structure seen in planet_clothpick_hueristic_reward
            if isinstance(action, dict) and 'norm_pixel_pick_and_place' in action:
                act_dict = action['norm_pixel_pick_and_place']
                pick_pos = act_dict['pick_0']
                place_pos = act_dict['place_0']
            else:
                # Fallback if action is passed as a flat array [pick_x, pick_y, place_x, place_y]
                flat_action = np.array(action).flatten()
                if len(flat_action) >= 4:
                    pick_pos = flat_action[:2]
                    place_pos = flat_action[2:4]
                else:
                    pick_pos, place_pos = np.zeros(2), np.zeros(2)

            action_dist = np.linalg.norm(np.array(place_pos) - np.array(pick_pos))

        drag_threshold = config.get('drag_penalty_threshold', 0.2)
        if is_success_prev and action_dist > drag_threshold:
            reward -= config.get('action_penalty_value', 1.0)

    # --- CLAMP REWARDS BETWEEN -1 and 1 ---
    reward = np.clip(reward, a_min=-1.0, a_max=1.0)

    # Remap reward range if specified
    map_min = config.get('map_min', None)
    map_max = config.get('map_max', None)

    if map_min is not None and map_max is not None:
        # Standard range mapping formula: (val - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        reward = (reward - (-1.0)) / (1.0 - (-1.0)) * (map_max - map_min) + map_min

    return reward