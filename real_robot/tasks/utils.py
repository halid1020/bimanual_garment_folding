import numpy as np

IOU_FLATTENING_TRESHOLD = 0.82
NC_FLATTENING_TRESHOLD = 0.95

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