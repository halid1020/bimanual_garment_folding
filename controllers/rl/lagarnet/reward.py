import torch
import numpy as np

def identity(rewards, observations, actions, config=None):
    return rewards

def map_reward_range(rewards, map_min, map_max, old_min=-1.0, old_max=1.0):
    """
    Linearly maps rewards from [old_min, old_max] to [map_min, map_max].
    """
    # Formula: (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    range_ratio = (map_max - map_min) / (old_max - old_min)
    
    # Works seamlessly for both PyTorch tensors and NumPy arrays
    mapped_rewards = (rewards - old_min) * range_ratio + map_min
    
    return mapped_rewards

def coverage_alignment_bonus_and_penalty(rewards, observations, actions, config):
    if isinstance(rewards, torch.Tensor):
        rewards_ = rewards.clone()
    else:
        rewards_ = rewards.copy()

    # Calculate differences
    dNC = observations['normalised_coverage'][:, 1:] - observations['normalised_coverage'][:, :-1]
    dIoU = observations['max_IoU_to_flattened'][:, 1:] - observations['max_IoU_to_flattened'][:, :-1]

    # Apply base reward alignment
    rewards_[:, 0] = 0
    alpha = config.get('alpha', 0.5)
    
    # Dense reward for transitions
    dense_reward = alpha * dNC + (1 - alpha) * dIoU
    rewards_[:, 1:] = dense_reward 

    NC_success_thresh = config.get('NC_success_threshold', 0.9)
    IoU_success_thresh = config.get('IoU_success_threshold', 0.8)

    # Calculate masks for t and t-1
    is_success_curr = (observations['normalised_coverage'][:, 1:] > NC_success_thresh) & \
                      (observations['max_IoU_to_flattened'][:, 1:] > IoU_success_thresh)
                      
    is_success_prev = (observations['normalised_coverage'][:, :-1] > NC_success_thresh) & \
                      (observations['max_IoU_to_flattened'][:, :-1] > IoU_success_thresh)

    # 1. Apply Success Bonus
    if config.get('apply_success_bonus', False):
        rewards_[:, 1:][is_success_curr] = 1.0
    
    # 2. Apply "Mess Up" Penalty (Additive)
    if config.get('apply_mess_up_penalty', False):
        messed_up = is_success_prev & ~is_success_curr
        penalty_value = config.get('mess_up_penalty_value', 1.0)
        rewards_[:, 1:][messed_up] -= penalty_value
    
    # 3. Apply Large Action Penalty in Success States (Additive)
    if config.get('apply_large_action_penalty_when_success', False):
        pick_pos = actions[:, :, :2]
        place_pos = actions[:, :, 2:4]
        
        if isinstance(actions, torch.Tensor):
            action_dist = torch.norm(place_pos - pick_pos, p=2, dim=-1, keepdim=True)
        else:
            action_dist = np.linalg.norm(place_pos - pick_pos, axis=-1, keepdims=True)
            
        drag_threshold = config.get('drag_penalty_threshold', 0.2)
        large_action = action_dist > drag_threshold
        
        action_penalty_value = config.get('action_penalty_value', 1.0)
        rewards_[:, 1:][is_success_prev & large_action] -= action_penalty_value

    # --- CLAMP REWARDS BETWEEN -1 and 1 ---
    if isinstance(rewards_, torch.Tensor):
        rewards_ = torch.clamp(rewards_, min=-1.0, max=1.0)
    else:
        rewards_ = np.clip(rewards_, a_min=-1.0, a_max=1.0)

    map_min = config.get('map_min', None)
    map_max = config.get('map_max', None)

    if map_min is not None and map_max is not None:
        rewards_ = map_reward_range(rewards_, map_min, map_max, old_min=-1.0, old_max=1.0)

    return rewards_

def apply_NC_bonus_and_penalty(rewards, observations, actions, config=None):
    if isinstance(rewards, torch.Tensor):
        rewards_ = rewards.clone()
    else:
        rewards_ = rewards.copy()
        
    above_0_9 = observations['normalised_coverage'][:, :-1] > 0.9
    below_0_9 = observations['normalised_coverage'][:, 1:] < 0.9
    
    # Safely check for terminal states to avoid key errors if not passed
    if 'terminal' in observations:
        first_state_no_term = observations['terminal'][:, :-1] == 0
        condition = above_0_9 & below_0_9 & first_state_no_term
    else:
        condition = above_0_9 & below_0_9

    rewards_[:, 1:][condition] = 0

    above_0_9_5 = observations['normalised_coverage'][:] > 0.95
    rewards_[above_0_9_5] = 0.7

    return rewards_

# Register the functions so RSSM can fetch them via self.config.reward_processor
REWARD = {
    'identity': identity,
    'coverage_alignment': coverage_alignment_bonus_and_penalty,
    'nc_bonus_penalty': apply_NC_bonus_and_penalty
}