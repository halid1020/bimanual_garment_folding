import numpy as np
def particle_distance_reward(mpd, threshold=0.05, scale=0.5):
    """
    Reward based on mean particle distance (mpd).

    - If mpd <= threshold → reward = 1
    - If mpd > threshold → reward decays toward 0

    :param mpd: mean particle distance (float)
    :param threshold: cutoff distance (float)
    :param scale: controls how fast reward decays (float)
    :return: reward in [0, 1]

    Example behavior (with scale=0.5):

        mpd = 0.03 → reward = 1.0

        mpd = 0.10 → reward = 0.9

        mpd = 0.20 → reward = 0.7

        mpd = 0.55 → reward = 0.0
    """
    
    if mpd <= threshold:
        return 1.0
    else:
        # Map linearly: higher mpd → lower reward
        # Example: mpd=threshold → 1, mpd=threshold+scale → 0
        reward = max(0.0, 1.0 - (mpd - threshold) / scale)
        return reward

def coverage_alignment_reward(last_info, action, info):
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
