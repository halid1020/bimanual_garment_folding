import numpy as np

import math

import math

def particle_distance_reward(mpd, threshold=0.05, k=233.00077621128227, p=1.9419474686965421):
    """
    Single-equation stretched-exponential reward:
      r = exp(-k * (mpd - threshold)^p)  for mpd > threshold
      r = 1.0                              for mpd <= threshold

    Calibrated so roughly:
      mpd = 0.05 -> 1.00
      mpd = 0.06 -> ~0.97
      mpd = 0.10 -> 0.50
      mpd = 0.2 -> 0.002
    """
    if mpd <= threshold:
        return 1.0
    return float(math.exp(-k * ((mpd - threshold) ** p)))


def coverage_alignment_reward(last_info, action, info):
    """
        In the original paper, it used a pretrained smoothness classifier to calculate the smoothness of the folding.
        Here, we use the max IoU to approximate the smoothness.
    """
    if last_info is None:
        last_info = info
    #print(info['evaluation'])
    delta_coverage = info['evaluation']['normalised_coverage'] - last_info['evaluation']['normalised_coverage'] # -1 to 1

    smoothness = info['evaluation']['max_IoU_to_flattened'] - last_info['evaluation']['max_IoU_to_flattened'] # -1 to 1

    alpha = 2
    beta = 1

    return max(np.tanh(alpha*delta_coverage + beta*smoothness), 0) # 0-1
