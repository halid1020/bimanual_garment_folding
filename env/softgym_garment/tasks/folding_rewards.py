import numpy as np

import math

import math


IOU_FLATTENING_TRESHOLD = 0.8
NC_FLATTENING_TRESHOLD = 0.95

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


def stable_nc_iou_reward(last_info, action, info):
    """
    Monotonic, cheat-proof reward using NC and IoU only.
    Range: 0 to 1
    """
    nc = info['evaluation']['normalised_coverage']
    iou = info['evaluation']['max_IoU_to_flattened']

    # --- Normalize NC and IoU into [0, 1] w.r.t thresholds ---
    nc_score  = min(nc / NC_FLATTENING_TRESHOLD, 1.0)
    iou_score = min(iou / IOU_FLATTENING_TRESHOLD, 1.0)

    # --- Combine them smoothly ---
    # harmonic mean avoids cheating and requires BOTH to be high
    if (nc_score + iou_score) == 0:
        combined = 0
    else:
        combined = 2 * (nc_score * iou_score) / (nc_score + iou_score)

    # Already in [0,1]
    return combined



def smooth_anti_cheat_reward(last_info, action, info,
                             nc_threshold=0.90,
                             iou_threshold=0.90,
                             w_abs=0.7,            # weight for absolute score
                             w_delta=0.3,          # weight for delta shaping
                             delta_cap=0.06,       # scale for tanh on positive delta (roughly the "useful" delta size)
                             max_step=0.20,        # delta sum beyond which we start penalising
                             step_penalty_factor=0.9):  # how strongly to punish huge steps
    """
    Reward in [0,1] combining absolute performance and bounded delta shaping,
    while penalising very large instantaneous jumps to avoid "huge-step" exploitation.
    """
    if last_info is None:
        last_info = info

    # Absolute scores (normalized to thresholds then harmonic mean -> [0,1])
    nc = info['evaluation']['normalised_coverage']
    iou = info['evaluation']['max_IoU_to_flattened']
    nc_score = min(nc / nc_threshold, 1.0)
    iou_score = min(iou / iou_threshold, 1.0)

    if (nc_score + iou_score) == 0:
        abs_score = 0.0
    else:
        abs_score = 2 * (nc_score * iou_score) / (nc_score + iou_score)  # harmonic mean in [0,1]

    # Success override
    if nc >= nc_threshold and iou >= iou_threshold:
        return 1.0

    # Deltas (could be negative)
    last_nc = last_info['evaluation']['normalised_coverage']
    last_iou = last_info['evaluation']['max_IoU_to_flattened']
    delta_nc = nc - last_nc
    delta_iou = iou - last_iou

    # Positive delta shaping: only reward positive improvements, average the two
    delta_positive = max(0.0, (delta_nc + delta_iou) / 2.0)

    # Bound and smooth the contribution of delta so large deltas saturate (no explosion)
    # delta_cap controls what size of delta maps to near-1 in tanh; keep small (~0.05-0.1)
    delta_bounded = np.tanh(delta_positive / max(1e-8, delta_cap))

    # Step penalty: if the absolute instantaneous change (sum of absolute deltas) is huge,
    # penalize the *delta* contribution (not the absolute score).
    abs_step = abs(delta_nc) + abs(delta_iou)
    # step_pen is 0 when within max_step, grows linearly when exceed
    step_pen = max(0.0, (abs_step / max_step) - 1.0)
    # multiplier in [0,1] that shrinks the delta contribution
    delta_multiplier = max(0.0, 1.0 - step_penalty_factor * step_pen)

    # Compose final reward and clip into [0,1]
    reward = w_abs * abs_score + w_delta * (delta_bounded * delta_multiplier)
    reward = float(np.clip(reward, 0.0, 1.0))
    return reward
