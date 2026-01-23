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