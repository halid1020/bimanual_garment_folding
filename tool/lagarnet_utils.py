import os
import matplotlib.pyplot as plt
from actoris_harena.utilities.visual_utils \
    import plot_pick_and_place_trajectory as pt
from actoris_harena.utilities.visual_utils \
    import plot_image_trajectory

obs_config = {
    'rgb': {'shape': (128, 128, 3), 'output_key': 'rgb'},
    'depth': {'shape': (128, 128, 1), 'output_key': 'depth'},
    'mask': {'shape': (128, 128, 1), 'output_key': 'mask'},

    ###############
    'goal-rgb': {'shape': (128, 128, 3), 'output_key': 'goal-rgb'},
    'goal-depth': {'shape': (128, 128, 1), 'output_key': 'goal-depth'},
    'goal-mask': {'shape': (128, 128, 1), 'output_key': 'goal-mask'},
    'success': {'shape': (1,), 'output_key': 'success'},
}

obs_config_ = {
    'rgb': {'shape': (128, 128, 3), 'output_key': 'rgb'}, 
    'depth': {'shape': (128, 128, 1), 'output_key': 'depth'},
    'mask': {'shape': (128, 128, 1), 'output_key': 'mask'},
    'goal-rgb': {'shape': (128, 128, 3), 'output_key': 'goal-rgb'},
    'goal-depth': {'shape': (128, 128, 1), 'output_key': 'goal-depth'},
    'goal-mask': {'shape': (128, 128, 1), 'output_key': 'goal-mask'},
    'success': {'shape': (1,), 'output_key': 'success'},
}
    
compress_obs_config = {
    'rgb': {'shape': (64, 64, 3), 'output_key': 'rgb'},
    'depth': {'shape': (64, 64, 1), 'output_key': 'depth'},
    'mask': {'shape': (64, 64, 1), 'output_key': 'mask'},

    ###############
    'goal-rgb': {'shape': (64, 64, 3), 'output_key': 'goal-rgb'},
    'goal-depth': {'shape': (64, 64, 1), 'output_key': 'goal-depth'},
    'goal-mask': {'shape': (64, 64, 1), 'output_key': 'goal-mask'},

    'success': {'shape': (1,), 'output_key': 'success'},
}

action_config = {
    'norm-pixel-pick-and-place': \
    {'shape': (2, 2), 'output_key': 'norm-pixel-pick-and-place'},
}

reward_names = [
    'clothfunnel_default',
    'clothfunnel_tanh_reward',
    'speedFolding_approx',
    'learningToUnfold_approx',
    'planet_clothpick_hueristic',


    'max_IoU_differance',
    'coverage_differance',
    'canon_IoU_differance',
    'coverage_aligment'
]

evaluation_names = [
    'normalised_coverage',
    'normalised_improvement',
    'max_IoU_to_flattened',
    'canon_IoU_to_flattened',
    'canon_l2_distance',
    # 'normalised_hausdorff_distance',
    'deform_l2_distance',
    'rigid_l2_distance',
]
evaluation_to_plot = evaluation_names.copy()

evaluation_labels = [
    'NC',
    'NI',
    'Max IoU',
    # 'Normalised Minimum\nHausdorff Distance',
    'Canon IoU',
    'L2',
    'Deform L2',
    'Rigid L2',
]


obs_config.update({key: {'shape': (1,), 'output_key': key} for key in reward_names})
obs_config.update({key: {'shape': (1,), 'output_key': key} for key in evaluation_names})
obs_config_.update({key: {'shape': (1,), 'output_key': key} for key in reward_names})
obs_config_.update({key: {'shape': (1,), 'output_key': key} for key in evaluation_names})
compress_obs_config.update({key: {'shape': (1,), 'output_key': key} for key in reward_names})
compress_obs_config.update({key: {'shape': (1,), 'output_key': key} for key in evaluation_names})

def plot_results(
    rgbs, depths, masks,
    goal_rgbs=None, goal_depths=None, goal_masks=None,             
    actions=None, reward_dict=None, evaluation_dict=None, 
    success=None, # <--- NEW ARGUMENT
    filename='example', sequence=-1):

    rgbs = rgbs[:sequence]
    depths = depths[:sequence]
    masks = masks[:sequence]
    if goal_rgbs is not None:
        goal_rgbs = goal_rgbs[:sequence]
    if goal_depths is not None:
        goal_depths = goal_depths[:sequence]
    if goal_masks is not None:
        goal_masks = goal_masks[:sequence]
    if actions is not None:
        actions = actions[:sequence]

    # --- NEW: Determine Status String ---
    status_suffix = ""
    if success is not None:
        # Handle if success is a list/array (trajectory) or single bool
        # We assume the last step represents the final episode outcome
        is_success = success[-1] if hasattr(success, '__getitem__') else success
        status_suffix = " [SUCCESS]" if is_success else " [FAIL]"
    # ------------------------------------

    T = len(rgbs)

    pt(
        rgbs, actions, 
        info = ['{}'.format(i) for i in range(T)],
        info_font=28,
        # --- NEW: Add status to title ---
        title='RGB {}{}'.format(filename, status_suffix), 
        # --------------------------------
        save_png = True, save_path=os.path.join('tmp', '{}_trajectory'.format(filename)), col=5)
    
    plot_image_trajectory(
        rgbs, 
        save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
        title='rgb_{}_pure'.format(filename))
    
    # ... (rest of the function remains exactly the same)