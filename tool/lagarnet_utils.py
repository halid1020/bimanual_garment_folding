import os
import matplotlib.pyplot as plt
from agent_arena.utilities.visual_utils \
    import plot_pick_and_place_trajectory as pt
from agent_arena.utilities.visual_utils \
    import plot_image_trajectory

obs_config = {
    'rgb': {'shape': (128, 128, 3), 'output_key': 'rgb'},
    'depth': {'shape': (128, 128, 1), 'output_key': 'depth'},
    'mask': {'shape': (128, 128, 1), 'output_key': 'mask'},

    ###############
    'goal-rgb': {'shape': (128, 128, 3), 'output_key': 'goal-rgb'},
    'goal-depth': {'shape': (128, 128, 1), 'output_key': 'goal-depth'},
    'goal-mask': {'shape': (128, 128, 1), 'output_key': 'goal-mask'},
}

obs_config_ = {
    'rgb': {'shape': (128, 128, 3), 'output_key': 'rgb'}, 
    'depth': {'shape': (128, 128, 1), 'output_key': 'depth'},
    'mask': {'shape': (128, 128, 1), 'output_key': 'mask'},
    'goal-rgb': {'shape': (128, 128, 3), 'output_key': 'goal-rgb'},
    'goal-depth': {'shape': (128, 128, 1), 'output_key': 'goal-depth'},
    'goal-mask': {'shape': (128, 128, 1), 'output_key': 'goal-mask'},
}
    
compress_obs_config = {
    'rgb': {'shape': (64, 64, 3), 'output_key': 'rgb'},
    'depth': {'shape': (64, 64, 1), 'output_key': 'depth'},
    'mask': {'shape': (64, 64, 1), 'output_key': 'mask'},

    ###############
    'goal-rgb': {'shape': (64, 64, 3), 'output_key': 'goal-rgb'},
    'goal-depth': {'shape': (64, 64, 1), 'output_key': 'goal-depth'},
    'goal-mask': {'shape': (64, 64, 1), 'output_key': 'goal-mask'},
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


    T = len(rgbs)

    pt(
        rgbs, actions, # TODO: this is envionrment specific
        info = ['{}'.format(i) for i in range(T)],
        info_font=28,
        title='RGB {}'.format(filename), 
        save_png = True, save_path=os.path.join('tmp', '{}_trajectory'.format(filename)), col=5)
    
    plot_image_trajectory(
        rgbs, 
        save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
        title='rgb_{}_pure'.format(filename))
    
    if goal_rgbs is not None:
        plot_image_trajectory(
            goal_rgbs,
            save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
            title='goal_rgb_pure'.format(filename))

    pt(
        depths, actions,
        title='Depth {}'.format(filename),  
        save_png = True, save_path=os.path.join('tmp', '{}_trajectory'.format(filename)), col=5)
    plot_image_trajectory(
        depths, 
        save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
        title='depth_{}_pure'.format(filename))
    
    if goal_depths is not None:

        plot_image_trajectory(
            goal_depths, 
            save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
            title='goal_depth_{}_pure'.format(filename))
    
    pt(
        masks, actions,
        title='Mask {}'.format(filename),
        save_png = True, save_path=os.path.join('tmp', '{}_trajectory'.format(filename)), col=5)
    
    plot_image_trajectory(
        masks, 
        save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
        title='mask_{}_pure'.format(filename)
    )

    if goal_masks is not None:
        plot_image_trajectory(
            goal_masks, 
            save_path=os.path.join('tmp', '{}_trajectory'.format(filename)),
            title='goal_mask_{}_pure'.format(filename
        ))
    
    if reward_dict is not None:
        # Figure 2: Reward plot
        fig_reward, ax_reward = plt.subplots(figsize=(8, 6))
        ## set font size
        font_size = 16
        col = len(rgbs)
        x = range(0, col)
        reward_to_plot = ['clothfunnel_default', 'learningToUnfold_approx', 
                        'planet_clothpick_hueristic', 'speedFolding_approx',
                        'coverage_aligment']
        labels = ['ClothFunnels', 'Learning2Unfold', 
                'PlaNet-ClothPick', 'SpeedFolding (Approx)',
                'Coverage-Alignment (ours)']
        for key, values in reward_dict.items():
            if key not in reward_to_plot:
                continue
            idx = reward_to_plot.index(key)
            ax_reward.plot(x, values[:sequence], marker='o', label=labels[idx])
        
        ax_reward.set_xlabel('State', fontsize=font_size)
        # ax_reward.set_ylabel('Reward')
        # ax_reward.set_title('Rewards')
        ax_reward.legend(fontsize=font_size, ncol=2, loc='lower center')
        ax_reward.set_ylim(-1.0, 1.0)
        ax_reward.grid(True)
        ax_reward.set_xticks(x)
        ## set font size for x-y ticks as well as the legend
        ax_reward.tick_params(axis='both', which='major', labelsize=font_size)


        plt.tight_layout()
        plt.savefig('tmp/{}_rewards.png'.format(filename), \
                    dpi=300, bbox_inches='tight')
        plt.close(fig_reward)

    if evaluation_dict is not None:
        ## Figure 3: Evaluation plot
        fig_evaluation, ax_evaluation = plt.subplots(figsize=(8, 6))
        col = len(rgbs)
        x = range(0, col)
        for key, values in evaluation_dict.items():
            if key not in evaluation_to_plot:
                continue
            idx = evaluation_to_plot.index(key)
            ax_evaluation.plot(x, values[:sequence], marker='o', label=evaluation_labels[idx])
        ax_evaluation.set_xlabel('State', fontsize=font_size)
        ax_evaluation.grid(True)
        # ax_evaluation.set_ylabel('Evaluation')
        ax_evaluation.set_ylim(-0.2, 1.05)
        ax_evaluation.legend(loc='lower center', ncol=3, fontsize=font_size)
        ax_evaluation.set_xticks(x)
        ax_evaluation.tick_params(axis='both', which='major', labelsize=font_size)

        plt.tight_layout()
        plt.savefig('tmp/{}_evaluation.png'.format(filename), \
                    dpi=300, bbox_inches='tight')
        plt.close(fig_evaluation)