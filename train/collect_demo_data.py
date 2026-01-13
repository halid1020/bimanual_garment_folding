import argparse
import os
import cv2

from dotmap import DotMap

# --- your project's imports (kept from original) ---
from env.single_garment_subgoal_initial_env import SingleGarmentSubGoalEnv
from env.tasks.garment_folding import GarmentFoldingTask
from env.tasks.garment_flattening import GarmentFlatteningTask
from env.multi_garment_env import MultiGarmentEnv
from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv

from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy import WaistLegFoldingStochasticPolicy
from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy import WaistHemAlignmentFoldingStochasticPolicy

from controllers.random.random_multi_primitive import RandomMultiPrimitive
from agent_arena.utilities.perform_single import perform_single
from controllers.human.human_multi_primitive import HumanMultiPrimitive
<<<<<<< HEAD
from controllers.human.human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
=======
from controllers.human.human_fold import HumanFold
>>>>>>> e8982bdda037099f737c014ffde53c5d35019faa

# --- dataset class: try plausible import; adjust if your project stores it elsewhere ---
from agent_arena.utilities.trajectory_dataset import TrajectoryDataset

# ---------------------------------------------------------------------------
# Observation / action configs (unchanged except typos fixed)
obs_config = {
    'rgb': {'shape': (128, 128, 3), 'output_key': 'rgb'},
    'semkey_pos': {'shape': (45,), 'output_key': 'semkey_pos'},
    'reward': {'shape': (1,), 'output_key': 'reward'},
}

action_config = {
    'norm-pixel-fold': {'shape': (8,), 'output_key': 'default'},
}
# ---------------------------------------------------------------------------

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='single_longsleeve_flattening')
    return parser.parse_args()

def main():
    arg_parser = argument_parser()

    task = 'flattening'
    garment_type = 'longsleeve'
    mode = 'train'
    reward_key = 'coverage_alignment_with_stretch_penality_high_coverage_bonus'
    arena_config = {
        'garment_type': garment_type,
        'picker_radius': 0.03,  # 0.015,
        # 'particle_radius': 0.00625,
        'picker_threshold': 0.0005,  # 0.05,
        'picker_low': (-5, 0, -5),
        'picker_high': (5, 5, 5),
        "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': os.path.join('assets', 'init_states'),
        # 'task': 'centre-sleeve-folding',
        'disp': False,
        'ray_id': 0,
        'horizon': 3,
        'track_semkey_on_frames': False,
        'readjust_pick': True,
        'provide_semkey_pos': True,
        'provide_flattened_semkey_pos': True,
        'grasp_mode': {'around': 1.0},
    }

    task_config = {
        'num_goals': 1,
        'garment_type': garment_type,
        'asset_dir': 'assets',
        'task_name': task,
        'debug': False,
        'alignment': 'simple_rigid',
        # Fixed missing comma between keys
        'overstretch_penality_scale': 0.1,
        'overstretch_penality_threshold': 0.1,
    }

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)

    # create task and environment
    task_obj = GarmentFlatteningTask(task_config)
    arena = SingleGarmentFixedInitialEnv(arena_config)
    arena.set_task(task_obj)

    # Use the human multi-primitive controller as the human agent (HumanFold was not defined)
<<<<<<< HEAD
    agent = HumanDualPickersPickAndPlace(DotMap())
=======
    agent = HumanFold(DotMap())
>>>>>>> e8982bdda037099f737c014ffde53c5d35019faa

    trials_configs = arena.get_train_configs()
    arena.set_train()

    dataset = TrajectoryDataset(
        data_path=arg_parser.data_path,
        data_dir='./data/datasets',
        io_mode='a',
        obs_config=obs_config,
        act_config=action_config,
        whole_trajectory=True
    )

    # iterate over trials (slice by number of requested trials)
    num_trj = dataset.num_trajectories()
    print('num trj', num_trj)
    for i in range(num_trj, arg_parser.trials):
        
        cfg = trials_configs[i%len(trials_configs)]
        res = perform_single(arena, agent, mode=mode, episode_config=cfg, collect_frames=False, debug=False)

        # prepare containers for this trajectory
        observations = {
            'rgb': [],
            'semkey_pos': [],
            'reward': []
        }

        actions = {
            'norm-pixel-fold': []
        }

        # res expected to have 'actions' (list) and 'information' (list of dicts)
        # iterate safely using enumerate
        for i, action in enumerate(res.get('actions', [])):
            # build vector action in consistent order of params
            vector_action = []
            for param_name in ['pick_0', 'pick_1', 'place_0', 'place_1']:
                # guard in case param missing
                val = action.get('norm-pixel-fold', {}).get(param_name, 0.0)
                vector_action.append(val)
            actions['norm-pixel-fold'].append(np.stack(vector_action).flatten())

            info = res.get('information', [])[i] if 'information' in res and i < len(res['information']) else {}
            obs = info.get('observation', {})

            # resize rgb if present, otherwise append a zero-array placeholder
            rgb = obs.get('rgb', None)
            if rgb is not None:
                rgb_resized = cv2.resize(rgb, (128, 128))
            else:
                # placeholder if missing
                rgb_resized = (255 * np.zeros((128, 128, 3), dtype=np.uint8))

            observations['rgb'].append(rgb_resized)

            # reward: prefer info['reward'] if present, otherwise 0.0
            reward = info['reward'][reward_key]
            observations['reward'].append(reward)

            semkey = obs.get('semkey_pos', None)
            observations['semkey_pos'].append(semkey)

        
        info = res.get('information', [])[-1] if 'information' in res and i < len(res['information']) else {}
        obs = info.get('observation', {})

        # resize rgb if present, otherwise append a zero-array placeholder
        rgb = obs.get('rgb', None)
        if rgb is not None:
            rgb_resized = cv2.resize(rgb, (128, 128))
        else:
            # placeholder if missing
            rgb_resized = (255 * np.zeros((128, 128, 3), dtype=np.uint8))

        observations['rgb'].append(rgb_resized)

        # reward: prefer info['reward'] if present, otherwise 0.0
        reward = info['reward'][reward_key]
        observations['reward'].append(reward)

        semkey = obs.get('semkey_pos', None)
        observations['semkey_pos'].append(semkey)

        # add the trajectory to dataset (API: dataset.add_trajectory(observations, actions))
        dataset.add_trajectory(observations, actions)

if __name__ == '__main__':
    # local import for numpy used for placeholder zeros
    import numpy as np
    main()
