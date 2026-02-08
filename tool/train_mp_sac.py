import argparse
import actoris_harena.api as ag_ar
from actoris_harena import train_and_evaluate
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os
import torch

from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_folding import GarmentFoldingTask
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.multi_primitive_sac.image_based_multi_primitive_SAC import ImageBasedMultiPrimitiveSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
from controllers.multi_primitive_sac.data_augmenter import PixelBasedPrimitiveDataAugmenter

def get_agent_config() -> DotMap:
    cfg = DotMap()
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.encoder = DotMap()
    cfg.encoder.out_dim = 256
    cfg.encoder.cnn_channels = [32, 64, 128]
    cfg.encoder.kernel = 3
    cfg.encoder.pool = 2
    cfg.obs_key = 'rgb'

    cfg.context_horizon = 4
    cfg.each_image_shape = (3, 64, 64) # RGB
    cfg.num_primitives = 5
    cfg.hidden_dim = 256

    cfg.actor_lr = 3e-4
    cfg.critic_lr = 3e-4
    cfg.encoder_lr = 3e-4
    cfg.alpha_lr = 3e-4
    cfg.tau = 0.005
    cfg.gamma = 0.99
    cfg.batch_size = 256

    cfg.replay_capacity = int(1e5)
    cfg.initial_act_steps = 100
    cfg.max_grad_norm = 10.0
    cfg.save_dir = None
    cfg.act_steps_per_update = 1
    cfg.num_primitives = 4
    cfg.action_dims = [4, 8, 6, 8]
    cfg.primitive_param = [
        ("norm-pixel-pick-and-fling", [("pick_0", 2), ("pick_1", 2)]),
        ('norm-pixel-pick-and-place', [("pick_0", 2), ("pick_1", 2), ("place_0", 2), ("place_1", 2)]),
        ('norm-pixel-pick-and-drag', [("pick_0", 2), ("pick_1", 2), ("place_0", 2)]),
        ('norm-pixel-fold',  [("pick_0", 2), ("pick_1", 2), ("place_0", 2), ("place_1", 2)])
    ]

    cfg.reward_key = 'multi_stage_reward'
    cfg.checkpoint_interval = 5
    cfg.add_reject_actions = True
    cfg.reject_action_reward = -1
    cfg.total_update_steps = int(1e5)
    cfg.explore_mode = 'e-greedy'

    return cfg

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mp_sac_v3",
                        help="Name of the experiment (used for save directory)")
    args = parser.parse_args()

    arena_config = {
        'object': 'longsleeve',
        'picker_radius': 0.03, #0.015,
        # 'particle_radius': 0.00625,
        'picker_threshold': 0.007, # 0.05,
        'picker_low': (-5, 0, -5),
        'picker_high': (5, 5, 5),
        'grasp_mode': {'closest': 1.0},
        "picker_initial_pos": [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': os.path.join('assets', 'init_states'),
        #'task': 'centre-sleeve-folding',
        'disp': False,
        'ray_id': 0,
        'horizon': 20,
        'track_semkey_on_frames': False,
        'readjust_pick': True
    }
    
    demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': False})) # TODO: create demonstrator for 'centre-sleeve-folding'
    
    task_config = {
        'num_goals': 10,
        'demonstrator': demonstrator,
        'object': 'longsleeve',
        'asset_dir': 'assets',
        'task_name': 'centre-sleeve-folding',
        'debug': False,
        'alignment': 'simple_rigid'
    }

    data_augmenter_config = {
        "device": "cuda:0",
        "rgb_noise_factor": 0.01,
        "random_rotation": True,
        "rotation_degree": 1,
        "vertical_flip": True,
    }


    validation_interval = int(1e3)
   
    eval_checkpoint = -1

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    data_augmenter_config = DotMap(data_augmenter_config)
    
    task = GarmentFoldingTask(task_config)
    arena = SingleGarmentFixedInitialEnv(arena_config)
    data_augmenter = PixelBasedPrimitiveDataAugmenter(data_augmenter_config)
    
    arena.set_task(task)

    agent_config = get_agent_config()
    agent = ImageBasedMultiPrimitiveSAC(config=agent_config)
    total_update_steps = agent_config.total_update_steps

    save_dir = os.path.join('/data/hcv530', 'garment_folding', args.exp_name)
    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)
    agent.set_data_augmenter(data_augmenter)
    
    res = ag_ar.train_and_evaluate(agent, arena,
        validation_interval, total_update_steps, eval_checkpoint)
    
    

if __name__ == '__main__':
    main()