import argparse
import agent_arena.api as ag_ar
from agent_arena import train_and_evaluate
import matplotlib.pyplot as plt
import numpy as np
from dotmap import DotMap
import os
import torch

from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_folding import GarmentFoldingTask
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.Image_based_multi_primitive_SAC import ImageBasedMultiPrimitiveSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy


def test_config() -> DotMap:
    cfg = DotMap()
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.encoder = DotMap()
    cfg.encoder.out_dim = 256
    cfg.encoder.cnn_channels = [32, 64, 128]
    cfg.encoder.kernel = 3
    cfg.encoder.pool = 2
    cfg.obs_key = 'rgb'

    cfg.context_horizon = 3
    cfg.each_image_shape = (3, 64, 64) # RGB
    cfg.num_primitives = 5
    cfg.hidden_dim = 256

    cfg.actor_lr = 3e-4
    cfg.critic_lr = 3e-4
    cfg.encoder_lr = 3e-4
    cfg.alpha_lr = 3e-4
    cfg.tau = 0.005
    cfg.gamma = 0.99
    cfg.batch_size = 2 #256

    cfg.replay_capacity = int(1e2)
    cfg.initial_act_steps = 5
    #cfg.target_entropy = -cfg.max_action_dim
    cfg.max_grad_norm = 10.0
    cfg.save_dir = None
    cfg.act_steps_per_update = 2
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

    return cfg

def main():

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
        'horizon': 8,
        'track_semkey_on_frames': True
    }
    
    demonstrator = CentreSleeveFoldingStochasticPolicy(DotMap({'debug': False})) # TODO: create demonstrator for 'centre-sleeve-folding'
    
    task_config = {
        'num_goals': 10,
        'demonstrator': demonstrator,
        'object': 'longsleeve',
        'asset_dir': 'assets',
        'task_name': 'centre-sleeve-folding',
        'debug': False,
        'alignment': 'deform'
    }

    validation_interval = int(1e3)
    total_update_steps = int(1e5)
    eval_checkpoint = -1

    arena_config = DotMap(arena_config)
    task_config = DotMap(task_config)
    
    task = GarmentFoldingTask(task_config)
    arena = SingleGarmentFixedInitialEnv(arena_config)
    
    arena.set_task(task)

    agent = ImageBasedMultiPrimitiveSAC(config=test_config())

    save_dir = os.path.join('./tmp', 'garment_folding', 'train_v0')
    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)
    
    res = ag_ar.train_and_evaluate(agent, arena,
        validation_interval, total_update_steps, eval_checkpoint)
    
    

if __name__ == '__main__':
    main()