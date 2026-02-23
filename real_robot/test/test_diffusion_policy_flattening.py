# main.py
from dotmap import DotMap
import time
import numpy as np
from hydra import compose, initialize
from actoris_harena.api import run
import actoris_harena.api as ag_ar
import os

from robot.dual_arm_arena import DualArmArena
from registration.agent import register_agents
from real_robot.tasks.garment_flattening_task import RealWorldGarmentFlatteningTask
from controllers.multi_primitive_diffusion.adapter import MultiPrimitiveDiffusionAdapter

def print_stats(name, data):
    """Helper to safely print mean and std of a list."""
    if not data or len(data) == 0:
        print(f"{name}: No data collected.")
        return
    
    avg = np.mean(data)
    std = np.std(data)
    # Print formatted as: Name: 0.1234s ± 0.0012s
    print(f"{name:<20}: {avg:.4f}s ± {std:.4f}s (n={len(data)})")

def main():
    measure_time = True
    debug = False
    project_name = 'bimanual_garment_folding'
    exp_name = 'human_real_world'

    anrea_config = {
        "ur5e_ip": "192.168.1.10",
        "ur16e_ip": "192.168.1.102",
        "dry_run": False,
        'action_horizon': 10,
        "debug": debug,
        'measure_time': measure_time,
        "snap_to_cloth_mask": True,
        "maskout_background": True,
    }
    task_config = {
        'debug': debug
    }
    agent_config_name = 'run_exp/diffusion_multi_primitive_multi_longsleeve_flattening_demo_100_workspace_snap_one_hot_goal_smaller_random_crop_goal_rot_trans_maskout_x2_training'

    save_dir = '/home/halid/project/garment_folding_data'

    with initialize(version_base=None, config_path=f"../conf"): 
        agent_config = compose(config_name=agent_config_name)
    
    agent_config.save_root = save_dir
    save_dir = os.path.join(agent_config.save_root, agent_config.exp_name)
    
    register_agents()
    policy = ag_ar.build_agent(
        agent_config.agent.name, 
        agent_config.agent,
        project_name=agent_config.project_name,
        exp_name=agent_config_name,
        save_dir= save_dir)
    policy.load_best()

    policy.set_log_dir(save_dir, project_name, exp_name)
    policy.reset([0])

    arena = DualArmArena(DotMap(anrea_config))
    task = RealWorldGarmentFlatteningTask(DotMap(task_config))
    arena.set_task(task)
    arena.set_log_dir(save_dir, project_name, exp_name)

    if measure_time:
        start_time = time.time()
    
    run(policy, arena, mode='eval', 
        episode_config={'eid': 0, 'save_video': False}, 
        checkpoint=-2, 
        policy_terminate=False, env_success_stop=False)
    
    if measure_time:
        duration = time.time() - start_time
        total_time = duration
    
  
    if measure_time:
        print("\n" + "="*40)
        print("PERFORMANCE TIMING REPORT")
        print("="*40)

        # Retrieve component timings
        # Note: We use act_durations from the HumanPolicy we modified previously
        inference_time = policy.internal_states[0]['inference_time']
        
        # Retrieve arena timings (assuming arena stores these as lists)
        primitive_time = getattr(arena, 'primitive_time', [])
        perception_time = getattr(arena, 'perception_time', [])
        process_action_time = getattr(arena, 'process_action_time', [])

        # Print Statistics
        print("Total Time for this Trajectory", total_time)
        print("-" * 40)
        print_stats("Policy Inference / Step", inference_time)
        print_stats("Perception / Step", perception_time)
        print_stats("Process Action / Step", process_action_time)
        print_stats("Primitives Exec / Step", primitive_time)
        print("="*40 + "\n")
    

if __name__ == "__main__":
    main()
