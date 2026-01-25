# main.py
from robot.dual_arm_arena import DualArmArena
from human_policy import HumanPolicy

from dotmap import DotMap
import time
import numpy as np
from agent_arena.api import run

from real_robot.tasks.garment_folding_task import GarmentFoldingTask

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
        'action_horizon': 3,
        "debug": debug,
        'measure_time': measure_time,
        "snap_to_cloth_mask": True
    }
    task_config = {
        'debug': debug,
        'goal_steps': 2,
        'num_goals': 3,
        'task_name': 'centre_sleeve_folding'
    }
    agent_config = {
        'debug': debug,
        'measure_time': measure_time
    }

    save_dir = './tmp'
    
    policy = HumanPolicy(DotMap(agent_config))
    policy.set_log_dir(save_dir, project_name, exp_name)
    policy.reset([0])

    arena = DualArmArena(DotMap(anrea_config))

    task_config = DotMap(task_config)
    task_config.demonstrator = policy
    task = GarmentFoldingTask(task_config)
    arena.set_task(task)
    arena.set_log_dir(save_dir, project_name, exp_name)

    if measure_time:
        start_time = time.time()
    
    run(policy, arena, mode='eval', episode_config={'eid': 0, 'save_video': False}, checkpoint=-1, policy_terminate=False, env_success_stop=False)
    
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
