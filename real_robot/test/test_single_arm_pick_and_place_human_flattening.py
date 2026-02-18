import time
import numpy as np
from dotmap import DotMap
from actoris_harena.api import run

# Import Single Arm Components
from real_robot.robot.single_arm_pick_and_place_arena import SingleArmPickAndPlaceArena
from controllers.human.real_world_human_single_arm_pick_and_place_policy import RealWorldSingleArmHumanPickAndPlacePolicy
from real_robot.tasks.garment_flattening_task import RealWorldGarmentFlatteningTask

def print_stats(name, data):
    """Helper to safely print mean and std of a list."""
    if not data or len(data) == 0:
        print(f"{name}: No data collected.")
        return
    
    avg = np.mean(data)
    std = np.std(data)
    print(f"{name:<25}: {avg:.4f}s ± {std:.4f}s (n={len(data)})")

def main():
    measure_time = True
    debug = True
    project_name = 'single_arm_garment_flattening'
    exp_name = 'human_real_world'

    # Single Arm Configuration
    arena_config = {
        "ur5e_ip": "192.168.1.10",
        "dry_run": False,
        'action_horizon': 5,
        "debug": debug,
        'measure_time': measure_time,
        "snap_to_cloth_mask": True,
    }
    
    task_config = {
        'debug': debug
    }
    
    agent_config = {
        'debug': debug,
        'measure_time': measure_time
    }

    save_dir = './tmp'
    
    # 1. Initialize Policy
    policy = RealWorldSingleArmHumanPickAndPlacePolicy(DotMap(agent_config))
    policy.set_log_dir(save_dir, project_name, exp_name)
    # The policy expects arena IDs, here we just use [0]
    policy.reset([0])

    # 2. Initialize Arena
    arena = SingleArmPickAndPlaceArena(DotMap(arena_config))
    
    # 4. Initialize Task
    # We reuse the garment flattening task as it depends on visual coverage, which is robot-agnostic
    task = RealWorldGarmentFlatteningTask(DotMap(task_config))
    arena.set_task(task)
    arena.set_log_dir(save_dir, project_name, exp_name)

    if measure_time:
        start_time = time.time()
    
    # 5. Run Evaluation
    print("\n[Main] Starting Single Arm Human Evaluation...")
    run(
        policy, 
        arena, 
        mode='eval', 
        episode_config={'eid': 0, 'save_video': False, 'garment_id': 'test_shirt'}, 
        checkpoint=-1, 
        policy_terminate=False, 
        env_success_stop=False
    )
    
    if measure_time:
        duration = time.time() - start_time
        total_time = duration
    
    # 6. Print Stats
    if measure_time:
        print("\n" + "="*50)
        print("SINGLE ARM PERFORMANCE TIMING REPORT")
        print("="*50)

        # Retrieve component timings
        # Note: 'internal_states' structure comes from the Agent/Policy parent class
        inference_time = policy.internal_states[0].get('inference_time', [])
        
        # Retrieve arena timings (assuming arena stores these as lists)
        # Check if the attributes exist, as SingleArmArena might not initialize them if measure_time was False initially
        primitive_time = getattr(arena, 'primitive_time', [])
        perception_time = getattr(arena, 'perception_time', [])
        process_action_time = getattr(arena, 'process_action_time', [])

        # Print Statistics
        print(f"Total Time for Trajectory   : {total_time:.4f}s")
        print("-" * 50)
        print_stats("Policy Inference / Step", inference_time)
        print_stats("Perception / Step", perception_time)
        print_stats("Process Action / Step", process_action_time)
        # Note: Primitive execution time might be None/Empty if using blocking calls inside step without specific timing wrappers
        if hasattr(arena, 'primitive_time'):
            print_stats("Primitives Exec / Step", primitive_time)
        
        print("="*50 + "\n")

if __name__ == "__main__":
    main()