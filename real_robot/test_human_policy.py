# main.py
from robot.dual_arm_arena import DualArmArena
from human_policy import HumanPolicy
from garment_flattening_task import GarmentFlatteningTask
from dotmap import DotMap
import time
import numpy as np

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

    anrea_config = {
        "ur5e_ip": "192.168.1.10",
        "ur16e_ip": "192.168.1.102",
        "dry_run": False,
        'action_horizon': 5,
        "debug": debug,
        'measure_time': measure_time
    }
    task_config = {
        'debug': debug
    }
    agent_config = {
        'debug': debug,
        'measure_time': measure_time
    }

    arena = DualArmArena(DotMap(anrea_config))
    policy = HumanPolicy(DotMap(agent_config))
    policy.reset([0])
    task = GarmentFlatteningTask(DotMap(task_config))
    arena.set_task(task)

    total_time = []
    info = arena.reset()

    print('info evaluate', info['evaluation'])
    print('info done', info['done'])
    while not info['done']:
        if measure_time:
            start_time = time.time()
        action = policy.single_act(info)
        info = arena.step(action)

        if measure_time:
            duration = time.time() - start_time
            total_time.append(duration)

        
        print('info evaluate', info['evaluation'])
    
  
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
        print_stats("Total Loop Time", total_time)
        print("-" * 40)
        print_stats("Policy Inference", inference_time)
        print_stats("Perception", perception_time)
        print_stats("Process Action", process_action_time)
        print_stats("Primitives Exec", primitive_time)
        print("="*40 + "\n")
    

if __name__ == "__main__":
    main()
