import time
import os
import socket
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotmap import DotMap

# Import your API and tools
import actoris_harena.api as ag_ar
from registration.agent import register_agents
from registration.real_arena import register_arenas
# You were missing this import for the build_task function used later
from registration.task import build_task 

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def print_stats(name, data):
    """Helper to safely print mean and std of a list."""
    if not data or len(data) == 0:
        print(f"{name:<25}: No data collected.")
        return
    
    avg = np.mean(data)
    std = np.std(data)
    # Print formatted as: Name: 0.1234s ± 0.0012s
    print(f"{name:<25}: {avg:.4f}s ± {std:.4f}s (n={len(data)})")

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------

# FIXED: Added config_name="config" so it loads conf/config.yaml
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Register Components
    register_agents()
    register_arenas()

    # 2. Automatic Save Root Detection based on Hostname
    hostname = socket.gethostname()
    
    # Define your save root logic here
    new_save_root = '/home/halid/project/garment_folding_data'

    # Update the config object with the detected path
    # We must unlock the struct to add a new key 'save_root' if it wasn't in the yaml,
    # or to modify it if it was.
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    OmegaConf.set_struct(cfg, True)

    print("\n" + "="*40)
    print("CONFIGURATION SETUP")
    print("="*40)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Detected Host: {hostname}")
    print(f"Using Save Root: {cfg.save_root}")
    # print(OmegaConf.to_yaml(cfg, resolve=True)) # Uncomment to debug full config
    print("-" * 40)

    # Prepare Save Directory
    # Now this will work because cfg is loaded correctly
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)

    # 3. Build Agent (Policy)
    # Uses cfg.agent.name or defaults to cfg.exp_name
    agent_name = cfg.agent.name
    print(f"[Building Agent]: {agent_name}")
    
    policy = ag_ar.build_agent(
        agent_name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir
    )

    # If the policy supports checkpoints (like diffusion), load the best one
    if hasattr(policy, 'load_best'):
        try:
            policy.load_best()
        except Exception as e:
            print(f"Warning: Could not load best checkpoint: {e}")

    policy.reset([0])

    # 4. Build Arena
    arena_name = cfg.arena.get('name', 'dual_arm_arena') 
    print(f"[Building Arena]: {arena_name}")
    
    arena = ag_ar.build_arena(
        arena_name, 
        cfg.arena,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir
    )

    # 5. Build Task
    print(f"[Building Task]")
    task = build_task(cfg.task)
    arena.set_task(task)

    # 6. Execution
    measure_time = cfg.get('measure_time', True)
    
    # Start Timer
    if measure_time:
        start_time = time.time()
    
    print("\nStarting Evaluation Run...")
    
    ag_ar.run(
        policy, 
        arena, 
        mode='eval', 
        episode_config={'eid': 0, 'save_video': False}, 
        checkpoint=cfg.get('checkpoint', -1), 
        policy_terminate=False, 
        env_success_stop=False
    )
    
    # End Timer
    if measure_time:
        duration = time.time() - start_time
        total_time = duration

    # 7. Performance Timing Report
    if measure_time:
        print("\n" + "="*40)
        print("PERFORMANCE TIMING REPORT")
        print("="*40)

        # Retrieve component timings
        # Try to access inference_time safely
        inference_time = []
        if hasattr(policy, 'internal_states') and len(policy.internal_states) > 0:
            if isinstance(policy.internal_states[0], dict):
                 inference_time = [s.get('inference_time') for s in policy.internal_states if 'inference_time' in s]
            elif hasattr(policy, 'inference_timer'): 
                inference_time = policy.inference_timer
        
        # Retrieve arena timings (assuming arena stores these as lists)
        primitive_time = getattr(arena, 'primitive_time', [])
        perception_time = getattr(arena, 'perception_time', [])
        process_action_time = getattr(arena, 'process_action_time', [])

        # Print Statistics
        print(f"Total Time for Trajectory : {total_time:.4f}s")
        print("-" * 40)
        print_stats("Policy Inference / Step", inference_time)
        print_stats("Perception / Step", perception_time)
        print_stats("Process Action / Step", process_action_time)
        print_stats("Primitives Exec / Step", primitive_time)
        print("="*40 + "\n")

if __name__ == "__main__":
    main()