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
    
    agent = ag_ar.build_agent(
        agent_name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir,
        disable_wandb=True
    )

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
    print(f"[Building Task] {cfg.task.task_name}")
    task = build_task(cfg.task)
    arena.set_task(task)

    
    print("\nStarting Evaluation Run...")

    ag_ar.evaluate(
        agent,
        arena,
        checkpoint=-2, # Load best checkpoint
        policy_terminate=False,
        env_success_stop=False,
        save_internal_states=True
    )

if __name__ == "__main__":
    main()