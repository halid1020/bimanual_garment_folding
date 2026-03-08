import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import actoris_harena.api as ag_ar


# Assuming these are your custom modules
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Agent Arena Environment")
    parser.add_argument('--arena_name', type=str, required=True, help="Name of the arena (e.g., 'standard_arena')")
    return parser.parse_args()

def main():
    args = parse_args()
    
    register_agents()
    register_arenas()

    # 1. Load the Arena Config
    arena_conf_path = os.path.join('./conf/', "arena", f"{args.arena_name}.yaml")
    
    if not os.path.exists(arena_conf_path):
        raise FileNotFoundError(f"Arena config not found at: {arena_conf_path}")
        
    print(f"Loading arena config from: {arena_conf_path}")
    arena_cfg = OmegaConf.load(arena_conf_path)

    # 2. Build Arena
    arena = ag_ar.build_arena(
        arena_cfg.name, 
        arena_cfg
    )

    # 3. Load Task Config
    task_conf_path = os.path.join('./conf/', "task", "flattening_overstretch_penalty_1_no_big_bonus.yaml")
    if not os.path.exists(task_conf_path):
        raise FileNotFoundError(f"Task config not found at: {task_conf_path}")
    
    print(f"Loading task config from: {task_conf_path}")
    task_cfg = OmegaConf.load(task_conf_path)

    task = build_task(task_cfg)
    arena.set_task(task)
    
    # 4. Collect Data
    episode_configs = arena.get_eval_configs()
    arena.set_eval()
    
    num_samples = len(episode_configs)
   
    for i in range(num_samples):
        eps_conf = episode_configs[i]
        info = arena.reset(eps_conf)

if __name__ == "__main__":
    main()