import os
from omegaconf import OmegaConf
import actoris_harena.api as ag_ar

# Custom modules
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_sim_task

def main():
    register_agents()
    register_arenas()

    # Target the Dress environment specifically
    arena_name = "magpie/multi_skirt_provide_semkey_pixel_no_success_stop_resol_128_workspace"
    
    print(f"Setting up environment: {arena_name}")
    
    # 1. Initialize environment
    arena_conf_path = os.path.join('./conf/', "arena", f"{arena_name}.yaml")
    arena_cfg = OmegaConf.load(arena_conf_path)
    
    arena_cfg.provide_flattened_semkey_norm_pixel = True
    
    arena = ag_ar.build_arena(arena_cfg.name, arena_cfg)
    arena.set_train()
    
    task_conf_path = os.path.join('./conf/', "task", "magpie", "flattening.yaml")
    task_cfg = OmegaConf.load(task_conf_path)
    arena.set_task(build_sim_task(task_cfg))
    
    
    for config in arena.get_train_configs():
        info = arena.reset(config)
        

    arena.close()
    print("Finished processing all episodes.")

if __name__ == "__main__":
    main()