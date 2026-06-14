import os
from omegaconf import OmegaConf
import actoris_harena.api as ag_ar

# Custom modules
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_sim_task

def main():
    # 1. Register custom components
    register_agents()
    register_arenas()

    # Define paths based on your provided config structure
    arena_name = "magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace"
    task_name = "canonicalisation_alignment_centre_sleeve_folding"
    
    print(f"Setting up environment: {arena_name}")
    print(f"Setting up task: {task_name}")
    
    # 2. Initialize Arena
    arena_conf_path = os.path.join('./conf/', "arena", f"{arena_name}.yaml")
    
    # Fallback if magpie directory isn't used in your conf structure
    if not os.path.exists(arena_conf_path):
        arena_conf_path = os.path.join('./conf/', "arena", "multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace.yaml")
        
    arena_cfg = OmegaConf.load(arena_conf_path)
    
    # Ensure any necessary observation flags are set
    arena_cfg.provide_flattened_semkey_norm_pixel = True
    
    arena = ag_ar.build_arena(arena_cfg.name, arena_cfg)
    
    # 3. Initialize Task
    task_conf_path = os.path.join('./conf/', "task", "magpie", f"{task_name}.yaml")
    
    if not os.path.exists(task_conf_path):
        task_conf_path = os.path.join('./conf/', "task", f"{task_name}.yaml")
        
    task_cfg = OmegaConf.load(task_conf_path)
    arena.set_task(build_sim_task(task_cfg))
    
    # 4. Generate Goals for all splits
    # This will trigger the Human Demonstrator UI if goals do not already exist in the asset_dir
    
    print("\n--- Generating Train Goals ---")
    arena.set_train()
    for config in arena.get_train_configs():
        print(f"Processing Train Episode: {config.get('eid')}")
        info = arena.reset(config)
        
    print("\n--- Generating Eval Goals ---")
    arena.set_eval()
    for config in arena.get_eval_configs():
        print(f"Processing Eval Episode: {config.get('eid')}")
        info = arena.reset(config)
        
    print("\n--- Generating Val Goals ---")
    arena.set_val()
    for config in arena.get_val_configs():
        print(f"Processing Val Episode: {config.get('eid')}")
        info = arena.reset(config)

    # Clean up
    arena.close()
    print("\nFinished processing all episodes and generating goals.")

if __name__ == "__main__":
    main()