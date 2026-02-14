import hydra
from omegaconf import DictConfig, OmegaConf
import os
import socket
import actoris_harena.api as ag_ar

from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task


# 1. Update config_path to point to the root 'conf' directory
@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    
    register_agents()
    register_arenas()

    # --- Automatic save_root detection ---
    hostname = socket.gethostname()
    
    if "pc282" in hostname:
        new_save_root = '/media/hcv530/T7/garment_folding_data'
    elif "thanos" in hostname:
        new_save_root = '/data/ah390/garment_folding_data'
    elif "viking" in hostname:
        new_save_root = '/mnt/scratch/users/hcv530/garment_folding_data'
    else:
        new_save_root = cfg.save_root # Fallback to config default

    # Update the config object (must unset 'struct' to modify)
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    OmegaConf.set_struct(cfg, True)
    # -------------------------------------

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Detected Host: {hostname}")
    print(f"Using Save Root: {cfg.save_root}")
    print("---------------------")


    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    
    # 2. Extract Names (Fallback to Exp Name if not in sub-config)
    # If your agent yaml doesn't have a 'name' field, we use cfg.exp_name or a default
    agent_name = cfg.agent.get('name', cfg.exp_name) 
    arena_name = cfg.arena.get('name', 'default_arena')

    # 3. Build Agent
    print(f"[hydra eval] Building Agent: {agent_name}")
    agent = ag_ar.build_agent(
        agent_name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir
    )
    
    # 4. Build Arena
    print(f"[hydra eval] Building Arena: {arena_name}")
    arena = ag_ar.build_arena(
        arena_name, 
        cfg.arena,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir
    )
    
    # 5. Build Task
    # Ensure build_task can handle the DictConfig object
    task = build_task(cfg.task)
    arena.set_task(task)

    # 6. Run Evaluation
    ag_ar.evaluate(
        agent,
        arena,
        checkpoint=-2, # Load best checkpoint
        policy_terminate=False,
        env_success_stop=False
    )

if __name__ == "__main__":
    main()