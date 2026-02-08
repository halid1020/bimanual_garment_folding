import hydra
from omegaconf import DictConfig, OmegaConf
import os

# Assuming these are your custom registration tools
from tool.utils import register_agent, register_arena, build_task
import actoris_harena.api as ag_ar

# 1. Update config_path to point to the root 'conf' directory
@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    register_agent()
    register_arena()

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
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