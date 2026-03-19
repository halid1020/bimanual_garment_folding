# tool/hydra_transfer_eval.py

import hydra
from hydra import compose
from omegaconf import DictConfig, OmegaConf
import os
import actoris_harena.api as ag_ar

from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task
from tool.utils import resolve_save_root

@hydra.main(config_path="../conf", config_name="transfer_eval/comp_gc_diffusion", version_base=None)
def main(cfg: DictConfig):
    register_agents()
    register_arenas()

    # 1. Turn off struct mode immediately so we can dynamically check keys without crashing
    OmegaConf.set_struct(cfg, False)

    print("\n[tool.hydra_transfer] --- Parsed Evaluation Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cfg = cfg.transfer_eval
    print("[tool.hydra_transfer] ---------------------------------------\n")

    # 2. Compose the training config dynamically
    # Using compose() here prevents the @package _global_ tag in the source config 
    # from accidentally overwriting this script's variables.
    print(f"[tool.hydra_transfer] Composing training config from: {cfg.train_exp_config}")
    train_cfg = compose(config_name=cfg.train_exp_config)

    new_save_root = resolve_save_root(train_cfg.save_root)
    print(f"[tool.hydra_transfer] Using Source Save Root: {train_cfg.save_root}")

    # Reconstruct the original save directory to load the checkpoint
    source_save_dir = os.path.join(new_save_root, train_cfg.exp_name)
    
    # Create the output directory for this transfer evaluation
    transfer_eval_dir = os.path.join(new_save_root, cfg.eval_name)
    os.makedirs(transfer_eval_dir, exist_ok=True)

    # 3. Build the Agent based on the training configuration
    agent = ag_ar.build_agent(
        train_cfg.agent.name, 
        train_cfg.agent,
        project_name=train_cfg.project_name,
        exp_name=train_cfg.exp_name,
        save_dir=source_save_dir,
        disable_wandb=True
    )
    
    # 4. Iterate through the assigned arenas and evaluate
    eval_arenas = cfg.get("eval_arenas", [])
    if not eval_arenas:
        print("WARNING: No eval_arenas found in your config! Check your YAML formatting.")
        return

    for i, eval_setup in enumerate(eval_arenas):
        
        # Dynamically compose the Arena and Task configs based on the strings in the YAML
        arena_cfg = compose(config_name=f"arena/{eval_setup.arena}")
        task_cfg = compose(config_name=f"task/{eval_setup.task}")
        
        # Safety Check: If the configs use @package directives, unwrap them
        if "arena" in arena_cfg and "name" not in arena_cfg:
            arena_cfg = arena_cfg.arena
        if "task" in task_cfg and "name" not in task_cfg:
            task_cfg = task_cfg.task

        print(f"\n>>> Starting Evaluation {i+1}/{len(eval_arenas)}")
        print(f">>> Arena: {arena_cfg.name} | Task: {task_cfg.task_name}")

        transfer_eval_dir_spe = os.path.join(transfer_eval_dir, eval_setup.arena)
        os.makedirs(transfer_eval_dir_spe, exist_ok=True)
        # Build Arena
        arena = ag_ar.build_arena(
            arena_cfg.name, 
            arena_cfg,
            project_name=train_cfg.project_name,
            exp_name=f"{cfg.eval_name}_arena_{i}", 
            save_dir=transfer_eval_dir_spe
        )
            
        # Build and set Task
        task = build_task(task_cfg)
        arena.set_task(task)

        # 5. Run the evaluation
        try:
            arena.reset()
            res = ag_ar.evaluate(
                agent,
                arena,
                checkpoint=-2, # Load best checkpoint
                policy_terminate=False,
                env_success_stop=False
            )
            print(f">>> Evaluation {i+1} completed. Results: {res}")
            
            # Clean up the C++ physics environment to prevent OpenGL memory leaks
            arena.close() 
            
        except AttributeError as e:
            print(f"ERROR: {e}")
            print("Please update the evaluation function call to match your actoris_harena evaluation API.")

if __name__ == "__main__":
    main()