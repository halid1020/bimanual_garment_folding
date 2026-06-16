# tool/hydra_train_and_transfer.py

"""
Unified Training and Transfer Evaluation Pipeline for Actoris Harena.

This module orchestrates the full lifecycle of a robotic manipulation agent. 
It first trains the policy using the primary environment configurations, and upon 
completion, dynamically loads a suite of transfer arenas to evaluate the zero-shot 
or fine-tuned performance of the resulting checkpoints.

Features:
    - Dynamic Hydra configuration composition for cross-domain evaluation.
    - Automatic memory management and C++ physics environment cleanup to prevent OpenGL leaks.
    - Support for both single and parallel distributed training environments.
"""

import os
from typing import List, Dict, Any

import hydra
from hydra import compose
from omegaconf import DictConfig, OmegaConf

import actoris_harena.api as ag_ar
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_sim_task
from tool.utils import resolve_save_root
from env.parallel import Parallel

def execute_training_phase(cfg: DictConfig, agent: Any, save_dir: str) -> None:
    """
    Executes the training loop based on the configuration parameters.
    
    Args:
        cfg (DictConfig): The global Hydra configuration object.
        agent (Any): The initialized Actoris Harena agent/policy.
        save_dir (str): Absolute path to the logging and checkpoint directory.
    
    Raises:
        ValueError: If the specified `train_and_eval` mode is not recognized.
    """
    print("Initializing Training Phase...")
    
    if cfg.train_and_eval == 'train_and_evaluate_single':
        arena = ag_ar.build_arena(
            cfg.arena.name, 
            cfg.arena,
            project_name=cfg.project_name,
            exp_name=cfg.exp_name,
            save_dir=save_dir
        )
            
        task = build_sim_task(cfg.task)
        arena.set_task(task)

        ag_ar.train_and_evaluate_single(
            agent,
            arena,
            cfg.agent.validation_interval,
            cfg.agent.total_update_steps,
            eval_last_check=True,
            eval_best_check=True,
            policy_terminate=False,
            env_success_stop=False
        )
        arena.close()

    elif cfg.train_and_eval == 'train_plural_eval_single':
        # NOTE: registered_arena mapping must be accessible or passed through ag_ar
        from train.utils import registered_arena 
        
        train_arenas = [registered_arena[cfg.arena.name](cfg.arena) for _ in range(cfg.num_train_envs)]
        train_arenas = [Parallel(arn, "process") for arn in train_arenas]
        task = build_sim_task(cfg.task)
        
        for i, arn in enumerate(train_arenas):
            arn.set_task(task)
            arn.set_log_dir(save_dir)
            arn.set_id(i)

        eval_arena = registered_arena[cfg.arena.name](cfg.arena)
        eval_arena.set_task(task)
        eval_arena.set_log_dir(save_dir)
        
        val_arena = registered_arena[cfg.arena.name](cfg.arena)
        val_arena.set_task(task)
        val_arena.set_log_dir(save_dir)

        ag_ar.train_plural_eval_single(
            agent,
            train_arenas,
            eval_arena,
            val_arena,
            cfg.agent.validation_interval,
            cfg.agent.total_update_steps,
            cfg.agent.eval_checkpoint,
            policy_terminate=False,
            env_success_stop=False
        )
        
        for arn in train_arenas:
            arn.close()
        eval_arena.close()
        val_arena.close()
        
    else:
        raise ValueError(f"Unrecognized training mode: {cfg.train_and_eval}")

def execute_transfer_evaluation_phase(transfer_cfg: DictConfig, train_cfg: DictConfig, agent: Any, save_root: str) -> None:
    """
    Evaluates the trained agent across multiple transfer arenas.
    
    Iterates through the requested transfer arenas, dynamically composes their 
    isolated configurations, and runs the evaluation protocol.
    
    Args:
        transfer_cfg (DictConfig): The transfer evaluation specific configuration block.
        train_cfg (DictConfig): The base training configuration (used for project metadata).
        agent (Any): The trained Actoris Harena agent.
        save_root (str): The resolved root directory for saving output data.
    """
    print("[INFO] Initializing Transfer Evaluation Phase...")

    eval_arenas = transfer_cfg.get("eval_arenas", [])
    if not eval_arenas:
        print("[WARNING] No 'eval_arenas' found in configuration. Skipping transfer phase.")
        return

    # Extract the base training configuration name and strip any preceding folders
    train_config_name = train_cfg.exp_name.split('/')[-1]

    for i, eval_setup in enumerate(eval_arenas):
        try:
            # 1. Compose configs using the FULL path string
            arena_cfg = compose(config_name=f"arena/{eval_setup.arena}")
            task_cfg = compose(config_name=f"task/{eval_setup.task}")
            
            # Safety Check: Unwrap @package directives if Hydra nested them
            arena_cfg = arena_cfg.get("arena", arena_cfg)
            task_cfg = task_cfg.get("task", task_cfg)

            print(f"[INFO] --- Starting Evaluation {i+1}/{len(eval_arenas)} ---")
            print(f"[INFO] Arena: {arena_cfg.name} | Task: {task_cfg.task_name}")

            # 2. Extract ONLY the last bit for the folder names
            clean_arena_name = eval_setup.arena.split('/')[-1]
            clean_task_name = eval_setup.task.split('/')[-1]

            # 3. Construct the flattened directory structure
            transfer_eval_dir_spe = os.path.join(
                save_root, 
                "transfer_eval", 
                train_config_name, 
                clean_arena_name, 
                clean_task_name
            )
            os.makedirs(transfer_eval_dir_spe, exist_ok=True)

            # Build Arena utilizing metadata from the base training config
            arena = ag_ar.build_arena(
                arena_cfg.name, 
                arena_cfg,
                project_name=train_cfg.project_name,
                exp_name=f"transfer_arena_{i}", 
                save_dir=transfer_eval_dir_spe
            )
                
            task = build_sim_task(task_cfg)
            arena.set_task(task)
            arena.reset()

            # Execute evaluation against the best checkpoint generated during the training phase
            res = ag_ar.evaluate(
                agent,
                arena,
                checkpoint=-2, 
                policy_terminate=False,
                env_success_stop=False
            )
            
            print(f"[INFO] Evaluation {i+1} completed. Results: {res}")

        except AttributeError as e:
            print(f"[ERROR] API Interface Error during transfer eval {i+1}: {e}")
            print("[ERROR] Please ensure the evaluation function signature matches the actoris_harena API.")
        except Exception as e:
            print(f"[ERROR] Unexpected error during transfer eval {i+1}: {e}")
        finally:
            # Clean up the C++ physics environment to prevent OpenGL memory leaks
            if 'arena' in locals():
                arena.close()

@hydra.main(config_path="../conf", config_name="train_and_transfer", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main execution entry point. Bootstraps environment, handles path resolutions,
    and sequentially triggers training and transfer evaluation.
    """
    os.environ['MEGPIE_ACTIVE_AGENT'] = cfg.agent.name
    register_agents()
    register_arenas()

    # 1. Resolve and lock paths
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = resolve_save_root(cfg.save_root)
    OmegaConf.set_struct(cfg, True)
    
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    print(f"Using Save Directory: {save_dir}")

    # 2. Build the Agent
    print(f"Building Agent: {cfg.agent.name}")
    agent = ag_ar.build_agent(
        cfg.agent.name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir
    )

    # 3. Execute Phases
    execute_training_phase(cfg, agent, save_dir)
    execute_transfer_evaluation_phase(cfg, agent, save_dir)

    print("Unified pipeline execution completed successfully.")

if __name__ == "__main__":
    main()