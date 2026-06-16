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


def execute_training_phase(train_cfg: DictConfig, agent: Any, save_dir: str) -> None:
    """
    Executes the training loop based on the configuration parameters.
    Catches SystemExit to prevent the API from forcefully closing the pipeline,
    and guarantees C++ physics cleanup via a finally block.
    """
    print("[INFO] Initializing Training Phase...", flush=True)
    
    # Initialize variables to None so we can safely check them during cleanup
    arena = None
    train_arenas = []
    eval_arena = None
    val_arena = None

    try:
        if train_cfg.train_and_eval == 'train_and_evaluate_single':
            arena = ag_ar.build_arena(
                train_cfg.arena.name, 
                train_cfg.arena,
                project_name=train_cfg.project_name,
                exp_name=train_cfg.exp_name,
                save_dir=save_dir
            )
                
            task = build_sim_task(train_cfg.task)
            arena.set_task(task)

            ag_ar.train_and_evaluate_single(
                agent,
                arena,
                train_cfg.agent.validation_interval,
                train_cfg.agent.total_update_steps,
                eval_last_check=True,
                eval_best_check=True,
                policy_terminate=False,
                env_success_stop=False
            )

        elif train_cfg.train_and_eval == 'train_plural_eval_single':
            from train.utils import registered_arena 
            
            train_arenas = [registered_arena[train_cfg.arena.name](train_cfg.arena) for _ in range(train_cfg.num_train_envs)]
            train_arenas = [Parallel(arn, "process") for arn in train_arenas]
            task = build_sim_task(train_cfg.task)
            
            for i, arn in enumerate(train_arenas):
                arn.set_task(task)
                arn.set_log_dir(save_dir)
                arn.set_id(i)

            eval_arena = registered_arena[train_cfg.arena.name](train_cfg.arena)
            eval_arena.set_task(task)
            eval_arena.set_log_dir(save_dir)
            
            val_arena = registered_arena[train_cfg.arena.name](train_cfg.arena)
            val_arena.set_task(task)
            val_arena.set_log_dir(save_dir)

            ag_ar.train_plural_eval_single(
                agent,
                train_arenas,
                eval_arena,
                val_arena,
                train_cfg.agent.validation_interval,
                train_cfg.agent.total_update_steps,
                train_cfg.agent.eval_checkpoint,
                policy_terminate=False,
                env_success_stop=False
            )
            
        else:
            raise ValueError(f"Unrecognized training mode: {train_cfg.train_and_eval}")

    except SystemExit as e:
        print(f"\n[INFO] Training API attempted an automatic system exit (code {e.code}).", flush=True)
        print("[INFO] Intercepting exit to proceed to the Transfer Evaluation phase...\n", flush=True)
    
    finally:
        # Guarantee that all C++ environments are destroyed before moving on
        print("[INFO] Cleaning up training environments...", flush=True)
        if arena is not None:
            try: arena.close()
            except Exception: pass
            
        for arn in train_arenas:
            try: arn.close()
            except Exception: pass
            
        if eval_arena is not None:
            try: eval_arena.close()
            except Exception: pass
            
        if val_arena is not None:
            try: val_arena.close()
            except Exception: pass

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
            
            # Safely extract names for logging without crashing Hydra
            safe_arena_name = arena_cfg.get("name", "Unknown_Arena")
            safe_task_name = task_cfg.get("name", task_cfg.get("    name", "Unknown_Task"))
            print(f"[INFO] Arena: {safe_arena_name} | Task: {safe_task_name}")

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
            print('About to Build arena')
            # Build Arena utilizing metadata from the base training config
            arena = ag_ar.build_arena(
                arena_cfg.name, 
                arena_cfg,
                project_name=train_cfg.project_name,
                exp_name=f"transfer_arena_{i}", 
                save_dir=transfer_eval_dir_spe
            )
            print('About to Build task')
            task = build_sim_task(task_cfg)
            arena.set_task(task)
            print('About to Build reset arena')
            arena.reset()
            print('Finished resetting arena')
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

@hydra.main(config_path="../conf", config_name="transfer_eval/comp_gc_diffusion", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main execution entry point. Bootstraps environment, handles path resolutions,
    dynamically loads the training config, and sequentially triggers training and transfer evaluation.
    """
    register_agents()
    register_arenas()

    # 1. Unlock Struct for Dynamic Access and parse Transfer Config
    OmegaConf.set_struct(cfg, False)
    transfer_cfg = cfg.transfer_eval if "transfer_eval" in cfg else cfg

    # 2. Dynamically compose the training configuration using the path in your transfer yaml
    print(f"[INFO] Composing training config from: {transfer_cfg.train_exp_config}")
    train_cfg = compose(config_name=transfer_cfg.train_exp_config)

    # 3. NOW we can safely expose the agent name because train_cfg has it
    os.environ['MEGPIE_ACTIVE_AGENT'] = train_cfg.agent.name

    # 4. Resolve and lock paths using the training config
    OmegaConf.set_struct(train_cfg, False)
    new_save_root = resolve_save_root(train_cfg.save_root)
    train_cfg.save_root = new_save_root
    OmegaConf.set_struct(train_cfg, True)
    
    save_dir = os.path.join(train_cfg.save_root, train_cfg.exp_name)
    print(f"[INFO] Using Save Directory: {save_dir}")

    # 5. Build the Agent
    print(f"[INFO] Building Agent: {train_cfg.agent.name}")
    agent = ag_ar.build_agent(
        train_cfg.agent.name, 
        train_cfg.agent,
        project_name=train_cfg.project_name,
        exp_name=train_cfg.exp_name,
        save_dir=save_dir
    )

    # 6. Execute Phases sequentially with the correct variables
    execute_training_phase(train_cfg, agent, save_dir)
    execute_transfer_evaluation_phase(transfer_cfg, train_cfg, agent, new_save_root)

    print("[INFO] Unified pipeline execution completed successfully.")

if __name__ == "__main__":
    main()