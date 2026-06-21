# tool/hydra_train_and_transfer.py

"""
Unified Training and Transfer Evaluation Pipeline for Actoris Harena.

This module orchestrates the full lifecycle of a robotic manipulation agent. 
It first trains the policy using the primary environment configurations, and upon 
completion, dynamically loads a suite of transfer arenas to evaluate the zero-shot 
or fine-tuned performance of the resulting checkpoints.
"""

import os
from typing import Any

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
    print("[INFO] Initializing Training Phase...", flush=True)
    
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
            arena.reset()

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
        print(f"\n[INFO] Training API exited (code {e.code}). Proceeding to transfer...", flush=True)
    
    finally:
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
    print("\n[INFO] Initializing Transfer Evaluation Phase...", flush=True)

    eval_arenas = transfer_cfg.get("eval_arenas", [])
    if not eval_arenas:
        print("[WARNING] No 'eval_arenas' found in configuration. Skipping transfer phase.", flush=True)
        return

    train_config_name = train_cfg.exp_name.split('/')[-1]

    for i, eval_setup in enumerate(eval_arenas):
        arena_cfg = compose(config_name=f"arena/{eval_setup.arena}")
        task_cfg = compose(config_name=f"task/{eval_setup.task}")
        
        # Mirroring the exact extraction logic from your standalone script
        if hasattr(arena_cfg, 'arena'):
            arena_cfg = arena_cfg.arena
            
        if hasattr(task_cfg, 'task') and hasattr(task_cfg.task, 'magpie'):
            task_cfg = task_cfg.task.magpie
        elif hasattr(task_cfg, 'magpie'):
            task_cfg = task_cfg.magpie

        print(f"\n>>> Starting Evaluation {i+1}/{len(eval_arenas)}")
        print(f">>> Arena: {arena_cfg.get('name', 'Unknown')} | Task: {task_cfg.get('task_name', 'Unknown')}")

        clean_arena_name = eval_setup.arena.split('/')[-1]
        clean_task_name = eval_setup.task.split('/')[-1]

        transfer_eval_dir_spe = os.path.join(
            save_root, 
            "transfer_eval", 
            train_config_name, 
            clean_arena_name, 
            clean_task_name
        )
        os.makedirs(transfer_eval_dir_spe, exist_ok=True)

        arena = ag_ar.build_arena(
            arena_cfg.name, 
            arena_cfg,
            project_name=train_cfg.project_name,
            exp_name=f"transfer_arena_{i}", 
            save_dir=transfer_eval_dir_spe
        )
            
        task = build_sim_task(task_cfg)
        arena.set_task(task)

        try:
            arena.reset()
            res = ag_ar.evaluate(
                agent,
                arena,
                checkpoint=-2, 
                policy_terminate=False,
                env_success_stop=False
            )
            print(f">>> Evaluation {i+1} completed. Results: {res}")
            
            arena.close() 
            
        except AttributeError as e:
            print(f"ERROR: {e}")
            print("Please update the evaluation function call to match your actoris_harena evaluation API.")


@hydra.main(config_path="../conf", config_name="transfer_eval/comp_gc_diffusion", version_base=None)
def main(cfg: DictConfig) -> None:
    register_agents()
    register_arenas()

    OmegaConf.set_struct(cfg, False)
    transfer_cfg = cfg.transfer_eval if "transfer_eval" in cfg else cfg

    print(f"[INFO] Composing training config from: {transfer_cfg.train_exp_config}", flush=True)
    train_cfg = compose(config_name=transfer_cfg.train_exp_config)

    os.environ['MEGPIE_ACTIVE_AGENT'] = train_cfg.agent.name

    OmegaConf.set_struct(train_cfg, False)
    new_save_root = resolve_save_root(train_cfg.save_root)
    train_cfg.save_root = new_save_root
    OmegaConf.set_struct(train_cfg, True)
    
    save_dir = os.path.join(train_cfg.save_root, train_cfg.exp_name)
    print(f"[INFO] Using Save Directory: {save_dir}", flush=True)

    print(f"[INFO] Building Agent: {train_cfg.agent.name}", flush=True)
    agent = ag_ar.build_agent(
        train_cfg.agent.name, 
        train_cfg.agent,
        project_name=train_cfg.project_name,
        exp_name=train_cfg.exp_name,
        save_dir=save_dir
    )

    execute_training_phase(train_cfg, agent, save_dir)
    execute_transfer_evaluation_phase(transfer_cfg, train_cfg, agent, new_save_root)

    print("\n[INFO] Unified pipeline execution completed successfully.", flush=True)

if __name__ == "__main__":
    main()