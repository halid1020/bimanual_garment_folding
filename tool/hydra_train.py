import hydra
from omegaconf import DictConfig, OmegaConf
import os
import socket
import actoris_harena.api as ag_ar

# from tool.utils import register_agent, register_arena, build_task
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task
from tool.utils import resolve_save_root

from env.parallel import Parallel

@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    register_agents()
    register_arenas()

    new_save_root = resolve_save_root(cfg.save_root)
    print(f"[tool.hydra_train] Using Save Root: {cfg.save_root}")

    # Update the config object (must unset 'struct' to modify)
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    OmegaConf.set_struct(cfg, True)
    # -------------------------------------

    print("[tool.hydra_train] --- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("[tool.hydra_train] ---------------------")

    save_dir = os.path.join(cfg.save_root, cfg.exp_name)

    agent = ag_ar.build_agent(
        cfg.agent.name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir)
    

    if cfg.train_and_eval == 'train_and_evaluate_single':
        # training

        arena = ag_ar.build_arena(
            cfg.arena.name, 
            cfg.arena,
            project_name=cfg.project_name,
            exp_name=cfg.exp_name,
            save_dir=save_dir)
            
        task = build_task(cfg.task)
        arena.set_task(task)

        res = ag_ar.train_and_evaluate_single(
            agent,
            arena,
            cfg.agent.validation_interval,
            cfg.agent.total_update_steps,
            eval_last_check=True,
            eval_best_check=True,
            policy_terminate=False,
            env_success_stop=False
            #eval_checkpoint=-2, # evaluate best model.
        )

    elif cfg.train_and_eval == 'train_plural_eval_single':

        train_arenas = [registered_arena[cfg.arena.name](cfg.arena) for _ in range(cfg.num_train_envs)]
        train_arenas = [Parallel(arn, "process") for arn in train_arenas]
        task = build_task(cfg.task) #TODO: this needs to become part of agent-arena.
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

        res = ag_ar.train_plural_eval_single(
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



if __name__ == "__main__":
    main()
