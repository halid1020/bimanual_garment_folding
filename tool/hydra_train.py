import hydra
from omegaconf import DictConfig, OmegaConf
import os
import socket
import agent_arena.api as ag_ar

from tool.utils import register_agent, register_arena, build_task
from env.parallel import Parallel

@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    register_agent()
    register_arena()

    # --- Automatic save_root detection ---
    hostname = socket.gethostname()
    
    if hostname == "pc282":
        new_save_root = '/media/hcv530/T7/garment_folding_data'
    elif hostname == "thanos":
        new_save_root = '/data/ah390/bimanual_garment_folding'
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
