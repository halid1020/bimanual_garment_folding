import hydra
from omegaconf import DictConfig, OmegaConf
import os

from train.utils import register_agent_arena, registered_arena, build_task
import agent_arena.api as ag_ar

@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):
    register_agent_arena()

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config

    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    agent = ag_ar.build_agent(
        cfg.agent.name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir)
    
    print('[hydra eval] agent', cfg.agent.name, agent)
    
    # data_augmenter
    
    # Data Augmenter should be part of agent.
    # augmenter = build_data_augmenter(cfg.data_augmenter)
    # agent.set_data_augmenter(augmenter)

    # logging
    
    arena = registered_arena[cfg.arena.name](cfg.arena) #We want to bulid this with agent arena.
    task = build_task(cfg.task)
    arena.set_task(task)
    arena.set_log_dir(save_dir, cfg.project_name, cfg.exp_name)
    
    

    # training
    ag_ar.evaluate(
        agent,
        arena,
        -1,
        load_best=True,
        policy_terminate=False,
        env_success_stop=False
    )


if __name__ == "__main__":
    main()
