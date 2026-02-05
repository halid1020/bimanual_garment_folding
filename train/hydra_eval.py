import hydra
from omegaconf import DictConfig, OmegaConf
import os

from train.utils import register_agent, register_arena, build_task
import agent_arena.api as ag_ar

@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):
    register_agent()
    register_arena()

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config

    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    agent = ag_ar.build_agent(
        cfg.agent.name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir)
    
    print('[hydra eval] agent', cfg.agent.name, agent)
    
    arena = ag_ar.build_arena(
        cfg.arena.name, 
        cfg.arena,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir)
    
    task = build_task(cfg.task)
    arena.set_task(task)
    
    # training
    ag_ar.evaluate(
        agent,
        arena,
        -2, # load best checkpoint
        policy_terminate=False,
        env_success_stop=False
    )


if __name__ == "__main__":
    main()
