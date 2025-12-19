import hydra
from omegaconf import DictConfig, OmegaConf
import os

from train.utils import register_agent_arena, registered_arena, build_task, build_data_augmenter
import agent_arena.api as ag_ar
from agent_arena import TrainableAgent
from train.utils import register_agent_arena,  build_data_augmenter

@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):
    register_agent_arena()

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config

    agent = ag_ar.build_agent(cfg.agent.name, cfg.agent)
    print('[hydra eval] agent', cfg.agent.name, agent)
    
    # data_augmenter
    
    if isinstance(agent, TrainableAgent):
        augmenter = build_data_augmenter(cfg.data_augmenter)
        agent.set_data_augmenter(augmenter)

    # logging
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    arena = registered_arena[cfg.arena.name](cfg.arena) #We want to bulid this with agent arena.
    task = build_task(cfg.task)
    arena.set_task(task)
    arena.set_log_dir(save_dir)
    
    agent.set_log_dir(save_dir)

    # training
    ag_ar.evaluate(
        agent,
        arena,
        -1,
        policy_terminate=False,
        env_success_stop=False
    )


if __name__ == "__main__":
    main()
