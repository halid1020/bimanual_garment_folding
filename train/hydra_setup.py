import hydra
from omegaconf import DictConfig, OmegaConf
import os
from tqdm import tqdm

from train.utils import register_agent_arena, registered_arena, build_task, build_data_augmenter
import agent_arena.api as ag_ar
from agent_arena import TrainableAgent
from train.utils import register_agent_arena,  build_data_augmenter

@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):
    register_agent_arena()

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config

    agent = ag_ar.build_agent(cfg.agent.name, cfg.agent)
    print('[hydra setup] agent', cfg.agent.name, agent)
    
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

    for mode in ['eval', 'val', 'train']:
        if mode == 'train':
            configs = arena.get_train_configs()
            arena.set_train()
        elif mode == 'val':
            configs = arena.get_val_configs()
            arena.set_val()
        elif mode == 'eval':
            configs = arena.get_eval_configs()
            arena.set_eval()

        for cfg in tqdm(configs, desc=f"Setup {mode} trials ..."):
            arena.reset(cfg)



if __name__ == "__main__":
    main()
