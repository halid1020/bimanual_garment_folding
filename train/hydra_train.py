import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dotmap import DotMap

import agent_arena.api as ag_ar
from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.tasks.garment_folding import GarmentFoldingTask
from env.tasks.garment_flattening import GarmentFlatteningTask
from controllers.multi_primitive_sac.image_based_multi_primitive_SAC import ImageBasedMultiPrimitiveSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy
from controllers.multi_primitive_sac.data_augmenter import PixelBasedPrimitiveDataAugmenter


@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config

    # arena
    arena = SingleGarmentFixedInitialEnv(cfg.arena)

    # task
    if cfg.task.task_name == 'centre-sleeve-folding':
        demonstrator = CentreSleeveFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**cfg.task, "demonstrator": demonstrator}))
        arena.set_task(task)
    elif cfg.task.task_name == 'flattening':
        task = GarmentFlatteningTask(cfg.task)
        arena.set_task(task)
    else:
        raise NotImplementedError(f"Task {cfg.task.task_name} not supported")

    # agent
    if cfg.agent.name == 'image-based-multi-primitive-sac':
        agent = ImageBasedMultiPrimitiveSAC(config=cfg.agent)
    else:
        raise NotImplementedError(f"Agent {cfg.agent.name} not supported")

    # data_augmenter
    if cfg.data_augmenter.name == 'pixel-based-primitive-data-augmenter':
        augmenter = PixelBasedPrimitiveDataAugmenter(cfg.data_augmenter)
        agent.set_data_augmenter(augmenter)
    else:
        raise NotImplementedError(f"Data augmenter {cfg.data_augmenter.name} not supported")

    # logging
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    arena.set_log_dir(save_dir)
    agent.set_log_dir(save_dir)

    # training
    res = ag_ar.train_and_evaluate(
        agent,
        arena,
        cfg.agent.validation_interval,
        cfg.agent.total_update_steps,
        cfg.agent.eval_checkpoint,
    )


if __name__ == "__main__":
    main()
