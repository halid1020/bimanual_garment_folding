import hydra
from omegaconf import DictConfig, OmegaConf
import os
from dotmap import DotMap

import agent_arena.api as ag_ar

from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.single_garment_vectorised_fold_prim_env import SingleGarmentVectorisedFoldPrimEnv
from env.multi_garment_env import MultiGarmentEnv

from env.tasks.garment_folding import GarmentFoldingTask
from env.tasks.garment_flattening import GarmentFlatteningTask

from controllers.rl.image_based_multi_primitive_sac import ImageBasedMultiPrimitiveSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy

from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter import PixelBasedMultiPrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_single_primitive_data_augmenter import PixelBasedSinglePrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_fold_data_augmenter import PixelBasedFoldDataAugmenter

from controllers.rl.vanilla_image_sac import VanillaImageSAC

from train.utils import register_agent_arena, registered_arena

@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):
    register_agent_arena()

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config

    # arena
    arena = registered_arena[cfg.arena.name](cfg.arena)

    # task
    if cfg.task.task_name == 'centre-sleeve-folding':
        demonstrator = CentreSleeveFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**cfg.task, "demonstrator": demonstrator}))
        arena.set_task(task)
    elif cfg.task.task_name == 'waist-leg-alignment-folding':
        from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy \
            import WaistLegFoldingStochasticPolicy
        demonstrator = WaistLegFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**cfg.task, "demonstrator": demonstrator}))
        arena.set_task(task)
    elif cfg.task.task_name == 'waist-hem-alignment-folding':
        from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy \
            import WaistHemAlignmentFoldingStochasticPolicy
        demonstrator = WaistHemAlignmentFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**cfg.task, "demonstrator": demonstrator}))
        arena.set_task(task)
    elif cfg.task.task_name == 'flattening':
        task = GarmentFlatteningTask(cfg.task)
        arena.set_task(task)
    elif cfg.task.task_name == 'dummy':
        pass
    else:
        raise NotImplementedError(f"Task {cfg.task.task_name} not supported")

    # agent
    #cfg.agent.exp_name = cfg.exp_name
    #cfg.agent.project_name = cfg.project_name
    agent = ag_ar.build_agent(cfg.agent.name, cfg.agent)
    print('agent', cfg.agent.name, agent)
    
    # data_augmenter
    if cfg.data_augmenter.name == 'pixel-based-multi-primitive-data-augmenter':
        augmenter = PixelBasedPrimitiveDataAugmenter(cfg.data_augmenter)
        agent.set_data_augmenter(augmenter)
    elif cfg.data_augmenter.name == 'pixel-based-fold-data-augmenter':
        data_augmenter = PixelBasedFoldDataAugmenter(cfg.data_augmenter)
        agent.set_data_augmenter(data_augmenter)
    elif cfg.data_augmenter.name == 'pixel-based-single-primitive-augmenter':
        data_augmenter = PixelBasedSinglePrimitiveDataAugmenter(cfg.data_augmenter)
        agent.set_data_augmenter(data_augmenter)
    elif cfg.data_augmenter.name == 'identity':
        pass
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
