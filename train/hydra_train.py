import hydra
from omegaconf import DictConfig, OmegaConf
import os
import agent_arena.api as ag_ar

from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter import PixelBasedMultiPrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_single_primitive_data_augmenter import PixelBasedSinglePrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_fold_data_augmenter import PixelBasedFoldDataAugmenter
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter_for_dreamer import PixelBasedMultiPrimitiveDataAugmenterForDreamer
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter_for_diffusion import PixelBasedMultiPrimitiveDataAugmenterForDiffusion
from train.utils import register_agent_arena, registered_arena, build_task
from env.parallel import Parallel

@hydra.main(config_path="../conf", config_name="mp_sac_v5", version_base=None)
def main(cfg: DictConfig):
    register_agent_arena()

    print(OmegaConf.to_yaml(cfg))  # sanity check merged config


    agent = ag_ar.build_agent(cfg.agent.name, cfg.agent)
    #print('agent', cfg.agent.name, agent)
    
    # data_augmenter
    if cfg.data_augmenter.name == 'pixel-based-multi-primitive-data-augmenter':
        augmenter = PixelBasedMultiPrimitiveDataAugmenter(cfg.data_augmenter)
    elif cfg.data_augmenter.name == 'pixel-based-fold-data-augmenter':
        augmenter = PixelBasedFoldDataAugmenter(cfg.data_augmenter)
    elif cfg.data_augmenter.name == 'pixel-based-single-primitive-augmenter':
        augmenter = PixelBasedSinglePrimitiveDataAugmenter(cfg.data_augmenter)
    elif cfg.data_augmenter.name == 'pixel-based-multi-primitive-data-augmenter-for-dreamer':
        augmenter = PixelBasedMultiPrimitiveDataAugmenterForDreamer(cfg.data_augmenter)
    elif cfg.data_augmenter.name == 'pixel-based-multi-primitive-data-augmenter-for-diffusion':
        augmenter = PixelBasedMultiPrimitiveDataAugmenterForDiffusion(cfg.data_augmenter)
    elif cfg.data_augmenter.name == 'identity':
        augmenter = lambda x: x
    else:
        raise NotImplementedError(f"Data augmenter {cfg.data_augmenter.name} not supported")

    agent.set_data_augmenter(augmenter)
    # logging
    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    
    agent.set_log_dir(save_dir)

    if cfg.train_and_eval == 'train_and_evaluate_single':
        # training

        arena = registered_arena[cfg.arena.name](cfg.arena) #We want to bulid this with agent arena.
        task = build_task(cfg.task)
        arena.set_task(task)
        arena.set_log_dir(save_dir)

        res = ag_ar.train_and_evaluate_single(
            agent,
            arena,
            cfg.agent.validation_interval,
            cfg.agent.total_update_steps,
            cfg.agent.eval_checkpoint,
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
        )



if __name__ == "__main__":
    main()
