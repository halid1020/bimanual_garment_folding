# Add this import block at the top of the second file
import os
from omegaconf import OmegaConf

from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter import PixelBasedMultiPrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_single_primitive_data_augmenter import PixelBasedSinglePrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_fold_data_augmenter import PixelBasedFoldDataAugmenter
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter_for_dreamer import PixelBasedMultiPrimitiveDataAugmenterForDreamer
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter_for_diffusion import PixelBasedMultiPrimitiveDataAugmenterForDiffusion


def build_data_augmenter(cfg_str):

    # load config from cfg_str
    config_path = os.path.join("conf", "data_augmenter", f"{cfg_str}.yaml")
    cfg = flattening_policy_config = OmegaConf.load(config_path)

    name = cfg.name

    if name == 'pixel-based-multi-primitive-data-augmenter':
        return PixelBasedMultiPrimitiveDataAugmenter(cfg)

    elif name == 'pixel-based-fold-data-augmenter':
        return PixelBasedFoldDataAugmenter(cfg)

    elif name == 'pixel-based-single-primitive-augmenter':
        return PixelBasedSinglePrimitiveDataAugmenter(cfg)

    elif name == 'pixel-based-multi-primitive-data-augmenter-for-dreamer':
        return PixelBasedMultiPrimitiveDataAugmenterForDreamer(cfg)

    elif name == 'pixel-based-multi-primitive-data-augmenter-for-diffusion':
        return PixelBasedMultiPrimitiveDataAugmenterForDiffusion(cfg)

    elif name == 'identity':
        return lambda x: x

    else:
        raise NotImplementedError(f"Data augmenter {name} not supported")
