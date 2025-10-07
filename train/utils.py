from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.single_garment_vectorised_fold_prim_env import SingleGarmentVectorisedFoldPrimEnv
from env.multi_garment_env import MultiGarmentEnv
from env.multi_garment_vectorised_fold_prim_env import MultiGarmentVectorisedFoldPrimEnv

from env.tasks.garment_folding import GarmentFoldingTask
from env.tasks.garment_flattening import GarmentFlatteningTask

from controllers.rl.image_based_multi_primitive_sac import ImageBasedMultiPrimitiveSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy import CentreSleeveFoldingStochasticPolicy

from controllers.rl.data_augmenter import PixelBasedPrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_fold_data_augmenter import PixelBasedFoldDataAugmenter

from controllers.rl.vanilla_image_sac import VanillaImageSAC
from controllers.rl.vanilla_sac import VanillaSAC

import agent_arena as ag_ar


registered_arena = {
    'single-garment-fixed-init-env':  SingleGarmentFixedInitialEnv,
    'single-garment-vectorised-fold-prim-env': SingleGarmentVectorisedFoldPrimEnv,
    'multi-garment-longsleeve-env': MultiGarmentEnv,
    'multi-garment-vectorised-fold-prim-env': MultiGarmentVectorisedFoldPrimEnv
}
def register_agent_arena():
    ag_ar.register_agent('centre_sleeve_folding_stochastic_policy', CentreSleeveFoldingStochasticPolicy)
    ag_ar.register_agent('image-based-multi-primitive-sac', ImageBasedMultiPrimitiveSAC)
    ag_ar.register_agent('vanilla-image-sac', VanillaImageSAC)
    ag_ar.register_agent('vanilla-sac', VanillaSAC)
