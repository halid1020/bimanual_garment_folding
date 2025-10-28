from env.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.single_garment_vectorised_fold_prim_env import SingleGarmentVectorisedFoldPrimEnv
from env.multi_garment_env import MultiGarmentEnv
from env.multi_garment_vectorised_fold_prim_env import MultiGarmentVectorisedFoldPrimEnv
from env.single_garment_subgoal_init_vectorised_fold_prim_env import SingleGarmentSubgoalInitVectorisedFoldPrimEnv
    

from env.tasks.garment_folding import GarmentFoldingTask
from env.tasks.garment_flattening import GarmentFlatteningTask


from controllers.rl.primitive_encoding_sac \
    import PrimitiveEncodingSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy \
    import CentreSleeveFoldingStochasticPolicy
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy \
    import WaistLegFoldingStochasticPolicy
from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy \
    import WaistHemAlignmentFoldingStochasticPolicy

from controllers.rl.vanilla_image_sac import VanillaImageSAC
from controllers.rl.vanilla_sac import VanillaSAC
from controllers.rl.image2state_sac import Image2State_SAC
from controllers.rl.image_based_multi_primitive_sac \
    import ImageBasedMultiPrimitiveSAC
from controllers.rl.primitive2vector_sac \
    import Primitive2VectorSAC

import agent_arena as ag_ar


registered_arena = {
    'single-garment-fixed-init-env':  SingleGarmentFixedInitialEnv,
    'single-garment-vectorised-fold-prim-env': SingleGarmentVectorisedFoldPrimEnv,
    'single-garment-subgoal-init-vectorised-fold-prim-env': SingleGarmentSubgoalInitVectorisedFoldPrimEnv,
    'multi-garment-env': MultiGarmentEnv,
    'multi-garment-vectorised-fold-prim-env': MultiGarmentVectorisedFoldPrimEnv
}
def register_agent_arena():
    ag_ar.register_agent('centre_sleeve_folding_stochastic_policy', CentreSleeveFoldingStochasticPolicy)
    ag_ar.register_agent('wasit_leg_alignment_folding_stochastic_policy', WaistLegFoldingStochasticPolicy)
    ag_ar.register_agent('wasit_hem_alignment_folding_stochastic_policy', WaistHemAlignmentFoldingStochasticPolicy)
    ag_ar.register_agent('image-based-multi-primitive-sac', ImageBasedMultiPrimitiveSAC)
    ag_ar.register_agent('primitive-encoding-sac', PrimitiveEncodingSAC)
    ag_ar.register_agent('vanilla-image-sac', VanillaImageSAC)
    ag_ar.register_agent('vanilla-sac', VanillaSAC)
    ag_ar.register_agent('image2state-sac', Image2State_SAC)
    ag_ar.register_agent('primitive2vector-sac', Primitive2VectorSAC)