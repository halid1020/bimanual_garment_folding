from controllers.rl.primitive_encoding_sac \
    import PrimitiveEncodingSAC
from controllers.demonstrators.centre_sleeve_folding_stochastic_policy \
    import CentreSleeveFoldingStochasticPolicy
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy \
    import WaistLegFoldingStochasticPolicy
from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy \
    import WaistHemAlignmentFoldingStochasticPolicy
from controllers.human.real_world_human_policy import RealWordHumanPolicy
from controllers.human.real_world_human_single_arm_pick_and_place_policy \
    import RealWorldSingleArmHumanPickAndPlacePolicy
from controllers.rl.vanilla_image_sac import VanillaImageSAC
from controllers.rl.vanilla_sac import VanillaSAC
from controllers.rl.image2state_sac import Image2State_SAC
from controllers.rl.primitive2vector_sac \
    import Primitive2VectorSAC
from controllers.rl.demo_sac \
    import DemoSAC
from controllers.rl.maple \
    import MAPLE
from controllers.rl.image2state_multi_primitive_sac \
    import Image2StateMultiPrimitiveSAC
from controllers.gpt_fabric.adapter import GPTFabricAdapter
from controllers.rl.dreamer_v3.adapter import DreamerV3Adapter
from controllers.human.human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
from controllers.human.human_single_picker_pick_and_place import HumanSinglePickerPickAndPlace
from controllers.human.human_multi_primitive import HumanMultiPrimitive
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.multi_primitive_diffusion.adapter import MultiPrimitiveDiffusionAdapter
from controllers.iou_based_stitching_policy import IoUBasedStitchingPolicy
from controllers.vlm_based_stitching_policy import VLMBasedStitchingPolicy
from controllers.rl.lagarnet.gc_rssm import GC_RSSM
from controllers.rl.lagarnet.cloth_mask_workspace_pick_and_place_mpc import ClothMaskWorkspacePickAndPlaceMPC
from controllers.rl.lagarnet.single_arm_mask_pick_and_place_mpc import SingleArmMaskPickAndPlaceMPC

import actoris_harena as athar

def register_agents():
    athar.register_agent('centre_sleeve_folding_stochastic_policy', CentreSleeveFoldingStochasticPolicy)
    athar.register_agent('wasit_leg_alignment_folding_stochastic_policy', WaistLegFoldingStochasticPolicy)
    athar.register_agent('wasit_hem_alignment_folding_stochastic_policy', WaistHemAlignmentFoldingStochasticPolicy)
    athar.register_agent('primitive-encoding-sac', PrimitiveEncodingSAC)
    athar.register_agent('vanilla-image-sac', VanillaImageSAC)
    athar.register_agent('vanilla-sac', VanillaSAC)
    athar.register_agent('image2state-sac', Image2State_SAC)
    athar.register_agent('primitive2vector-sac', Primitive2VectorSAC)
    athar.register_agent('demo-sac', DemoSAC)
    athar.register_agent('maple', MAPLE)
    athar.register_agent('image2state-multi-primitive-sac', Image2StateMultiPrimitiveSAC)
    athar.register_agent('gpt-fabric', GPTFabricAdapter)
    athar.register_agent('dreamerV3', DreamerV3Adapter)
    athar.register_agent('human-dual-pickers-pick-and-place', HumanDualPickersPickAndPlace)
    athar.register_agent('human-single-picker-pick-and-place', HumanSinglePickerPickAndPlace)
    athar.register_agent('human-multi-primitive', HumanMultiPrimitive)
    athar.register_agent('real-world-human', RealWordHumanPolicy)
    athar.register_agent('real-world-human-single-arm-pick-and-place', RealWorldSingleArmHumanPickAndPlacePolicy)
    athar.register_agent('random-multi-primitive', RandomMultiPrimitive)
    athar.register_agent('multi-primitive-diffusion', MultiPrimitiveDiffusionAdapter)
    athar.register_agent('iou-based-stitching-policy', IoUBasedStitchingPolicy)
    athar.register_agent('vlm-based-stitching-policy', VLMBasedStitchingPolicy)
    athar.register_agent('lagarnet', GC_RSSM)
    athar.register_agent('cloth_mask_workspace_pick_and_place_mpc', ClothMaskWorkspacePickAndPlaceMPC)
    athar.register_agent('single_arm_mask_pick_and_place_mpc', SingleArmMaskPickAndPlaceMPC)