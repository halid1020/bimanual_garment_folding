import os
from packaging import version
import torch 

from controllers.demonstrators.centre_sleeve_folding_stochastic_policy \
    import CentreSleeveFoldingPolicy
from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy \
    import WaistLegFoldingStochasticPolicy
from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy \
    import WaistHemAlignmentFoldingStochasticPolicy
from controllers.human.real_world_human_policy import RealWordHumanPolicy
from controllers.human.real_world_human_single_arm_pick_and_place_policy \
    import RealWorldSingleArmHumanPickAndPlacePolicy

from controllers.rl.dreamer_v3.adapter import DreamerV3Adapter
from controllers.human.human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
from controllers.human.human_single_picker_pick_and_place import HumanSinglePickerPickAndPlace
from controllers.human.human_multi_primitive import HumanMultiPrimitive
from controllers.random.random_multi_primitive import RandomMultiPrimitive
from controllers.random.dual_arm_random_pick_and_place import DualArmRandomPickAndPlace
from controllers.random.noise_injected_policy import NoiseInjectedPolcy

from controllers.iou_based_stitching_policy import IoUBasedStitchingPolicy

from controllers.rl.lagarnet.cloth_mask_workspace_pick_and_place_mpc import ClothMaskWorkspacePickAndPlaceMPC
from controllers.rl.lagarnet.single_arm_mask_pick_and_place_mpc import SingleArmMaskPickAndPlaceMPC
from controllers.rl.lagarnet.dual_arm_mask_pick_and_place_mpc import DualArmMaskPickAndPlaceMPC
from controllers.rl.cloth_mate.adapter import ClothMateAdapter
from controllers.rl.cloth_funnels.adapter import ClothFunnelsAdapter

import actoris_harena as athar

def register_agents():
    athar.register_agent('centre_sleeve_folding_policy', CentreSleeveFoldingPolicy)
    athar.register_agent('wasit_leg_alignment_folding_stochastic_policy', WaistLegFoldingStochasticPolicy)
    athar.register_agent('wasit_hem_alignment_folding_stochastic_policy', WaistHemAlignmentFoldingStochasticPolicy)
    
    athar.register_agent('dreamerV3', DreamerV3Adapter)
    athar.register_agent('human-dual-pickers-pick-and-place', HumanDualPickersPickAndPlace)
    athar.register_agent('human-single-picker-pick-and-place', HumanSinglePickerPickAndPlace)
    athar.register_agent('human-multi-primitive', HumanMultiPrimitive)
    athar.register_agent('real-world-human', RealWordHumanPolicy)
    athar.register_agent('real-world-human-single-arm-pick-and-place', RealWorldSingleArmHumanPickAndPlacePolicy)
    athar.register_agent('random-multi-primitive', RandomMultiPrimitive)
    athar.register_agent('iou-based-stitching-policy', IoUBasedStitchingPolicy)
   
    athar.register_agent('cloth_mask_workspace_pick_and_place_mpc', ClothMaskWorkspacePickAndPlaceMPC)
    athar.register_agent('single_arm_mask_pick_and_place_mpc', SingleArmMaskPickAndPlaceMPC)
    athar.register_agent('dual_arm_mask_pick_and_place_mpc', DualArmMaskPickAndPlaceMPC)
    athar.register_agent('clothmate', ClothMateAdapter)
    athar.register_agent('cloth-funnels', ClothFunnelsAdapter)
    
    athar.register_agent('noise-injected-policy', NoiseInjectedPolcy)
    athar.register_agent('dual-arm-random-pick-and-place', DualArmRandomPickAndPlace)

    active_agent = os.environ.get('MEGPIE_ACTIVE_AGENT', '')
    
    if active_agent == 'unifolding':
        from controllers.rl.unifolding.adapter import UniFoldingAdapter
        athar.register_agent('unifolding', UniFoldingAdapter)

    

    # PyTorch 2.0+ specific agents
    is_torch_2 = version.parse(torch.__version__) >= version.parse("2.0.0")
    if is_torch_2:
        from controllers.vlm_based_stitching_policy import VLMBasedStitchingPolicy
        athar.register_agent('vlm-based-stitching-policy', VLMBasedStitchingPolicy)

        from controllers.gpt_fabric.adapter import GPTFabricAdapter
        athar.register_agent('gpt-fabric', GPTFabricAdapter)

        from controllers.magpie.magpie_agent import MagpieAgent
        athar.register_agent('magpie', MagpieAgent)

        from controllers.rl.lagarnet.gc_rssm import GC_RSSM
        from controllers.rl.lagarnet.rssm import RSSM
        athar.register_agent('lagarnet', GC_RSSM)
        athar.register_agent('rssm', RSSM)

        if active_agent == 'vcd':
            from controllers.rl.vcd.adapter import VCDAdapter
            athar.register_agent('vcd', VCDAdapter)

        