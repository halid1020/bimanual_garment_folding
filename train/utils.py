import agent_arena as ag_ar
from dotmap import DotMap


from env.softgym_garment.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.softgym_garment.single_garment_vectorised_fold_prim_env import SingleGarmentVectorisedFoldPrimEnv
from env.softgym_garment.multi_garment_env import MultiGarmentEnv
from env.softgym_garment.multi_garment_vectorised_fold_prim_env import MultiGarmentVectorisedFoldPrimEnv
from env.softgym_garment.single_garment_subgoal_init_vectorised_fold_prim_env import SingleGarmentSubgoalInitVectorisedFoldPrimEnv
from env.softgym_garment.single_garment_second_last_goal_vectorised_fold_prim_env import SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv    
from env.robosuite_env.robosuite_arena import RoboSuiteArena    
from env.robosuite_env.robosuite_skill_arena import RoboSuiteSkillArena
from env.dm_control.dmc_arena import DMC_Arena

from env.softgym_garment.tasks.garment_folding import GarmentFoldingTask
from env.softgym_garment.tasks.garment_flattening import GarmentFlatteningTask


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
from controllers.human.human_fold import HumanFold
from controllers.human.human_multi_primitive import HumanMultiPrimitive
from controllers.multi_primitive_diffusion.adapter import MultiPrimitiveDiffusionAdapter

# Add this import block at the top of the second file
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter import PixelBasedMultiPrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_single_primitive_data_augmenter import PixelBasedSinglePrimitiveDataAugmenter
from controllers.data_augmentation.pixel_based_fold_data_augmenter import PixelBasedFoldDataAugmenter
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter_for_dreamer import PixelBasedMultiPrimitiveDataAugmenterForDreamer
from controllers.data_augmentation.pixel_based_multi_primitive_data_augmenter_for_diffusion import PixelBasedMultiPrimitiveDataAugmenterForDiffusion



registered_arena = {
    'single-garment-fixed-init-env':  SingleGarmentFixedInitialEnv,
    'single-garment-vectorised-fold-prim-env': SingleGarmentVectorisedFoldPrimEnv,
    'single-garment-subgoal-init-vectorised-fold-prim-env': SingleGarmentSubgoalInitVectorisedFoldPrimEnv,
    'multi-garment-env': MultiGarmentEnv,
    'multi-garment-vectorised-fold-prim-env': MultiGarmentVectorisedFoldPrimEnv,
    'robosuite-env': RoboSuiteArena,
    'robosuite-skill-env': RoboSuiteSkillArena,
    'single-garment-second-last-goal-init-vectorised-fold-prim-env': SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv,
    'dm_control': DMC_Arena
}
def register_agent_arena():
    ag_ar.register_agent('centre_sleeve_folding_stochastic_policy', CentreSleeveFoldingStochasticPolicy)
    ag_ar.register_agent('wasit_leg_alignment_folding_stochastic_policy', WaistLegFoldingStochasticPolicy)
    ag_ar.register_agent('wasit_hem_alignment_folding_stochastic_policy', WaistHemAlignmentFoldingStochasticPolicy)
    ag_ar.register_agent('primitive-encoding-sac', PrimitiveEncodingSAC)
    ag_ar.register_agent('vanilla-image-sac', VanillaImageSAC)
    ag_ar.register_agent('vanilla-sac', VanillaSAC)
    ag_ar.register_agent('image2state-sac', Image2State_SAC)
    ag_ar.register_agent('primitive2vector-sac', Primitive2VectorSAC)
    ag_ar.register_agent('demo-sac', DemoSAC)
    ag_ar.register_agent('maple', MAPLE)
    ag_ar.register_agent('image2state-multi-primitive-sac', Image2StateMultiPrimitiveSAC)
    ag_ar.register_agent('gpt-fabric', GPTFabricAdapter)
    ag_ar.register_agent('dreamerV3', DreamerV3Adapter)
    ag_ar.register_agent('human-fold', HumanFold)
    ag_ar.register_agent('human-multi-primitive', HumanMultiPrimitive)
    ag_ar.register_agent('multi-primitive-diffusion', MultiPrimitiveDiffusionAdapter)


def build_task(task_cfg):
    # task
    if task_cfg.task_name == 'centre-sleeve-folding':
        demonstrator = HumanMultiPrimitive({"debug": False})
        task = GarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
       
    elif task_cfg.task_name == 'waist-leg-alignment-folding':
        from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy \
            import WaistLegFoldingStochasticPolicy
        demonstrator = WaistLegFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
       
    elif task_cfg.task_name == 'waist-hem-alignment-folding':
        from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy \
            import WaistHemAlignmentFoldingStochasticPolicy
        demonstrator = WaistHemAlignmentFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
        
    elif task_cfg.task_name == 'flattening':
        task = GarmentFlatteningTask(task_cfg)
       
    elif task_cfg.task_name == 'dummy':
        task = None
    else:
        raise NotImplementedError(f"Task {task_cfg.task_name} not supported")
    return task

def build_data_augmenter(cfg):
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
