
from dotmap import DotMap
import agent_arena as ag_ar

from env.softgym_garment.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.softgym_garment.single_garment_vectorised_fold_prim_env import SingleGarmentVectorisedFoldPrimEnv
from env.softgym_garment.multi_garment_env import MultiGarmentEnv
from env.softgym_garment.multi_garment_vectorised_fold_prim_env import MultiGarmentVectorisedFoldPrimEnv
from env.softgym_garment.multi_garment_vectorised_single_picker_pick_and_place_env import MultiGarmentVectorisedSinglePickerPickAndPlaceEnv
from env.softgym_garment.multi_garment_vectorised_dual_picker_pick_and_place_env import MultiGarmentVectorisedDualPickerPickAndPlaceEnv
from env.softgym_garment.single_garment_subgoal_init_vectorised_fold_prim_env import SingleGarmentSubgoalInitVectorisedFoldPrimEnv
from env.softgym_garment.single_garment_second_last_goal_vectorised_fold_prim_env import SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv    
from env.robosuite_env.robosuite_arena import RoboSuiteArena    
from env.robosuite_env.robosuite_skill_arena import RoboSuiteSkillArena
from env.dm_control.dmc_arena import DMC_Arena

from env.softgym_garment.tasks.garment_folding import GarmentFoldingTask
from env.softgym_garment.tasks.garment_flattening import GarmentFlatteningTask



# registered_arena = {
#     'single-garment-fixed-init-env':  SingleGarmentFixedInitialEnv,
#     'single-garment-vectorised-fold-prim-env': SingleGarmentVectorisedFoldPrimEnv,
#     'single-garment-subgoal-init-vectorised-fold-prim-env': SingleGarmentSubgoalInitVectorisedFoldPrimEnv,
#     'multi-garment-env': MultiGarmentEnv,
#     'multi-garment-vectorised-dual-picker-pick-and-place-env': MultiGarmentVectorisedDualPickerPickAndPlaceEnv,
#     'multi-garment-vectorised-single-picker-pick-and-place-env': MultiGarmentVectorisedSinglePickerPickAndPlaceEnv,
#     'robosuite-env': RoboSuiteArena,
#     'robosuite-skill-env': RoboSuiteSkillArena,
#     'single-garment-second-last-goal-init-vectorised-fold-prim-env': SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv,
#     'dm_control': DMC_Arena
# }

def register_arena():
    ag_ar.register_arena('single-garment-fixed-init-env', SingleGarmentFixedInitialEnv)
    ag_ar.register_arena('single-garment-vectorised-fold-prim-env', SingleGarmentVectorisedFoldPrimEnv)
    ag_ar.register_arena('single-garment-subgoal-init-vectorised-fold-prim-env', SingleGarmentSubgoalInitVectorisedFoldPrimEnv)
    ag_ar.register_arena('multi-garment-env', MultiGarmentEnv)
    ag_ar.register_arena('multi-garment-vectorised-dual-picker-pick-and-place-env', MultiGarmentVectorisedDualPickerPickAndPlaceEnv)
    ag_ar.register_arena('multi-garment-vectorised-single-picker-pick-and-place-env', MultiGarmentVectorisedSinglePickerPickAndPlaceEnv)
    ag_ar.register_arena('robosuite-env', RoboSuiteArena)
    ag_ar.register_arena('robosuite-skill-env', RoboSuiteSkillArena)
    ag_ar.register_arena('single-garment-second-last-goal-init-vectorised-fold-prim-env', SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv)
    ag_ar.register_arena('dm_control',  DMC_Arena)


def register_agent():
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
    ag_ar.register_agent('human-dual-pickers-pick-and-place', HumanDualPickersPickAndPlace)
    ag_ar.register_agent('human-single-picker-pick-and-place', HumanSinglePickerPickAndPlace)
    ag_ar.register_agent('human-multi-primitive', HumanMultiPrimitive)
    ag_ar.register_agent('random-multi-primitive', RandomMultiPrimitive)
    ag_ar.register_agent('multi-primitive-diffusion', MultiPrimitiveDiffusionAdapter)
    ag_ar.register_agent('iou-based-stitching-policy', IoUBasedStitchingPolicy)
    ag_ar.register_agent('vlm-based-stitching-policy', VLMBasedStitchingPolicy)
    ag_ar.register_agent('lagarnet', GC_RSSM)

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

