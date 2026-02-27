import actoris_harena as ag_ar

from env.softgym_garment.single_garment_fixed_initial_env import SingleGarmentFixedInitialEnv
from env.softgym_garment.single_garment_vectorised_fold_prim_env import SingleGarmentVectorisedFoldPrimEnv
from env.softgym_garment.multi_garment_env import MultiGarmentEnv
# from env.softgym_garment.multi_garment_vectorised_fold_prim_env import MultiGarmentVectorisedFoldPrimEnv
from env.softgym_garment.multi_garment_vectorised_single_picker_pick_and_place_env \
    import MultiGarmentVectorisedSinglePickerPickAndPlaceEnv, MultiGarmentVectorisedSinglePickerPickAndPlaceEnvRay
from env.softgym_garment.multi_garment_vectorised_dual_picker_pick_and_place_env import MultiGarmentVectorisedDualPickerPickAndPlaceEnv
from env.softgym_garment.single_garment_subgoal_init_vectorised_fold_prim_env import SingleGarmentSubgoalInitVectorisedFoldPrimEnv
from env.softgym_garment.single_garment_second_last_goal_vectorised_fold_prim_env import SingleGarmentSecondLastGoalInitVectorisedFoldPrimEnv    
from env.robosuite_env.robosuite_arena import RoboSuiteArena    
from env.robosuite_env.robosuite_skill_arena import RoboSuiteSkillArena
from env.dm_control.dmc_arena import DMC_Arena

def register_arenas():
    ag_ar.register_arena('single-garment-fixed-init-env', SingleGarmentFixedInitialEnv)
    ag_ar.register_arena('single-garment-vectorised-fold-prim-env', SingleGarmentVectorisedFoldPrimEnv)
    ag_ar.register_arena('single-garment-subgoal-init-vectorised-fold-prim-env', SingleGarmentSubgoalInitVectorisedFoldPrimEnv)
    ag_ar.register_arena('multi-garment-env', MultiGarmentEnv)
    ag_ar.register_arena('multi-garment-vectorised-dual-picker-pick-and-place-env', MultiGarmentVectorisedDualPickerPickAndPlaceEnv)
    ag_ar.register_arena('multi-garment-vectorised-single-picker-pick-and-place-env', MultiGarmentVectorisedSinglePickerPickAndPlaceEnv)
    ag_ar.register_arena('multi-garment-vectorised-single-picker-pick-and-place-env-ray', MultiGarmentVectorisedSinglePickerPickAndPlaceEnvRay)
    ag_ar.register_arena('robosuite-env', RoboSuiteArena)
    ag_ar.register_arena('robosuite-skill-env', RoboSuiteSkillArena)
    ag_ar.register_arena('dm_control',  DMC_Arena)