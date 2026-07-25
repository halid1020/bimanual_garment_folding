import actoris_harena as ag_ar
from real_robot.robot.dual_arm_arena import DualArmArena
from real_robot.robot.single_arm_pick_and_place_arena import SingleArmPickAndPlaceArena
from real_robot.robot.xarm_dual_arm_arena import XArmDualArmArena
from real_robot.robot.xarm_single_arm_pick_and_place_arena import XArmSingleArmPickAndPlaceArena

def register_arenas():
    ag_ar.register_arena('real-world-dual-arm-multi-primitive', DualArmArena)
    ag_ar.register_arena('real-world-single-arm-pick-and-place', SingleArmPickAndPlaceArena)
    ag_ar.register_arena('real-world-xarm-dual-arm-multi-primitive', XArmDualArmArena)
    ag_ar.register_arena('real-world-xarm-single-arm-pick-and-place', XArmSingleArmPickAndPlaceArena)