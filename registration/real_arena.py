import actoris_harena as ag_ar
from real_robot.robot.dual_arm_arena import DualArmArena
from real_robot.robot.single_arm_pick_and_place_arena import SingleArmPickAndPlaceArena

def register_arenas():
    ag_ar.register_arena('real-world-dual-arm-multi-primitive', DualArmArena)
    ag_ar.register_arena('real-world-single-arm-pick-and-place', SingleArmPickAndPlaceArena)