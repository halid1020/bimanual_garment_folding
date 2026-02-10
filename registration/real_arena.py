import actoris_harena as ag_ar
from real_robot.robot.dual_arm_arena import DualArmArena

def register_arenas():
    ag_ar.register_arena('real-world-dual-arm-multi-primitive', DualArmArena)