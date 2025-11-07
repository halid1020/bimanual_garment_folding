# skills/pick_and_fling_skill.py
import math
import numpy as np
import time


class PickAndFlingSkill:
    """
    Pick-and-Fling skill primitive.
    Step executes a fling motion with pick coordinates.
    """

    def __init__(self, scene):
        self.scene = scene
        self.vertical_rotvec = [math.pi, 0.0, 0.0]
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_dist = 0.1
        self.move_speed = 0.2
        self.move_acc = 0.2

    def reset(self):
        print("[PickAndFlingSkill] Resetting...")
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def step(self, action):
        """
        Args:
            action: dict with keys: pick_0, pick_1 -> np.array([x, y, z])
        """
        scene = self.scene
        rot = self.vertical_rotvec

        p0, p1 = np.array(action["pick_0"]), np.array(action["pick_1"])
        def approach(p): return p + [0, 0, self.approach_dist]
        def lift(p): return p + [0, 0, self.lift_dist]

        print("[PickAndFlingSkill] Executing pick-and-fling...")

        scene.both_movel(np.concatenate([approach(p0), rot]),
                         np.concatenate([approach(p1), rot]),
                         self.move_speed, self.move_acc)

        scene.both_movel(np.concatenate([p0, rot]), np.concatenate([p1, rot]), 0.08, 0.05)
        scene.both_close_gripper(); time.sleep(1)
        scene.both_movel(np.concatenate([lift(p0), rot]), np.concatenate([lift(p1), rot]), 0.2, 0.1)

        # Perform fling
        scene.dual_arm_stretch_and_fling(lift(p0), np.copy(lift(p1)))
        scene.both_home()
        print("✅ [PickAndFlingSkill] Completed.")
