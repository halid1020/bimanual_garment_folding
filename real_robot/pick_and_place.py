# skills/pick_and_place_skill.py
import math
import numpy as np
import time


class PickAndPlaceSkill:
    """
    Pick-and-Place skill primitive.
    Each step executes a pick-place with given pick/place coordinates.
    """

    def __init__(self, scene):
        self.scene = scene
        self.vertical_rotvec = [math.pi, 0.0, 0.0]
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_dist = 0.08
        self.move_speed = 0.2
        self.move_acc = 0.2
        self.home_after = True

    def reset(self):
        """Reset robot state for the skill."""
        print("[PickAndPlaceSkill] Resetting...")
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def step(self, action):
        """
        Args:
            action: dict with keys:
                pick_0, pick_1, place_0, place_1 -> each np.array([x, y, z])
        """
        scene = self.scene
        rot = self.vertical_rotvec

        p0p, p1p = np.array(action["pick_0"]), np.array(action["pick_1"])
        p0pl, p1pl = np.array(action["place_0"]), np.array(action["place_1"])

        def approach(p): return p + [0, 0, self.approach_dist]
        def lift(p): return p + [0, 0, self.lift_dist]

        print("[PickAndPlaceSkill] Executing pick-and-place action...")

        scene.both_movel(np.concatenate([approach(p0p), rot]),
                         np.concatenate([approach(p1p), rot]),
                         self.move_speed, self.move_acc)

        scene.both_movel(np.concatenate([p0p, rot]), np.concatenate([p1p, rot]), 0.08, 0.05)
        scene.both_close_gripper()
        time.sleep(1.0)

        scene.both_movel(np.concatenate([lift(p0p), rot]), np.concatenate([lift(p1p), rot]), 0.2, 0.1)

        scene.both_movel(np.concatenate([approach(p0pl), rot]),
                         np.concatenate([approach(p1pl), rot]),
                         0.2, 0.1)

        scene.both_movel(np.concatenate([p0pl, rot]), np.concatenate([p1pl, rot]), 0.08, 0.05)
        scene.both_open_gripper()
        time.sleep(0.5)

        scene.both_movel(np.concatenate([approach(p0pl), rot]),
                         np.concatenate([approach(p1pl), rot]),
                         0.2, 0.1)

        if self.home_after:
            scene.both_home()
        print("✅ [PickAndPlaceSkill] Completed.")
