# policies/human_policy.py
import numpy as np
from human_utils import click_points_pick_and_place, click_points_pick_and_fling

class HumanPolicy:
    """Interactive human-in-the-loop policy for giving pick/place actions."""

    def __init__(self, arena):
        self.arena = arena

    def single_act(self, info):
        """Get an action dict from user clicks."""
        while True:
            cmd = input("\nSkill [1=pick-place, 2=pick-fling, q=quit]: ").strip().lower()
            if cmd in ("q", "quit"):
                return None
            elif cmd == "1":
                skill_type = "pick_and_place"
                break
            elif cmd == "2":
                skill_type = "pick_and_fling"
                break
            else:
                print("Invalid command. Please enter 1, 2, or q.")

        # Unpack scene info
        rgb, depth, mask = info["rgb"], info["depth"], info["mask"]
        workspace_mask_0, workspace_mask_1 = info["workspace_mask_0"], info["workspace_mask_1"]

        h, w = rgb.shape[:2]

        # -------------------------------
        # Pick & Place
        # -------------------------------
        if skill_type == "pick_and_place":
            clicks = click_points_pick_and_place("Pick & Place", rgb)

            pick_0, place_0, pick_1, place_1 = clicks

            # Normalize pixel coordinates to [-1, 1]
            def norm_xy(pt):
                x, y = pt
                return ((x / w) * 2 - 1, (y / h) * 2 - 1)

            pick_0_norm = norm_xy(pick_0)
            place_0_norm = norm_xy(place_0)
            pick_1_norm = norm_xy(pick_1)
            place_1_norm = norm_xy(place_1)

            return {
                "norm-pixel-pick-and-place": {
                    "pick_0": pick_0_norm,
                    "place_0": place_0_norm,
                    "pick_1": pick_1_norm,
                    "place_1": place_1_norm,
                },
            }

        # -------------------------------
        # Pick & Fling
        # -------------------------------
        elif skill_type == "pick_and_fling":
            clicks = click_points_pick_and_fling("Pick & Fling", rgb)

            pick_0, pick_1 = clicks

            def norm_xy(pt):
                x, y = pt
                return ((x / w) * 2 - 1, (y / h) * 2 - 1)

            pick_0_norm = norm_xy(pick_0)
            pick_1_norm = norm_xy(pick_1)

            return {
                "norm-pixel-pick-and-fling": {
                    "pick_0": pick_0_norm,
                    "pick_1": pick_1_norm,
                }
            }

        else:
            raise ValueError(f"Unknown skill type: {skill_type}")
