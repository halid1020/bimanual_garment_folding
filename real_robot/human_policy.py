# policies/human_policy.py
import numpy as np
from human_utils import click_points_pick_and_place, click_points_pick_and_fling
from save_utils import save_colour, save_mask

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
        
        print('rgb shape', rgb.shape)
        save_colour(rgb, 'policy_input_rgb', './tmp')
        print('mask shape', mask.shape)
        save_mask(mask, 'policy_input_mask', './tmp')
        save_mask(workspace_mask_0, 'policy_input_workspace_mask_0', './tmp')
        save_mask(workspace_mask_1, 'policy_input_workspace_mask_1', './tmp')
        display_rgb = self.apply_workspace_masks(rgb, workspace_mask_0, workspace_mask_1)

            
        h, w = rgb.shape[:2]

        # -------------------------------
        # Pick & Place
        # -------------------------------
        if skill_type == "pick_and_place":
            clicks = click_points_pick_and_place("Pick & Place", display_rgb, mask)

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
                "norm-pixel-pick-and-place": \
                    np.concatenate([pick_0_norm, pick_1_norm, place_0_norm, place_1_norm])
                
            }

        # -------------------------------
        # Pick & Fling
        # -------------------------------
        elif skill_type == "pick_and_fling":
            clicks = click_points_pick_and_fling("Pick & Fling", display_rgb, mask)

            pick_0, pick_1 = clicks

            def norm_xy(pt):
                x, y = pt
                return ((x / w) * 2 - 1, (y / h) * 2 - 1)

            pick_0_norm = norm_xy(pick_0)
            pick_1_norm = norm_xy(pick_1)

            return {
                "norm-pixel-pick-and-fling": \
                    np.concatenate([pick_0_norm, pick_1_norm])
            }

        else:
            raise ValueError(f"Unknown skill type: {skill_type}")

    def apply_workspace_masks(self, rgb, robot_0_mask, robot_1_mask):
        # Ensure masks are boolean and 2D
        robot_0_mask = robot_0_mask.astype(bool)
        robot_1_mask = robot_1_mask.astype(bool)
        
        # Remove channel dimension if any (H, W, 1) → (H, W)
        if robot_0_mask.ndim == 3:
            robot_0_mask = robot_0_mask[:, :, 0]
        if robot_1_mask.ndim == 3:
            robot_1_mask = robot_1_mask[:, :, 0]

        combined_mask = robot_0_mask | robot_1_mask

        rgb_f = rgb.astype(np.float32) / 255.0

        robot_0_tint = np.array([0.2, 0.5, 1.0])
        robot_1_tint = np.array([1.0, 0.4, 0.4])
        gray_tint  = np.array([0.5, 0.5, 0.5])

        blend_outside = 0.3
        blend_robot_0 = 0.7
        blend_robot_1 = 0.7

        shaded_rgb = rgb_f.copy()

        # --- FIXED LINE ---
        shaded_rgb[~combined_mask] = (
            rgb_f[~combined_mask] * blend_outside + gray_tint * (1 - blend_outside)
        )

        robot_0_only = robot_0_mask & ~robot_1_mask
        shaded_rgb[robot_0_only] = (
            rgb_f[robot_0_only] * blend_robot_0 + robot_0_tint * (1 - blend_robot_0)
        )

        robot_1_only = robot_1_mask & ~robot_0_mask
        shaded_rgb[robot_1_only] = (
            rgb_f[robot_1_only] * blend_robot_1 + robot_1_tint * (1 - blend_robot_1)
        )

        overlap_mask = robot_0_mask & robot_1_mask
        if np.any(overlap_mask):
            purple_tint = np.array([0.7, 0.4, 0.9])
            blend_overlap = 0.4
            shaded_rgb[overlap_mask] = (
                rgb_f[overlap_mask] * blend_overlap + purple_tint * (1 - blend_overlap)
            )

        shaded_rgb = (np.clip(shaded_rgb, 0, 1) * 255).astype(np.uint8)
        return shaded_rgb
