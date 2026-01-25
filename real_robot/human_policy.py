import time
import numpy as np
from real_robot.utils.human_utils \
    import click_points_pick_and_place, click_points_pick_and_fling
from real_robot.utils.save_utils import save_colour, save_mask

from agent_arena import Agent

class HumanPolicy(Agent):
    """Interactive human-in-the-loop policy for giving pick/place actions."""

    def __init__(self, config):
        super().__init__(config)
        self.measure_time = config.get('measure_time', False)

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        for arena_id in arena_ids:
            self.internal_states[arena_id]['inference_time'] = []

    def single_act(self, info, update=False):
        """Get an action dict from user clicks."""
        # --- Start Timer ---
        if self.measure_time:
            start_time = time.time()
            arena_id = info['arena_id']

        while True:
            cmd = input("\nSkill [1=pick-fling, 2=pick-place, 3=no-operation, q=quit]: ").strip().lower()
            if cmd in ("q", "quit"):
                return None
            elif cmd == "1":
                prim_type = "pick_and_fling"
                break
            elif cmd == "2":
                prim_type = "pick_and_place"
                break
            elif cmd == '3':
                prim_type = "no_operation"
            else:
                print("Invalid command. Please enter 1, 2, 3 or q.")

        # Unpack scene info
        rgb, depth = info['observation']["rgb"], info['observation']["depth"]
        mask = info['observation']["mask"]
        workspace_mask_0, workspace_mask_1 = info['observation']["robot0_mask"], info['observation']["robot1_mask"]
        
        if self.config.debug:
            save_colour(rgb, 'policy_input_rgb', './tmp')
            save_mask(mask, 'policy_input_mask', './tmp')
            save_mask(workspace_mask_0, 'policy_input_workspace_mask_0', './tmp')
            save_mask(workspace_mask_1, 'policy_input_workspace_mask_1', './tmp')
        
        display_rgb = self.apply_workspace_masks(rgb, workspace_mask_0, workspace_mask_1)

            
        h, w = rgb.shape[:2]

        # -------------------------------
        # Pick & Place
        # -------------------------------
        if prim_type == "pick_and_place":
            clicks = click_points_pick_and_place("Pick & Place", display_rgb, mask)
            #clicks = click_points_pick_and_place("Pick & Place", display_rgb)

            pick_0, place_0, pick_1, place_1 = clicks

            # Normalize pixel coordinates to [-1, 1]
            def norm_xy(pt):
                x, y = pt
                return ((x / w) * 2 - 1, (y / h) * 2 - 1)

            pick_0_norm = norm_xy(pick_0)
            place_0_norm = norm_xy(place_0)
            pick_1_norm = norm_xy(pick_1)
            place_1_norm = norm_xy(place_1)

            action = {
                "norm-pixel-pick-and-place": \
                    np.concatenate([pick_0_norm, pick_1_norm, place_0_norm, place_1_norm])
                
            }

        # -------------------------------
        # Pick & Fling
        # -------------------------------
        elif prim_type == "pick_and_fling":
            clicks = click_points_pick_and_fling("Pick & Fling", display_rgb, mask)

            pick_0, pick_1 = clicks

            def norm_xy(pt):
                x, y = pt
                return ((x / w) * 2 - 1, (y / h) * 2 - 1)

            pick_0_norm = norm_xy(pick_0)
            pick_1_norm = norm_xy(pick_1)

            action = {
                "norm-pixel-pick-and-fling": \
                    np.concatenate([pick_0_norm, pick_1_norm])
            }

        elif prim_type == 'no_operation':
            action = {"no-operation": np.zeros(8)}

        else:
            raise ValueError(f"Unknown skill type: {prim_type}")
        

        # --- End Timer & Store Duration ---
        if self.measure_time:
            self.internal_states[arena_id]['inference_time'].append(time.time() - start_time)
        
        return action

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
