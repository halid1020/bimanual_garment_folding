import os
import json
from datetime import datetime

import numpy as np
import cv2

from agent_arena import Agent

from .pick_and_fling.pixel_human import PixelHumanFling
from .human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
from .no_operation import NoOperation

from .utils import draw_text_top_right, apply_workspace_shade


def draw_reasoning_text(img, text, max_lines=4):
    """
    Draw human reasoning at the bottom-left of the image.
    """
    if not text:
        return img

    lines = text.split(". ")
    lines = lines[:max_lines]

    y = img.shape[0] - 20
    for line in reversed(lines):
        cv2.putText(
            img,
            line.strip(),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        y -= 18

    return img


import numpy as np



import numpy as np

def action_to_pixel_coords(
    action_1d,
    image_size=(128, 128),
    dtype=np.float32,
    dtype_output=np.int32,
):
    """
    Convert normalized action coordinates to OpenCV pixel coordinates.

    Normalized space (as per diagram):
      - norm_x in [-1, 1] : top (-1) → bottom (+1)
      - norm_y in [-1, 1] : left (-1) → right (+1)

    OpenCV space:
      - (0,0) top-left
      - x right, y down
    """

    action = np.asarray(action_1d, dtype=dtype)

    if action.ndim != 1 or action.size % 2 != 0:
        raise ValueError(
            f"Expected 1D array with even length, got shape {action.shape}"
        )

    H, W = image_size

    # (N, 2): [norm_x, norm_y]
    pts = action.reshape(-1, 2)

    # norm_y → horizontal (x)
    x_px = (pts[:, 1] + 1.0) * 0.5 * W

    # norm_x → vertical (y)
    y_px = (pts[:, 0] + 1.0) * 0.5 * H

    return np.stack([x_px, y_px], axis=1).astype(dtype_output)




class HumanMultiPrimitiveReasoning(Agent):
    """
    Human-in-the-loop multi-primitive controller with reasoning capture.
    """

    def __init__(self, config):
        super().__init__(config)

        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place",
            "no-operation"
        ]

        self.primitive_instances = [
            PixelHumanFling(config),
            HumanDualPickersPickAndPlace(config),
            NoOperation(config)
        ]

        self.human_cfg = config.get("human_reasoning", {})
        self.enabled = self.human_cfg.get("enabled", True)

        self._episode_counter = 0
        self._step_counter = 0



        self.save_dir = self.human_cfg.get("save_dir", "logs/human_reasoning")
        os.makedirs(self.save_dir, exist_ok=True)

        self.last_primitive = None

        print(
            "[human-multi-primitive-reasoning] "
            f"Human reasoning enabled = {self.enabled}"
        )

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        self.last_primitive = None
        self._episode_counter += 1
        self._step_counter = 0

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def act(self, info_list, update=False):
        actions = []
        for info in info_list:
            actions.append(self.single_act(info))
        return actions

    def single_act(self, state, update=False):
        """
        Show image + goals, let human choose primitive, capture reasoning,
        save everything, then delegate.
        """

        # -----------------------------
        # Image preparation
        # -----------------------------
        rgb = state["observation"]["rgb"]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512))
        H, W = rgb.shape[:2]

        if "robot0_mask" in state["observation"]:
            mask0 = state["observation"]["robot0_mask"].astype(bool)
            if mask0.shape[:2] != (H, W):
                mask0 = cv2.resize(
                    mask0.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            rgb = apply_workspace_shade(rgb, mask0, color=(255, 0, 0), alpha=0.2)

        if "robot1_mask" in state["observation"]:
            mask1 = state["observation"]["robot1_mask"].astype(bool)
            if mask1.shape[:2] != (H, W):
                mask1 = cv2.resize(
                    mask1.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            rgb = apply_workspace_shade(rgb, mask1, color=(0, 0, 255), alpha=0.2)

        if "evaluation" in state and state["evaluation"]:
            success = state.get("success", False)
            text_lines = [
                (f"Success: {success}", (0, 255, 0) if success else (0, 0, 255)),
                (f"IoU(flat): {state['evaluation'].get('max_IoU_to_flattened', 0):.3f}",
                 (255, 255, 255))
            ]
            if "max_IoU" in state["evaluation"]:
                text_lines.append(
                    (f"IoU(fold): {state['evaluation']['max_IoU']:.3f}",
                     (255, 255, 255))
                )
            draw_text_top_right(rgb, text_lines)

        img = rgb.copy()

        # -----------------------------
        # Goal image grid
        # -----------------------------
        goal_rgb = None
        if "goals" in state:
            rgbs = []
            for goal in state["goals"][:4]:
                g = goal["observation"]["rgb"]
                g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
                g = cv2.resize(g, (256, 256))
                rgbs.append(g)

            while len(rgbs) < 4:
                rgbs.append(np.zeros((256, 256, 3), dtype=np.uint8))

            top = np.concatenate([rgbs[0], rgbs[1]], axis=1)
            bottom = np.concatenate([rgbs[2], rgbs[3]], axis=1)
            goal_rgb = np.concatenate([top, bottom], axis=0)

            img = np.concatenate([img, goal_rgb], axis=1)
            cv2.line(img, (512, 0), (512, img.shape[0]), (255, 255, 255), 2)

        # -----------------------------
        # Save preview image
        # -----------------------------
        preview_path = "tmp/human_rgb.png"
        cv2.imwrite(preview_path, img)
        print(f"[human] Preview saved to {preview_path}")

        # -----------------------------
        # Primitive selection
        # -----------------------------
        while True:
            print("\nChoose a primitive:")
            for i, p in enumerate(self.primitive_names):
                print(f"{i + 1}. {p}")
            try:
                choice = int(input("> ")) - 1
                if 0 <= choice < len(self.primitive_names):
                    chosen_primitive = self.primitive_names[choice]
                    primitive = self.primitive_instances[choice]
                    break
            except ValueError:
                pass
            print("Invalid choice.")

        # -----------------------------
        # Human reasoning
        # -----------------------------
        reasoning = ""
        if self.enabled:
            print(
                "\nExplain your reasoning (1–3 sentences).\n"
                "Focus on cloth state, risk, and expected effect."
            )
            while not reasoning:
                reasoning = input("> ").strip()

        # Draw reasoning on image
        img = draw_reasoning_text(img, reasoning)

        episode_id = self._episode_counter
        step_id = self._step_counter
        self._step_counter += 1

        

        # -----------------------------
        # Delegate
        # -----------------------------
        action = primitive.single_act(state)
        self.last_primitive = chosen_primitive

        print("\n" + "=" * 50)
        print(state["observation"]["rgb"].shape)

        # scaling = state["observation"]["rgb"].shape[1]


        print("Action (normalized):", action)
        # print(state["observation"].keys())
        # print(state["observation"]["picker_norm_pixel_pos"].keys())
        # print(state["observation"])
        # print(state.keys())

        new_pts = action_to_pixel_coords(
            action,
            image_size=(128, 128),
            dtype=np.float32,
            dtype_output=np.int32
        )

        print(f"[human-multi-primitive-reasoning] Chosen primitive: {chosen_primitive} -> Action: {new_pts}")

        # if new_pts.shape[0] == 8:
        # testing_image = state["observation"]["rgb"].copy()
        pick_a = None
        pick_b = None
        place_a = None
        place_b = None
        fling = None

        if len(new_pts) == 4:
            pick_a = new_pts[0]
            pick_b = new_pts[1]
            place_a = new_pts[2]
            place_b = new_pts[3]
        elif len(new_pts) == 2:
            pick_a, fling = new_pts[0], new_pts[1]


        


        # -----------------------------
        # Save decision
        # -----------------------------
        if self.enabled:
            self._save_decision(
                episode_id=episode_id,
                step_id=step_id,
                state=state,
                img=img,
                raw_rgb=state["observation"]["rgb"],
                goal_rgb=goal_rgb,
                chosen_primitive=chosen_primitive,
                reasoning=reasoning,
                pick_a=pick_a,
                pick_b=pick_b,
                place_a=place_a,
                place_b=place_b,
                fling=fling
            )
        
        
        return {chosen_primitive: action}

    def _save_decision(self, episode_id, step_id, state, img, raw_rgb, goal_rgb,
                       chosen_primitive, reasoning, pick_a, pick_b, place_a, place_b, fling):
        """
        Save images + JSON record atomically.
        """
        # episode_id = state.get("episode_id", 0)
        # step_id = state.get("step_id", 0)
        # episode_id = episode_id
        # step_id = step_id


        ep_dir = os.path.join(self.save_dir, f"episode_{episode_id}")
        step_dir = os.path.join(ep_dir, f"step_{step_id}")
        os.makedirs(step_dir, exist_ok=True)

        cv2.imwrite(os.path.join(step_dir, "rendered.png"), img)
        cv2.imwrite(os.path.join(step_dir, "raw_rgb.png"), raw_rgb)

        if goal_rgb is not None:
            cv2.imwrite(os.path.join(step_dir, "goal_rgb.png"), goal_rgb)


        if pick_a is not None and pick_b is not None and place_a is not None and place_b is not None:
            
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "episode_id": episode_id,
                "step_id": step_id,
                "chosen_primitive": chosen_primitive,
                "reasoning": reasoning,
                "pick_a": pick_a.tolist(),
                "pick_b": pick_b.tolist(),
                "place_a": place_a.tolist(),
                "place_b": place_b.tolist(),
                "success": state.get("success"),
                "evaluation": state.get("evaluation", {})
            }
        elif pick_a is not None and fling is not None:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "episode_id": episode_id,
                "step_id": step_id,
                "chosen_primitive": chosen_primitive,
                "reasoning": reasoning,
                "pick": pick_a.tolist(),
                "fling": fling.tolist(),
                "success": state.get("success"),
                "evaluation": state.get("evaluation", {})
            }
        else:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "episode_id": episode_id,
                "step_id": step_id,
                "chosen_primitive": chosen_primitive,
                "reasoning": reasoning,
                "success": state.get("success"),
                "evaluation": state.get("evaluation", {})
            }

        with open(os.path.join(step_dir, "decision.json"), "w") as f:
            json.dump(record, f, indent=2)

    def terminate(self):
        return {
            arena_id: (self.last_primitive == "no-operation")
            for arena_id in self.internal_states.keys()
        }
