# gpt_fabric_agent.py
from typing import Dict, List, Any, Optional
import json
import time
import base64
from dotmap import DotMap

import openai
from your_package.agents.agent import Agent  # adjust import as needed


class GPTFabricAgent(Agent):
    """
    GPT-Fabric-style agent supporting multimodal (text + image) prompting and structured JSON action output.
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "gpt_fabric_agent"
        self.config = config
        self.internal_states = {}
        self.model_name = self.config.get("model_name", "gpt-4o")
        self.max_tokens = int(self.config.get("max_tokens", 512))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.openai_api_key = self.config.get("openai_api_key", None)

        if self.openai_api_key and openai:
            openai.api_key = self.openai_api_key

    # -------------------------------------------------------------------------
    # Initialization / Update
    # -------------------------------------------------------------------------
    def init(self, info_list):
        results = []
        for info in info_list:
            arena_id = info.get("arena_id", "default_arena")
            self.internal_states[arena_id] = {
                "history": [],
                "last_action": None,
                "last_step_info": None,
                "steps": 0,
                "stage": info.get("stage", "smoothing")
            }
            results.append(True)
        return results

    def update(self, info_list, actions):
        out = []
        for info, action in zip(info_list, actions):
            arena_id = info.get("arena_id", "default_arena")
            st = self.internal_states.setdefault(arena_id, {"history": [], "steps": 0})
            entry = {"info": info, "action": action, "timestamp": time.time()}
            st["history"].append(entry)
            st["last_action"] = action
            st["steps"] = st.get("steps", 0) + 1
            out.append(True)
        return out

    # -------------------------------------------------------------------------
    # Main decision method
    # -------------------------------------------------------------------------
    def single_act(self, info, update=False):
        """
        Builds multimodal prompt (text + image) and queries GPT to get the next pick-and-pull action.
        """
        rgb = info['rgb']
        depth = info['depth']
        corners, img=self.get_corners_img(rgb=rgb,depth=depth,specifier=specifier,corner_limit=corner_limit)
        center_point_pixel,preprocessed_img=self.get_center_point_bounding_box(rgb=rgb,depth=depth,need_box=need_box)

        return action

    # -------------------------------------------------------------------------
    # Prompt construction (multimodal)
    # -------------------------------------------------------------------------
    def _build_multimodal_prompt(self,
                                 encoded_image: Optional[str],
                                 corners: List[Any],
                                 center_point_pixel: List[float],
                                 curr_coverage: float,
                                 last_step_info: Optional[Dict[str, Any]]):
        """
        Construct the multimodal prompt used for GPT communication.
        Produces a list of 'content' items (text + optional image_url).
        """

        text_lines = []
        text_lines.append(f"The current cloth coverage is {curr_coverage:.3f}.")

        # corners summary
        if corners:
            corner_str = ", ".join([f"[{c[0]}, {c[1]}]" for c in corners])
            text_lines.append(f"The detected Shi-Tomasi corner points (blue dots) are: {corner_str}.")

        # center point
        text_lines.append(f"The black point is the cloth center: [{center_point_pixel[0]}, {center_point_pixel[1]}].")

        # previous step info
        if last_step_info is not None:
            last_pick = last_step_info.get("place_pixel", None)
            if last_pick:
                last_pick_oppo = [
                    center_point_pixel[0] * 2 - last_pick[0],
                    center_point_pixel[1] * 2 - last_pick[1],
                ]
                text_lines.append(f"Last pick point was {last_pick}, symmetric opposite point is {last_pick_oppo}. Avoid nearby areas.")

        # instructions
        text_lines.append("")
        text_lines.append("Please propose the next action following this schema (JSON only):")
        text_lines.append("""{
            "type": "pick_and_pull",
            "grasp": {"x": <float>, "y": <float>, "z": null, "frame": "image"},
            "target": {"x": <float>, "y": <float>, "z": null, "frame": "image"},
            "parameters": {"pull_strength": <float>},
            "explanation": "<short reason>"
        }""")
        text_lines.append("Keep the explanation under 30 words and output only JSON.")

        # assemble content
        content = [{"type": "text", "text": "\n".join(text_lines)}]
        if encoded_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": "high"
                }
            })
        return content

    # -------------------------------------------------------------------------
    # Model call (chat multimodal)
    # -------------------------------------------------------------------------
    def _call_model_multimodal(self, messages: List[Dict[str, Any]]) -> str:
        """
        Send multimodal messages (text + image) to GPT model and return response text.
        """
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------
    def _parse_model_response(self, text: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Robustly parse model output JSON. Fall back to heuristic defaults if parsing fails.
        """
        txt = text.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    txt = p
                    break
        try:
            parsed = json.loads(txt)
            if "type" not in parsed:
                parsed["type"] = "pick_and_pull"
            return parsed
        except Exception:
            try:
                start, end = txt.find("{"), txt.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(txt[start:end+1])
            except Exception:
                pass

        # fallback
        bbox = info.get("cloth_bbox", [0, 0, 0, 0])
        return {
            "type": "pick_and_pull",
            "grasp": {"x": bbox[0], "y": bbox[1], "z": None, "frame": "image"},
            "target": {"x": bbox[2], "y": bbox[3], "z": None, "frame": "image"},
            "parameters": {"pull_strength": 0.5},
            "explanation": "fallback action due to parse error"
        }
