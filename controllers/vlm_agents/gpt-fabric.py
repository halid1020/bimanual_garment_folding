# gpt_fabric_agent.py
from typing import Dict, List, Any, Optional
import json
import time
from dotmap import DotMap

# Import your Agent base (adapt import path)
from your_package.agents.agent import Agent  # <- adjust path to where Agent class lives

# Optionally import an LLM client. This implementation uses openai as an example.
# If you use a different LLM wrapper in your codebase, replace call_model accordingly.
import openai


class GPTFabricAgent(Agent):
    """
    Concrete Agent implementing the GPT-Fabric style action generation.

    Expected info dicts (example keys, adapt to your env):
      - 'rgb_image_path' or 'rgb_image' (optional)
      - 'depth_image_path' or 'depth_image' (optional)
      - 'keypoints': list of named keypoints {name: str, x: float, y: float, z?: float}
      - 'cloth_bbox': [x_min, y_min, x_max, y_max] (image coords)
      - 'stage' : optional string describing stage ("smoothing" or "folding")
      - 'meta' : other metadata (step count, trial id, etc)
    """

    def __init__(self, config: DotMap):
        super().__init__(config)
        self.name = "gpt_fabric_agent"
        self.config = config or DotMap()
        self.internal_states = {}  # per-arena internal state
        # model configuration (defaults)
        self.model_name = self.config.get("model_name", "gpt-4o")  # recommended by repo readme
        self.max_tokens = int(self.config.get("max_tokens", 256))
        self.temperature = float(self.config.get("temperature", 0.0))
        # optional API key environment handling - user may prefer to manage externally
        self.openai_api_key = self.config.get("openai_api_key", None)
        if self.openai_api_key and openai:
            openai.api_key = self.openai_api_key

    # --- lifecycle methods -------------------------------------------------
    def init(self, info_list: List[InformationType]) -> List[bool]:
        """
        Initialize internal states for each incoming arena info.
        """
        results = []
        for info in info_list:
            arena_id = info.get("arena_id", "default_arena")
            self.internal_states[arena_id] = {
                "history": [],
                "last_action": None,
                "steps": 0,
                "stage": info.get("stage", "smoothing")
            }
            results.append(True)
        return results

    def update(self, info_list: List[InformationType], actions: List[ActionType]) -> List[bool]:
        """
        Optionally incorporate environment feedback (e.g., success/failure).
        We'll store action/results in history for the arena.
        """
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

    # --- core decision: single_act ----------------------------------------
    def single_act(self, info: InformationType, update: bool = False) -> ActionType:
        """
        Convert the incoming info into an LLM prompt and call the model to get an action.
        The return ActionType is a dict describing a grasp/pull or higher-level command.
        """
        arena_id = info.get("arena_id", "default_arena")
        # ensure arena exists in internal state
        st = self.internal_states.setdefault(arena_id, {"history": [], "steps": 0})

        # Build structured prompt
        prompt = self._build_prompt(info, st)

        # call LLM (implement your own call_model if you don't want openai here)
        try:
            model_response = self.call_model(prompt)
        except Exception as e:
            # On failure, return a safe no-op or last-known action (you can customize)
            return {"type": "noop", "reason": f"model_error: {e}"}

        # parse the response into a structured action
        action = self._parse_model_response(model_response, info)

        # update internal state if requested
        if update:
            st["last_action"] = action
            st["history"].append({"info_snapshot": info, "action": action, "time": time.time()})
            st["steps"] = st.get("steps", 0) + 1

        return action

    # --- helpers -----------------------------------------------------------
    def _build_prompt(self, info: InformationType, state: Dict[str, Any]) -> str:
        """
        Create a textual prompt to send to the LLM describing the visual state.

        The repository/paper used GPT vision models and sent both visual + textual context.
        Here we produce a textual, structured prompt suitable for text-only models (or for
        text+vision models you can include an image reference or attach the bytes as your client supports).
        """

        lines = []
        lines.append("You are GPT-Fabric, an agent that chooses a single pick-and-pull action for a fabric.")
        lines.append("Respond with JSON describing the action. Always follow the schema described below.")
        lines.append("")
        lines.append("Schema (JSON):")
        lines.append("""{
            "type": "pick_and_pull",   // or "fold", "grasp_and_place", "noop"
            "grasp": {"x": <float>, "y": <float>, "z": <float_or_null>, "frame": "world|image"},
            "target": {"x": <float>, "y": <float>, "z": <float_or_null>, "frame": "world|image"},
            "parameters": {"pull_strength": <float>, "pull_direction": [dx,dy,dz], ...},
            "explanation": "<short explanation of why>"
            }""")
        lines.append("")
        # Include environment-specific observations
        lines.append("Observation:")
        # Add cloth bbox if present
        if "cloth_bbox" in info:
            cb = info["cloth_bbox"]
            lines.append(f"- cloth_bbox (image coords): {cb}")
        if "keypoints" in info:
            kp = info["keypoints"]
            lines.append(f"- keypoints (name,x,y,(z optional)):")
            for k in kp:
                # k may be dict {'name':..,'x':..,'y':..}
                if isinstance(k, dict):
                    lines.append(f"  - {k.get('name','kp')} : ({k.get('x')}, {k.get('y')}, {k.get('z','null')})")
                else:
                    lines.append(f"  - {k}")
        # include optional meta
        if "meta" in info:
            lines.append(f"- meta: {info['meta']}")
        lines.append("")
        # Provide stable instructions (encourage JSON only)
        lines.append("Important instructions:")
        lines.append("- Output only valid JSON with the schema above (no additional surrounding text).")
        lines.append("- Coordinates should be in the frame you specify (image coords ok if you cannot convert).")
        lines.append("- Keep explanation < 30 words.")
        lines.append("")
        # optionally include prior actions
        if state.get("last_action"):
            lines.append("- Last action:")
            lines.append(json.dumps(state["last_action"]))
        # close
        return "\n".join(lines)

    def _call_model(self, prompt: str) -> str:
        """
        Call an LLM and return its text output. Default uses OpenAI chat completions (if available).
        You can override this method to integrate with a different client or a local LLM.

        NOTE: The official GPT-Fabric repo used GPT-vision models in their experiments and
        recommended replacing deprecated gpt-4-vision-preview with gpt-4o. See repo README.
        See: https://github.com/slurm-lab-usc/GPT-Fabric-Smoothing and paper. :contentReference[oaicite:1]{index=1}
        """
        if openai is None:
            raise RuntimeError("openai package not available. Provide a custom call_model implementation.")

        # We use ChatCompletion for compatibility: send prompt as a 'system' + 'user' pair or just user.
        messages = [
            {"role": "system", "content": "You are a robot action generator for fabric manipulation."},
            {"role": "user", "content": prompt}
        ]
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        # Get the assistant content
        text = resp["choices"][0]["message"]["content"]
        return text

    def _parse_model_response(self, text: str, info: InformationType) -> ActionType:
        """
        Parse the LLM response (prefer JSON). Provide robust fallbacks.
        """
        # Try to parse JSON directly
        text_stripped = text.strip()
        # sometimes model returns triple-backticks
        if text_stripped.startswith("```"):
            # remove code fences
            parts = text_stripped.split("```")
            # pick longest chunk that looks like JSON
            candidate = None
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    candidate = p
                    break
            if candidate:
                text_stripped = candidate

        # attempt JSON parse
        try:
            parsed = json.loads(text_stripped)
            # minimal validation
            if "type" not in parsed:
                parsed.setdefault("type", "pick_and_pull")
            return parsed
        except Exception:
            # fallback: try to find JSON substring
            try:
                start = text_stripped.find("{")
                end = text_stripped.rfind("}")
                if start != -1 and end != -1 and end > start:
                    sub = text_stripped[start:end+1]
                    parsed = json.loads(sub)
                    return parsed
            except Exception:
                pass

        # last resort: simple heuristic parse (very basic)
        return {
            "type": "pick_and_pull",
            "grasp": {"x": info.get("cloth_bbox", [0,0,0])[0], "y": info.get("cloth_bbox", [0,0,0])[1], "z": None, "frame": "image"},
            "target": {"x": info.get("cloth_bbox", [0,0,0])[2], "y": info.get("cloth_bbox", [0,0,0])[3], "z": None, "frame": "image"},
            "parameters": {"pull_strength": 0.5},
            "explanation": "fallback action due to parsing failure"
        }
