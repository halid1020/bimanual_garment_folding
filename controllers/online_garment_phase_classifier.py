import base64
import io
import re
from PIL import Image
from openai import OpenAI


class OnlineGarmentPhaseClassifier:
    def __init__(self, config):
        """
        config expects:
          - model_id (e.g. "gpt-4.1-mini", "gpt-4.1")
          - api_key (optional if env var is set)
          - use_definitions
          - use_goal_hints
          - use_demo
          - use_history
          - use_reasoning
        """
        self.config = config
        self.model_id = config.get("model_id", "gpt-4.1-mini")
        self.base_url = config.get("base_url", "https://api.openai.com/v1/")

        self.client = OpenAI(api_key=config.get("api_key"))

        self.definitions_text = (
            "CONTEXT:\n"
            "- Action Space: Bimanual robot arms (pick-and-place).\n"
            "- Observation Space: Top-down RGB camera.\n"
        )

        self.goal_hints_text = (
            "GOALS:\n"
            "- Flattening: garment fully spread, no wrinkles.\n"
            "- Folding: garment folded into demonstrated configuration.\n"
        )

    # -----------------------------
    # Utilities
    # -----------------------------

    def _image_to_base64(self, image):
        """Convert PIL or numpy image to base64 PNG"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # -----------------------------
    # Prompt construction
    # -----------------------------

    def _build_message_content(self, current_rgb, history_images, demo_images):
        """
        EXACT port of HF prompt logic.
        No text rewritten.
        """
        content = []

        instruction = "You are controlling a robot to do garment folding. Classify the current phase based on the images provided."

        if self.config.use_definitions:
            instruction += f"\n\n{self.definitions_text}"
        if self.config.use_goal_hints:
            instruction += f"\n{self.goal_hints_text}"

        content.append({"type": "text", "text": instruction})

        # Reference demo
        if self.config.use_demo and demo_images:
            content.append({"type": "text", "text": "\nREFERENCE DEMO SEQUENCE (Success path):"})
            for img in demo_images:
                content.append({
                    "type": "input_image",
                    "image_base64": self._image_to_base64(img)
                })

        # History
        if self.config.use_history and history_images:
            content.append({"type": "text", "text": "\nPAST TRAJECTORY IMAGES (Previous states):"})
            for img in history_images:
                content.append({
                    "type": "input_image",
                    "image_base64": self._image_to_base64(img)
                })

        # Current
        content.append({"type": "text", "text": "\nCURRENT OBSERVATION (Classify this):"})
        content.append({
            "type": "input_image",
            "image_base64": self._image_to_base64(current_rgb)
        })

        # Output formatting
        if self.config.use_reasoning:
            format_instruction = (
                "\n\nFirst, explain your reasoning based on the visual evidence, history, and demo reference."
                "\nThen, conclude with exactly: 'Phase: <phase>'."
                "\nAllowed phases: flattening, folding."
            )
        else:
            format_instruction = "\n\nAnswer with exactly one word: 'flattening' or 'folding'."

        content.append({"type": "text", "text": format_instruction})

        return content


    # -----------------------------
    # Inference
    # -----------------------------

    def predict_phase(self, current_rgb, history_images=None, demo_images=None):
        history_images = history_images or []
        demo_images = demo_images or []

        

        messages = [
            {
                "role": "user",
                "content": self._build_message_content(
                    current_rgb, history_images, demo_images
                )
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
            # max_output_tokens=200 if self.config.get("use_reasoning") else 20,
        )

        output_text = response.output_text.strip()
        return self._parse_output(output_text)

    # -----------------------------
    # Output parsing
    # -----------------------------

    def _parse_output(self, output_text):
        text = output_text.lower()

        if self.config.get("use_reasoning"):
            match = re.search(r"phase:\s*(flattening|folding)", text)
            if match:
                return match.group(1), output_text

            # Fallbacks
            if "folding" in text and "flattening" not in text:
                return "folding", output_text
            return "flattening", output_text

        else:
            return ("folding" if "folding" in text else "flattening"), None
