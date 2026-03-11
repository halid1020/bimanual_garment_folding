import base64
import io
import os
import re
from PIL import Image
from openai import OpenAI
import base64
import cv2


def load_image_as_base64(image_array):


    _, buffer = cv2.imencode(".png", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buffer).decode("utf-8")

    data_url = f"data:image/png;base64,{b64}"
    return {"url": data_url}


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
        self.model_id = config.model_id if config.model_id else "gpt-4.1-mini"
        self.base_url = config.base_url if config.base_url else "https://api.openai.com/v1/"

        print("NOW ONLINE CLASSIFIER", self.base_url)
        api_key = os.environ[config.api_key] if config.api_key else None # if config.api_key else None

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

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

        self.how_hints_text = (
            "GOALS:\n"
            "- If the clothing is not fully flat, and there are wrinkles: Flattening\n"
            f"- If the clothing is fully flat or in the folding process {'(Check the demo images)' if self.config.use_goal_demo_steps else ''}: Folding\n"
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

    def _build_message_content(self, current_rgb, demo_images, history_images, human_demo_steps, history_labels=None):
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
        if self.config.use_how_hints:
            instruction += f"\n{self.how_hints_text}"

        content.append({"type": "text", "text": instruction})

        # Reference demo
        if self.config.use_goal_demo_steps and demo_images:
            content.append({"type": "text", "text": "\nREFERENCE DEMO SEQUENCE (Success path):"})
            for img in demo_images:
                content.append({
                    "type": "image_url",
                    "image_url": load_image_as_base64(img)
                })

        # History
        if self.config.use_trajectory_history and history_images:
            content.append({"type": "text", "text": "\nPAST TRAJECTORY IMAGES (Previous states):"})
            for i, img in enumerate(history_images):
                if self.config.use_vlm_history_images_labels and history_labels and i < len(history_labels):
                    content.append({"type": "text", "text": f"Phase: {history_labels[i]}"})
                content.append({
                    "type": "image_url",
                    "image_url": load_image_as_base64(img)
                })


        # Human reasoning
        if self.config.use_human_demo_steps and human_demo_steps:
            content.append({"type": "text", "text": "\nREFERENCE REASONING FROM HUMAN DEMO (Visual evidence, thought process, and conclusion from human demo):"})
            i = 1
            for img, label, reasoning in human_demo_steps:
                content.append({"type": "text", "text": f"Image {i}: {label}\nReasoning: {reasoning}"})
                content.append({
                    "type": "image_url",
                    "image_url": load_image_as_base64(img)
                })
                i += 1

        # Current
        content.append({"type": "text", "text": "\nCURRENT OBSERVATION (Classify this):"})
        content.append({
            "type": "image_url",
            "image_url": load_image_as_base64(current_rgb)
        })

        # Output formatting
        if self.config.use_vlm_reasoning:
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

    def predict_phase(self, current_rgb, history_images=None, human_demo_steps=None, demo_images=None, history_labels=None):
        history_images = history_images or []
        demo_images = demo_images or []
        human_demo_steps = human_demo_steps or []
        history_labels = history_labels or []

        messages = [
            {
                "role": "user",
                "content": self._build_message_content(
                    current_rgb, demo_images, history_images, human_demo_steps, history_labels
                )
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
            # max_output_tokens=200 if self.config.get("use_reasoning") else 20,
        )

        # print("RAW MODEL OUTPUT:", response)

        # output_text = response.output_text.strip()
        output_text = response.choices[0].message.content.strip()
        return self._parse_output(output_text)

    # -----------------------------
    # Output parsing
    # -----------------------------

    def _parse_output(self, output_text):
        text = output_text.lower()

        if self.config.use_vlm_reasoning:
            match = re.search(r"phase:\s*(flattening|folding)", text)
            if match:
                return match.group(1), output_text

            # Fallbacks
            if "folding" in text and "flattening" not in text:
                return "folding", output_text
            return "flattening", output_text

        else:
            return ("folding" if "folding" in text else "flattening"), None
