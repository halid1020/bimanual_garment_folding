import torch
import re
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

class GarmentPhaseClassifier:
    def __init__(self, config):
        self.device = config.get('device', "cuda:0")
        self.model_id = config.get('model_id', "google/gemma-3-4b-it")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, device_map=self.device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.config = config

        self.definitions_text = (
            "CONTEXT:\n"
            "- Action Space: Bimanual robot arms (pick-and-place) within the observation space.\n" 
            "- Observation Space: Top-down RGB camera.\n"
        )

        self.goal_hints_text = (
            "GOALS:\n"
            "- Flattening: Success when the garment is fully spread with no wrinkles.\n" 
            "- Folding: Success when the garment is folded into the demonstrated configuration from a flattened state.\n" 
        )

    def _build_prompt_content(self, history_images, demo_images):
        """
        Dynamically builds a multimodal message content list.
        Order: Demo (Reference) -> History (Context) -> Current (Target)
        """
        content = []
        
        # 1. Base Instruction
        instruction = "You are controlling a robot to do garment folding. Classify the current phase based on the images provided."

        if self.config.use_definitions:
            instruction += f"\n\n{self.definitions_text}"
        if self.config.use_goal_hints:
            instruction += f"\n{self.goal_hints_text}"

        content.append({"type": "text", "text": instruction})

        # 2. Add Reference Demo Images (TODO: Fixed)
        if self.config.use_demo and demo_images:
            content.append({"type": "text", "text": "\nREFERENCE DEMO SEQUENCE (Success path):"})
            for _ in demo_images:
                content.append({"type": "image"})

        # 3. Add Trajectory History Images (TODO: Fixed)
        if self.config.use_history and history_images:
            content.append({"type": "text", "text": "\nPAST TRAJECTORY IMAGES (Previous states):"})
            for _ in history_images:
                content.append({"type": "image"})

        # 4. Add Current Observation
        content.append({"type": "text", "text": "\nCURRENT OBSERVATION (Classify this):"})
        content.append({"type": "image"})

        # 5. Output Formatting
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

    @torch.no_grad()
    def predict_phase(self, current_rgb, history_images=None, demo_images=None):
        """
        Args:
            current_rgb: PIL.Image or np.array (The latest state)
            history_images: List of PIL.Images (Past states in current episode)
            demo_images: List of PIL.Images (Images from an expert demonstration)
        """
        history_images = history_images or []
        demo_images = demo_images or []

        # 1. Assemble the list of all images in the order they appear in the prompt
        all_images = []
        if self.config.use_demo:
            all_images.extend(demo_images)
        if self.config.use_history:
            all_images.extend(history_images)
        all_images.append(current_rgb)

        # 2. Build the prompt structure
        messages = [{
            "role": "user", 
            "content": self._build_prompt_content(history_images, demo_images)
        }]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # 3. Process inputs (Processor handles the list of images)
        inputs = self.processor(
            text=prompt,
            images=all_images,
            return_tensors="pt"
        ).to(self.device)

        # 4. Generate
        max_tokens = 150 if self.config.use_reasoning else 10
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        # 5. Decode
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        full_output = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

        return self._parse_output(full_output)

    def _parse_output(self, output_text):
        output_lower = output_text.lower()
        if self.config.use_reasoning:
            match = re.search(r"phase:\s*(flattening|folding)", output_lower)
            if match:
                return match.group(1), output_text
            
            # Simple keyword fallbacks
            if "folding" in output_lower and "flattening" not in output_lower:
                return "folding", output_text
            return "flattening", output_text
        else:
            return ("folding" if "folding" in output_lower else "flattening"), None