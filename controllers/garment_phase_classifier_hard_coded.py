import torch
import re
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration

class GarmentPhaseClassifier:
    def __init__(self, config):
        self.device = config.get('device', "cpu")
        self.model_id = config.get('model_id', "Qwen/Qwen3-VL-8B-Instruct")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
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

        # print(f"########################\n\n [GarmentPhaseClassifier] Number of images in prompt: {len(all_images)} \n\n {inputs} \n\n########################")
        # inputs.pop("image_grid_thw", None)
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
    
    def let_human_reason_and_decide(self, current_rgb, history_images=None, demo_images=None):
        """
        Args:
            current_rgb: PIL.Image or np.array (The latest state)
            history_images: List of PIL.Images (Past states in current episode)
            demo_images: List of PIL.Images (Images from an expert demonstration)
        """
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        import threading
        
        history_images = history_images or []
        demo_images = demo_images or []

        # Create result container
        result = {"action": None, "reasoning": None}
        
        def create_ui():
            root = tk.Tk()
            root.title("Human Decision Interface - Garment Manipulation")
            root.geometry("800x700")
            
            # Main frame
            main_frame = ttk.Frame(root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Display current image
            ttk.Label(main_frame, text="Current Garment State:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
            
            # Convert PIL image to PhotoImage for Tkinter
            from PIL import ImageTk, Image
            # img_display = current_rgb.copy() if hasattr(current_rgb, 'copy') else current_rgb
            img_display = Image.fromarray(current_rgb) if not isinstance(current_rgb, Image.Image) else current_rgb
            img_display = img_display.resize((400, 300), Image.Resampling.LANCZOS)
            # img_display = img_display.resize((400, 300))
            photo = ImageTk.PhotoImage(img_display)
            
            img_label = ttk.Label(main_frame, image=photo)
            img_label.image = photo  # Keep reference to prevent garbage collection
            img_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
            
            # Action selection
            ttk.Label(main_frame, text="Choose Action:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
            
            action_var = tk.StringVar(value="flattening")
            action_frame = ttk.Frame(main_frame)
            action_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
            
            ttk.Radiobutton(action_frame, text="Flattening", variable=action_var, value="flattening").pack(side=tk.LEFT, padx=(0, 20))
            ttk.Radiobutton(action_frame, text="Folding", variable=action_var, value="folding").pack(side=tk.LEFT)
            
            # Reasoning input
            ttk.Label(main_frame, text="Reasoning/Explanation:", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
            
            reasoning_text = scrolledtext.ScrolledText(main_frame, height=8, width=70, wrap=tk.WORD)
            reasoning_text.grid(row=5, column=0, columnspan=2, pady=(0, 20))
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=6, column=0, columnspan=2, pady=(0, 10))
            
            def submit_decision():
                result["action"] = action_var.get()
                result["reasoning"] = reasoning_text.get("1.0", tk.END).strip()
                root.quit()  # Close the UI
                root.destroy()
            
            def cancel_decision():
                result["action"] = None
                result["reasoning"] = None
                root.quit()
                root.destroy()
            
            ttk.Button(button_frame, text="Submit Decision", command=submit_decision).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Cancel", command=cancel_decision).pack(side=tk.LEFT)
            
            # Center the window
            root.update_idletasks()
            x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
            y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
            root.geometry(f"+{x}+{y}")
            
            root.mainloop()
        
        # Run UI in separate thread to avoid blocking
        ui_thread = threading.Thread(target=create_ui)
        ui_thread.daemon = True
        ui_thread.start()
        ui_thread.join()  # Wait for user input
        
        # Return the human decision in the same format as AI output
        # {"action": None, "reasoning": None}
        return result["action"], result["reasoning"]

# def _parse_output(self, human_result):
#     """Parse human result to match AI output format"""
#     if human_result["action"] is None:
#         return {"action": None, "reasoning": None}
    
#     return {
#         "action": human_result["action"],
#         "reasoning": human_result["reasoning"]
#     }


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