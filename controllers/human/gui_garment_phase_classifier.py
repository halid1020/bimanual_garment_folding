import datetime
import cv2
import torch
import re
import base64
from transformers import AutoProcessor, Gemma3nForConditionalGeneration, Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoModelForImageTextToText

from PIL import ImageTk, Image
from pathlib import Path
import json

def load_image_array_to_url(image_array):

    _, buffer = cv2.imencode('.png', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"



class GarmentPhaseClassifier:
    def __init__(self, config):
        
        self.config = config

        
    
    def let_human_reason_and_decide(self, current_rgb):
        """
        Args:
            current_rgb: PIL.Image or np.array (The latest state)
            history_images: List of PIL.Images (Past states in current episode)
            demo_images: List of PIL.Images (Images from an expert demonstration)
        """
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        import threading
        


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
        

    

    # def save_image_data(
    #     self,
    #     save_dir: str,
    #     image,
    #     phase: str,
    #     reasoning_skill: str = None,
    # ) -> None:
    #     """
    #     Saves an image and a linked JSON file containing text metadata.

    #     Files produced:
    #         base_name.png
    #         base_name.json
    #     """
    #     save_dir = Path(save_dir)
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     basename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    #     # --- Save image ---
    #     image_path = save_dir / f"image_{basename}.png"
    #     Image.fromarray(image).save(image_path)

    #     # --- Save linked metadata ---
    #     metadata = {
    #         "image_file": image_path.name,
    #         "phase": phase,
    #         "reasoning": reasoning_skill,
    #     }


    #     metadata_path = save_dir / f"metadata_{basename}.json"
    #     with open(metadata_path, "w", encoding="utf-8") as f:
    #         json.dump(metadata, f, indent=2, ensure_ascii=False)


    import json
    import datetime
    from pathlib import Path
    from PIL import Image
    from typing import Optional, List, Tuple, Any

    def save_image_data(
        self,
        save_dir: str,
        image: Any,
        phase: str,
        reasoning_skill: Optional[str] = None,
        prim_type: Optional[str] = None,
        coords: Optional[List[Tuple[int, int]]] = None,
        reasoning_prim: Optional[str] = None,
    ) -> None:
        """
        Saves an image and a linked JSON file containing text metadata.

        Files produced:
            image_{timestamp}.png
            metadata_{timestamp}.json

        Args:
            save_dir: Directory to save the files.
            image: Numpy array or similar image data.
            phase: The phase of the process (e.g., 'train', 'val').
            reasoning_skill: High-level reasoning skill associated with the image.
            prim_type: Type of primitive action. Options:
                    - 'pick_and_place'
                    - 'pick_and_fling'
                    - 'no-operation'
            coords: List of coordinate tuples associated with the primitive.
                    None if prim_type is 'no-operation'.
            reasoning_prim: Reasoning specific to the primitive action.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        basename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # --- Save image ---
        image_path = save_dir / f"image_{basename}.png"
        Image.fromarray(image).save(image_path)

        # --- Save linked metadata ---
        metadata = {
            "image_file": image_path.name,
            "phase": phase,
            "reasoning": reasoning_skill,
            # New fields
            "prim_type": prim_type,
            "coords": coords,
            "reasoning_prim": reasoning_prim,
        }

        metadata_path = save_dir / f"metadata_{basename}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)