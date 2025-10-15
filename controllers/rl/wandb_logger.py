import os
import numpy as np
from agent_arena.utilities.logger.logger_interface import Logger

class WandbLogger(Logger):
    def __init__(self, project="garment-folding", name=None, config=None):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config or {})
        except Exception:
            self.wandb = None
            self.run = None
        self.log_dir = None

    def set_log_dir(self, log_dir) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, metrics, step=None, fps=10) -> None:
        """
        Logs scalars, images, and videos (supports numpy arrays or file paths).
        - Scalars: logged directly.
        - File paths: logged as wandb.Image or wandb.Video depending on file extension.
        - NumPy arrays: automatically converted to wandb.Video.
        """
        if self.wandb is None or self.run is None:
            return
        
        processed_metrics = {}
        for key, value in metrics.items():
            # 1. Handle numpy arrays (e.g., video frames)
            if isinstance(value, np.ndarray):
                # Expected shape: (T, H, W, C)
                if value.ndim == 4 and value.dtype in [np.uint8, np.float32, np.float64]:
                    # Ensure dtype uint8 in [0,255]
                    if value.dtype != np.uint8:
                        value = np.clip(value * 255, 0, 255).astype(np.uint8)
                    processed_metrics[key] = self.wandb.Video(value, fps=fps, format="mp4")
                else:
                    processed_metrics[key] = value  # fallback, maybe scalar array
                
            # 2. Handle file paths (e.g., image or video files)
            elif isinstance(value, str) and os.path.exists(value):
                ext = os.path.splitext(value)[-1].lower()
                if ext in [".mp4", ".avi", ".mov"]:
                    processed_metrics[key] = self.wandb.Video(value, fps=fps, format="mp4")
                elif ext in [".gif", ".png", ".jpg", ".jpeg"]:
                    processed_metrics[key] = self.wandb.Image(value)
                else:
                    processed_metrics[key] = value  # unsupported file type
            
            # 3. Scalars or other numeric values
            else:
                processed_metrics[key] = value

        self.wandb.log(processed_metrics, step=step)

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()