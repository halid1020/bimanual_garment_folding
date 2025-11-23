import os
import numpy as np
from agent_arena.utilities.logger.logger_interface import Logger
import wandb

class WandbLogger(Logger):
    def __init__(self, project="garment-folding", name=None, config=None, run_id=None, resume=False):
        self.project = project
        self.name = name
        self.config = config
        #self.log_dir = log_dir

        self.wandb = wandb.init(
            project=project,
            name=name,
            config=config,
            id=run_id,          # allow restoring
            resume="must" if resume else "never",
            dir='/mnt/ssd'
        )

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
        if self.wandb is None:
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

    def log_frames(self, frames, key="video", step=None, fps=100, format="mp4"):
        """
        Logs video to WandB.

        Final logged shape is always: T×C×H×W

        Accepts:
            - list of H×W×C or C×H×W frames
            - numpy T×H×W×C
            - numpy T×C×H×W
            - single frame in either format (expanded to T=1)
        """

        if self.wandb is None or frames is None:
            return

        # --------------------------------------------------------
        # 1. Convert input to numpy array
        # --------------------------------------------------------
        if isinstance(frames, list):
            if len(frames) == 0:
                return

            processed = []

            for f in frames:
                f = np.asarray(f)

                # CHW → HWC
                if f.ndim == 3 and f.shape[0] in [1, 3]:
                    f = np.transpose(f, (1, 2, 0))

                processed.append(f)

            video = np.stack(processed, axis=0)  # Now T×H×W×C

        else:
            video = np.asarray(frames)

            # Single frame: expand T
            if video.ndim == 3:
                # CHW → HWC
                if video.shape[0] in [1, 3]:
                    video = np.transpose(video, (1, 2, 0))
                video = video[None]  # T=1

        # --------------------------------------------------------
        # 2. Validate now (should be T×H×W×C)
        # --------------------------------------------------------
        if video.ndim != 4:
            raise ValueError(
                f"log_frames(): Expected 4D input (T,H,W,C), got {video.shape}"
            )

        T, H, W, C = video.shape

        if C not in [1, 3]:
            raise ValueError(
                f"log_frames(): expected channel dim 1 or 3, got {C}"
            )

        # --------------------------------------------------------
        # 3. Convert to T×C×H×W
        # --------------------------------------------------------
        video = np.transpose(video, (0, 3, 1, 2))  # (T,H,W,C) → (T,C,H,W)

        # --------------------------------------------------------
        # 4. Float → uint8 normalization
        # --------------------------------------------------------
        if video.dtype != np.uint8:
            video = np.clip(video * 255, 0, 255).astype(np.uint8)

        # --------------------------------------------------------
        # 5. Log to WandB
        # --------------------------------------------------------
        self.wandb.log(
            {
                key: wandb.Video(video, fps=fps, format=format)
            },
            step=step
        )


    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()
        
    def get_run_id(self):
        return self.wandb.id