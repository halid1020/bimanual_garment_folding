import os

from agent_arena.utilities.logger.logger_interface import Logger

class WandbLogger(Logger):
    def __init__(self, project="garment-folding", name= None, config= None):
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

    def log(self, metrics, step= None) -> None:
        if self.wandb is not None and self.run is not None:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()