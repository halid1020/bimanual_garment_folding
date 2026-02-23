import hydra
from omegaconf import DictConfig, OmegaConf
import os
import socket
import actoris_harena.api as ag_ar

from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task
from tool.utils import resolve_save_root


# 1. Update config_path to point to the root 'conf' directory
@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):


    new_save_root = resolve_save_root(cfg.save_root)
    print(f"[tool.hydra_train] Using Save Root: {cfg.save_root}")

    # Update the config object (must unset 'struct' to modify)
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    OmegaConf.set_struct(cfg, True)
    # -------------------------------------

    print("[tool.hydra_check] --- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("[tool.hydra_check] ---------------------")


if __name__ == "__main__":
    main()