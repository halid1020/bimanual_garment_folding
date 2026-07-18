# tool/hydra_horizon_ablation.py

"""
Standalone MPC-horizon ablation for LaGarNet.

Loads the best checkpoint of a trained LaGarNet experiment and evaluates it on
the configured arenas across the planning variants below (horizons H = 1, 2,
3, 5 plus an unconstrained-CEM variant at H = 1), reporting both task
performance (via the standard arena logger -> performance.csv) and per-step
planning runtime (runtime.csv). Results land in
<save_root>/<eval_name>_<tag>/<arena>/eval_checkpoint_-2/ so the analysis
notebook can read them with the same layout as other transfer evaluations.
"""

import copy
import os

import hydra
import numpy as np
import pandas as pd
from hydra import compose
from omegaconf import DictConfig, OmegaConf

import actoris_harena.api as ag_ar
from actoris_harena.api import run
from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_sim_task
from tool.utils import resolve_save_root

# The ablations this script exists for; edit here, not in the config.
# (tag, planning_horizon, constrain_actions)
VARIANTS = [
    ('H1', 1, True),
    ('H2', 2, True),
    ('H3', 3, True),
    ('H5', 5, True),
    ('H1_unconstrained', 1, False),  # CEM without cloth/workspace mask rejection
]


def _unwrap(cfg: DictConfig, marker_keys) -> DictConfig:
    """Descend through single-child group nesting until a marker key appears."""
    while not any(k in cfg for k in marker_keys):
        keys = list(cfg.keys())
        if len(keys) != 1:
            raise KeyError(
                f"Cannot locate {marker_keys} in config; got keys {keys}")
        cfg = cfg[keys[0]]
    return cfg


def _append_runtime_row(runtime_csv, episode_config, horizon, constrained, durations):
    row = pd.DataFrame([{
        'tier': episode_config['tier'],
        'episode_id': episode_config['eid'],
        'planning_horizon': horizon,
        'constrain_actions': constrained,
        'num_steps': len(durations),
        'planning_time_mean_s': float(np.mean(durations)),
        'planning_time_std_s': float(np.std(durations)),
        'planning_times_s': list(np.round(durations, 4)),
    }])
    written = os.path.exists(runtime_csv)
    row.to_csv(runtime_csv, mode=('a' if written else 'w'),
               header=(not written), index=False)


@hydra.main(config_path="../conf",
            config_name="transfer_eval/lagarnet/final_lagarnet_horizon_ablation",
            version_base=None)
def main(cfg: DictConfig) -> None:
    register_agents()
    register_arenas()

    OmegaConf.set_struct(cfg, False)
    cfg = _unwrap(cfg, ['eval_name'])

    print(f"[horizon_ablation] Composing training config: {cfg.train_exp_config}")
    train_cfg = compose(config_name=cfg.train_exp_config)
    os.environ['MEGPIE_ACTIVE_AGENT'] = train_cfg.agent.name
    OmegaConf.set_struct(train_cfg, False)

    save_root = resolve_save_root(train_cfg.save_root)
    source_save_dir = os.path.join(save_root, train_cfg.exp_name)
    print(f"[horizon_ablation] Checkpoint dir: {source_save_dir}")

    for tag, horizon, constrained in VARIANTS:
        print(f"\n{'='*60}\n[horizon_ablation] Variant {tag}: H = {horizon}, "
              f"constrain_actions = {constrained}\n{'='*60}")

        agent_cfg = copy.deepcopy(train_cfg.agent)
        agent_cfg.policy.params.planning_horizon = horizon
        agent_cfg.policy.params.constrain_actions = constrained

        agent = ag_ar.build_agent(
            agent_cfg.name,
            agent_cfg,
            project_name=train_cfg.project_name,
            exp_name=f"{train_cfg.exp_name}_{tag}",
            save_dir=source_save_dir,
            disable_wandb=True
        )
        checkpoint = agent.load_best()
        if checkpoint != -2:
            raise FileNotFoundError(
                f"Best checkpoint not found under {source_save_dir}/checkpoints/"
                "model_best.pth; refusing to evaluate random weights.")

        for i, eval_setup in enumerate(cfg.eval_arenas):
            arena_cfg = _unwrap(compose(config_name=f"arena/{eval_setup.arena}"),
                                ['name'])
            task_cfg = _unwrap(compose(config_name=f"task/{eval_setup.task}"),
                               ['task_name'])

            clean_arena_name = eval_setup.arena.split('/')[-1]
            out_dir = os.path.join(save_root, f"{cfg.eval_name}_{tag}",
                                   clean_arena_name)
            os.makedirs(out_dir, exist_ok=True)
            runtime_csv = os.path.join(out_dir, 'eval_checkpoint_-2', 'runtime.csv')

            print(f"\n>>> {tag} | Arena: {arena_cfg.name} | "
                  f"Task: {task_cfg.task_name}\n>>> Output: {out_dir}")

            arena = ag_ar.build_arena(
                arena_cfg.name,
                arena_cfg,
                project_name=train_cfg.project_name,
                exp_name=f"{cfg.eval_name}_{tag}_arena_{i}",
                save_dir=out_dir
            )
            task = build_sim_task(task_cfg)
            arena.set_task(task)
            arena.reset()

            for episode_config in arena.get_eval_configs():
                ran, res = run(
                    agent, arena, 'eval', episode_config,
                    checkpoint=-2,
                    policy_terminate=False,
                    save_internal_states=True,
                    env_success_stop=False
                )
                if not ran:
                    print(f">>> Skipping already-logged episode {episode_config['eid']}")
                    continue
                os.makedirs(os.path.dirname(runtime_csv), exist_ok=True)
                _append_runtime_row(runtime_csv, episode_config, horizon,
                                    constrained, res['action_durations'])

            arena.close()  # release the PyFlex/OpenGL context between runs

    print("\n[horizon_ablation] All variants completed.")


if __name__ == "__main__":
    main()
