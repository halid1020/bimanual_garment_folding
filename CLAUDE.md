# CLAUDE.md — Bimanual Garment Folding (MAGPIE)

Project guide for Claude. Read this first to understand the repo and how to work in it.

## Operating principles (follow these)
1. **Reflect code in the paper.** Any change to the codebase that affects method, architecture,
   hyperparameters, metrics, or evaluation must be reflected at an appropriate level of detail in
   the paper (`paper/example.tex`). Keep the paper consistent with the code that actually runs.
2. **Be surgical, and ask before modifying.** Make the smallest change that fixes the problem.
   Investigate (read-only) first, then propose the exact change and ask for approval before
   editing code or the paper. Prefer config-only solutions over code changes when possible.
3. **Verify before concluding.** After a change, confirm it: `python -m py_compile` (or a Hydra
   `--cfg job` dry-run) for Python, and `latexmk -pdf -bibtex` for the paper. State results
   plainly; don't claim done without checking. *(Third rule was left open by the user — refine as
   they specify.)*

## What this is
A data-driven system for **bimanual (dual-arm) garment manipulation**: flattening/canonicalisation-
alignment and folding from crumpled states. The headline method is **MAGPIE** — a multi-primitive
imitation-learning policy: an MLP classifier picks one of 4 primitives, and per-primitive
Optimal-Transport Conditional Flow-Matching (COT-FM) networks generate the continuous parameters.
The paper is in `paper/` (IEEE RA-L / `IEEEtran`; build with `latexmk -pdf -bibtex example.tex`).

## Environment / how to run
- Conda env **`magpie`**; always `source ./setup.sh` first (activates conda, sources
  `../softgym/setup.sh` for PyFlex, exports `PYTHONPATH`/`MP_FOLD_PATH`/`REAL_ROBOT_PATH`).
- Hybrid-GPU laptop: default GL is AMD Mesa (PyFlex's legacy geometry shaders crash it). Run with
  **`nvrun`** = `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` to use the NVIDIA GPU.
  In a non-interactive shell, `conda` needs `source ~/anaconda3/etc/profile.d/conda.sh` first.
- Local eval:  `nvrun ./job_scripts/submit_evaluating_locally.sh magpie/<exp> f`
- Local train: `nvrun ./job_scripts/submit_training_locally.sh magpie/<exp>`
- Viking (SLURM): `./job_scripts/generate_and_submit_viking_job.sh magpie/<exp> -c 6 -m 24G -p gpu -t 72:00:00 -a`
  (`-a` → `hydra_train_and_transfer.py` + `transfer_eval/` configs: train, then zero-shot transfer
  eval across a list of arenas/tasks; adds headless `QT_QPA_PLATFORM=offscreen`, `SDL_VIDEODRIVER=dummy`.)

## Pipeline (both train & eval)
Entry points `tool/hydra_{train,eval,train_and_transfer}.py` (Hydra, `config_path=../conf`):
compose config → `registration.{agent,sim_arena,task}` register classes into external framework
**`actoris_harena`** (`ag_ar.build_agent/build_arena`, `evaluate`, `train_and_evaluate_single`) →
`arena.set_task(task)` → run.

- **Configs** (`conf/`): a `sim_exp/<group>/<exp>.yaml` (`# @package _global_`) composes
  `agent/<...>`, `arena/<...>`, `task/<...>` sub-configs, plus `exp_name`, `project_name`,
  `save_root` (`/mnt/ssd/garment_folding_data`, remapped by `resolve_save_root`), `train_and_eval`.
- **Agents** (`controllers/`): `magpie` → `MagpieAgent` (+`MagpieTrainer`, `magpie_network_builder`);
  `human-multi-primitive` → `HumanMultiPrimitive` (interactive human baseline / demonstrator);
  `iou-based-stitching-policy` → `IoUBasedStitchingPolicy` (runs a flattening policy, then hands off
  to a folding policy when `multi_stage_reward` completes the alignment stage);
  demonstrators like `centre_sleeve_folding_policy` (keypoint heuristic folder).
- **Arena** (`env/softgym_garment/`): `multi-garment-env` → `MultiGarmentEnv(GarmentEnv(Arena))`.
  `GarmentEnv.step` → `HybridActionPrimitive.step` (executes the primitive in PyFlex) → `_process_info`
  (builds `info['observation']`, `info['evaluation']`, `info['success']`, `info['reward']`,
  `info['goals']`, `done`). `arena.evaluate()` delegates to `task.evaluate(arena)`.
- **Tasks** (`env/softgym_garment/tasks/`): `build_sim_task` maps `task_name` →
  `GarmentFoldingTask` (folding, incl. `canonicalisation_alignment_centre_sleeve_folding`),
  `AlignmentTask`/`CanonicalisationAlignmentTask`, `GarmentFlatteningTask`.
- **Action primitives** (normalised pixel `[-1,1]`, dict `{prim_name: params}`):
  `norm-pixel-pick-and-fling` (4), `norm-pixel-dual-pick-and-place` (8),
  `norm-pixel-single-pick-and-place` (4), `no-operation` (0).

## MAGPIE current config: `magpie_ctr_align_all_sim_garments_p4_v126_hindsight`
This is the config the paper's Method/Appendix A describe. Verified from code:
- Vision: ResNet-18 + GroupNorm → 512-d; MLP projector `[512,512]` GELU → **512-d** embedding.
- Primitive classifier: MLP `[256,128]` + LayerNorm, dropout 0.5, K=4.
- Flow matching: **MLP backbone (`ConditionalMLP1D`)** `[1024,1024,1024,512]` GELU dropout 0.1,
  `separate_networks` (one per primitive; no-op skipped), **4-step Euler** inference,
  conditioning by **concatenation** (not FiLM). `loss_type: ot_flow_match`.
- **No representation/semantic-keypoint head** (`rep_learn` unset → `none`; rep loss = 0).
- Data: `dataset_mode: hindsight` (`HindsightDataset`, 0.8 future-goal relabel), all-garment human
  multi-primitive demos; AdamW lr 3e-4, wd 1e-2, batch 1024, 120k steps, warmup 3k, EMA 0.9999.
- ⚠️ The paper previously described a 1D U-Net + a semantic-keypoint head + different numbers;
  it has been corrected to match v126. Keep it in sync on future config changes.

## Task subgoal / evaluation notes (`GarmentFoldingTask`)
- `canonicalisation_alignment_centre_sleeve_folding`: `goal_steps: 2` → **3 subgoals** =
  initial flattened/canonical state + 2 fold steps (`K = goal_steps + 1`; `iou_thresholds` length 3).
- Goals are generated once (demonstrator rollout) and cached under `assets/goals/.../eid_*/goal_*`;
  the loader reads `range(goal_steps+1)` RGB/particle/depth frames.
- Subgoal feeding: `evaluate()` computes `active_subgoal_idx` from the **current** state (final goal
  is shown only if the current mask still matches it — do NOT gate on the latched `has_succeeded`,
  which is updated later in `success()` and lags by one step). The human overlay
  (`controllers/human/utils.py`) shows `state['goals'][active_subgoal_idx]`.
- Order in `GarmentEnv._process_info`: `evaluate()` runs before `success()`; keep active-subgoal
  logic independent of `has_succeeded` to avoid a one-step lag.

## Paper
- `paper/example.tex` (RA-L/IEEEtran, two-column). Build: `cd paper && latexmk -pdf -bibtex -interaction=nonstopmode example.tex`.
  `.vscode/settings.json` (repo root) has a LaTeX-Workshop recipe; `paper/.latexmkrc` pins the target.
- `example.bib` must stay free of duplicate keys / unescaped math in titles (BibTeX aborts otherwise).
- VS Code LaTeX build works via the `latexmk (RA-L)` recipe.

## External code (not in this repo)
`actoris_harena` (agent-arena framework: base `Agent`/`TrainableAgent`/`Arena`, `build_*`,
`evaluate`, `train_and_evaluate_single`, `perform_single`, `TrajectoryDataset`) and `../softgym`
(PyFlex sim) live outside this repo. Respect any read-scope restrictions the user sets on them.
