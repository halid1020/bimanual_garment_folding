"""
Offline Action-Prediction Evaluation (sim-to-real screening).

Scores a trained MAGPIE checkpoint by how closely its *inferred* actions match the
*ground-truth human* actions recorded in a real-world demonstration dataset — without
touching the robot. Closed-loop simulation success rate has proven to be a poor proxy
for real-world behaviour, so this gives a cheap, directly comparable signal for every
agent before committing hardware time.

The forward pass is `MagpieTrainer.predict_batch`, i.e. exactly the inference path used
at deployment (primitive classification followed by the full flow/diffusion integration),
so online validation and this offline evaluation can never drift apart.

Metrics
-------
prim_accuracy / confusion matrix
    Does the classifier pick the primitive the human picked? The `not_rotate_primitives`
    exemption in the augmenter correlates scene orientation with primitive identity, which
    would show up here as a bias toward the exempted primitive on always-upright real frames.
action_mse / oracle_action_mse
    Masked to the ground-truth primitive's valid dimensions. `oracle_*` routes to the flow
    network of the *ground-truth* primitive, separating parameter error from routing error.
pick_err_px / place_err_px, raw and swap-invariant
    Euclidean error in pixels. `random_swap_actions` trains the two grippers' parameter
    slots as interchangeable; a large gap between the raw and swap-invariant numbers means
    the policy is predicting the correct grasps in the opposite order.
pred_pair_spread_px vs gt_pair_spread_px
    Distance between the two grasp points. If the predicted spread collapses relative to
    ground truth, the policy is averaging the two swap modes into a single midpoint grasp.
pick_on_mask_rate
    Fraction of predicted pick points landing on the garment mask. Needs no ground truth,
    so it is a sanity check that survives even where the human action is ambiguous.

Usage
-----
    python ./tool/magpie/eval_action_prediction.py \
        --config-name sim_exp/magpie/magpie_ctr_align_all_sim_garments_p4_v150_hindsight

Outputs `<save_root>/offline_eval/<exp_name>/<dataset>/action_prediction.csv` (one summary
row), `confusion_matrix.csv` and `per_sample.csv`.
"""

import os
import socket

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

import actoris_harena.api as ag_ar
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset

from registration.agent import register_agents
from tool.utils import resolve_save_root


# Real-world demonstrations carry no semantic keypoints (the converter fills them with
# -1 dummies) and use 15 rather than the simulator's 17, so they are dropped outright.
# They are not part of `input_obs` and are only read when `rep_learn: 'predict-state'`.
SEMKEY_OBS_KEYS = (
    'semkey_norm_pixel',
    'flattened_semkey_norm_pixel',
    'goal_semkey_norm_pixel',
    'flattened_goal_semkey_norm_pixel',
)

# Which action slots hold pick / place points, per primitive index. Actions are
# (y_norm, x_norm) pairs in [-1, 1]; see `MagpieAgent._score_pick_on_mask`.
PRIMITIVE_POINT_LAYOUT = {
    'pick-and-fling':          {'pick': [(0, 1), (2, 3)], 'place': []},
    'dual-pick-and-place':     {'pick': [(0, 1), (2, 3)], 'place': [(4, 5), (6, 7)]},
    'single-pick-and-place':   {'pick': [(0, 1)],         'place': [(2, 3)]},
    'no-operation':            {'pick': [],               'place': []},
}

# Gripper-order symmetry groups used by the `random_swap_actions` augmentation, keyed by
# primitive index. Swapping these index groups describes the same physical action.
SWAP_GROUPS = {
    0: ([0, 1], [2, 3]),
    1: ([0, 1, 4, 5], [2, 3, 6, 7]),
}


def primitive_name(agent, prim_id):
    """Returns the registered name of primitive `prim_id`."""
    p_obj = agent.primitives[prim_id]
    return p_obj['name'] if isinstance(p_obj, dict) else p_obj.name


def point_layout(agent, prim_id):
    """Maps a primitive index to its pick/place slot layout."""
    name = primitive_name(agent, prim_id)
    for key, layout in PRIMITIVE_POINT_LAYOUT.items():
        if key in name:
            return layout
    return {'pick': [], 'place': []}


def points_from(action_vec, slots):
    """Extracts (y, x) point pairs from a flat action vector."""
    return np.array([[action_vec[yi], action_vec[xi]] for yi, xi in slots], dtype=np.float64)


def apply_swap(action_vec, prim_id):
    """Returns a copy of `action_vec` with the two grippers' parameter slots exchanged."""
    if prim_id not in SWAP_GROUPS:
        return action_vec
    group_a, group_b = SWAP_GROUPS[prim_id]
    swapped = action_vec.copy()
    swapped[group_a], swapped[group_b] = action_vec[group_b], action_vec[group_a]
    return swapped


def point_error_px(pred_vec, gt_vec, slots, px_scale):
    """Mean Euclidean distance in pixels between the predicted and ground-truth points."""
    if not slots:
        return np.nan
    pred_pts = points_from(pred_vec, slots)
    gt_pts = points_from(gt_vec, slots)
    return float(np.linalg.norm(pred_pts - gt_pts, axis=-1).mean() * px_scale)


def pair_spread_px(action_vec, slots, px_scale):
    """Distance in pixels between the first two points; NaN when the primitive has one."""
    if len(slots) < 2:
        return np.nan
    pts = points_from(action_vec, slots)
    return float(np.linalg.norm(pts[0] - pts[1]) * px_scale)


def picks_on_mask(pred_vec, slots, mask2d):
    """Fraction of predicted pick points landing on the garment mask."""
    if not slots or mask2d is None:
        return np.nan
    H, W = mask2d.shape[:2]
    hits = 0
    for yi, xi in slots:
        row = int(np.clip((pred_vec[yi] + 1.0) * 0.5 * H, 0, H - 1))
        col = int(np.clip((pred_vec[xi] + 1.0) * 0.5 * W, 0, W - 1))
        hits += int(bool(mask2d[row, col]))
    return hits / len(slots)


def build_eval_dataset(agent_cfg, data_path, data_dir):
    """
    Opens the evaluation store read-only, dropping keypoint entries.

    Uses the plain `TrajectoryDataset` rather than `HindsightDataset`: real deployment
    conditions on a fixed target image, so relabelling goals with sampled future frames
    would measure something the robot never sees.
    """
    dataset_config = OmegaConf.to_container(agent_cfg.dataset_config, resolve=True) \
        if not isinstance(agent_cfg.dataset_config, dict) else dict(agent_cfg.dataset_config)

    obs_config = {k: v for k, v in dataset_config['obs_config'].items()
                  if k not in SEMKEY_OBS_KEYS}

    return TrajectoryDataset(
        data_path=data_path,
        data_dir=data_dir,
        # 'r' is mandatory: 'w' deletes the store on open.
        io_mode='r',
        seq_length=dataset_config.get('seq_length', 1),
        split_ratios=dataset_config.get('split_ratios', [0.0, 0.0, 1.0]),
        cache_in_memory=True,
        obs_config=obs_config,
        act_config=dataset_config['act_config'],
        sample_mode='all',
    )


def evaluate(agent, dataset, batch_size, px_scale, use_mask):
    """Runs the full inference path over `dataset` and returns per-sample records."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    trainer = agent.trainer
    agent.nets.eval()
    records = []

    with torch.no_grad():
        for nbatch in dataloader:
            raw_masks = None
            if use_mask and 'mask' in nbatch['observation']:
                raw_masks = np.asarray(nbatch['observation']['mask'])

            prepared = trainer.prepare_batch(nbatch, train=False)
            out = trainer.predict_batch(prepared)

            # Second pass conditioned on the ground-truth primitive isolates the
            # action-parameter error from the classifier's routing error.
            oracle = trainer.predict_batch(
                trainer.prepare_batch(nbatch, train=False), use_gt_primitive=True)

            pred_np = out['pred_action'][:, 0].detach().cpu().numpy()
            gt_np = out['gt_action'][:, 0].detach().cpu().numpy()
            oracle_np = oracle['pred_action'][:, 0].detach().cpu().numpy()
            pred_prims = out['pred_prim_ids'].detach().cpu().numpy()
            gt_prims = out['gt_prim_ids'].detach().cpu().numpy()

            for i in range(len(gt_prims)):
                gt_prim = int(gt_prims[i])
                layout = point_layout(agent, gt_prim)
                pick_slots, place_slots = layout['pick'], layout['place']

                pred_vec, gt_vec = pred_np[i], gt_np[i]
                swapped_vec = apply_swap(pred_vec, gt_prim)

                pick_raw = point_error_px(pred_vec, gt_vec, pick_slots, px_scale)
                pick_swap = point_error_px(swapped_vec, gt_vec, pick_slots, px_scale)
                place_raw = point_error_px(pred_vec, gt_vec, place_slots, px_scale)
                place_swap = point_error_px(swapped_vec, gt_vec, place_slots, px_scale)

                # Pick the assignment with the smaller pick error, then report the
                # matching place error under that same assignment.
                use_swapped = (not np.isnan(pick_swap)) and (pick_swap < pick_raw)
                place_best = place_swap if use_swapped else place_raw

                mask2d = None
                if raw_masks is not None:
                    m = raw_masks[i]
                    m = m[0] if m.ndim == 4 else m          # take the first frame of the sequence
                    m = m[..., 0] if m.ndim == 3 else m     # drop a trailing singleton channel
                    mask2d = m > 0.5

                records.append({
                    'gt_primitive': gt_prim,
                    'pred_primitive': int(pred_prims[i]),
                    'prim_correct': int(pred_prims[i] == gt_prim),
                    'action_sq_err': float(((pred_vec - gt_vec) ** 2).mean()),
                    'oracle_action_sq_err': float(((oracle_np[i] - gt_vec) ** 2).mean()),
                    'pick_err_px_raw': pick_raw,
                    'pick_err_px_swapinv': min(pick_raw, pick_swap) if not np.isnan(pick_swap) else pick_raw,
                    'place_err_px_raw': place_raw,
                    'place_err_px_swapinv': place_best,
                    'pred_pair_spread_px': pair_spread_px(pred_vec, pick_slots, px_scale),
                    'gt_pair_spread_px': pair_spread_px(gt_vec, pick_slots, px_scale),
                    'pick_on_mask': picks_on_mask(pred_vec, pick_slots, mask2d),
                })

    return pd.DataFrame(records)


def summarise(per_sample, agent, exp_name, dataset_name, num_samples, in_sample):
    """Collapses the per-sample table into the single row the notebook aggregates."""
    def mean(col):
        vals = per_sample[col].dropna()
        return float(vals.mean()) if len(vals) else np.nan

    row = {
        'exp_name': exp_name,
        'dataset': dataset_name,
        'num_samples': num_samples,
        # True when these episodes are inside the agent's own training split, which makes
        # the numbers a training-set fit rather than a held-out generalisation measure.
        'in_sample': in_sample,
        'prim_accuracy': float(per_sample['prim_correct'].mean()),
        'action_mse': mean('action_sq_err'),
        'oracle_action_mse': mean('oracle_action_sq_err'),
        'pick_err_px_raw': mean('pick_err_px_raw'),
        'pick_err_px_swapinv': mean('pick_err_px_swapinv'),
        'place_err_px_raw': mean('place_err_px_raw'),
        'place_err_px_swapinv': mean('place_err_px_swapinv'),
        'pred_pair_spread_px': mean('pred_pair_spread_px'),
        'gt_pair_spread_px': mean('gt_pair_spread_px'),
        'pick_on_mask_rate': mean('pick_on_mask'),
    }

    # A spread ratio well below 1 is the signature of the two swap modes being averaged
    # into a single midpoint grasp.
    if row['gt_pair_spread_px'] and not np.isnan(row['gt_pair_spread_px']):
        row['pair_spread_ratio'] = row['pred_pair_spread_px'] / row['gt_pair_spread_px']

    # A large raw-vs-swap-invariant gap means the grasps are right but ordered wrong.
    row['swap_gap_px'] = row['pick_err_px_raw'] - row['pick_err_px_swapinv']

    for prim_id in range(agent.K):
        subset = per_sample[per_sample['gt_primitive'] == prim_id]
        row[f'n_gt_prim_{prim_id}'] = int(len(subset))
        row[f'recall_prim_{prim_id}'] = \
            float(subset['prim_correct'].mean()) if len(subset) else np.nan

    return row


@hydra.main(config_path="../../conf", version_base=None)
def main(cfg: DictConfig):
    os.environ['MEGPIE_ACTIVE_AGENT'] = cfg.agent.name
    register_agents()

    new_save_root = resolve_save_root(cfg.save_root)
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    OmegaConf.set_struct(cfg, True)

    offline_cfg = cfg.get('offline_eval', {})
    data_path = offline_cfg.get('data_path', 'real_world_longsleeve')
    data_dir = offline_cfg.get('data_dir', cfg.agent.dataset_config.get('data_dir', './data/datasets'))
    batch_size = int(offline_cfg.get('batch_size', 64))
    image_size = int(offline_cfg.get('image_size', 128))
    checkpoint = offline_cfg.get('checkpoint', 'best')

    # Normalised coordinates span [-1, 1] across the image, so one unit is image_size/2 px.
    px_scale = image_size / 2.0

    save_dir = os.path.join(cfg.save_root, cfg.exp_name)
    print(f"[eval_action_prediction] Host: {socket.gethostname()}")
    print(f"[eval_action_prediction] Experiment: {cfg.exp_name}")
    print(f"[eval_action_prediction] Checkpoint dir: {save_dir}")
    print(f"[eval_action_prediction] Dataset: {os.path.join(data_dir, data_path)}")

    agent = ag_ar.build_agent(
        cfg.agent.name,
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir,
        disable_wandb=True,
    )

    if checkpoint == 'best':
        loaded = agent.load_best()
        if loaded == 0 and not agent.loaded:
            print("[eval_action_prediction] No 'best' checkpoint; falling back to the latest.")
            agent.load()
    else:
        agent.load_checkpoint(checkpoint)

    if not agent.loaded:
        raise FileNotFoundError(
            f"[eval_action_prediction] No checkpoint found under {save_dir}/checkpoints. "
            f"Has this experiment been trained?"
        )

    dataset = build_eval_dataset(cfg.agent, data_path, data_dir)
    print(f"[eval_action_prediction] {len(dataset)} samples "
          f"across {dataset.num_trajectories()} trajectories.")

    per_sample = evaluate(
        agent, dataset, batch_size, px_scale,
        use_mask=bool(cfg.agent.get('use_mask', False)),
    )

    # The combined sim+real store contains these very episodes in its training split, so
    # a mix-trained agent is being scored on data it has already seen.
    train_data_path = str(cfg.agent.dataset_config.get('data_path', ''))
    in_sample = ('real' in train_data_path) and (data_path in train_data_path
                                                 or 'combined' in train_data_path)

    summary = summarise(per_sample, agent, cfg.exp_name, data_path, len(dataset), in_sample)

    out_dir = os.path.join(cfg.save_root, 'offline_eval', cfg.exp_name, data_path)
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'action_prediction.csv'), index=False)
    per_sample.to_csv(os.path.join(out_dir, 'per_sample.csv'), index=False)

    confusion = pd.crosstab(
        per_sample['gt_primitive'], per_sample['pred_primitive'],
        rownames=['gt'], colnames=['pred'], dropna=False,
    ).reindex(index=range(agent.K), columns=range(agent.K), fill_value=0)
    confusion.to_csv(os.path.join(out_dir, 'confusion_matrix.csv'))

    print(f"\n[eval_action_prediction] Written to {out_dir}")
    for key, value in summary.items():
        print(f"  {key:<26} {value}")
    print("\n[eval_action_prediction] Primitive confusion (rows=gt, cols=pred):")
    print(confusion)


if __name__ == '__main__':
    main()
