"""Shared helpers for persisting per-step evaluation data from the simulation
loggers, mirroring the real-world ``PixelBasedPrimitiveImpEnvLogger`` layout so
sim and real produce byte-compatible ``episode_{eid}/step_{i}/`` folders.
"""
import os
import json
import numpy as np

from real_robot.utils.save_utils import (
    save_colour, save_depth, save_mask, save_action_json, NumpyEncoder
)


def _resolve_garment_name(info, episode_config):
    arena = info.get('arena') if isinstance(info, dict) else None
    if arena is not None and hasattr(arena, 'garment_type'):
        return getattr(arena, 'garment_type', 'unknown')
    if 'garment_id' in episode_config:
        return episode_config['garment_id']
    if 'garment_type' in episode_config:
        return episode_config['garment_type']
    return 'unknown'


def _save_observation(obs, directory):
    if 'rgb' in obs:
        save_colour(obs['rgb'], filename='rgb', directory=directory, rgb2bgr=True)
    if 'depth' in obs:
        depth = np.asarray(obs['depth'])
        # Sim depth is (H, W, 1); squeeze the trailing channel for cv2.imwrite.
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        save_depth(depth, filename='depth', directory=directory)
    if 'mask' in obs:
        save_mask(obs['mask'], filename='mask', directory=directory)
    if 'robot0_mask' in obs:
        save_mask(obs['robot0_mask'], filename='robot0_mask', directory=directory)
    if 'robot1_mask' in obs:
        save_mask(obs['robot1_mask'], filename='robot1_mask', directory=directory)


def _dump_info(info, directory, eid, default_done):
    step_info = {
        'evaluation': info.get('evaluation', {}),
        'success': info.get('success', False),
        'reward': info.get('reward', 0.0),
        'done': info.get('done', default_done),
        'eid': eid,
    }
    with open(os.path.join(directory, 'info.json'), 'w') as f:
        json.dump(step_info, f, indent=4, cls=NumpyEncoder)


def save_step_data(step_dir, info, action, eid, episode_config):
    """Persist rgb/depth/mask(s), action, info and garment name for one step."""
    os.makedirs(step_dir, exist_ok=True)
    _save_observation(info.get('observation', {}), step_dir)
    if action is not None:
        save_action_json(action, filename='action', directory=step_dir)
    _dump_info(info, step_dir, eid, default_done=False)
    with open(os.path.join(step_dir, 'garment_name.txt'), 'w') as f:
        f.write(str(_resolve_garment_name(info, episode_config)))


def save_final_state(episode_data_dir, last_info, eid, episode_config):
    """Persist the terminal observation into ``episode_data_dir/final_state``."""
    final_state_dir = os.path.join(episode_data_dir, 'final_state')
    os.makedirs(final_state_dir, exist_ok=True)
    _save_observation(last_info.get('observation', {}), final_state_dir)
    _dump_info(last_info, final_state_dir, eid, default_done=True)
    with open(os.path.join(final_state_dir, 'garment_name.txt'), 'w') as f:
        f.write(str(_resolve_garment_name(last_info, episode_config)))
