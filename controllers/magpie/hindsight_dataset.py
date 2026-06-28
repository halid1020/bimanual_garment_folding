from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset
import numpy as np

class HindsightDataset(TrajectoryDataset):
    """
    A tailored dataset for Hindsight Experience Replay (HER) in RL pipelines.
    
    Overrides observation fetching to dynamically sample goal states from 
    future time steps within the same trajectory. It mixes hindsight goals 
    with actual goals based on a probability threshold.
    """
    def __init__(self, future_goal_mapping=None, hindsight_prob=0.8, **kwargs):
        # Pop the mapping before passing the rest to TrajectoryDataset
        self.future_goal_mapping = future_goal_mapping or {
            'rgb': 'goal_rgb',
            'depth': 'goal_depth',
            'mask': 'goal_mask',
            'semkey_norm_pixel': 'flattened_goal_semkey_norm_pixel'
        }
        # Probability of using a hindsight goal vs the actual goal
        self.hindsight_prob = hindsight_prob
        super().__init__(**kwargs)

    def __getitem__(self, idx: int):
        # 1. Temporarily disable transforms to retrieve the raw data.
        current_transform = self.transform
        self.transform = None
        ret = super().__getitem__(idx)  # This fetches the actual goals by default
        self.transform = current_transform
        
        # 2. Recalculate trajectory bounds to find a valid future index
        idx_offset = idx + self.start_sample
        
        if self.whole_trajectory:
            traj_idx = idx_offset
            start_idx = self.traj_starts[traj_idx]
            end_idx = start_idx + self.traj_lengths[traj_idx] - 1
        elif self.cross_trajectory:
            start_idx = idx_offset if self.return_trj_last else self.valid_indices[idx_offset]
            traj_idx = np.searchsorted(self.traj_starts, start_idx, side='right') - 1
            end_idx = start_idx + self.seq_length
        else:
            traj_idx, start_idx = self.flat_ranges[idx_offset]
            end_idx = start_idx + self.seq_length

        # 3. Determine the effective bounds of the current trajectory
        # traj_lengths includes the padded terminal state, so valid observations end at (length - 2)
        effective_traj_end = self.traj_starts[traj_idx] + self.traj_lengths[traj_idx] - 2
        
        # strictly in the future of the current sequence (avoids distance 0)
        min_future_idx = end_idx
        
        # 4. Decide whether to use Hindsight or the Actual Goal
        # We can only use hindsight if there are future frames available to sample.
        can_sample_future = min_future_idx <= effective_traj_end
        use_hindsight = can_sample_future and (np.random.rand() < self.hindsight_prob)
        
        if use_hindsight:
            # Sample a strictly future state index
            future_idx = np.random.randint(min_future_idx, effective_traj_end + 1)
            
            # Overwrite the static goal keys in the observation dict with the sampled future observation
            for src_key, goal_key in self.future_goal_mapping.items():
                if src_key in self.obs_source and goal_key in ret['observation']:
                    # Extract the single raw future observation frame
                    future_obs = self.obs_source[src_key][future_idx].reshape(-1, *self.obs_shapes[src_key])
                    
                    # The returned observation dictionary contains sequences of shape (seq_length, ...)
                    # The goal should remain constant across the sequence steps, so we tile it.
                    seq_len = ret['observation'][goal_key].shape[0]
                    ret['observation'][goal_key] = np.repeat(future_obs, seq_len, axis=0)
            
            dist = future_idx - start_idx
        else:
            # Use the Actual Goal (keep the original 'ret' as fetched by super().__getitem__)
            # We assume the actual goal is the end of the trajectory for distance calculations
            dist = max(0, effective_traj_end - start_idx)

        # 5. Apply the transform now that the goals have been resolved
        if self.transform is not None:
            ret_ = self.transform(ret)
            for key in ret['observation'].keys():
                if key not in ret_:
                    ret_[key] = ret['observation'][key]
            ret = ret_

        # Record the distance for visualizer or network ingestion
        ret['goal_distance'] = np.array(dist, dtype=np.int32)
        return ret