import unittest
import numpy as np
import os
import shutil
import tempfile
from torch.utils.data import DataLoader

# --- Mocking the class import for the purpose of this script ---
# Replace this with your actual import path
from actoris_harena import TrajectoryDataset
from hindsight_dataset import HindsightDataset 

class TestHindsightDataset(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_path = "test_hindsight_dataset.zarr"
        self.full_path = os.path.join(self.test_dir, self.data_path)

        # Standard Configs matching your YAML
        self.obs_config = {
            'rgb': {'shape': (3, 128, 128), 'output_key': 'rgb'},
            'depth': {'shape': (1, 128, 128), 'output_key': 'depth'},
            'semkey_norm_pixel': {'shape': (17, 2), 'output_key': 'semkey_norm_pixel'},
            # The static goals that should be OVERWRITTEN by the hindsight logic
            'goal_rgb': {'shape': (3, 128, 128), 'output_key': 'goal_rgb'},
            'flattened_goal_semkey_norm_pixel': {'shape': (17, 2), 'output_key': 'flattened_goal_semkey_norm_pixel'}
        }
        
        self.act_config = {
            'default': {'shape': (9,), 'output_key': 'default'}
        }
        
        self.future_goal_mapping = {
            'rgb': 'goal_rgb',
            'semkey_norm_pixel': 'flattened_goal_semkey_norm_pixel'
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def generate_temporally_predictable_data(self, num_steps=10):
        """
        CRITICAL: Instead of random noise, we fill the tensors with their timestep index 't'.
        This allows the tests to inspect the sampled goal and know exactly which future 
        frame the stochastic sampler selected.
        """
        obs = {
            'rgb': np.zeros((num_steps + 1, 3, 128, 128), dtype=np.float32),
            'depth': np.zeros((num_steps + 1, 1, 128, 128), dtype=np.float32),
            'semkey_norm_pixel': np.zeros((num_steps + 1, 17, 2), dtype=np.float32),
            
            # Fill original static goals with -1.0 so we can prove they get overwritten
            'goal_rgb': np.full((num_steps + 1, 3, 128, 128), -1.0, dtype=np.float32),
            'flattened_goal_semkey_norm_pixel': np.full((num_steps + 1, 17, 2), -1.0, dtype=np.float32)
        }
        
        for t in range(num_steps + 1):
            obs['rgb'][t] += t
            obs['semkey_norm_pixel'][t] += (t * 10) # Distinct scalar for keypoints
            
        act = {
            'default': np.zeros((num_steps, 9), dtype=np.float32)
        }
        
        return obs, act

    # =========================================================================
    # TEST 1: HER Mapping and Shape Consistency
    # =========================================================================
    def test_hindsight_mapping_and_shapes(self):
        print("\n--- Test 1: HER Mapping and Shape Consistency ---")
        
        dataset_w = HindsightDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='w',
            obs_config=self.obs_config, act_config=self.act_config,
            future_goal_mapping=self.future_goal_mapping
        )
        
        # Add trajectory of 10 actions (11 observations stored due to pad)
        obs, act = self.generate_temporally_predictable_data(10)
        dataset_w.add_trajectory(obs, act)

        seq_len = 2
        dataset_r = HindsightDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='r',
            obs_config=self.obs_config, act_config=self.act_config,
            seq_length=seq_len, cross_trajectory=False, return_trj_last=False,
            future_goal_mapping=self.future_goal_mapping
        )

        item = dataset_r[0] # Sampling start_idx = 0, end_idx = 2
        
        # 1. Verify shapes. Observation sequences should be (seq_len + 1, ...)
        expected_obs_seq_len = seq_len + 1
        self.assertEqual(item['observation']['goal_rgb'].shape, (expected_obs_seq_len, 3, 128, 128))
        self.assertEqual(item['observation']['flattened_goal_semkey_norm_pixel'].shape, (expected_obs_seq_len, 17, 2))

        # 2. Verify static goals were overwritten (no -1.0 values remain)
        self.assertTrue(np.all(item['observation']['goal_rgb'] >= 0))
        
        # 3. Verify temporal tiling. The goal should be identical across the sequence dimension.
        # Check if the first frame of the goal equals the last frame of the goal sequence
        np.testing.assert_array_equal(item['observation']['goal_rgb'][0], item['observation']['goal_rgb'][-1])
        
        print("Mapping and shape constraints verified.")

    # =========================================================================
    # TEST 2: Rigorous Stochastic Future Boundaries & Synchronization
    # =========================================================================
    def test_future_sampling_bounds_rigorous(self):
        print("\n--- Test 2: Rigorous Stochastic Future Boundaries ---")
        
        dataset_w = HindsightDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='w',
            obs_config=self.obs_config, act_config=self.act_config,
            future_goal_mapping=self.future_goal_mapping
        )
        
        # Use a slightly longer trajectory to give the sampler a wider variance
        traj_len = 15
        obs, act = self.generate_temporally_predictable_data(traj_len)
        dataset_w.add_trajectory(obs, act)

        seq_len = 4
        dataset_r = HindsightDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='r',
            obs_config=self.obs_config, act_config=self.act_config,
            seq_length=seq_len, cross_trajectory=False, return_trj_last=False,
            future_goal_mapping=self.future_goal_mapping
        )

        # effective_traj_end avoids the dummy pad (traj_len - 1 = 14)
        effective_traj_end = traj_len - 1 

        # We will test EVERY valid starting index in the trajectory
        num_valid_starts = len(dataset_r)
        
        for start_idx in range(num_valid_starts):
            end_idx = start_idx + seq_len
            expected_min = min(end_idx, effective_traj_end)
            expected_max = effective_traj_end
            
            valid_future_range = list(range(expected_min, expected_max + 1))
            sampled_t_rgb_counts = {t: 0 for t in valid_future_range}
            
            # Sample enough times to guarantee hitting every uniform choice.
            # Using 200 samples makes the probability of missing a valid bin astronomically small.
            num_samples = 200 
            
            for _ in range(num_samples):
                item = dataset_r[start_idx]
                
                # Check RGB temporal mark
                sampled_t_rgb = int(np.mean(item['observation']['goal_rgb']))
                
                # Check Semkey temporal mark. 
                # Recall we multiplied semkey by 10 in generate_temporally_predictable_data
                sampled_t_semkey = int(np.mean(item['observation']['flattened_goal_semkey_norm_pixel'])) / 10.0
                
                # --- RIGOR CHECK 1: Cross-Modality Synchronization ---
                self.assertEqual(
                    sampled_t_rgb, int(sampled_t_semkey), 
                    f"Temporal mismatch! RGB goal is from t={sampled_t_rgb} but Semkey goal is from t={sampled_t_semkey}"
                )
                
                # --- RIGOR CHECK 2: Absolute Boundaries ---
                self.assertGreaterEqual(
                    sampled_t_rgb, expected_min, 
                    f"Time Travel Bug: Sampled past index {sampled_t_rgb} for start {start_idx} (min allowed: {expected_min})"
                )
                self.assertLessEqual(
                    sampled_t_rgb, expected_max, 
                    f"Out of Bounds Bug: Sampled index {sampled_t_rgb} for start {start_idx} (max allowed: {expected_max})"
                )
                
                sampled_t_rgb_counts[sampled_t_rgb] += 1

            # --- RIGOR CHECK 3: Uniform Coverage ---
            # Ensure every single valid future index was sampled at least once
            if expected_min < expected_max: # If there's an actual range to sample from
                for t, count in sampled_t_rgb_counts.items():
                    self.assertGreater(
                        count, 0, 
                        f"Coverage Failure: At start_idx={start_idx}, valid future index {t} was NEVER sampled after {num_samples} tries. Sampler is biased."
                    )
                    
        print(f"Rigorous future boundary, synchronization, and coverage checks passed across all {num_valid_starts} sliding windows.")

    # =========================================================================
    # TEST 3: Edge Case - Sampling at the absolute end of the trajectory
    # =========================================================================
    def test_trajectory_end_edge_case(self):
        print("\n--- Test 3: Trajectory End Edge Case ---")
        
        dataset_w = HindsightDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='w',
            obs_config=self.obs_config, act_config=self.act_config,
            future_goal_mapping=self.future_goal_mapping
        )
        
        traj_len = 5
        obs, act = self.generate_temporally_predictable_data(traj_len)
        dataset_w.add_trajectory(obs, act)

        seq_len = 1
        dataset_r = HindsightDataset(
            data_path=self.data_path, data_dir=self.test_dir, io_mode='r',
            obs_config=self.obs_config, act_config=self.act_config,
            seq_length=seq_len, cross_trajectory=False, return_trj_last=False,
            future_goal_mapping=self.future_goal_mapping
        )

        # Max valid start_idx = (traj_len) - seq_len = 5 - 1 = 4
        # At start_idx = 4, end_idx = 5.
        # effective_traj_end is 4 (indices 0,1,2,3,4 are valid obs, 5 is pad).
        # In this edge case, end_idx > effective_traj_end, so the sampler MUST 
        # clamp to effective_traj_end (4) without throwing a ValueError.
        max_idx = len(dataset_r) - 1
        
        try:
            item = dataset_r[max_idx]
            sampled_t = int(np.mean(item['observation']['goal_rgb']))
            
            # The only valid fallback is clamping to the final valid observation
            self.assertEqual(sampled_t, 4, "Edge case clamping failed to select the final valid frame.")
            print("Edge case clamping successful.")
        except ValueError as e:
            self.fail(f"Sampler crashed at the end of the trajectory: {e}")

if __name__ == '__main__':
    unittest.main()