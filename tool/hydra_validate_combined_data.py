import hydra
from hydra import compose
from omegaconf import DictConfig, OmegaConf
import os
import sys
import numpy as np
from tqdm import tqdm

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from actoris_harena.utilities.trajectory_dataset import TrajectoryDataset

def get_time_steps(data_obj):
    """Safely extracts the temporal length whether the object is an array or a nested dictionary."""
    if isinstance(data_obj, dict):
        # If it's a dict (like bimanual actions), check the length of the first item inside it
        first_val = next(iter(data_obj.values()))
        return first_val.shape[0] if hasattr(first_val, 'shape') else len(first_val)
    elif hasattr(data_obj, 'shape'):
        return data_obj.shape[0]
    return len(data_obj)

@hydra.main(config_path="../conf", config_name="data_combination/combine_datasets", version_base=None)
def main(cfg: DictConfig):
    print("--- Dataset Validation Tool ---")
    print(f"Target Path: {cfg.target_data_path}")
    print("-------------------------------\n")

    try:
        dataset = TrajectoryDataset(
            data_path=cfg.target_data_path,
            data_dir=cfg.target_data_dir,
            io_mode='r',
            obs_config=OmegaConf.to_container(cfg.dataset.obs_config, resolve=True),
            act_config=OmegaConf.to_container(cfg.dataset.action_config, resolve=True),
            whole_trajectory=True
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    num_trajs = dataset.num_trajectories()
    print(f"Found {num_trajs} trajectories. Commencing audit...\n")

    if num_trajs == 0:
        print("❌ Dataset is empty!")
        return

    errors = []
    expected_image_keys = ['rgb', 'depth', 'mask', 'goal-rgb', 'goal-depth', 'goal-mask']

    for i in tqdm(range(num_trajs), desc="Validating Trajectories"):
        try:
            trajectory = dataset.get_trajectory(i)
            obs = trajectory.get('observation', {})
            act = trajectory.get('action', None)

            if not obs or act is None:
                errors.append(f"Traj {i}: Missing 'observation' or 'action' dict.")
                continue

            # 1. Temporal Consistency Check
            time_steps = None
            for key in expected_image_keys:
                if key in obs:
                    time_steps = get_time_steps(obs[key])
                    break
            
            if time_steps is not None:
                act_steps = get_time_steps(act)
                if act_steps != time_steps:
                    errors.append(f"Traj {i}: Length mismatch! Obs T={time_steps}, Act T={act_steps}")

            # 2. Image and Mask Checks
            for key in expected_image_keys:
                if key in obs:
                    data = obs[key]
                    
                    # If data is somehow a dict here, we extract the first array to check
                    if isinstance(data, dict):
                        data = next(iter(data.values()))
                        
                    if np.isnan(data).any() or np.isinf(data).any():
                        errors.append(f"Traj {i}: {key} contains NaN or Inf values.")

                    if 'mask' in key:
                        unique_vals = np.unique(data)
                        if data.dtype.kind == 'f':
                            if not np.all(np.isin(unique_vals, [0.0, 1.0])):
                                errors.append(f"Traj {i}: {key} has corrupted continuous values from resizing (e.g., {unique_vals[1:3]}).")
                        
            # 3. Success Key Checks
            if 'success' in obs:
                succ_data = obs['success']
                if isinstance(succ_data, dict):
                    succ_data = next(iter(succ_data.values()))
                if np.isnan(succ_data).any():
                    errors.append(f"Traj {i}: 'success' contains NaN.")

        except Exception as e:
            # We print the full traceback of the first error to the console so you can see exactly what failed
            if len(errors) == 0:
                import traceback
                print(f"\n[Detailed Error on Traj {i}]:")
                traceback.print_exc()
            errors.append(f"Traj {i}: Threw exception during read -> {str(e)}")

    # --- Summary Report ---
    print("\n\n" + "="*30)
    print("      VALIDATION REPORT")
    print("="*30)
    
    if len(errors) == 0:
        print("✅ SUCCESS! All trajectories passed soundness checks.")
        print(f"Total valid trajectories: {num_trajs}")
    else:
        print(f"❌ FAILED! Found {len(errors)} issues.")
        print("First 10 errors:")
        for err in errors[:10]:
            print(f"  - {err}")

if __name__ == '__main__':
    main()