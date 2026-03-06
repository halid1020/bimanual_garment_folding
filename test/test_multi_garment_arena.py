import os
import cv2
import numpy as np

# Adjust this import to match your project structure
from env.softgym_garment.multi_garment_env import MultiGarmentEnv

class MockConfig(dict):
    """
    A simple dictionary wrapper that allows dot notation access 
    to mimic Hydra's DictConfig behavior.
    """
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value

class DummyTask:
    """
    A minimal mock task to prevent AttributeErrors during env.reset(), 
    since reset() accesses self.task.name and self.task.semkey2pid.
    """
    name = "dummy_task"
    semkey2pid = {}
    
    def reset(self, env): pass
    def evaluate(self, env): return {}
    def get_goals(self): return []
    def reward(self, last_info, action, info): return {}
    def success(self, arena): return False

def main():
    # 1. Setup output directory
    output_dir = "./tmp/test_arena_cleanup"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Define the garments you want to iterate through
    garments_to_test = ['trousers', 'longsleeve'] 

    for garment in garments_to_test:
        print(f"\n{'='*40}")
        print(f"=== Testing Initialization for: {garment.upper()} ===")
        print(f"{'='*40}")

        # 3. Create a mock configuration
        # IMPORTANT: Update 'init_state_path' to point to your actual HDF5 data directory.
        cfg = MockConfig({
            'garment_type': garment,
            'init_state_path': '../assets/init_states', 
            'disp': False,  # Headless mode
            'picker_radius': 0.05,
            'picker_threshold': 0.005,
            'picker_low': [-1.0, 0.0, -1.0],
            'picker_high': [1.0, 1.0, 1.0],
            'picker_initial_pos': [[0.0, 0.5, 0.0], [0.1, 0.5, 0.0]],
            'action_horizon': 20,
            'track_semkey_on_frames': False,
            'num_eval_trials': 1,
            'num_train_trials': 1,
            'num_val_trials': 1,
            'observation_image_shape': (128, 128, 3),
            'frame_resolution': [128, 128],
            'image_resolution': [128, 128],
            'stop_on_success': False,
            'apply_workspace': False,
        })

        try:
            # 4. Initialize Environment
            print("[1/5] Building environment...")
            env = MultiGarmentEnv(cfg)
            env.mode = 'eval' 
            
            # Attach dummy task to satisfy env.reset() logic
            env.task = DummyTask()
            
            # 5. Reset to trigger PyFlex geometry loading
            print("[2/5] Resetting environment to load initial state...")
            info = env.reset({'eid': 0, 'save_video': False})
            
            # 6. Extract and save the RGB observation
            print("[3/5] Extracting observation...")
            rgb_obs = info['observation']['rgb']
            
            save_path = os.path.join(output_dir, f"init_obs_{garment}.png")
            # Convert RGB (PyFlex/Gym standard) to BGR (OpenCV standard) for saving
            bgr_obs = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_obs)
            print(f"[4/5] Successfully saved initial observation to: {save_path}")
            
        except Exception as e:
            print(f"ERROR during execution for {garment}: {e}")
            
        finally:
            # 7. Test the teardown process
            print("[5/5] Closing environment and destroying PyFlex instance...")
            if 'env' in locals():
                env.close() 
            print("Cleanup complete.")

if __name__ == "__main__":
    main()