import os
import time
import numpy as np
from dotmap import DotMap
import cv2

# Adjust this import to match your actual module path
from env.softgym_garment.multi_garment_vectorised_single_picker_pick_and_place_env \
    import MultiGarmentVectorisedSinglePickerPickAndPlaceEnv
from env.softgym_garment.tasks.garment_flattening \
    import GarmentFlatteningTask  

from actoris_harena import Agent

def resample_sequence(sequence, num_samples=100):
    """
    Subsamples or duplicates instances in a sequence to yield exactly `num_samples` items.
    Safe for both numpy arrays and lists of variable-length numpy arrays.
    """
    if sequence is None or len(sequence) == 0:
        return sequence
        
    T = len(sequence)
    if T == num_samples:
        return sequence
        
    # Generate exactly `num_samples` indices spanning 0 to T-1
    indices = np.linspace(0, T - 1, num_samples).astype(int)
    
    if isinstance(sequence, np.ndarray):
        return sequence[indices]
    
    # If it's a list (like variable-length point clouds), extract by index
    return [sequence[i] for i in indices]


class SinglePickerRandomAgent(Agent):
    """Adjusted Random Agent for a 4D action space (Single Picker)"""
    def __init__(self, config):
        super().__init__(config)
        self.name = "random-single-pixel-pick-and-place"

    def act(self, info_list, update=False):
        return [self.single_act(info) for info in info_list]

    def single_act(self, state, update=False):
        mask = state['observation']['mask']
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        H, W = mask.shape
        mask_coords = np.argwhere(mask > 0)

        def random_norm_xy():
            return np.random.uniform(-1, 1, size=2)

        if len(mask_coords) == 0:
            pick0 = random_norm_xy()
        else:
            p0 = mask_coords[np.random.randint(len(mask_coords))]
            pick0 = np.array([
                p0[0] / W * 2 - 1,
                p0[1] / H * 2 - 1
            ], dtype=np.float32)

        place0 = random_norm_xy()

        action = np.concatenate([pick0, place0], axis=0)
        return action

    def init(self, state): pass
    def update(self, state, action): pass

def main():
    config_dict = {
        'garment_type': 'longsleeve',
        'name': 'multi-garment-vectorised-single-picker-pick-and-place-env',
        'picker_radius': 0.03,
        'picker_threshold': 0.007,
        'picker_low': [-5, 0, -5],
        'picker_high': [5, 5, 5],
        'grasp_mode': {'around': 1.0},
        'picker_initial_pos': [[0.7, 0.2, 0.7], [-0.7, 0.2, 0.7]],
        'init_state_path': 'assets/init_states',
        'disp': False,
        'ray_id': 0,
        'action_horizon': 30,
        'image_resolution': [480, 480],
        'track_semkey_on_frames': False,
        'num_val_trials': 10,
        'readjust_pick_poss': 1.0,
        'provide_semkey_pos': False,
        'provide_flattened_semkey_pos': False,
        'provide_semkey_norm_pixel': True,
        'provide_flattened_semkey_norm_pixel': True,
        'stop_on_success': False,
        'add_final_goal_to_obs': True,
        'num_train_trials': 200,
        'collect_control_data': True 
    }

    task_config = {
        'num_goals': 1,
        'garment_type': 'longsleeve',
        'asset_dir': 'assets',
        'name': 'flattening', 
        'debug': False,
        'alignment': 'simple_rigid',
        'overstretch_penalty_scale': 1,
        'overstretch_penalty_threshold': 0.1,
        'affordance_penalty_scale': 0,
        'big_success_bonus': False
    }
    
    print("Initialising Environment...")
    env = MultiGarmentVectorisedSinglePickerPickAndPlaceEnv(DotMap(config_dict))
    task = GarmentFlatteningTask(DotMap(task_config))
    
    if hasattr(env, 'set_task'):
        env.set_task(task)
    else:
        env.task = task
        
    agent = SinglePickerRandomAgent(DotMap({}))

    save_dir = "./tmp/control_data_output"
    os.makedirs(save_dir, exist_ok=True)

    info = env.reset()
    
    num_steps = 100 
    print(f"Starting data collection for {num_steps} trajectories...")

    for step in range(num_steps):
        # 1. Get Action & Step
        action_batch = agent.act([info])
        action = action_batch[0]
        
        info = env.step(action)
        obs = info['observation']
        
        # 2. Extract Data
        rgb_img = obs.get('rgb')
        depth_img = obs.get('depth')

        if rgb_img is not None:
            # Create a separate folder for images to keep things clean
            img_debug_dir = os.path.join(save_dir, "debug_images")
            os.makedirs(img_debug_dir, exist_ok=True)
            
            img_path = os.path.join(img_debug_dir, f"step_{step:03d}.png")
            
            # Convert RGB (Env) to BGR (OpenCV)
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # Save to disk
            cv2.imwrite(img_path, bgr_img)
        
        # Snapshot at the end of the action
        particle_positions = obs.get('particle_positions')
        visible_pc = obs.get('visible_point_cloud', np.array([]))
        
        # Extract low-level trajectory buffers
        raw_ll_traj = obs.get('picker_norm_pixel_pos', np.array([]))
        raw_ll_mesh = obs.get('low_level_mesh_particles', [])
        raw_ll_vpc  = obs.get('low_level_visible_pcs', [])
        
        # 3. Resample to exactly 100 instances
        resampled_traj = resample_sequence(raw_ll_traj, num_samples=100)
        resampled_ll_mesh = resample_sequence(np.array(raw_ll_mesh), num_samples=100)
        
        # For ragged lists (visible pc has varying shapes), keep as object array
        resampled_ll_vpc = resample_sequence(raw_ll_vpc, num_samples=100)
        if isinstance(resampled_ll_vpc, list):
            resampled_ll_vpc_array = np.empty(len(resampled_ll_vpc), dtype=object)
            resampled_ll_vpc_array[:] = resampled_ll_vpc
        else:
            resampled_ll_vpc_array = resampled_ll_vpc
        
        # 4. Save and Monitor Time
        save_path = os.path.join(save_dir, f"step_{step:03d}.npz")
        start_time = time.perf_counter()
        
        np.savez_compressed(
            save_path,
            rgb=rgb_img,
            depth=depth_img,
            particle_positions=particle_positions,
            visible_point_cloud=visible_pc,
            action_applied=action,
            low_level_trajectory=resampled_traj,
            low_level_mesh_particles=resampled_ll_mesh,
            low_level_visible_pcs=resampled_ll_vpc_array,
            raw_trajectory_length=len(raw_ll_traj) if raw_ll_traj is not None else 0
        )
        end_time = time.perf_counter()
        
        save_duration = end_time - start_time
        print(f"[{step+1}/{num_steps}] Saved data to {save_path} in {save_duration:.4f} seconds.")

        # Calculate shapes for the summary
        traj_shape = resampled_traj.shape if resampled_traj is not None else "None"
        mesh_shape = resampled_ll_mesh.shape if len(resampled_ll_mesh) > 0 else "None"
        
        # For variable-length point clouds, we show the count and the shape of the first sample
        vpc_count = len(resampled_ll_vpc)
        first_vpc_shape = resampled_ll_vpc[0].shape if vpc_count > 0 else "N/A"

        print("-" * 50)
        print(f"[{step+1}/{num_steps}] TRAJECTORY SAVED: {save_path}")
        print(f"  > Time Taken:  {save_duration:.4f}s")
        print(f"  > RGB-D:       {rgb_img.shape} | {depth_img.shape}")
        print(f"  > Action Vec:  {action.shape}")
        print(f"  > LL Traj:     {traj_shape} (Picker Poses)")
        print(f"  > LL Mesh:     {mesh_shape} (Full Garment)")
        print(f"  > LL Vis PC:   {vpc_count} frames (First frame shape: {first_vpc_shape})")
        print("-" * 50)
        
        if info.get('done', False):
            info = env.reset()

    print("Data collection complete.")
    env.close()

if __name__ == "__main__":
    main()