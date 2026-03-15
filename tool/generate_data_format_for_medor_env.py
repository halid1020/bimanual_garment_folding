import os
import pickle
import numpy as np
from copy import deepcopy
from dotmap import DotMap

# Adjust these imports to match your actual module paths
import pyflex
from env.softgym_garment.multi_garment_vectorised_single_picker_pick_and_place_env \
    import MultiGarmentVectorisedSinglePickerPickAndPlaceEnv
from env.softgym_garment.tasks.garment_flattening \
    import GarmentFlatteningTask  

def generate_softgym_cache(env, output_path="converted_flat_states.pkl", num_episodes=None, scale_factor=0.5):
    """
    Steps through MultiGarmentEnv and extracts the data into the 
    SoftGym (AnyClothFlattenEnv) pickle format.
    """
    print(f"Starting conversion. Output will be saved to: {output_path}")
    print(f"Applying scale factor of: {scale_factor}")
    
    generated_configs = []
    generated_states = []
    
    # if num_episodes is None:
    #     num_episodes = env.get_num_episodes()
    env.set_eval()
    for i in range(30):
        # Reset the environment to the specific episode ID
        env.reset({'eid': i})
        
        old_config = env.init_state_params

        # Extract the integer ID from the filename (e.g., '07037_Tshirt.obj.pkl' -> 7037)
        instance_name = old_config.get('cloth_instance', '0_')
        real_cloth_id = int(instance_name.split('_')[0])

        # Number of actual vertices in the mesh
        num_particles = old_config['mesh_verts'].reshape(-1, 3).shape[0]

        # ---------------------------------------------------------
        # APPLY SCALE FACTOR TO GEOMETRY
        # ---------------------------------------------------------
        # 1. Scale the raw vertices in the configuration
        scaled_verts = old_config['mesh_verts'].reshape(-1, 3) * scale_factor
        
        # 2. Scale the area metrics (Area scales quadratically)
        area_scale = scale_factor ** 2
        scaled_flatten_area = env.flatten_coverage * area_scale
        scaled_init_area = env.init_coverae * area_scale

        # 1. Map Configuration Dictionary
        config = {
            'cloth_id': real_cloth_id,
            
            'v': scaled_verts,
            'f': old_config['mesh_faces'].reshape(-1, 3),
            
            # FIX 1: Reshape edges to Nx2
            'stretch_e': old_config['mesh_stretch_edges'].reshape(-1, 2),
            'bend_e': old_config['mesh_bend_edges'].reshape(-1, 2),
            'shear_e': old_config['mesh_shear_edges'].reshape(-1, 2),
            
            'mass': 0.0003,
            'radius': 0.005,
            # FIX 2: Ensure stiffness is a list
            'stiff': [1.5, 0.6, 1.0], 
            'damping': 1.0, 
            'dyn_fric': 1.0, 
            'particle_fric': 1.2, 
            'gravity': -9.8,
            'vel': 0.0,
            'rot': 0.0,
            
            'cloth_size': [-1, -1], 
            'cloth_type': 'Tshirt',
            
            'camera_name': 'default_camera',
            'camera_params': {
                'default_camera': {
                    'pos': env.camera_pos,
                    'angle': env.camera_angle,
                    'width': env.camera_size[0],
                    'height': env.camera_size[1]
                }
            },
            
            'flatten_area': scaled_flatten_area,
            'init_area': scaled_init_area
        }
        
        # 2. Map State Dictionary
        canon_poses = env.flattened_obs['observation']['particle_positions']
        # 3. Scale Canonical Poses
        canon_poses = np.expand_dims(canon_poses, axis=0) * scale_factor

        # FIX 3: Slice to remove padding, and flatten to 1D
        raw_pos = pyflex.get_positions().reshape(-1, 4)
        raw_vel = pyflex.get_velocities().reshape(-1, 3)
        
        # 4. Scale the XYZ coordinates of the live particle positions
        # Note: We must NOT scale the 4th column (inverse mass)!
        scaled_clean_pos = raw_pos[:num_particles, :].copy()
        scaled_clean_pos[:, :3] = scaled_clean_pos[:, :3] * scale_factor
        scaled_clean_pos = scaled_clean_pos.flatten()
        
        # 5. (Optional) Scale velocities if you want them mathematically consistent 
        # with the smaller spatial scale.
        scaled_clean_vel = raw_vel[:num_particles, :] * scale_factor
        scaled_clean_vel = scaled_clean_vel.flatten()
        
        state = {
            'config_id': i,
            'particle_pos': scaled_clean_pos,     
            'particle_vel': scaled_clean_vel,    
            'shape_pos': pyflex.get_shape_states(),
            'phase': pyflex.get_phases(),
            'camera_params': config['camera_params'],
            'canon_poses': canon_poses
        }

        # =====================================================================
        # NEW PRINT BLOCK: Garment & Data Format Details (For bridging verification)
        # =====================================================================
        print("\n" + "="*60)
        print(f" EXPORTING CONFIGURATION (Config ID: {i})")
        print("="*60)
        
        print("[Garment Geometry Information]")
        print(f"  Cloth Type:    {config.get('cloth_type', 'N/A')}")
        print(f"  Cloth ID:      {config.get('cloth_id', 'N/A')}")
        
        if config['v'].size > 0:
            print(f"  Vertices (v):  {config['v'].shape[0]} points -> Data Shape: {config['v'].shape}")
            print(f"  Faces (f):     {config['f'].shape[0]} triangles -> Data Shape: {config['f'].shape}")
            
            se_count = config['stretch_e'].shape[0] if config['stretch_e'].size > 0 else 0
            be_count = config['bend_e'].shape[0] if config['bend_e'].size > 0 else 0
            she_count = config['shear_e'].shape[0] if config['shear_e'].size > 0 else 0
            
            print(f"  Stretch Edges: {se_count} connections -> Data Shape: {config['stretch_e'].shape}")
            print(f"  Bend Edges:    {be_count} connections -> Data Shape: {config['bend_e'].shape}")
            print(f"  Shear Edges:   {she_count} connections -> Data Shape: {config['shear_e'].shape}")
            
        print("\n[Physics & Scene Properties]")
        print(f"  Particle Mass: {config.get('mass', 'N/A')}")
        print(f"  Radius:        {config.get('radius', 'N/A')}")
        print(f"  Friction:      Dynamic: {config.get('dyn_fric')}, Particle: {config.get('particle_fric')}")
        print(f"  Stiffness:     {config.get('stiff', 'N/A')} (Stretch, Bend, Shear)")
        print(f"  Gravity:       {config.get('gravity', 'N/A')}")
        print(f"  Applied Scale: {scale_factor}x")
        
        print("\n[PyFlex Live Data Formats]")
        print(f"  Particle Pos:  {state['particle_pos'].shape} (1D array. Reshapes to Nx4: x,y,z, 1/mass)")
        print(f"  Particle Vel:  {state['particle_vel'].shape} (1D array. Reshapes to Nx3: vx,vy,vz)")
        print(f"  Canon Poses:   {state['canon_poses'].shape} (Shape: K x N x 3 | K=ambiguity bins, N=particles)")
            
        print("\n[Task Metrics]")
        print(f"  Init Area:     {config.get('init_area', 0):.5f} (Scaled by {area_scale})")
        print(f"  Target Area:   {config.get('flatten_area', 0):.5f} (Scaled by {area_scale})")
        
        mask_h = config['camera_params']['default_camera']['height']
        mask_w = config['camera_params']['default_camera']['width']
        print(f"  Target Mask:   ({mask_h}, {mask_w}) (Computed at runtime by SoftGym)")
        print("="*60 + "\n")
        # =====================================================================

        generated_configs.append(deepcopy(config))
        generated_states.append(deepcopy(state))
        
    # 3. Save to Pickle
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((generated_configs, generated_states), f, protocol=4)
        
    print(f"Successfully saved {num_episodes} configurations and states to {output_path}!")

def main():
    # 1. Define Environment Configuration
    config_dict = {
        'mode': 'eval',
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
        'num_eval_trials': 30,
        'num_val_trials': 10,
        'num_train_trials': 200,
    }

    task_config = {
        'num_goals': 1,
        'garment_type': 'longsleeve',
        'asset_dir': 'assets',
        'name': 'flattening', 
        'debug': False,
    }
    
    print("Initialising Environment for Cache Generation...")
    env = MultiGarmentVectorisedSinglePickerPickAndPlaceEnv(DotMap(config_dict))
    task = GarmentFlatteningTask(DotMap(task_config))
    
    if hasattr(env, 'set_task'):
        env.set_task(task)
    else:
        env.task = task

    output_filename = f"./tmp/medor/{config_dict['garment_type']}_hard_v4.pkl"
    
    generate_softgym_cache(
        env=env, 
        output_path=output_filename, 
        num_episodes=None 
    )
    
    print("Finished.")
    env.close()

if __name__ == "__main__":
    main()