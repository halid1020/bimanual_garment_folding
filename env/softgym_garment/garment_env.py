import os
import h5py
import numpy as np
import cv2
import json
import pyflex
from actoris_harena import Arena
from tqdm import tqdm
from scipy.spatial.distance import cdist
import gym

from .action_primitives.picker import Picker
from .action_primitives.hybrid_action_primitive import HybridActionPrimitive
from .pixel_based_primitive_env_logger import PixelBasedPrimitiveEnvLogger
from .utils.env_utils import set_scene, get_coverage
from .utils.camera_utils import get_camera_matrix

global ENV_NUM
ENV_NUM = 0

class GarmentEnv(Arena):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.info = {}
        self.draw_fatten_contour = False
        self.all_infos = []
        self.sim_step = 0
        self.save_video = False
        self.debug = config.get("debug", False)
        self.add_final_goal_to_obs = config.get('add_final_goal_to_obs', False)
        
        self.random_reset = False
        self.set_id(0)
        self.name = config.name
        self.frame_resolution = config.get("frame_resolution", [256, 256])
        self.obs_resolution =  config.get("image_resolution", [128, 128])
        self.stop_on_success = config.get('stop_on_success', True)
        self.hard_shift_x = self.config.get('hard_shift_x', 0.0)
        
        # Softgym Setup
        self._get_sim_config()
        self._setup_camera()
        
        self.scene_config = self.default_config['scene_config']
        self.workspace_mask = None
        
        headless = not config.disp

        pyflex.init(
            headless, 
            True, 
            self.camera_config['cam_size'][0], 
            self.camera_config['cam_size'][1],
        )
        
        self.pickers = Picker(
            2, picker_radius=self.config.picker_radius, 
            particle_radius=self.scene_config['radius'],
            picker_threshold=self.config.picker_threshold,
            picker_low=self.config.picker_low, 
            picker_high=self.config.picker_high,
            grasp_mode=(self.config.grasp_mode if 'grasp_mode' in self.config.keys() else {'closest': 1.0}),
        )
        self.particle_radius = self.scene_config['radius']
        
        self.apply_workspace = config.get('apply_workspace', False)
        if self.apply_workspace:
            self.readjust_to_workspace = config.get('readjust_to_workspace', False)
            self.robot1_radius = config.robot1_radius
            self.robot0_radius = config.robot0_radius
            self.robot0_base = config.robot0_base
            self.robot1_base = config.robot1_base
            self._calculate_workspace_masks(self.camera_config['cam_size'])

        self.action_tool = HybridActionPrimitive(
            readjust_pick_poss=self.config.get('readjust_pick_poss', 0),
            apply_workspace=self.apply_workspace,
            drag_vel=0.01
        )
        
        self.save_each_action_picker_poses = True
        self.logger = PixelBasedPrimitiveEnvLogger()

        self._observation_image_shape = config.get('observation_image_shape', (480, 480, 3))
        self.init_state_path = config.init_state_path
        self.garment_type = self.config.get('garment_type', 'longsleeve')
        self._get_init_state_keys()

        self.evaluate_result = None
        self.track_semkey_on_frames = self.config.get('track_semkey_on_frames', False)
        self.init_mode = self.config.get('init_mode', 'crumpled')
        self.action_step = 0
        self.last_info = None
        self.action_horizon = self.config.action_horizon

        self.overstretch = 0
        
    def _get_garment_colour(self):
        """
        Extracts the median RGB colour of the garment using the rendered mask.
        Returns a list: [R, G, B]
        """
        rgb = self.info['observation']['rgb']
        mask = self.info['observation']['mask']
        cloth_pixels = rgb[mask]
        
        if len(cloth_pixels) > 0:
            return np.median(cloth_pixels, axis=0).astype(int).tolist()
        return [0, 0, 0]
    
    def _calculate_workspace_masks(self, resolution):
        """
        Computes the reachable workspace for robot arms on a 2D pixel grid.
        It un-projects the 2D camera pixels back into 3D space, finds where those 
        camera rays intersect the table plane (z=0), and then checks if those 3D 
        table points are within the physical reach radius of the robot bases.
        """
        W, H = resolution[0], resolution[1]

        # 1. Generate pixel grid and format as homogeneous 2D points (u, v, 1)
        u, v = np.arange(W), np.arange(H)
        uu, vv = np.meshgrid(u, v)
        # Note: vv (Y) is stacked first to match the pre-swapped intrinsic matrix
        pixels = np.stack([vv.ravel(), uu.ravel(), np.ones(H * W)], axis=1)

        # 2. Un-project: Multiply by Inverse Intrinsic Matrix to get 3D rays in Camera Space
        K_inv = np.linalg.inv(self.camera_intrinsic_matrix)
        rays_cam = (K_inv @ pixels.T).T
        rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True) # Normalize rays

        # 3. Transform rays from Camera Space to World Space
        T_world_cam = self.camera_extrinsic_matrix
        R, cam_origin = T_world_cam[:3, :3], T_world_cam[:3, 3]
        rays_world = (R @ rays_cam.T).T

        # 4. Ray-Plane Intersection: Find exactly where rays hit the table (z = 0)
        # Formula: origin.z + ray.z * s = 0  =>  s = -origin.z / ray.z
        dz = rays_world[:, 2]
        valid = np.abs(dz) > 1e-6 # Avoid division by zero for rays parallel to the table
        s = np.full(dz.shape, np.nan)
        s[valid] = -cam_origin[2] / dz[valid]
        world_points = cam_origin + rays_world * s[:, None]

        # 5. Measure planar distances (XY only) from each Robot Base to the Table Points
        robot0_dist = np.linalg.norm(world_points[:, :2] - np.asarray(self.robot0_base)[:2], axis=1)
        robot1_dist = np.linalg.norm(world_points[:, :2] - np.asarray(self.robot1_base)[:2], axis=1)

        # 6. Create binary masks based on minimum/maximum reach radiuses
        robot0_mask = (robot0_dist >= self.robot0_radius[0]) & (robot0_dist <= self.robot0_radius[1])
        robot1_mask = (robot1_dist >= self.robot1_radius[0]) & (robot1_dist <= self.robot1_radius[1])

        # 7. Reshape back to the 2D image dimension
        self.robot0_mask_full = robot0_mask.reshape(H, W)
        self.robot1_mask_full = robot1_mask.reshape(H, W)
        
        # Aliases
        self.robot0_mask = self.robot0_mask_full
        self.robot1_mask = self.robot1_mask_full

    def _setup_camera(self):
        self.default_camera = self.default_config['camera_name']
        self.camera_config = self.default_config['camera_params'][self.default_camera]
        self.camera_config['cam_position'][1] = self.config.get('camera_height', self.camera_config['cam_position'][1])
        
        if 'camera_resolution' in self.config:
            self.camera_config['cam_size'] = list(self.config.camera_resolution)
            
        if 'field_of_view' in self.config:
            fov_x_rad = self.config.field_of_view[0] * np.pi / 180.0
            fov_y_rad = self.config.field_of_view[1] * np.pi / 180.0
            self.camera_config['cam_fov'] = [fov_x_rad, fov_y_rad]

        camera_pos = self.camera_config['cam_position'].copy()
        camera_pos[1], camera_pos[2] = camera_pos[2], camera_pos[1]  # swap y and z
        self.camera_height = camera_pos[2]
        
        camera_angle = self.camera_config['cam_angle'].copy()
        camera_angle[1], camera_angle[2] = camera_angle[2], camera_angle[1]
        camera_angle[0] = np.pi + camera_angle[0]
        camera_angle[2] = 4*np.pi/2 - camera_angle[2]

        self.picker_initial_pos = self.config.picker_initial_pos
        self.camera_intrinsic_matrix, self.camera_extrinsic_matrix = get_camera_matrix(
            camera_pos, camera_angle, self.camera_config['cam_size'], self.camera_config['cam_fov'])
            
        self.camera_pos = camera_pos
        self.camera_angle = camera_angle
        self.camera_size = self.camera_config['cam_size']
        self.picker_poses = []

    def _get_sim_config(self):
        from .utils.env_utils import get_default_config
        
        # Extract variables from config, falling back to defaults if not present
        p_radius = self.config.get('particle_radius', 0.0175)
        c_stiffness = self.config.get('cloth_stiffness', [0.75, 0.02, 0.02])
        scale = self.config.get('scale', 0.8)
        c_distance = self.config.get('collisionDistance', 0.0006)
        
        self.default_config = get_default_config(
            particle_radius=p_radius,
            cloth_stiffness=c_stiffness,
            scale=scale,
            collision_distance=c_distance
        )

    def reset(self, episode_config=None):
        if episode_config is None:
            episode_config = {'eid': None, 'save_video': False}
        
        self.eid = episode_config['eid']
        self.save_video = episode_config.get('save_video', False)
        self.episode_config = episode_config

        init_state_params = self._get_init_state_params(self.eid)
        self.draw_fatten_contour = ('alignment' in self.task.name)
        self.sim_step = 0
        self.video_frames = []

        init_state_params['scene_config'] = self.scene_config
        init_state_params.update(self.default_config)
        set_scene(config=init_state_params, state=init_state_params)
        
        self.num_mesh_particles = int(len(init_state_params['mesh_verts'])/3)
        self.init_state_params = init_state_params

        self.pickers.reset(self.picker_initial_pos)
        self.info = {}
        self.last_info = None
        self.action_tool.reset(self)
        self._step_sim()
           
        self.last_flattened_step = -100
        self.flattened_obs = None
        self.get_flattened_obs()
        
        self.task.reset(self)
        self.action_step = 0
        self.evaluate_result = None
        self._initialise_trajectory()
        self.init_coverae = self._get_coverage()
        self.is_recording_low_level = False

        self.info = self._process_info({})
        self.clear_frames()
        self.picker_poses = []
        self.all_infos = [self.info]
        
        return self.info

    def set_to_flatten(self, re_process_info=False):
        pyflex.set_positions(self.episode_params['init_particle_pos'].flatten())
        self.wait_until_stable()
        self._get_particle_distance_matrix()

        self.info = self._process_info({}, task_related=re_process_info, flatten_obs=False)
        return self.info
    
    def set_to_canon_flatten(self, re_process_info=False):
        self.set_to_flatten()
        goal_particles = self.get_mesh_particles_positions()

        theta = np.pi
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        center = goal_particles.mean(axis=0)
        goal_particles -= center
        goal_particles = goal_particles @ rotation_matrix.T 

        displacement = [0, 0, 0]
        displacement[0] += self.config.get('hard_shift_x', 0.0)
        
        goal_particles += displacement
        self.set_mesh_particles_positions(goal_particles)
        self.wait_until_stable()

        self.info = self._process_info({}, task_related=re_process_info, flatten_obs=False)
        return self.info

    def set_to_random_flatten(self, re_process_info=False):
        self.set_to_flatten()
        goal_particles = self.get_mesh_particles_positions()

        rng = np.random.RandomState(self.eid)
        theta = rng.uniform(0, 2 * np.pi) 
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        center = goal_particles.mean(axis=0)
        goal_particles -= center
        goal_particles = goal_particles @ rotation_matrix.T 

        disp_range = self.config.get('random_displacement_range', [-0.3, 0.3])
        displacement = rng.uniform(disp_range[0], disp_range[1], size=3)
        displacement[2] = 0
        displacement[0] += self.config.get('hard_shift_x', 0.0)
        
        goal_particles += displacement
        self.set_mesh_particles_positions(goal_particles)
        self.wait_until_stable()

        self.info = self._process_info({}, task_related=re_process_info, flatten_obs=False)
        return self.info
    
    def _initialise_trajectory(self):
        if self.init_mode == 'flattened':
            particles = self.flattened_obs['observation']['particle_positions']
            particles += [self.config.get('hard_shift_x', 0.0), 0, 0]
            self.set_mesh_particles_positions(particles)
            self.wait_until_stable()
            self.last_flattened_step = 0
            
        elif self.init_mode == 'random_flattened':
            rng = np.random.RandomState(self.eid)
            theta = rng.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            goal_particles = self.flattened_obs['observation']['particle_positions']
            center = goal_particles.mean(axis=0)
            goal_particles -= center
            goal_particles = goal_particles @ rotation_matrix.T 

            displacement = rng.uniform(-0.3, 0.3, size=3)
            displacement[2] = 0
            goal_particles += displacement

            self.set_mesh_particles_positions(goal_particles)
            self.wait_until_stable()
            self.last_flattened_step = 0

    def get_episode_config(self):
        return self.episode_config
    
    def get_num_episodes(self):
        return 1 if self.mode in ['eval', 'train'] else NotImplementedError

    def get_info(self, new=False):
        if new:
            self.info = self._process_info({})
        return self.info
    
    def get_trajectory_infos(self):
        return self.all_infos
    
    def _process_info(self, info, task_related=True, flatten_obs=True):
        info.update({
            'observation': self._get_obs(flatten_obs),
            'arena': self,
            'arena_id': self.aid,
            'action_space': self.get_action_space(),
            'overstretch': self.overstretch,
            'sim_steps': self.sim_step,
            'draw_flatten_contour': self.draw_fatten_contour
        })
        
        info['observation']['low_level_mesh_particles'] = getattr(self, 'low_level_mesh_particles', [])
        info['observation']['low_level_visible_pcs'] = getattr(self, 'low_level_visible_pcs', [])
        
        if self.save_each_action_picker_poses and self.mode != 'train':
            if len(self.picker_poses) > 0:
                picker_poses = np.stack(self.picker_poses)
                picker_poses = picker_poses[:, :, [0, 2, 1]].reshape(-1, 3)
                picker_pixel_poses, _ = self.get_visibility(picker_poses)
                
                if hasattr(self, 'crop_size') and hasattr(self, 'x1'):
                    picker_pixels_crop = picker_pixel_poses - np.array([self.y1, self.x1])
                    norm_pixels = (picker_pixels_crop / self.crop_size) * 2.0 - 1.0
                else:
                    norm_pixels = picker_pixel_poses / np.array(self.camera_size) * 2 - 1
                
                info['observation']['picker_norm_pixel_pos'] = norm_pixels.reshape(-1, 2, 2)
            else: 
                info['observation']['picker_norm_pixel_pos'] = None

        if flatten_obs:
            info['flattened_obs'] = self.get_flattened_obs()
            for k, v in info['flattened_obs']['observation'].items():
                info['observation'][f'flattened-{k}'] = v

        is_truncated = self.action_step >= self.action_horizon
        is_terminated = False
        info['discount'] = 1.0 
        
        # Out of view threshold check
        cloth_pixel_count = np.sum(info['observation']['mask'])
        out_of_view = cloth_pixel_count < 10 
        info['out_of_view'] = out_of_view
        
        if out_of_view:
            is_terminated = True
        
        if task_related:
            info['evaluation'] = self.evaluate()
            if info['evaluation'].get('normalised_coverage', 0) > 0.9:
                self.last_flattened_step = self.action_step

            info['observation']['last_flattened_step'] = self.last_flattened_step
            info['success'] = self.success()
            
            if info['success'] and self.stop_on_success:
                is_terminated = True
                
            if info['evaluation']:
                info['reward'] = self.task.reward(self.last_info, None, info)
                for k, v in info['reward'].items():
                    info['evaluation'][f'reward/{k}'] = v
            
            goals = self.task.get_goals()
            if goals:
                goal_obs = goals[0][-1]['observation']
                info['goal'] = goal_obs.copy()
                info['goals'] = goals[0]
                if self.add_final_goal_to_obs:
                    for k, v in goal_obs.items():
                        info['observation'][f'goal_{k}'] = v
                        info['observation'][f'goal-{k}'] = v

        info['terminated'] = is_terminated
        info['truncated'] = is_truncated
        info['done'] = is_terminated or is_truncated

        return info
    
    def step(self, action): 
        self.last_info = self.info
        self.evaluate_result = None
        self.picker_poses = []
        self.low_level_mesh_particles = []
        self.low_level_visible_pcs = []
        self.is_recording_low_level = True 

        self.overstretch = 0
        self.sim_step = 0
        self.info = self.action_tool.step(self, action)
        
        self.action_step += 1
        self.all_infos.append(self.info)
        self.info = self._process_info(self.info)

        self.info['observation']['is_first'] = False
        self.info['observation']['is_terminal'] = self.info['done']

        if self.debug and self.video_frames:
            from actoris_harena.utilities.visual_utils import save_numpy_as_gif
            save_numpy_as_gif(self.video_frames, './tmp', 'debug-frames')
        
        return self.info
    
    @property
    def observation_space(self):
        spaces = {"image": gym.spaces.Box(0, 255, tuple(self.obs_resolution) + (3,), dtype=np.uint8)}
        return gym.spaces.Dict(spaces)

    def clear_frames(self):
        self.video_frames = []
    
    def get_action_space(self):
        return self.action_tool.get_action_space()
    
    def get_frames(self):
        return self.video_frames
    
    def _get_particle_distance_matrix(self):
        mesh_particles = self.get_mesh_particles_positions()
        self.particle_dist_matrix = cdist(mesh_particles, mesh_particles)

    def _compute_flattened_state(self, setup_method):
        """Helper to reduce duplication between flattened observation generators"""
        if self.flattened_obs is None:
            current_particle_pos = pyflex.get_positions()
            self.flattened_obs = setup_method()
            self.flatten_coverage = self._get_coverage()
            pyflex.set_positions(current_particle_pos)
            self.wait_until_stable()
        return self.flattened_obs

    def get_flattened_obs(self):
        return self._compute_flattened_state(self.set_to_flatten)
    
    def get_random_flattened_obs(self):
        return self._compute_flattened_state(self.set_to_random_flatten)

    def get_caon_flattened_obs(self):
        return self._compute_flattened_state(self.set_to_canon_flatten)

    def get_no_op(self):
        return self.action_tool.get_no_op()
    
    def set_disp(self, disp):
        pass

    def evaluate(self):
        if self.evaluate_result is None or self.action_step == 0:
            self.evaluate_result = self.task.evaluate(self)
        return self.evaluate_result
        
    def observation_shape(self):
        return {'rgb': self._observation_image_shape, 'depth': self._observation_image_shape}

    def sample_random_action(self):
        return self.action_tool.sample_random_action()

    def get_picker_position(self):
        p = self._get_picker_position()
        p[:, [1, 2]] = p[:, [2, 1]]
        return p
    
    def control_picker(self, signal, process_info=True):
        signal = signal[:, [0, 2, 1, 3]]
        new_picker_pos = self.pickers.step(signal, self)
        self._step_sim()
        
        if getattr(self, 'is_recording_low_level', True):
            self.picker_poses.append(new_picker_pos)
            
            mesh_pos = self.get_mesh_particles_positions()
            self.low_level_mesh_particles.append(mesh_pos.copy()) 
            
            _, visible_mask = self.get_visibility(mesh_pos)
            self.low_level_visible_pcs.append(mesh_pos[visible_mask].copy())

        info = {}
        if process_info:
            info = self._process_info_({'observation': self._get_obs()})
        return info
    
    def wait_until_stable(self, max_wait_step=200, stable_vel_threshold=0.0006, target_point=None, target_pos=None):
        stable_step = 0
        last_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        
        for _ in range(max_wait_step):
            cur_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            curr_vel = np.linalg.norm(cur_pos - last_pos, axis=1)
            
            if target_point is not None:
                cur_poss = pyflex.get_positions()
                curr_vell = pyflex.get_velocities()
                cur_poss[target_point * 4: target_point * 4 + 3] = target_pos
                curr_vell[target_point * 3: target_point * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(cur_poss.flatten())
                pyflex.set_velocities(curr_vell)
                curr_vel = curr_vell

            self._step_sim()
            
            if stable_step > 10:
                break
            if np.max(curr_vel) < stable_vel_threshold:
                stable_step += 1
            else:
                stable_step = 0

            last_pos = cur_pos

    def _get_coverage(self):
        particle_positions = self.get_mesh_particles_positions()
        particle_positions[:, [1, 2]] = particle_positions[:, [2, 1]]
        return get_coverage(particle_positions, self.particle_radius)

    def get_particle_positions(self):
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3].copy()
        pos[:, [1, 2]] = pos[:, [2, 1]]
        return pos
    
    def get_mesh_particles_positions(self):
        return self.get_particle_positions()[:self.num_mesh_particles]
    
    def get_algn_mesh_particles_positions(self):
        return self.flattened_obs['observation']['particle_positions']

    def set_particle_positions(self, particle_positions):
        particle_positions[:, [1, 2]] = particle_positions[:, [2, 1]]
        pos = pyflex.get_positions().reshape(-1, 4).copy()
        pos[:, :3] = particle_positions.copy()
        pyflex.set_positions(pos.flatten())
    
    def set_mesh_particles_positions(self, mesh_particle_positions):
        particle_positions = self.get_particle_positions()
        particle_positions[:self.num_mesh_particles] = mesh_particle_positions
        self.set_particle_positions(particle_positions)
    
    def _get_picker_pos(self):
        return self.pickers.get_picker_pos()
    
    def _get_picker_position(self):
        return self._get_picker_pos()[:, :3].copy()
        
    def _step_sim(self):
        pyflex.step()
        
        if self.save_video:
            rgb = self._render('rgb', background=True, resolution=self.frame_resolution)
            
            if self.track_semkey_on_frames and self.task.semkey2pid:
                particle_pos = self.get_mesh_particles_positions()
                semkey2pid = self.task.semkey2pid
                keypids = list(semkey2pid.values())
                keynames = list(semkey2pid.keys())

                key_particles = particle_pos[keypids]
                key_pixels, visibility = self.get_visibility(key_particles, resolution=self.frame_resolution)

                K = len(keynames)
                hsv_colors = [(int(i * 180 / K), 255, 255) for i in range(K)]
                bgr_colors = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2BGR)[0,0].tolist() for c in hsv_colors]

                debug_frame = rgb.copy()
                for (v, u), vis, name, color in zip(key_pixels, visibility, keynames, bgr_colors):
                    color = tuple(map(int, color))
                    cv2.circle(debug_frame, (int(u), int(v)), 4, color, -1)
                    cv2.putText(debug_frame, name, (int(u)+5, int(v)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                self.video_frames.append(debug_frame)
            else:
                self.video_frames.append(rgb)

        self.sim_step += 1

    def _process_info_(self, info):
        return info

    def _render(self, mode='rgb', camera_name='default_camera', resolution=None, background=False):
        if not background:
            img, depth_img = pyflex.render_cloth()
        else:
            img, depth_img = pyflex.render()

        CAMERA_WIDTH = self.camera_config['cam_size'][0]
        CAMERA_HEIGHT = self.camera_config['cam_size'][1]

        img = img.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 4)[::-1, :, :3]  
        depth_img = depth_img.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 1)[::-1, :, :1]
        
        if mode == 'rgbd': img = np.concatenate((img, depth_img), axis=2)
        elif mode == 'depth': img = depth_img
        elif mode == 'mask': img = img.sum(axis=2) > 0
        
        if resolution is not None and (CAMERA_HEIGHT != resolution[0] or CAMERA_WIDTH != resolution[1]):
            img = cv2.resize(img, resolution)

        return img
    
    def _get_obs(self, flatten_obs=True):
        obs = {}
        
        W_full, H_full = self.camera_size[0], self.camera_size[1]
        self.crop_size = min(H_full, W_full)
        
        offset_x = self.config.get('crop_offset_x', 0) 
        self.x1 = W_full // 2 - self.crop_size // 2 + offset_x
        self.y1 = H_full // 2 - self.crop_size // 2 
        x2, y2 = self.x1 + self.crop_size, self.y1 + self.crop_size

        rgbd_full = self._render(mode='rgbd') 
        self.cloth_mask_full = rgbd_full[:, :, :3].sum(axis=2) > 0
        
        rgbd_crop = rgbd_full[self.y1:y2, self.x1:x2]
        self.cloth_mask_crop = self.cloth_mask_full[self.y1:y2, self.x1:x2]
        
        if self.apply_workspace:
            self.robot0_mask_crop = self.robot0_mask_full[self.y1:y2, self.x1:x2]
            self.robot1_mask_crop = self.robot1_mask_full[self.y1:y2, self.x1:x2]

        rgbd_resized = cv2.resize(rgbd_crop, tuple(self.obs_resolution), interpolation=cv2.INTER_LINEAR)

        obs['rgb'] = obs['image'] = rgbd_resized[:, :, :3].astype(np.uint8)
        obs['depth'] = rgbd_resized[:, :, 3:]
        
        cloth_mask_uint8 = self.cloth_mask_crop.astype(np.uint8) * 255
        obs['mask'] = cv2.resize(cloth_mask_uint8, tuple(self.obs_resolution), interpolation=cv2.INTER_NEAREST) > 0
        
        if self.apply_workspace:
            r0_uint8 = self.robot0_mask_crop.astype(np.uint8) * 255
            r1_uint8 = self.robot1_mask_crop.astype(np.uint8) * 255
            obs['robot0_mask'] = cv2.resize(r0_uint8, tuple(self.obs_resolution), interpolation=cv2.INTER_NEAREST) > 0
            obs['robot1_mask'] = cv2.resize(r1_uint8, tuple(self.obs_resolution), interpolation=cv2.INTER_NEAREST) > 0

        obs['raw_rgb'] = rgbd_full[:, :, :3].astype(np.uint8)
        self.cloth_mask = self.cloth_mask_crop

        if self.debug:
            from actoris_harena.utilities.save_utils import save_mask 
            step_idx, ep_id = getattr(self, 'action_step', 0), getattr(self, 'eid', 'unknown')
            save_mask(mask=obs['mask'], filename=f"mask_ep{ep_id}_step{step_idx}", directory="tmp/debug_garment_env")
            
        obs['particle_positions'] = self.get_mesh_particles_positions()
        obs['semkey2pid'] = self.task.semkey2pid

        # if self.config.get('collect_control_data', False):
        _, visible_mask = self.get_visibility(obs['particle_positions'])
        obs['visible_point_cloud'] = obs['particle_positions'][visible_mask]
        if hasattr(self, 'init_state_params') and 'mesh_faces' in self.init_state_params:
            obs['mesh_faces'] = self.init_state_params['mesh_faces']
        
        if flatten_obs and obs['semkey2pid']:
            keypids = list(obs['semkey2pid'].values())
            
            # Current State Keys
            if self.config.get("provide_semkey_pos", False):
                obs['semkey_pos'] = obs['particle_positions'][keypids].astype(np.float32)
            
            # Flattened State Keys
            if self.config.get("provide_flattened_semkey_pos", False):
                obs['flattened_semkey_pos'] = self.flattened_obs['observation']['particle_positions'][keypids].astype(np.float32)

            # Normalised Pixel Keys (Current)
            if self.config.get("provide_semkey_norm_pixel", False):
                key_particles = obs['particle_positions'][keypids]                 
                key_pixels_full, _ = self.get_visibility(key_particles)  
                key_pixels_crop = key_pixels_full - np.array([self.y1, self.x1])
                obs['semkey_norm_pixel'] = ((key_pixels_crop / self.crop_size) * 2.0 - 1.0).flatten()
                obs['key_pixels'] = key_pixels_crop

            # Normalised Pixel Keys (Flattened)
            if self.config.get("provide_flattened_semkey_norm_pixel", False):
                flat_particles = self.flattened_obs['observation']['particle_positions'][keypids]                 
                flat_pixels_full, _ = self.get_visibility(flat_particles)  
                flat_pixels_crop = flat_pixels_full - np.array([self.y1, self.x1])
                obs['flattened_semkey_norm_pixel'] = ((flat_pixels_crop / self.crop_size) * 2.0 - 1.0).flatten()

        obs['action_step'] = self.action_step
        return obs

    def _get_cloth_mask(self, camera_name='default_camera', resolution=None, crop=True):
        rgb = self._render(camera_name=camera_name, mode='rgb', resolution=resolution)
        mask = rgb.sum(axis=2) > 0
        if crop and hasattr(self, 'x1'):
            mask = mask[self.y1:self.y1+self.crop_size, self.x1:self.x1+self.crop_size]
        return mask
    
    def _get_init_keys_helper(self, hdf5_path, key_file, difficulties=['hard', 'easy']):
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return json.load(f)
        else:
            with h5py.File(hdf5_path, 'r') as tasks:
                eval_keys = [key for key in tasks if tasks[key].attrs['task_difficulty'] in difficulties]
            keys = []
            for key in tqdm(eval_keys, desc='Filtering keys'):
                with h5py.File(hdf5_path, 'r') as tasks:
                    group = tasks[key]
                    episode_params = dict(group.attrs)
                    for dataset_name in group.keys():
                        episode_params[dataset_name] = group[dataset_name][()]
                    
                    episode_params['scene_config'] = self.scene_config
                    episode_params.update(self.default_config)
                    set_scene(config=episode_params, state=episode_params)
                    
                    pyflex.step()
                    cloth_mask = self._get_cloth_mask()
                    if self.workspace_mask is not None:
                        cloth_mask = cloth_mask & self.workspace_mask
                    if cloth_mask.sum() > 500:
                        keys.append(key)
                        
            with open(key_file, 'w') as f:
                json.dump(keys, f)
        return keys

    def _get_init_state_keys(self):
        path = os.path.join(self.init_state_path, f'multi-{self.garment_type}-eval.hdf5')
        key_file = os.path.join(self.init_state_path, f'{self.name}-eval.json')
        self.keys = self._get_init_keys_helper(path, key_file, difficulties=['hard'])
        self.num_trials = 1 

    def _get_init_state_params(self, eid):
        raise NotImplementedError
    
    def get_visibility(self, particle_positions, resolution=None):
        """
        Takes 3D particles in the environment and computes their (Y, X) pixel 
        coordinates on the camera sensor, returning a boolean mask identifying 
        which points are actually visible (i.e. not occluded by cloth overlaps).
        """
        assert particle_positions.shape[1] == 3, "particle_positions must be (N, 3)"
        N = particle_positions.shape[0]

        # 1. Transform World Points to Camera Frame (Multiply by Inverse Extrinsic)
        particles_world_h = np.hstack([particle_positions, np.ones((N, 1), dtype=np.float32)])
        T_cam_world = np.linalg.inv(self.camera_extrinsic_matrix)
        cam_pts = (T_cam_world @ particles_world_h.T).T[:, :3]

        # 2. Camera Frame to Image Plane (Multiply by Intrinsic Matrix)
        if resolution is not None:
            camera_intrinsic_matrix, _ = get_camera_matrix(
                self.camera_pos, self.camera_angle, resolution, self.camera_config['cam_fov'])
        else:
            camera_intrinsic_matrix = self.camera_intrinsic_matrix
            
        proj = (camera_intrinsic_matrix @ cam_pts.T).T  
        z = proj[:, 2]
        z_safe = np.where(z == 0, 1e-6, z)
        
        # 3. Divide by Depth (Z) to get final pixel coordinates
        pixels = proj[:, :2] / z_safe[:, None]  
        W, H = resolution if resolution is not None else self.camera_size             

        # Swapped Intrinsic K means index 0 is V (Y-axis) and index 1 is U (X-axis)
        v, u = pixels[:, 0], pixels[:, 1]

        # 4. Filter 1: Frustum Check (Is it behind the camera or off screen?)
        visible = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # 5. Filter 2: Occlusion Check (Depth Buffer Comparison)
        # Fetch the rendered z-buffer (what the camera actually sees right now)
        depth_img = self._render('depth', resolution=resolution).reshape(H, W)
        u_int = np.clip(np.round(u).astype(int), 0, W - 1)
        v_int = np.clip(np.round(v).astype(int), 0, H - 1)
        depth_buffer = depth_img[v_int, u_int]

        # --- FIX 1: Use planar Z-depth (from step 2) instead of radial Euclidean distance ---
        particle_depth = z

        # --- FIX 2: Relax the epsilon tolerance to 0.015 to fix self-occlusion / Z-fighting ---
        eps = 0.015
        
        # If the front of the particle is <= the depth buffer, it is visible!
        visible &= (particle_depth - self.particle_radius) <= (depth_buffer + eps)

        return pixels, visible
    
    def get_cloth_area(self):
        return self.episode_params['cloth_height'] * self.episode_params['cloth_width']

    def compare(self, results_1, results_2):
        return self.task.compare(results_1, results_2)

    def close(self):
        """
        Shuts down the PyFlex simulation, freeing up GPU/CPU memory.
        """
        pyflex.clean()