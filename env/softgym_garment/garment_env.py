import os
import h5py
import numpy as np
import cv2
import json


# from softgym.utils.env_utils import get_coverage
import pyflex
from agent_arena import Arena
from tqdm import tqdm
from scipy.spatial.distance import cdist
import gym

from .action_primitives.picker import Picker
from .action_primitives.hybrid_action_primitive import HybridActionPrimitive
# from ..video_logger import VideoLogger
from .pixel_based_primitive_env_logger import PixelBasedPrimitiveEnvLogger
from .utils.env_utils import set_scene
from .utils.camera_utils import get_camera_matrix

global ENV_NUM
ENV_NUM = 0

def get_coverage(positions, particle_radius, resolution=500):
    """
    Fast approximate coverage estimation using vectorized rasterization.
    Computes a binary mask of covered cells on a 2D grid.
    """
    pos2d = positions[:, [0, 2]]

    min_x, max_x = np.min(pos2d[:, 0]), np.max(pos2d[:, 0])
    min_y, max_y = np.min(pos2d[:, 1]), np.max(pos2d[:, 1])

    # Slightly pad the area to include edge particles
    pad = particle_radius * 1.1
    min_x -= pad; max_x += pad
    min_y -= pad; max_y += pad

    grid_x = np.linspace(min_x, max_x, resolution)
    grid_y = np.linspace(min_y, max_y, resolution)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    cell_area = dx * dy

    mask = np.zeros((resolution, resolution), dtype=bool)

    # Convert particle positions to grid coordinates once
    gx = ((pos2d[:, 0] - min_x) / (max_x - min_x) * (resolution - 1)).astype(int)
    gy = ((pos2d[:, 1] - min_y) / (max_y - min_y) * (resolution - 1)).astype(int)
    r_pix = int(np.ceil(particle_radius / dx))  # particle radius in pixels

    # Only update local neighborhoods
    for px, py in zip(gx, gy):
        x_low = max(px - r_pix, 0)
        x_high = min(px + r_pix + 1, resolution)
        y_low = max(py - r_pix, 0)
        y_high = min(py + r_pix + 1, resolution)

        sub_x = np.arange(x_low, x_high)
        sub_y = np.arange(y_low, y_high)
        sx, sy = np.meshgrid(sub_x, sub_y, indexing='ij')

        dist2 = (sx - px)**2 + (sy - py)**2
        mask[x_low:x_high, y_low:y_high] |= dist2 <= r_pix**2

    return np.sum(mask) * cell_area

# @ray.remote
class GarmentEnv(Arena):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.info = {}
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
            self.readjust_to_workspace =  config.get('readjust_to_workspace', False) # <--- Store flag
            self.robot1_radius = config.robot1_radius
            self.robot0_radius = config.robot0_radius
            self.robot0_base = config.robot0_base
            self.robot1_base = config.robot1_base
            self._calculate_workspace_masks(self.camera_config['cam_size'])

        self.action_tool = HybridActionPrimitive(
            readjust_pick_poss=self.config.get('readjust_pick_poss', 0),
            apply_workspace=self.apply_workspace,
            drag_vel=0.01)
        self.save_each_action_picker_poses = True
        self.logger = PixelBasedPrimitiveEnvLogger()

        self._observation_image_shape = config.observation_image_shape \
            if 'observation_image_shape' in config else (480, 480, 3)

        self.init_state_path = config.init_state_path
        self.garment_type = self.config.get('garment_type', 'longsleeve')
        self._get_init_state_keys()

        self.evaluate_result = None
        self.track_semkey_on_frames = self.config.track_semkey_on_frames
        self.init_mode = self.config.get('init_mode', 'crumpled')
        self.action_step = 0
        self.last_info = None
        self.action_horizon = self.config.action_horizon

        self.overstretch = 0

        
    
    def _calculate_workspace_masks(self, resolution):
        """
        Compute workspace masks for robot0 and robot1.
        Pixels are projected to world coordinates (z=0 plane),
        then distances to robot bases are computed in the XY plane.
        """
        H, W = resolution

        # ---- 1. Generate pixel grid ----
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(H * W)], axis=1)  # (N, 3)

        # ---- 2. Pixel → Camera ray ----
        K_inv = np.linalg.inv(self.camera_intrinsic_matrix)
        rays_cam = (K_inv @ pixels.T).T
        rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)

        # ---- 3. Camera → World ----
        T_world_cam = self.camera_extrinsic_matrix
        R = T_world_cam[:3, :3]
        cam_origin = T_world_cam[:3, 3]

        rays_world = (R @ rays_cam.T).T

        # ---- 4. Ray–plane intersection (z = 0) ----
        dz = rays_world[:, 2]
        valid = np.abs(dz) > 1e-6

        s = -cam_origin[2] / dz
        s[~valid] = np.nan

        world_points = cam_origin + rays_world * s[:, None]  # (N, 3)

        # ---- 5. World → Robot frame (translation only) ----
        robot0_base = np.asarray(self.robot0_base)
        robot1_base = np.asarray(self.robot1_base)

        robot0_pts = world_points - robot0_base
        robot1_pts = world_points - robot1_base

        # ---- 6. Planar distances ----
        robot0_dist = np.linalg.norm(robot0_pts[:, :2], axis=1)
        robot1_dist = np.linalg.norm(robot1_pts[:, :2], axis=1)

        # ---- 7. Workspace masks ----
        robot0_mask = (
            (robot0_dist >= self.robot0_radius[0]) &
            (robot0_dist <= self.robot0_radius[1])
        )

        robot1_mask = (
            (robot1_dist >= self.robot1_radius[0]) &
            (robot1_dist <= self.robot1_radius[1])
        )

        # ---- 8. Reshape ----
        self.robot0_mask = robot0_mask.reshape(H, W)
        self.robot1_mask = robot1_mask.reshape(H, W)


    def _setup_camera(self):
        
        self.default_camera = self.default_config['camera_name']
        self.camera_config = self.default_config['camera_params'][self.default_camera]
        camera_pos = self.camera_config['cam_position'].copy()
        # swap y and z
        camera_pos[1], camera_pos[2] = camera_pos[2], camera_pos[1]
        #print('camera_pos', camera_pos)
        self.camera_height = camera_pos[2]
        camera_angle = self.camera_config['cam_angle'].copy()
        camera_angle[1], camera_angle[2] = camera_angle[2], camera_angle[1]
        camera_angle[0] = np.pi + camera_angle[0]
        camera_angle[2] = 4*np.pi/2 - camera_angle[2]
        #print('camera_angle', camera_angle)
        self.picker_initial_pos = self.config.picker_initial_pos
        self.camera_intrinsic_matrix, self.camera_extrinsic_matrix = \
            get_camera_matrix(
                camera_pos, 
                camera_angle, 
                self.camera_config['cam_size'], 
                self.camera_config['cam_fov'])
        self.camera_pos = camera_pos
        self.camera_angle = camera_angle
        
        self.camera_size = self.camera_config['cam_size']
        self.picker_poses = []

    def _get_sim_config(self):
        from .utils.env_utils import get_default_config
        self.default_config = get_default_config()



    ## TODO: if eid is out of range, we need to raise an error.   
    def reset(self, episode_config=None):
        if episode_config == None:
            episode_config = {
                'eid': None,
                'save_video': False
            }
        if 'save_video' not in episode_config:
            episode_config['save_video'] = False
        
        episode_config['eid'] = episode_config['eid']
        self.eid = episode_config['eid']
        init_state_params = self._get_init_state_params(episode_config['eid'])



        self.sim_step = 0
        self.video_frames = []
        self.save_video = episode_config['save_video']

        self.episode_config = episode_config

        init_state_params['scene_config'] = self.scene_config
        init_state_params.update(self.default_config)
        set_scene(
            config=init_state_params, 
            state=init_state_params)
        self.num_mesh_particles = int(len(init_state_params['mesh_verts'])/3)
        self.init_state_params = init_state_params

        
        #print('set scene done')
        #print('pciker initial pos', self.picker_initial_pos)
        self.pickers.reset(self.picker_initial_pos)
        #print('picker reset done')

        self.init_coverae = self._get_coverage()
        self.flattened_obs = None
        self.get_flattened_obs()
        #self.flatten_coverage = init_state_params['flatten_area']
        
        self.info = {}
        self.last_info = None
        self.action_tool.reset(self) # get out of camera view, and open the gripper
        self._step_sim()
        
        self.last_flattened_step = -100
        self.task.reset(self)
        self.action_step = 0

        self.evaluate_result = None
        
        self._initialise_trajectory()

        self.info = self._process_info()
        self.clear_frames()

        self.picker_poses = []

        self.all_infos = [self.info]
        
        return self.info
    
    def _initialise_trajecotry(self):
        if self.init_mode == 'flattened':
            self.set_to_flatten()
            self.last_flattened_step = 0
        elif self.init_mode == 'random_flattened':
            rng = np.random.RandomState(self.eid)
            theta = rng.uniform(0, 2 * np.pi)  # random angle in radians
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ])
            goal_particles = self.flattened_obs['observation']['particle_positions']

            center = goal_particles.mean(axis=0)
            goal_particles -= center
            
            goal_particles = goal_particles @ rotation_matrix.T  # rotate

            # Random displacement within ±0.5 range per axis
            displacement = rng.uniform(-0.5, 0.5, size=3)
            displacement[2] = 0
            goal_particles += displacement

            self.set_mesh_particles_positions(goal_particles)
            self.wait_until_stable()
            self.last_flattened_step = 0
        elif self.init_mode == 'crumpled':
            pass
    
    def get_episode_config(self):
        return self.episode_config
    
   

    def get_num_episodes(self):
        if self.mode == 'eval':
            return 1
        elif self.mode == 'train':
            return 1
        else:
            raise NotImplementedError

    def get_info(self, new=False):
        if new:
            self.info = self._process_info({})
            return self.info
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
        })
        
        if self.save_each_action_picker_poses and self.mode != 'train':
            #print('save pixel poses!!!', len(self.picker_poses))
            if len(self.picker_poses) > 0:
                picker_poses = np.stack(self.picker_poses) #T, 2, 3
                picker_poses = picker_poses[:, :, [0, 2, 1]].reshape(-1, 3)
                picker_pixel_poses, _ = self.get_visibility(picker_poses)
                H, W = self.camera_size
                norm_pixels = picker_pixel_poses/np.array([H, W]) * 2 - 1
                
                info['observation']['picker_norm_pixel_pos'] = norm_pixels.reshape(-1, 2, 2)
            else: 
                info['observation']['picker_norm_pixel_pos'] = None

        if flatten_obs:
            info['flattened_obs'] = self.get_flattened_obs()

            for k, v in info['flattened_obs']['observation'].items():
                info['observation'][f'flattened-{k}'] = v

        info['done'] = self.action_step >= self.action_horizon
        info['discount'] = 1.0 # For dreamer
        
        if task_related:
            info['evaluation'] = self.evaluate()
            if info['evaluation'].get('normalised_coverage', 0) > 0.9:
                self.last_flattened_step = self.action_step

            
           
            info['observation']['last_flattened_step'] = self.last_flattened_step
            
            info['success'] =  self.success()
            if info['success'] and self.stop_on_success:
                info['done'] = True
            
            #print('ev', info['evaluation'])
            if info['evaluation'] != {}:
                #print('self.last_info', self.last_info)
                #print(info['evaluation'])
                info['reward'] = self.task.reward(self.last_info, None, info)

                for k, v in info['reward'].items():
                    info['evaluation'][f'reward/{k}'] = v
            

            goals = self.task.get_goals()
            #print('len goals', len(goals))
            if len(goals) > 0:
                goal = goals[0]
                info['goal'] = {}
                for k, v in goal[-1]['observation'].items():
                    info['goal'][k] = v
                info['goals'] = goals[0]
            
            if self.add_final_goal_to_obs:
                for k, v in goal[-1]['observation'].items():
                    info['observation'][f'goal_{k}'] = v
                    info['observation'][f'goal-{k}'] = v

        return info
    
    def step(self, action): ## get action for hybrid action primitive, action defined in the observation space
        self.last_info = self.info
        self.evaluate_result = None
        self.picker_poses = []
        #print('action step', self.action_step)
        self.overstretch = 0
        self.sim_step = 0
        self.info = self.action_tool.step(self, action)
        #print('applied aciton', info['applied_action'])
        
        self.action_step += 1
        self.all_infos.append(self.info)
        self.info = self._process_info(self.info)

        self.info['observation']['is_first'] = False
        self.info['observation']['is_terminal'] = self.info['done']

        if self.debug and len(self.video_frames) > 0:
            #print('[GarmentEnv] debug!')
            # save gif to a directory
            from agent_arena.utilities.visual_utils import save_numpy_as_gif
            save_numpy_as_gif(self.video_frames, './tmp', 'debug-frames')
        
        #print('reward', info['reward'])
        return self.info
    
    @property
    def observation_space(self):
        spaces = {}
        spaces["image"] = gym.spaces.Box(0, 255, tuple(self.obs_resolution) + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    def clear_frames(self):
        self.video_frames = []
    
    def get_action_space(self):
        return self.action_tool.get_action_space()
    
    def get_frames(self):
        return self.video_frames
    
    def _get_particle_distance_matrix(self):
        mesh_particles = self.get_mesh_particles_positions()
        # Only use xyz coordinates (ignore mass or extra channels if present)
        #positions = mesh_particles[:, :3]

        # Compute pairwise Euclidean distances
        self.particle_dist_matrix = cdist(mesh_particles, mesh_particles)

        #return self.particle_dist_matrix

    def set_to_flatten(self, re_process_info=False):
        pyflex.set_positions(self.episode_params['init_particle_pos'].flatten())
        self.wait_until_stable()

        self._get_particle_distance_matrix()

        self.info = self._process_info({}, task_related=False, flatten_obs=False)
        if re_process_info:
            self.info = self._process_info({}, task_related=True, flatten_obs=False)
        return self.info
    
    def get_flattened_obs(self):
        
        if self.flattened_obs == None:
            current_particl_pos = pyflex.get_positions()
            # pyflex.set_positions(self.episode_params['init_particle_pos'].flatten())
            # self.wait_until_stable()
            self.flattened_obs = self.set_to_flatten()
            self.flatten_coverage = self._get_coverage()
            #print('flatten coverage', self._get_coverage())
            pyflex.set_positions(current_particl_pos)
            self.wait_until_stable()
        
        return self.flattened_obs
    
    def get_no_op(self):
        return self.action_tool.get_no_op()
    
    def set_disp(self, disp):
        # This function is disabled for this environment
        pass

    def evaluate(self):
        if (self.evaluate_result is None) or (self.action_step == 0):
            self.evaluate_result = self.task.evaluate(self)
        return self.evaluate_result
        
    
    def observation_shape(self):
        return {'rgb': self._observation_image_shape, 
                'depth': self._observation_image_shape}

    def sample_random_action(self):
        return self.action_tool.sample_random_action()

    # TODO: we may need to modify this.
    
    
    # these funcitons is required by the action_tool
    def get_picker_position(self):
        p = self._get_picker_position()
        #print('p', p)
        # swap y and z
        p[:, [1, 2]] = p[:, [2, 1]]
        return p
    
    def control_picker(self, signal, process_info=True):
        
        signal = signal[:, [0, 2, 1, 3]]
        new_picker_pos = self.pickers.step(signal, self)
        self._step_sim()
        self.picker_poses.append(new_picker_pos)
        
        info = {}
        if process_info:
            #print('here')
            info = {
                'observation': self._get_obs(),
            }
            info = self._process_info_(info)
        #self.info = info
        return info
    
    def wait_until_stable(self, max_wait_step=200, stable_vel_threshold=0.0006):
        wait_steps = self._wait_to_stabalise(max_wait_step=max_wait_step, stable_vel_threshold=stable_vel_threshold)
        # print('wait steps', wait_steps)
        # obs = self._get_obs()
        # return {
        #     'observation': obs,
        #     'done': False,
        #     'wait_steps': wait_steps
        # }
    
    ## Helper Functions
    def _wait_to_stabalise(self, max_wait_step=300, stable_vel_threshold=0.0006,
            target_point=None, target_pos=None):
        t = 0
        stable_step = 0
        #print('stable vel threshold', stable_vel_threshold)
        last_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        for j in range(0, max_wait_step):
            t += 1

           
            cur_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
            curr_vel = np.linalg.norm(cur_pos - last_pos, axis=1)
            if target_point != None:
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
            
        return t
    
    
    def _get_coverage(self):
        particle_positions = self.get_mesh_particles_positions()
        # swap y and z
        particle_positions[:, [1, 2]] = particle_positions[:, [2, 1]]
        coverage = get_coverage(particle_positions, self.particle_radius)
        #print('coverage', get_coverage(particle_positions, self.particle_radius))
        return coverage

    def get_particle_positions(self): # standard x y z
        pos = pyflex.get_positions().reshape(-1, 4)[:, :3].copy()
        # swap y and z
        pos[:, [1, 2]] = pos[:, [2, 1]]
        #print('len particles', len(pos))
        #print('pos', pos[0])
        return pos
    
    def get_mesh_particles_positions(self):
        particles = self.get_particle_positions()
        return particles[:self.num_mesh_particles]
    
    # Input is N*3
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
        pos = self._get_picker_pos()
        return pos[:, :3].copy()
        
    def _step_sim(self):
        pyflex.step()
        

        if self.save_video:
            #print('save')
            #print('save here')
            rgb = self._render('rgb', background=True, resolution=self.frame_resolution)
            if self.track_semkey_on_frames and self.task.semkey2pid:
                # --- Track semantic keypoints ---
                particle_pos = self.get_mesh_particles_positions()          # (N, 3)
                semkey2pid = self.task.semkey2pid                     # dict {name: pid}
                keypids = list(semkey2pid.values())
                keynames = list(semkey2pid.keys())

                key_particles = particle_pos[keypids]                 # (K, 3)
                key_pixels, visibility = self.get_visibility(key_particles)  # (K,2), (K,)

                # Assign unique colors to each keypoint (HSV evenly spaced)
                K = len(keynames)
                hsv_colors = [(int(i * 180 / K), 255, 255) for i in range(K)]  # HSV values
                bgr_colors = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2BGR)[0,0].tolist()
                            for c in hsv_colors]

                # Overlay keypoints
                debug_frame = rgb.copy()
                for (v, u), vis, name, color in zip(key_pixels, visibility, keynames, bgr_colors):
                    color = tuple(map(int, color))  # ensure int
                    #print('v, u, name', v, u, name)
                    cv2.circle(debug_frame, (int(u), int(v)), 4, color, -1)
                    cv2.putText(debug_frame, name, (int(u)+5, int(v)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                self.video_frames.append(debug_frame)

            else:
                # just save plain RGB frame
                self.video_frames.append(rgb)

        self.sim_step += 1



    def _process_info_(self, info):
        #print('here process')
        assert 'observation' in info.keys()
        assert 'rgb' in info['observation'].keys()
        H, W = self.observation_shape()['rgb'][0], self.observation_shape()['rgb'][1]
        info['observation']['rgb'] = cv2.resize(info['observation']['rgb'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['depth'] = cv2.resize(info['observation']['depth'], (H, W), interpolation=cv2.INTER_LINEAR).reshape(H, W, -1)
        info['observation']['mask'] = self._get_cloth_mask(resolution=(H, W))
        
        #info['normalised_coverage'] = self._get_normalised_coverage()
        return info

    
    def _render(self, mode='rgb', camera_name='default_camera', resolution=None, background=False):

        if not background:
            img, depth_img = pyflex.render_cloth()
        else:
            img, depth_img = pyflex.render()

        CAMERA_WIDTH = self.camera_config['cam_size'][0]
        CAMERA_HEIGHT = self.camera_config['cam_size'][1]

        img = img.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 4)[::-1, :, :3]  # Need to reverse the height dimension
        depth_img = depth_img.reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 1)[::-1, :, :1]
        

        if mode == 'rgbd':
            img =  np.concatenate((img, depth_img), axis=2)

        elif mode == 'rgb':
            pass
        elif mode == 'depth':
            img = depth_img
        elif mode == 'mask':
            img = img.sum(axis=2) > 0
        else:
            raise NotImplementedError
        
        if resolution is None:
            return img
        
        if CAMERA_HEIGHT != resolution[0] or CAMERA_WIDTH != resolution[1]:
            #print('resizing asked resolution', resolution)
            img = cv2.resize(img, resolution)

        return img
    
    def _get_obs(self, flatten_obs=True):
        obs = {}
        # print('get obs here')
        rgbd = self._render(mode='rgbd', resolution=self.obs_resolution)
        obs['rgb'] = rgbd[:, :, :3].astype(np.uint8)
        obs['image'] = obs['rgb']
        obs['depth'] = rgbd[:, :, 3:]
        obs['mask'] = obs['rgb'].sum(axis=2) > 0 #self._get_cloth_mask()
        self.cloth_mask = obs['mask']
        obs['particle_positions'] = self.get_mesh_particles_positions()
        obs['semkey2pid'] = self.task.semkey2pid
        if self.apply_workspace:
            #print('[GarmentEnv] add workspace')
            obs['robot0_mask'] = self.robot0_mask
            obs['robot1_mask'] = self.robot1_mask
        
        if flatten_obs and self.config.get("provide_semkey_pos", False) and obs['semkey2pid']:
            semkey_positions = []
            for key in obs['semkey2pid'].keys():
                pid = obs['semkey2pid'][key]
                pos = obs['particle_positions'][pid]
                semkey_positions.append(pos)
            obs['semkey_pos'] = np.concatenate(semkey_positions, axis=0).astype(np.float32)
        
        if flatten_obs and self.config.get("provide_flattened_semkey_pos", False) and obs['semkey2pid']:
            semkey_positions = []
            for key in obs['semkey2pid'].keys():
                pid = obs['semkey2pid'][key]
                pos = self.flattened_obs['observation']['particle_positions'][pid]
                semkey_positions.append(pos)
            obs['flattened_semkey_pos'] = np.concatenate(semkey_positions, axis=0).astype(np.float32)

        if flatten_obs and self.config.get("provide_semkey_norm_pixel", False) and obs['semkey2pid']:
            particle_pos = obs['particle_positions']          # (N, 3)
            semkey2pid = obs['semkey2pid']                    # dict {name: pid}
            keypids = list(semkey2pid.values())
            key_particles = particle_pos[keypids]                 # (K, 3)
            key_pixels, visibility = self.get_visibility(key_particles)  # (K,2), (K,)
            H, W = self.camera_size
            norm_pixels = key_pixels/np.array([H, W]) * 2 - 1
            obs['semkey_norm_pixel'] = norm_pixels.flatten()

        if flatten_obs and self.config.get("provide_flattened_semkey_norm_pixel", False) and obs['semkey2pid']:
            particle_pos = self.flattened_obs['observation']['particle_positions']          # (N, 3)
            semkey2pid = obs['semkey2pid']                    # dict {name: pid}
            keypids = list(semkey2pid.values())
            key_particles = particle_pos[keypids]                 # (K, 3)
            key_pixels, visibility = self.get_visibility(key_particles)  # (K,2), (K,)
            H, W = self.camera_size
            norm_pixels = key_pixels/np.array([H, W]) * 2 - 1
            obs['flattened_semkey_norm_pixel'] = norm_pixels.flatten()


        obs['action_step'] = self.action_step

        return obs

    def _get_cloth_mask(self, camera_name='default_camera', resolution=None):
        # print('cloth mask here')
        rgb = self._render(camera_name=camera_name, mode='rgb', resolution=resolution)
        
        return rgb.sum(axis=2) > 0
    
    def _get_init_keys_helper(self, hdf5_path, key_file, difficulties=['hard', 'easy']):

        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return json.load(f)
        else:
            with h5py.File(hdf5_path, 'r') as tasks:
                eval_keys = \
                    [key for key in tasks if tasks[key].attrs['task_difficulty'] in difficulties]
                # print('total evalal keys', len(eval_keys))
            keys = []
            for key in tqdm(eval_keys, desc='Filtering keys'):
                with h5py.File(hdf5_path, 'r') as tasks:
                    group = tasks[key]
                    episode_params = dict(group.attrs)
                    for dataset_name in group.keys():
                        episode_params[dataset_name] = group[dataset_name][()]
                    
                    episode_params['scene_config'] = self.scene_config
                    episode_params.update(self.default_config)
                    set_scene(
                        config=episode_params, 
                        state=episode_params)
                    
                    pyflex.step()
                    cloth_mask = self._get_cloth_mask()
                    if self.workspace_mask is not None:
                        cloth_mask = cloth_mask & self.workspace_mask
                    if cloth_mask.sum() > 500:
                        
                        keys.append(key)
                        
            # save the keys
            with open(key_file, 'w') as f:
                json.dump(keys, f)
        return keys

    def _get_init_state_keys(self):
        
        path = os.path.join(self.init_state_path, f'multi-{self.garment_type}-eval.hdf5')
        #train_path = os.path.join(self.init_state_path, f'multi-{self.garment_type}-train.hdf5')

        key_file = os.path.join(self.init_state_path, f'{self.name}-eval.json')
        #train_key_file = os.path.join(self.init_state_path, f'{self.name}-train.json')

        self.keys = self._get_init_keys_helper(path, key_file, difficulties=['hard'])
        
        # print len of keys
        self.num_trials = 1 #len(self.eval_keys)

    def _get_init_state_params(self, eid):
        raise NotImplementedError
    
    def get_visibility(self, particle_positions, resolution=None):
        """
        Project world-space particle positions into image pixels
        and check if each particle is visible in the current camera view.

        Args:
            particle_positions (np.ndarray): (N, 3) world-space particle positions

        Returns:
            pixels (np.ndarray): (N, 2) array of (u, v) pixel coordinates
            visible (np.ndarray): (N,) boolean array, True if visible
        """
        assert particle_positions.shape[1] == 3, "particle_positions must be (N, 3)"

        # ---- World → Camera ----
        N = particle_positions.shape[0]
        ones = np.ones((N, 1), dtype=np.float32)
        particles_world_h = np.hstack([particle_positions, ones])  # (N, 4)

        # Inverse extrinsic: world → camera
        T_cam_world = np.linalg.inv(self.camera_extrinsic_matrix)
        cam_pts_h = (T_cam_world @ particles_world_h.T).T  # (N, 4)
        cam_pts = cam_pts_h[:, :3]  # (N, 3)

        # ---- Camera → Pixel ----
        if resolution is not None:
            camera_intrinsic_matrix, _ = \
                get_camera_matrix(
                    self.camera_pos, 
                    self.camera_angle, 
                    resolution, 
                    self.camera_config['cam_fov'])
        else:
            camera_intrinsic_matrix = self.camera_intrinsic_matrix
            
        proj = (camera_intrinsic_matrix @ cam_pts.T).T  # (N, 3)
        z = proj[:, 2]
        # avoid div by zero
        z_safe = np.where(z == 0, 1e-6, z)
        pixels = proj[:, :2] / z_safe[:, None]  # (u, v) = (x/z, y/z)

        # ---- Visibility ----
        if resolution is None:
            H, W = self.camera_size
            
        else:
            H, W = resolution
        # print('Visibility H, W', H, W)
        u, v = pixels[:, 0], pixels[:, 1]

        # Rendered depth map (same size as camera)
        # print('vis here')
        depth_img = self._render('depth', resolution=resolution).reshape(H, W)  # (H, W, 1)

        # Conditions: in front of camera and inside image bounds
        visible = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

       # Convert (u,v) to integer pixel indices
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        u_int = np.clip(u_int, 0, W - 1)
        v_int = np.clip(v_int, 0, H - 1)

        # Depth from depth buffer (closest surface)
        depth_buffer = depth_img[v_int, u_int]

        
        # Particle depth = Euclidean distance in world space
        cam_world_pos = self.camera_extrinsic_matrix[:3, 3]  # (3,)
        particle_depth = np.linalg.norm(particle_positions - cam_world_pos, axis=1)

        # Occlusion check: particle must not be behind depth buffer
        eps = 1e-6
        visible &= (particle_depth - self.particle_radius) < (depth_buffer - eps)


        return pixels, visible
    
    def get_cloth_area(self):
        return self.episode_params['cloth_height'] * self.episode_params['cloth_width']

    def compare(self, results_1, results_2):
        return self.task.compare(results_1, results_2)