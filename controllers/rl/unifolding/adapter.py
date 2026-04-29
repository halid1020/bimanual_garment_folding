import os
import numpy as np
import torch
import cv2
import open3d as o3d
from actoris_harena import TrainableAgent
from autolab_core import RigidTransform

# Import the original UniFolding classes so we can inherit from them
from .manipulation.experiment_virtual import ExperimentVirtual, ExperimentVirtualTransforms
from .learning.inference_3d import Inference3D
from .common.datamodels import ActionTypeDef, ObservationMessage

# UniFolding expects an ObservationMessage object. 
# We create a simple mock struct to bridge your dictionary obs to their required format.
class ObservationMessage:
    def __init__(self, pcd):
        self.valid_virtual_pts = pcd
        self.raw_virtual_pts = pcd  # Fallback to the same point cloud if you don't have a separate raw one


class HarenaTransforms(ExperimentVirtualTransforms):
    def __post_init__(self):
        """
        OVERRIDE: Do not call super().__post_init__(). 
        This entirely bypasses UniFolding's attempt to load JSON files.
        """
        if self.option is not None:
            machine_cfg = self.option.machine
            cam_cfg = self.option.compat.camera
            
            # 1. Set Robot Bases
            self.left_robot_base_pos = np.array(machine_cfg.left_robot_base)
            self.right_robot_base_pos = np.array(machine_cfg.right_robot_base)
            
            # Create 4x4 transform matrices for the robots
            self.left_robot_to_world_transform = np.eye(4)
            self.left_robot_to_world_transform[:3, 3] = self.left_robot_base_pos
            self.world_to_left_robot_transform = np.linalg.inv(self.left_robot_to_world_transform)
            
            self.right_robot_to_world_transform = np.eye(4)
            self.right_robot_to_world_transform[:3, 3] = self.right_robot_base_pos
            self.world_to_right_robot_transform = np.linalg.inv(self.right_robot_to_world_transform)
            
            # 2. Set Camera Matrix
            self.camera_to_world_transform = np.eye(4)
            self.camera_to_world_transform[:3, 3] = np.array(cam_cfg.pos)
            self.world_to_camera_transform = np.linalg.inv(self.camera_to_world_transform)
            
            # 3. Maintain UniFolding's expected Virtual Frame rotation
            self.virtual_to_world_transform = np.array([
                [0., 1., 0., 0.],
                [-1., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ])
            self.world_to_virtual_transform = np.linalg.inv(self.virtual_to_world_transform)
            self.virtual_to_camera_transform = self.world_to_camera_transform @ self.virtual_to_world_transform
            self.camera_to_virtual_transform = np.linalg.inv(self.virtual_to_camera_transform)


class HarenaExperiment(ExperimentVirtual):
    def _init_env(self):
        """OVERRIDE: Use our custom transforms and skip loading the Unity environment"""
        self.transforms = HarenaTransforms(option=self.option)
        self.env = None 
        return None

    def is_pose_reachable(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        """
        OVERRIDE: Replaces the 'TODO'. Calculates if the distance 
        from the robot base to the pick point is within your ring limits.
        """
        machine_cfg = self.option.machine 
        target_pos = pose.translation
        
        if is_left_robot:
            base_pos = self.transforms.left_robot_base_pos
            min_r, max_r = machine_cfg.left_workspace  # [0.25, 0.9]
        else:
            base_pos = self.transforms.right_robot_base_pos
            min_r, max_r = machine_cfg.right_workspace # [0.1, 0.85]
            
        # Calculate Euclidean distance on the 2D table plane (XY axis only)
        dist = np.linalg.norm(target_pos[:2] - base_pos[:2])
        
        return min_r <= dist <= max_r

    def is_pose_within_workspace(self, pose: RigidTransform) -> bool:
        """OVERRIDE: Check if the AI's predicted point actually falls on your table"""
        machine_cfg = self.option.machine
        x, y, z = pose.translation
        
        in_x = machine_cfg.x_lim_m[0] <= x <= machine_cfg.x_lim_m[1]
        in_y = machine_cfg.y_lim_m[0] <= y <= machine_cfg.y_lim_m[1]
        
        return in_x and in_y
    
    def is_pose_safe(self, pose1: RigidTransform, pose2: RigidTransform) -> bool:
        """OVERRIDE: Use the flattened config structure to find safe_distance_m"""
        dist = np.linalg.norm(pose1.translation - pose2.translation)
        return dist > self.option.machine.safe_distance_m


class UniFoldingAdapter(TrainableAgent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'unifolding'
        self.debug = config.get('debug', False)
        
        # Save the config so we can access it later in set_log_dir
        self.agent_config = config 
        
        # Instantiate your custom environment rules
        self.experiment = HarenaExperiment(config.experiment)
        
        # We do NOT initialize Inference3D here anymore because self.save_dir isn't assigned yet!
        self.inference = None

    def load_best(self, path=None) -> int:
        
        # --- DYNAMIC MODEL PATH LOGIC ---
        # 1. Grab the garment type (e.g., 'tshirt_long') from your config
        garment_type = self.agent_config.experiment.compat.garment_type
        
        # 2. Combine the dynamically generated save_dir and the garment_type
        dynamic_model_path = os.path.join(self.save_dir, garment_type)
        
        # 3. Override the placeholder model_path in the config
        inference_kwargs = dict(self.agent_config.inference)
        inference_kwargs['model_path'] = dynamic_model_path
        
        # 4. Safely initialize Inference3D now that the path is absolute and correct!
        self.inference = Inference3D(experiment=self.experiment, **inference_kwargs)
        
        if self.debug:
            print(f"[Unifolding Adapter] Dynamically loaded model from: {dynamic_model_path}")

        return True
    
    def single_act(self, info, update=False):
        obs = info['observation']
        
        # Grab the environment instance to access exact camera matrices
        env = info.get('arena', None)
        
        # Extract the point cloud (World Space)
        pointcloud = obs.get('visible_point_cloud', obs.get('particle_positions'))
        
        # ==========================================
        # World Space -> Virtual Space Conversion
        # ==========================================
        w2v_matrix = self.experiment.transforms.world_to_virtual_transform
        
        N = pointcloud.shape[0]
        pc_homogeneous = np.hstack((pointcloud, np.ones((N, 1))))
        virtual_pointcloud = (w2v_matrix @ pc_homogeneous.T).T[:, :3].astype(np.float32)

        # Pass the Virtual pointcloud to the network
        obs_msg = ObservationMessage(virtual_pointcloud)

        action_type = self.inference.predict_raw_action_type(obs_msg)
        prediction_message, action_message, err = self.inference.predict_action(obs_msg, action_type=action_type)

        if err is not None and self.debug:
            print(f"[Unifolding Adapter] Inference Exception: {err}")

        # ==========================================
        # Action Conversion: 3D World -> 2D Normalized Pixel
        # ==========================================
        
        def get_translation_2d(transform):
            if transform is None: return None
            x, y, z = transform.translation
            
            # Use SoftGym's EXACT intrinsic and extrinsic matrices if available
            if env is not None and hasattr(env, 'camera_extrinsic_matrix'):
                pt_h = np.array([[x, y, z, 1.0]], dtype=np.float32)
                
                # 1. World to Camera Frame
                T_cam_world = np.linalg.inv(env.camera_extrinsic_matrix)
                cam_pts = (T_cam_world @ pt_h.T).T[:, :3]
                
                # 2. Camera Frame to Image Pixels
                proj = (env.camera_intrinsic_matrix @ cam_pts.T).T
                z_cam = proj[0, 2]
                if z_cam == 0: z_cam = 1e-6
                pixels = proj[0, :2] / z_cam
                
                # SoftGym Swapped Intrinsic K means: index 0 is V (Y-axis), index 1 is U (X-axis)
                pixel_y, pixel_x = pixels[0], pixels[1]
                
                # 3. Apply SoftGym's internal crop logic
                pixel_x_crop = pixel_x - env.x1
                pixel_y_crop = pixel_y - env.y1
                
                # 4. Normalize based on the cropped square bounds
                norm_x = (pixel_x_crop / env.crop_size) * 2.0 - 1.0
                norm_y = (pixel_y_crop / env.crop_size) * 2.0 - 1.0
                
            else:
                print("[Unifolding Adapter] WARNING: Missing Softgym environment. Output will be broken.")
                norm_y, norm_x = 0, 0
                
            return np.array([norm_y, norm_x], dtype=np.float32)

        # Helper to safely extract 3D points for debugging
        def get_3d(transform):
            return transform.translation if transform is not None else None

        p0_3d = get_3d(action_message.left_pick_pt)
        p1_3d = get_3d(action_message.right_pick_pt)

        p0 = get_translation_2d(action_message.left_pick_pt)
        p1 = get_translation_2d(action_message.right_pick_pt)
        l0 = get_translation_2d(action_message.left_place_pt)
        l1 = get_translation_2d(action_message.right_place_pt)

        # ==========================================
        # VISUALIZATION BLOCK (DEBUG)
        # ==========================================
        if self.debug:
            os.makedirs('./tmp/unifolding', exist_ok=True)
            
            # --- 1. Save 3D Point Cloud ---
            # 1a. Original garment (White)
            pcd_garment = o3d.geometry.PointCloud()
            pcd_garment.points = o3d.utility.Vector3dVector(pointcloud)
            pcd_garment.paint_uniform_color([0.9, 0.9, 0.9]) 
            
            # 1b. Transformed Virtual garment (Green)
            pcd_garment_v = o3d.geometry.PointCloud()
            pcd_garment_v.points = o3d.utility.Vector3dVector(virtual_pointcloud)
            pcd_garment_v.paint_uniform_color([0.0, 1.0, 0.0])
            
            combined_pcd = pcd_garment + pcd_garment_v
            
            pick_pts_3d = []
            if p0_3d is not None: pick_pts_3d.append(p0_3d)
            if p1_3d is not None: pick_pts_3d.append(p1_3d)
            
            if pick_pts_3d:
                pick_arr = np.array(pick_pts_3d)
                
                # 1c. Original World action points (Red)
                pcd_picks = o3d.geometry.PointCloud()
                pcd_picks.points = o3d.utility.Vector3dVector(pick_arr)
                pcd_picks.paint_uniform_color([1.0, 0.0, 0.0])
                
                # 1d. Transformed Virtual action points (Blue)
                picks_homo = np.hstack((pick_arr, np.ones((len(pick_arr), 1))))
                virtual_picks = (w2v_matrix @ picks_homo.T).T[:, :3]
                
                pcd_picks_v = o3d.geometry.PointCloud()
                pcd_picks_v.points = o3d.utility.Vector3dVector(virtual_picks)
                pcd_picks_v.paint_uniform_color([0.0, 0.0, 1.0])
                
                combined_pcd += pcd_picks + pcd_picks_v
                
            o3d.io.write_point_cloud('./tmp/unifolding/debug_before_3d.ply', combined_pcd)
            
            # --- 2. Save 2D RGB Image ---
            rgb_img = obs.get('rgb')
            if rgb_img is not None:
                img_draw = rgb_img.copy()
                if img_draw.dtype != np.uint8:
                    img_draw = (img_draw * 255).astype(np.uint8)
                
                img_draw_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                H, W = img_draw_bgr.shape[:2]
                
                def draw_2d_pick(pt_2d, name, color):
                    if pt_2d is not None:
                        py = int((pt_2d[0] + 1.0) / 2.0 * H)
                        px = int((pt_2d[1] + 1.0) / 2.0 * W)
                        cv2.circle(img_draw_bgr, (px, py), radius=4, color=color, thickness=-1)
                        cv2.circle(img_draw_bgr, (px, py), radius=5, color=(255,255,255), thickness=1)
                        cv2.putText(img_draw_bgr, name, (px + 6, py - 6), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                draw_2d_pick(p0, 'L', (0, 0, 255)) 
                draw_2d_pick(p1, 'R', (255, 0, 0))
                
                cv2.imwrite('./tmp/unifolding/debug_after_2d.png', img_draw_bgr)

        # ==========================================
        # Action Dictionary Assembly
        # ==========================================
        act_type = action_message.action_type
        converted_action = {}

        if act_type == ActionTypeDef.FLING:
            converted_action['norm-pixel-pick-and-fling'] = {'pick_0': p0, 'pick_1': p1}
        elif act_type in [ActionTypeDef.PICK_AND_PLACE, ActionTypeDef.FOLD_1, ActionTypeDef.FOLD_2, ActionTypeDef.DRAG, ActionTypeDef.DRAG_HYBRID]:
            converted_action['norm-pixel-dual-pick-and-place'] = {'pick_0': p0, 'pick_1': p1, 'place_0': l0, 'place_1': l1}
        elif act_type == ActionTypeDef.PICK_AND_PLACE_SINGLE:
            converted_action['norm-pixel-single-pick-and-place'] = {'pick_0': p0, 'place_0': l0}
        else:
            converted_action['no-operation'] = True

        return converted_action
            
    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir

    def save(self):
        pass

    def set_eval(self):
        pass
    
    def set_train(self):
        pass