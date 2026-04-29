import os
import numpy as np
import torch
from actoris_harena import TrainableAgent
import numpy as np
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
        machine_cfg = self.option.machine  # <--- REMOVED .experiment here
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
        machine_cfg = self.option.machine  # <--- REMOVED .experiment here
        x, y, z = pose.translation
        
        in_x = machine_cfg.x_lim_m[0] <= x <= machine_cfg.x_lim_m[1]
        in_y = machine_cfg.y_lim_m[0] <= y <= machine_cfg.y_lim_m[1]
        
        return in_x and in_y
    
    def is_pose_safe(self, pose1: RigidTransform, pose2: RigidTransform) -> bool:
        """OVERRIDE: Use the flattened config structure to find safe_distance_m"""
        # Look directly in self.option.machine instead of self.option.compat.machine
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
        # Result: /media/hcv530/T71/.../unifolding_folding_from_crumpled/tshirt_long
        dynamic_model_path = os.path.join(self.save_dir, garment_type)
        
        # 3. Override the placeholder model_path in the config
        # We convert it to a standard dict to ensure we can modify it
        inference_kwargs = dict(self.agent_config.inference)
        inference_kwargs['model_path'] = dynamic_model_path
        
        # 4. Safely initialize Inference3D now that the path is absolute and correct!
        self.inference = Inference3D(experiment=self.experiment, **inference_kwargs)
        
        if self.debug:
            print(f"[Unifolding Adapter] Dynamically loaded model from: {dynamic_model_path}")

        return True
    
    def single_act(self, info, update=False):
        obs = info['observation']
        
        # Extract the point cloud
        pointcloud = obs.get('visible_point_cloud', obs.get('particle_positions'))
        
        # Wrap the numpy array in the expected ObservationMessage struct
        obs_msg = ObservationMessage(pointcloud)

        # Get raw action type prediction
        action_type = self.inference.predict_raw_action_type(obs_msg)

        # Get the full action prediction (poses)
        prediction_message, action_message, err = self.inference.predict_action(obs_msg, action_type=action_type)

        if err is not None and self.debug:
            print(f"[Unifolding Adapter] Inference Exception: {err}")

        # ==========================================
        # Action Conversion: 3D World -> 2D Normalized Pixel
        # ==========================================
        
        def get_translation_2d(transform):
            if transform is None: return None
            x, y, z = transform.translation
            
            # Fetch camera parameters from your config
            cam_pos = self.agent_config.experiment.compat.camera.pos
            fov_x, fov_y = self.agent_config.experiment.compat.camera.field_of_view
            
            # 1. Calculate distance from camera to the pick point (Z-depth)
            depth = cam_pos[2] - z
            
            # 2. Calculate the physical width and height of the camera's view at this depth
            half_w = depth * np.tan(np.deg2rad(fov_x) / 2.0)
            half_h = depth * np.tan(np.deg2rad(fov_y) / 2.0)
            
            # 3. Normalize coordinates to [-1, 1]
            norm_x = (x - cam_pos[0]) / half_w
            
            # Image Y (pixel rows) goes DOWN, while World Y goes UP. Invert it:
            norm_y = -(y - cam_pos[1]) / half_h
            
            # Clip them just to be safe
            norm_x = np.clip(norm_x, -1.0, 1.0)
            norm_y = np.clip(norm_y, -1.0, 1.0)
            
            # The environment explicitly multiplies the action by np.array([H, W]).
            # This mathematically implies it expects the layout as [Y, X] (row, col).
            return np.array([norm_y, norm_x], dtype=np.float32)

        # If no valid action message or the action is terminal/failed, trigger no-op
        if action_message is None or action_message.action_type in [ActionTypeDef.DONE, ActionTypeDef.FAIL]:
            return {'no-operation': True}

        # Extract 2D projected points
        p0 = get_translation_2d(action_message.left_pick_pt)
        p1 = get_translation_2d(action_message.right_pick_pt)
        l0 = get_translation_2d(action_message.left_place_pt)
        l1 = get_translation_2d(action_message.right_place_pt)

        act_type = action_message.action_type
        converted_action = {}

        # 1. Fling
        if act_type == ActionTypeDef.FLING:
            converted_action['norm-pixel-pick-and-fling'] = {
                'pick_0': p0,
                'pick_1': p1
            }
            
        # 2. Dual-arm operations (Pick&Place, Folds, Drags)
        elif act_type in [
            ActionTypeDef.PICK_AND_PLACE, 
            ActionTypeDef.FOLD_1, 
            ActionTypeDef.FOLD_2, 
            ActionTypeDef.DRAG, 
            ActionTypeDef.DRAG_HYBRID
        ]:
            converted_action['norm-pixel-dual-pick-and-place'] = {
                'pick_0': p0,
                'pick_1': p1,
                'place_0': l0,
                'place_1': l1
            }
            
        # 3. Single-arm operations
        elif act_type == ActionTypeDef.PICK_AND_PLACE_SINGLE:
            converted_action['norm-pixel-single-pick-and-place'] = {
                'pick_0': p0,
                'place_0': l0  # Mapping place to left_place/0
            }
            
        # 4. Fallback
        else:
            if self.debug:
                print(f"[Unifolding Adapter] Unmapped ActionTypeDef: {act_type}. Defaulting to no-op.")
            converted_action['no-operation'] = True

        return converted_action

            
    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir

    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir
        
        

    def save(self):
        pass

    def set_eval(self):
        pass
    
    def set_train(self):
        pass