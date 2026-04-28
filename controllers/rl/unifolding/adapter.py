# This file is adapted from the running script of unifolding repository
# pip install autolab-core
# pip install loguru>=0.7.0
# pip install easydict
# pip install pyrfuniverse==0.10.1 --> this leads to numpy incompatiblity for pybullet version of raven.
# pip install rich
# pip install pymongo
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
            machine_cfg = self.option.experiment.machine
            cam_cfg = self.option.experiment.compat.camera
            
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
        machine_cfg = self.option.experiment.machine
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
        machine_cfg = self.option.experiment.machine
        x, y, z = pose.translation
        
        in_x = machine_cfg.x_lim_m[0] <= x <= machine_cfg.x_lim_m[1]
        in_y = machine_cfg.y_lim_m[0] <= y <= machine_cfg.y_lim_m[1]
        
        return in_x and in_y

class UniFoldingAdapter(TrainableAgent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'unifolding'
        self.debug = config.get('debug', False)
        
        # Instantiate your custom environment rules
        self.experiment = HarenaExperiment(config)
        
        # Pass it to Inference3D so it filters out bad poses correctly
        self.inference = Inference3D(experiment=self.experiment, **config.inference)


    def load_best(self, path=None) -> int:
        # Construct the path to load from
        load_path = path if path is not None else self.save_dir
        load_path = os.path.join(load_path, 'checkpoints', 'model_best.pth')
        
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        
        if getattr(self, 'policy', None) is not None:
            # We don't need to strip the 'value_net.' prefix anymore because MaximumValuePolicy expects it!
            state_dict = checkpoint.get('net', checkpoint)
            self.policy.load_state_dict(state_dict, strict=False)
            self.policy.eval()
            
        if getattr(self, 'keypoint_detector', None) is not None:
            kps_model_path = os.path.join(path if path is not None else self.save_dir, 'checkpoints', 'keypoint_model.pth')
            
            if os.path.exists(kps_model_path):
                kps_ckpt = torch.load(kps_model_path, map_location="cpu", weights_only=True)
                self.keypoint_detector.load_state_dict(kps_ckpt.get('model_state_dict', kps_ckpt))
                self.keypoint_detector.eval()

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
        # Action Conversion: UniFolding -> HybridActionPrimitive
        # ==========================================
        
        # Helper to safely extract the 3D translation from a RigidTransform
        def get_translation(transform):
            return transform.translation if transform is not None else None

        # If no valid action message or the action is terminal/failed, trigger no-op
        if action_message is None or action_message.action_type in [ActionTypeDef.DONE, ActionTypeDef.FAIL]:
            return {'no-operation': True}

        # Extract points
        p0 = get_translation(action_message.left_pick_pt)
        p1 = get_translation(action_message.right_pick_pt)
        l0 = get_translation(action_message.left_place_pt)
        l1 = get_translation(action_message.right_place_pt)

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

    def save(self):
        pass

    def set_eval(self):
        pass
    
    def set_train(self):
        pass