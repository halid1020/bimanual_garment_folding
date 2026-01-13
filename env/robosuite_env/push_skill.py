from .base_skill import BaseSkill
import numpy as np

from scipy.spatial.transform import Rotation as R # Required for orientation math

class PushSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PUSHED']

    def __init__(
            self,
            skill_type,
            max_ac_calls=20,
            use_ori_params=True,
            ori_threshold_rad=0.05, 
            **config
    ):
        super().__init__(
            skill_type,
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            **config
        )
        self._config['ori_threshold_rad'] = ori_threshold_rad 
        # State variables for orientation control (copied from ReachSkill)
        self._initial_roll_pitch_deg = None 
        self._target_rotation = None        

    def get_param_dim(self):
        # 3 (XYZ reach pos) + 1 (Yaw) + 3 (XYZ push delta) = 7 parameters
        return 7

    def reset(self, params, skill_config_update, info):
        super().reset(params, skill_config_update, info)
        
        # Orientation Capture (Copied from ReachSkill)
        world_rotmat = info.get('cur_ee_rotmat')
        if world_rotmat is None:
            raise ValueError("cur_ee_rotmat is missing from info. SkillController failed to provide current EE rotation matrix.")
        
        # Convert to Euler (XYZ)
        initial_euler_deg = R.from_matrix(world_rotmat).as_euler('xyz', degrees=True)
        
        # Store only Roll (X) and Pitch (Y). Yaw (Z) is set by the parameter.
        self._initial_roll_pitch_deg = initial_euler_deg[:2]
        self._target_rotation = None 


    def update_state(self, info):
        cur_pos = info['cur_ee_pos']
        src_pos = self._get_reach_pos(info)
        target_pos = self._get_push_pos(info)

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_src_xy = (np.linalg.norm(cur_pos[0:2] - src_pos[0:2]) < th)
        reached_src_xyz = (np.linalg.norm(cur_pos - src_pos) < th)
        reached_target_xyz = (np.linalg.norm(cur_pos - target_pos) < th)
        reached_ori = self._reached_goal_ori(info)

        if self._state in ['REACHED', 'PUSHED'] and reached_target_xyz:
            self._state = 'PUSHED'
        else:
            if self._state == 'REACHED' or (reached_src_xyz and reached_ori):
                self._state = 'REACHED'
            else:
                if reached_src_xy and reached_ori:
                    self._state = 'HOVERING'
                else:
                    if reached_lift:
                        self._state = 'LIFTED'
                    else:
                        self._state = 'INIT'

        assert self._state in PushSkill.STATES

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        src_pos = self._get_reach_pos(info)
        target_pos = self._get_push_pos(info)

        is_delta = False
        if self._state == 'INIT':
            # Lift to clearance height first
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            # Move X/Y to start position at clearance height
            pos = src_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING':
            # Move down to final Z of the start position
            pos = src_pos.copy()
        elif self._state == 'REACHED':
            # Execute the push by moving to the target position
            pos = target_pos.copy()
        elif self._state == 'PUSHED':
            # Hold at the target position
            pos = target_pos.copy()
        else:
            raise NotImplementedError

        return pos, is_delta

    def get_ori_ac(self, info):
        # Orientation logic (Copied from ReachSkill)
        
        # Target Yaw is the 4th parameter (index 3)
        normalized_yaw = self._params[3] if len(self._params) > 3 else 0.0
        target_yaw_deg = normalized_yaw * 180.0
        
        # Use initial Roll (X) and Pitch (Y) stored during reset
        target_roll = self._initial_roll_pitch_deg[0]
        target_pitch = self._initial_roll_pitch_deg[1]
        
        # Construct the full target Euler vector (Roll, Pitch, Yaw)
        target_euler_deg = np.array([target_roll, target_pitch, target_yaw_deg])
        
        # Convert target Euler (degrees) to Rotation object
        self._target_rotation = R.from_euler('xyz', target_euler_deg, degrees=True)
        
        # Convert Rotation object to Axis-Angle (RotVec) vector for the controller
        ori = self._target_rotation.as_rotvec()
        is_delta = False
       
            
        return ori, is_delta

    def get_gripper_ac(self, info):
        # Gripper must remain open (1) for pushing
        rg_dim = self._config['robot_gripper_dim']
<<<<<<< HEAD
        gripper_action = -np.ones(rg_dim) 
=======
        gripper_action = np.ones(rg_dim) 
>>>>>>> e8982bdda037099f737c014ffde53c5d35019faa
        return gripper_action

    def _get_reach_pos(self, info):
        # Uses the first 3 parameters (index 0, 1, 2) for the start of the push
        params = self._params
        pos = self._get_unnormalized_pos(
            params[:3], self._config['global_xyz_bounds'])
        return pos.copy()

    def _get_push_pos(self, info):
        params = self._params

        # Start position of the push
        src_pos = self._get_reach_pos(info)
        pos = src_pos.copy()

        # Delta position uses the LAST 3 parameters (index 4, 5, 6)
        delta_pos = params[-3:].copy()
        delta_pos = np.clip(delta_pos, -1, 1)
        delta_pos *= self._config['delta_xyz_scale']
        
        # Final target position
        pos += delta_pos

        return pos

    def is_success(self, info):
        return self._state == 'PUSHED'

    def _reached_goal_ori(self, info):
        # Orientation success logic (Copied from ReachSkill)
        if not self._config['use_ori_params'] or self._target_rotation is None:
            return True
            
        cur_rotmat = info.get('cur_ee_rotmat')
        if cur_rotmat is None:
            return False 
        
        cur_rotation = R.from_matrix(cur_rotmat)
        
        # Calculate angle error
        rot_diff = cur_rotation.inv() * self._target_rotation
        angle_error_rad = rot_diff.magnitude()
        
        ori_threshold = self._config.get('ori_threshold_rad', 0.05)
        
        return angle_error_rad < ori_threshold

    def _get_aff_centers(self, info):
        aff_centers = info.get('push_pos', []) # Affinity is checked against the final push position
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)
