import numpy as np
# Assuming BaseSkill is defined in a sibling module or globally imported in the runtime environment
# For a runnable example, we define a minimal BaseSkill placeholder
from .base_skill import BaseSkill
from scipy.spatial.transform import Rotation as R # Required for orientation math

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED']

    def __init__(
            self,
            skill_type,
            use_gripper_params=True,
            use_ori_params=True, 
            max_ac_calls=15,
            ori_threshold_rad=0.05, 
            **config
    ):
        super().__init__(
            skill_type,
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            **config,
        )
        self._config['ori_threshold_rad'] = ori_threshold_rad 
        # Internal state to store initial Roll/Pitch and target rotation
        self._initial_roll_pitch_deg = None
        self._target_rotation = None

    def get_param_dim(self):
        # 3 (XYZ pos) + 1 (Yaw angle) = 4 parameters
        return 4

    def reset(self, params, skill_config_update, info):
        super().reset(params, skill_config_update, info)
        
        # --- FIX START: Use rotation matrix directly from the info dict ---
        # SkillController ensures 'cur_ee_rotmat' is available in 'info'
        world_rotmat = info.get('cur_ee_rotmat')
        
        if world_rotmat is None:
            # Fallback error check if SkillController didn't provide rotation info
            raise ValueError("cur_ee_rotmat is missing from info. SkillController failed to provide current EE rotation matrix.")
        
        # Convert to Euler (XYZ)
        initial_euler_deg = R.from_matrix(world_rotmat).as_euler('xyz', degrees=True)
        
        # Store only Roll (X) and Pitch (Y). We assume the Yaw (Z) is set by the parameter.
        self._initial_roll_pitch_deg = initial_euler_deg[:2]
        self._target_rotation = None 
        # --- FIX END ---


    def update_state(self, info):
        cur_pos = info['cur_ee_pos']
        # Get current rotation matrix from info
        cur_rotmat = info.get('cur_ee_rotmat')
        
        goal_pos = self._get_reach_pos(info)

        th = self._config['reach_threshold']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        
        reached_ori = self._reached_goal_ori(info) 

        if reached_xyz and reached_ori:
            self._state = 'REACHED'
        else:
            if reached_xy and reached_ori:
                self._state = 'HOVERING'
            else:
                if reached_lift:
                    self._state = 'LIFTED'
                else:
                    self._state = 'INIT'

        assert self._state in ReachSkill.STATES

    def get_pos_ac(self, info):
        cur_pos = info['cur_ee_pos']
        # The goal pos is defined by the first 3 parameters
        goal_pos = self._get_reach_pos(info)

        is_delta = False
        if self._state == 'INIT':
            # Lift to clearance height first
            pos = cur_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'LIFTED':
            # Move X/Y to target at clearance height
            pos = goal_pos.copy()
            pos[2] = self._config['lift_height']
        elif self._state == 'HOVERING':
            # Move down to final Z position
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            # Hold position
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos, is_delta

    def get_ori_ac(self, info):
        # The action to the controller is expected to be Axis-Angle (RotVec)
        #if self._config['use_ori_params']:
        # Target Yaw is the 4th parameter (index 3)
        # Check if params has at least 4 elements, otherwise default to 0 (straight down)
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
        #print('ori', ori)
        return ori, is_delta

    def get_gripper_ac(self, info):
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = np.zeros(rg_dim) # Default to maintaining current gripper state (gripper 0)

        return gripper_action
    
    def _get_reach_pos(self, info):
        # Uses the first 3 parameters (XYZ)
        params = self._params
        pos = self._get_unnormalized_pos(
            params[:3], self._config['global_xyz_bounds'])
        return pos

    def is_success(self, info):
        return self._state == 'REACHED'

    def _reached_goal_ori(self, info):
        if not self._config['use_ori_params'] or self._target_rotation is None:
            return True
            
        # Get current EE orientation rotation matrix from SkillController info
        cur_rotmat = info.get('cur_ee_rotmat')
        if cur_rotmat is None:
            return False 
        
        cur_rotation = R.from_matrix(cur_rotmat)
        
        # Calculate angle error (in radians) using Scipy's robust magnitude method
        # R_error = R_current_inverse * R_target
        rot_diff = cur_rotation.inv() * self._target_rotation
        angle_error_rad = rot_diff.magnitude()
        
        ori_threshold = self._config.get('ori_threshold_rad', 0.05)
        
        return angle_error_rad < ori_threshold

    def _get_aff_centers(self, info):
        aff_centers = info.get('reach_pos', [])
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)
