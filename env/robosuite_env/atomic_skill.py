from .base_skill import BaseSkill
import numpy as np

from scipy.spatial.transform import Rotation as R # Required for orientation math


class AtomicSkill(BaseSkill):
    """
    An atomic skill that directly outputs action control signals as deltas.
    It takes the full action vector (pos_delta, ori_delta, gripper_ac) from its parameters.
    """
    def __init__(
            self,
            skill_type,
            use_ori_params=True,
            use_gripper_params=True,
            **config
    ):
        super().__init__(
            skill_type,
            use_ori_params=use_ori_params,
            use_gripper_params=use_gripper_params,
            # Atomic skills execute in one step
            max_ac_calls=1,
            **config
        )

    def get_param_dim(self):
        # Assumes the parameters map directly to the controller and gripper dimensions.
        # Defaults to 6 for controller (3 pos, 3 ori) + 1 for gripper if not specified in config.
        controller_dim = self._config.get('robot_controller_dim', 6)
        gripper_dim = self._config.get('robot_gripper_dim', 1)
        return controller_dim + gripper_dim

    def get_pos_ac(self, info):
        params = self._params
        # The first 3 parameters are the positional delta
        pos = params[:3].copy()
        is_delta = True
        return pos, is_delta

    def get_ori_ac(self, info):
        ori_dim = 3
        # Orientation parameters start at index 3 (after 3 positional params)
        start_idx = 3 
        
        if self._config['use_ori_params']:
            # Extract the 3 orientation delta parameters (Axis-Angle / RotVec)
            ori = self._params[start_idx : start_idx + ori_dim].copy()
        else:
            # If not using ori params, the delta is zero
            ori = np.zeros(ori_dim)
            
        is_delta = True
        return ori, is_delta

    def get_gripper_ac(self, info):
        rg_dim = self._config.get('robot_gripper_dim', 1)
        
        # Calculate the starting index for the gripper action parameter(s)
        # Assumes parameter order: [pos(3), ori(3, if used), gripper(rg_dim)]
        start_idx = 3 + (3 if self._config['use_ori_params'] else 0)
        
        if self._config['use_gripper_params']:
            # The gripper action is the last parameter(s) in the expected sequence
            gripper_action = self._params[start_idx : start_idx + rg_dim].copy()
        else:
            # If not using params, default to open (1)
            gripper_action = np.ones(rg_dim) 

        return gripper_action

    def is_success(self, info):
        # Atomic skills execute in one step and immediately succeed
        return True

    def _get_aff_centers(self, info):
        # Affinity centers are not applicable for an atomic delta skill
        return None

    def _get_reach_pos(self, info):
        # Returns current position, as the action is a delta from here
        return info['cur_ee_pos']