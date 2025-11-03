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
        return 5

    def get_pos_ac(self, info):
        params = self._params
        # The first 3 parameters are the positional delta
        pos = params[:3].copy()
        is_delta = True
        return pos, is_delta

    def get_ori_ac(self, info):
        normalized_yaw = self._params[3] if len(self._params) > 3 else 0.0
        delta_yaw_deg = normalized_yaw * 20
        
        # Construct the full target Euler vector (Roll, Pitch, Yaw)
        delta_euler_deg = np.array([0, 0, delta_yaw_deg])
        
        # Convert target Euler (degrees) to Rotation object
        self._target_rotation = R.from_euler('xyz', delta_euler_deg, degrees=True)
        
        # Convert Rotation object to Axis-Angle (RotVec) vector for the controller
        ori = self._target_rotation.as_rotvec()
        is_delta = True
        #print('ori', ori)
        return ori, is_delta

    def get_gripper_ac(self, info):

        gripper_action = self._params[-1:].copy()

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