from .base_skill import BaseSkill
import numpy as np

from scipy.spatial.transform import Rotation as R # Required for orientation math



class GripperSkill(BaseSkill):
    """
    A skill that performs a simple gripper open or close action over a fixed number of timesteps.
    It does not take any skill parameters, only controls the gripper, and keeps EE position fixed.
    """
    def __init__(
            self,
            skill_type,
            max_ac_calls=4,
            **config
    ):
        super().__init__(
            skill_type,
            use_ori_params=False, # Does not use orientation parameters
            use_gripper_params=False, # Controls gripper internally, not via parameters
            max_ac_calls=max_ac_calls,
            **config
        )
        self._num_steps_steps = 0

    def get_param_dim(self):
        # No parameters needed for this skill
        return 0

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._num_steps_steps = 0

    def update_state(self, info):
        # Increment step count to track progress
        self._num_steps_steps += 1

    def get_pos_ac(self, info):
        # Keep the current position (delta 0)
        pos = np.zeros(3)
        is_delta = True
        return pos, is_delta

    def get_ori_ac(self, info):
        # Keep the current orientation (delta 0). 
        # Note: self._config['robot_controller_dim'] might be greater than 3.
        controller_dim = self._config.get('robot_controller_dim', 6)
        ori_dim = controller_dim - 3 # Typically 3 (for RotVec/Axis-Angle)
        
        ori = np.zeros(ori_dim)
        ori[:] = 0.0
        is_delta = True
        return ori, is_delta

    def get_gripper_ac(self, info):
        rg_dim = self._config['robot_gripper_dim']
        gripper_action = np.zeros(rg_dim)
        
        # Determine action based on the skill_type string
        if self._skill_type in ['close', 'close_pos']:
            # Assume -1 is 'close' or 'grip' in the controller
            gripper_action[:] = -1
        elif self._skill_type in ['open', 'open_pos']:
            # Assume 1 is 'open' or 'release' in the controller
            gripper_action[:] = 1
        else:
            raise ValueError(f"Invalid skill_type for GripperSkill: {self._skill_type}")

        return gripper_action

    def is_success(self, info):
        # Success once the maximum number of action calls is reached
        return (self._num_steps_steps >= self._config['max_ac_calls'])

    def _get_aff_centers(self, info):
        # No affinity centers needed for a simple gripper action
        return None

    def _get_reach_pos(self, info):
        # The target position is just the current EE position since it remains fixed
        return info['cur_ee_pos']
