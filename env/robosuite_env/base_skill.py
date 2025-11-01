import numpy as np

class BaseSkill:
    def __init__(
            self,
            skill_type,

            ### common settings ###
            global_xyz_bounds=np.array([
                [-0.30, -0.30, 0.80],
                [0.15, 0.30, 0.90]
            ]),
            delta_xyz_scale=np.array([0.15, 0.15, 0.05]),
            local_xyz_scale=np.array([0.05, 0.05, 0.05]),
            lift_height=0.95,
            reach_threshold=0.02,
            aff_threshold=0.08,
            aff_type=None,

            binary_gripper=True,

            aff_tanh_scaling=10.0,

            **config
    ):
        self._skill_type = skill_type
        self._config = dict(
            global_xyz_bounds=global_xyz_bounds,
            delta_xyz_scale=delta_xyz_scale,
            local_xyz_scale=local_xyz_scale,
            lift_height=lift_height,
            reach_threshold=reach_threshold,
            aff_threshold=aff_threshold,
            aff_type=aff_type,
            binary_gripper=binary_gripper,
            aff_tanh_scaling=aff_tanh_scaling,
            **config
        )

        for k in ['global_xyz_bounds', 'delta_xyz_scale', 'local_xyz_scale']:
            assert self._config[k] is not None
            self._config[k] = np.array(self._config[k])

        assert self._config['aff_type'] in [None, 'sparse', 'dense']

    def get_param_dim(self, base_param_dim):
        # This should be implemented by the subclass
        raise NotImplementedError

    def update_state(self, info):
        pass

    def reset(self, params, config_update, info):
        self._params = params
        self._state = None
        self._config.update(config_update)
        self._aff_reward, self._aff_success = \
            self._compute_aff_reward_and_success(info)

    def get_pos_ac(self, info):
        raise NotImplementedError

    def get_ori_ac(self, info):
        # --- EDITED: Revert to abstract method. Subclasses must implement specific logic. ---
        # The previous implementation relying on parameter slicing was incorrect for skills 
        # that calculate orientation targets (like ReachSkill) using Axis-Angle control.
        raise NotImplementedError

    def get_gripper_ac(self, info):
        # This should be implemented by the subclass
        raise NotImplementedError

    def get_max_ac_calls(self):
        return self._config['max_ac_calls']

    def get_aff_reward(self):
        return self._aff_reward

    def get_aff_success(self):
        return self._aff_success

    def _get_unnormalized_pos(self, pos, bounds):
        pos = np.clip(pos, -1, 1)
        pos = (pos + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * pos

    def _reached_goal_ori(self, info):
        # --- EDITED: Revert to abstract method. Subclasses must implement robust check. ---
        if not self._config['use_ori_params']:
            return True
        raise NotImplementedError
        
    # --- REMOVED: get_global_euler_from_ori_ac is removed as it relied on old, flawed orientation logic. ---
    
    def _compute_aff_reward_and_success(self, info):
        if self._config['aff_type'] is None:
            return 1.0, True

        aff_centers = self._get_aff_centers(info)
        reach_pos = self._get_reach_pos(info)

        if aff_centers is None:
            return 1.0, True

        if len(aff_centers) == 0:
            return 0.0, False

        th = self._config['aff_threshold']
        within_th = (np.abs(aff_centers - reach_pos) <= th)
        aff_success = np.any(np.all(within_th, axis=1))

        if self._config['aff_type'] == 'dense':
            if aff_success:
                aff_reward = 1.0
            else:
                dist = np.clip(np.abs(aff_centers - reach_pos) - th, 0, None)
                min_dist = np.min(np.sum(dist, axis=1))
                aff_reward = 1.0 - np.tanh(self._config['aff_tanh_scaling'] * min_dist)
        else:
            aff_reward = float(aff_success)

        return aff_reward, aff_success

    def _get_aff_centers(self, info):
        raise NotImplementedError

    def _get_reach_pos(self, info):
        raise NotImplementedError
