"""Real dual-arm arena for two xArm Lite 6 arms.

Subclasses ``DualArmArena`` and overrides ONLY ``__init__`` to build the xArm
scene and xArm skills under the same attribute names the parent uses
(``self.dual_arm``, ``self.pick_and_place_skill``, ``self.pick_and_fling_skill``).
Every other method — ``reset``, ``step``, ``_process_info``, ``evaluate``,
``success`` — is robot-agnostic and inherited unchanged.

We deliberately do NOT call ``super().__init__`` (the parent constructor connects
the UR5e+UR16e); the ~config-plumbing below is replicated so the UR file stays
untouched.
"""
import os
import numpy as np

from real_robot.robot.dual_arm_arena import DualArmArena
from real_robot.robot.xarm_dual_arm_scene import XArmDualArmScene
from real_robot.primitives.xarm_pick_and_place import XArmPickAndPlaceSkill
from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
from real_robot.loggers.pixel_based_primitive_env_imp_logger import PixelBasedPrimitiveImpEnvLogger
from real_robot.utils.mask_utils import get_mask_generator
from real_robot.utils.xarm_constants import XARM_WORKSPACE_RADIUS


class XArmDualArmArena(DualArmArena):
    def __init__(self, config):
        self.name = "xarm_dual_arm_garment_arena"
        self.config = config
        self.draw_fatten_contour = False
        self.measure_time = config.get('measure_time', False)
        self.roi_off_x = config.get('roi_off_x', 60)

        dry_run = config.get("dry_run", False)
        self.dual_arm = XArmDualArmScene(
            left_robot_ip=config.get("left_ip", "192.168.1.201"),
            right_robot_ip=config.get("right_ip", "192.168.1.202"),
            dry_run=dry_run,
            workspace_radius=config.get("workspace_radius", XARM_WORKSPACE_RADIUS),
        )

        self.pick_and_place_skill = XArmPickAndPlaceSkill(self.dual_arm)
        self.pick_and_fling_skill = XArmPickAndFlingSkill(self.dual_arm)
        self.logger = PixelBasedPrimitiveImpEnvLogger()

        self.mask_generator = get_mask_generator()

        self.num_train_trials = config.get("num_train_trials", 100)
        self.num_val_trials = config.get("num_val_trials", 10)
        self.num_eval_trials = config.get("num_eval_trials", 30)
        self.action_horizon = config.get("action_horizon", 20)
        self.snap_to_cloth_mask = config.get("snap_to_cloth_mask", False)
        self.init_from = config.get("init_from", "crumpled")
        self.maskout_background = config.get("maskout_background", False)
        self.use_sim_workspace = config.get("use_sim_workspace", False)
        self.asset_dir = f"{os.environ['MP_FOLD_PATH']}/assets"
        self.track_trajectory = config.get("track_trajectory", False)

        self.current_episode = None
        self.frames = []
        self.all_infos = []
        self.goal = None
        self.debug = config.get("debug", False)

        self.resolution = (512, 512)
        self.action_step = 0
        self.evaluate_result = None
        self.last_flattened_step = -1
        self.id = 0
        self.init_coverage = None
        self.flattened_obs = None

        print('[XArmDualArmArena] Finished init.')
