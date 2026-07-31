"""Real single-arm arena for one xArm Lite 6 (pick-and-place only).

Subclasses ``SingleArmPickAndPlaceArena`` and overrides ONLY ``__init__`` to build
the xArm scene and skill under the same attribute names the parent uses
(``self.single_arm``, ``self.pick_and_place_skill``). Every other method is
robot-agnostic and inherited unchanged.

The parent's ``__init__`` calls ``super().__init__(config)`` (the actoris_harena
``Arena`` base) and then connects a UR5e; here we call the ``Arena`` base
directly and connect an xArm instead, so the UR file stays untouched.
"""
import os
import numpy as np
from actoris_harena import Arena

from real_robot.robot.single_arm_pick_and_place_arena import SingleArmPickAndPlaceArena
from real_robot.robot.xarm_single_arm_scene import XArmSingleArmScene
from real_robot.primitives.xarm_single_arm_pick_and_place import XArmSingleArmPickAndPlaceSkill
from real_robot.loggers.single_arm_pixel_logger import SingleArmPixelLogger
from real_robot.utils.mask_utils import get_mask_generator
from real_robot.utils.xarm_constants import XARM_WORKSPACE_RADIUS


class XArmSingleArmPickAndPlaceArena(SingleArmPickAndPlaceArena):
    def __init__(self, config):
        Arena.__init__(self, config)
        self.name = "xarm_single_arm_garment_pick_and_place_arena"
        self.config = config
        self.measure_time = config.get('measure_time', False)

        dry_run = config.get("dry_run", False)
        self.single_arm = XArmSingleArmScene(
            robot_ip=config.get("robot_ip", config.get("xarm_ip", "192.168.1.201")),
            dry_run=dry_run,
            radius=config.get('radius', XARM_WORKSPACE_RADIUS),
            side=config.get('arm_side', 'left'),
        )

        self.pick_and_place_skill = XArmSingleArmPickAndPlaceSkill(self.single_arm)
        self.logger = SingleArmPixelLogger()
        self.mask_generator = get_mask_generator()

        self.num_train_trials = config.get("num_train_trials", 100)
        self.num_val_trials = config.get("num_val_trials", 10)
        self.num_eval_trials = config.get("num_eval_trials", 30)
        self.action_horizon = config.get("action_horizon", 20)
        self.snap_to_cloth_mask = config.get("snap_to_cloth_mask", False)
        self.init_from = config.get("init_from", "crumpled")
        self.maskout_background = config.get("maskout_background", False)
        self.use_sim_workspace = config.get("use_sim_workspace", False)
        self.asset_dir = f"{os.environ.get('MP_FOLD_PATH', '.')}/assets"
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

        print('[XArmSingleArmPickAndPlaceArena] Finished init.')
