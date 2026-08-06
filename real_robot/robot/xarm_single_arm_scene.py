"""Single-arm control scene for one UFACTORY xArm Lite 6 with a top-down camera.

Mirrors the public interface of ``SingleArmScene`` (the UR5e single-arm scene) so
that ``XArmSingleArmPickAndPlaceArena`` can inherit
``SingleArmPickAndPlaceArena``'s robot-agnostic logic unchanged. The world frame
is the arm's own base frame.
"""
import os
import numpy as np

from real_robot.utils.motion_utils import safe_movel, safe_gripper, safe_home, safe_out_scene
from real_robot.utils.scene_utils import load_camera_to_base
from real_robot.utils.transform_utils import pixels2base_on_table
from real_robot.robot.xarm_lite6 import XArmLite6
from real_robot.robot.realsense_camera import RealsenseCamera
from real_robot.utils.xarm_constants import XARM_TABLE_Z, XARM_WORKSPACE_RADIUS

# What `source ./setup.sh xarm` exports, with the literal every xArm test script
# falls back to. The 192.168.1.201 that used to sit in this signature was never an
# address of either controller.
DEFAULT_IP = os.environ.get('XARM_LEFT_IP', '192.168.1.155')


class XArmSingleArmScene:
    def __init__(self, robot_ip=DEFAULT_IP, dry_run=False, radius=XARM_WORKSPACE_RADIUS,
                 side='left'):
        self.dry_run = dry_run
        self.gripper_type = 'lite6'
        self.side = side
        calib_name = 'xarm-left-calib.yaml' if side == 'left' else 'xarm-right-calib.yaml'
        self.calib_file = f"{os.environ['MP_FOLD_PATH']}/real_robot/calibration/{calib_name}"
        self.radius = tuple(radius)

        if not dry_run:
            # side= also selects this arm's virtual walls (see xarm_walls.py).
            self.arm = XArmLite6(robot_ip, self.gripper_type, side=side)
            self.camera = RealsenseCamera(debug=True)
            self.intr = self.camera.get_intrinsic()
            self.arm.open_gripper()
            self.arm.home()

            self.T_cam = load_camera_to_base(self.calib_file)
            self.T_world = np.eye(4)   # single arm: world == base
            rgb, _ = self.take_rgbd()
            self._calculate_wrkspace_masks(rgb)
        else:
            print("[XArmSingleArmScene][Dry-run] Skipping robot and camera init.")
            self.arm = None
            self.camera = None
            self.intr = np.eye(3)
            self.T_cam = np.eye(4)
            self.T_world = np.eye(4)

        print('[XArmSingleArmScene] Finished init.')

    # ------------------------------------------------------------------
    # Compatibility wrappers (mimic DualArmScene's single-arm surface)
    # ------------------------------------------------------------------
    def movel(self, pose, speed, acc, blocking=True, record=False):
        return safe_movel(self.arm, pose, speed, acc, blocking, self.dry_run)

    # home()/out_scene() are JOINT moves, so their speed is rad/s and rad/s^2 -- NOT
    # the Cartesian m/s the rest of this class takes. Defaulting to None lets
    # XArmLite6.movej apply XARM_JOINT_SPEED / XARM_JOINT_ACC. Passing a Cartesian
    # speed here is what made homing crawl: 0.2 m/s was read as 0.2 rad/s (11 deg/s)
    # and 0.5 m/s^2 as 0.5 rad/s^2, a tenth of the intended joint acceleration.
    def home(self, speed=None, acc=None, blocking=True):
        return safe_home(self.arm, speed, acc, blocking, self.dry_run)

    def out_scene(self, speed=None, acc=None, blocking=True):
        return safe_out_scene(self.arm, speed, acc, blocking, self.dry_run)

    def open_gripper(self):
        return safe_gripper(self.arm, "open", self.dry_run)

    def close_gripper(self):
        return safe_gripper(self.arm, "close", self.dry_run)

    # ------------------------------------------------------------------
    def take_rgbd(self):
        return self.camera.take_rgbd()

    def restart_camera(self):
        self.camera.restart()

    def get_workspace_mask(self):
        return self.mask

    def _calculate_wrkspace_masks(self, rgb):
        H, W, _ = rgb.shape
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        pixel_points = np.stack([uu.ravel(), vv.ravel()], axis=1)

        base_points = pixels2base_on_table(pixel_points, self.intr, self.T_cam, XARM_TABLE_Z)
        dist = np.linalg.norm(base_points[:, :2], axis=1)
        mask = (dist >= self.radius[0]) & (dist <= self.radius[1])
        self.mask = mask.reshape(H, W)
