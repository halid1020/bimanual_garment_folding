"""Dual-arm control scene for two symmetric UFACTORY xArm Lite 6 arms with a
top-down camera. Mirrors the public interface of ``DualArmScene`` (the UR5e+UR16e
scene) so that ``XArmDualArmArena`` can inherit ``DualArmArena``'s robot-agnostic
``step``/``_process_info`` unchanged.

Differences from the UR scene:
  * two IDENTICAL arms (``left``/``right``), bases ~35 cm apart;
  * strictly top-down camera;
  * annular reach workspace masks (no UR shoulder-singularity cylinder);
  * no force/torque based helpers.

The world frame sits at the midpoint between the two bases, at the table height,
with axes following the LEFT arm's base frame (the analogue of the UR5e).
"""
import os
import time
import numpy as np

from real_robot.utils.thread_utils import ThreadWithResult
from real_robot.utils.motion_utils import safe_movel, safe_gripper, safe_home, safe_out_scene
from real_robot.utils.scene_utils import load_camera_to_base
from real_robot.utils.transform_utils import (
    tcp_pose_to_transform, pixels2base_on_table, transform_pose, matrix_to_pose,
)
from real_robot.robot.xarm_lite6 import XArmLite6
from real_robot.robot.realsense_camera import RealsenseCamera
from real_robot.utils.xarm_camera import crop_window, describe_orientation
from real_robot.utils.xarm_constants import (
    XARM_TABLE_Z, XARM_WORKSPACE_RADIUS, XARM_MOVE_SPEED, XARM_MOVE_ACC,
    XARM_BASE_SEPARATION,
)


class XArmDualArmScene:
    def __init__(self, left_robot_ip="192.168.1.201", right_robot_ip="192.168.1.202",
                 dry_run=False, workspace_radius=XARM_WORKSPACE_RADIUS):
        self.dry_run = dry_run
        self.gripper_type = 'lite6'
        self.last_trajectory = None

        # Eye-to-hand (static top-down camera) calibration, one file per arm.
        self.left_calib_file = f"{os.environ['MP_FOLD_PATH']}/real_robot/calibration/xarm-left-calib.yaml"
        self.right_calib_file = f"{os.environ['MP_FOLD_PATH']}/real_robot/calibration/xarm-right-calib.yaml"

        self.left_radius = tuple(workspace_radius)
        self.right_radius = tuple(workspace_radius)

        if not dry_run:
            # side= gives each driver the virtual walls in its OWN base frame
            # (the right base is yawed 180 deg, so the table box transforms).
            self.left = XArmLite6(left_robot_ip, self.gripper_type, side='left')
            self.right = XArmLite6(right_robot_ip, self.gripper_type, side='right')
            self.camera = RealsenseCamera(debug=True)
            self.intr = self.camera.get_intrinsic()
            self.both_open_gripper()
            self.both_home()

            self.T_left_cam = load_camera_to_base(self.left_calib_file)
            self.T_right_cam = load_camera_to_base(self.right_calib_file)
            self.T_left_right = self.T_left_cam @ np.linalg.inv(self.T_right_cam)

            # Everything downstream sees a square window of the table centred
            # between the two arms, with the principal point shifted to match, so
            # crop pixels invert straight back to the same metres. take_rgbd() is
            # the only place the crop is applied -- self.intr and every image this
            # scene hands out belong to the same frame by construction.
            self.crop = crop_window(self.intr, self.T_left_cam,
                                    separation=XARM_BASE_SEPARATION,
                                    table_z=XARM_TABLE_Z, verbose=True)
            self.intr = self.crop.intrinsic(self.intr)

            # How the mount actually came out. Reported, not enforced -- see
            # describe_orientation.
            print(describe_orientation(self.T_left_cam, self.intr,
                                       XARM_BASE_SEPARATION, XARM_TABLE_Z)[1])

            # World frame at the base midpoint, at table height, LEFT-aligned axes.
            self.T_left_world = np.eye(4)
            self.T_left_world[:3, 3] = self.T_left_right[:3, 3] / 2.0
            self.T_left_world[2, 3] = XARM_TABLE_Z
            self.T_right_world = np.linalg.inv(self.T_left_right) @ self.T_left_world

            self.T_world_cam = np.linalg.inv(self.T_left_world) @ self.T_left_cam
            self.T_cam_world = np.linalg.inv(self.T_world_cam)

            rgb, _ = self.take_rgbd()
            self._calculate_wrkspace_masks(rgb)
        else:
            print("[XArmDualArmScene][Dry-run] Skipping robot and camera init.")
            self.left = self.right = self.camera = None
            self.crop = None            # no intrinsic to crop against
            self.intr = np.eye(3)
            self.T_left_cam = np.eye(4); self.T_left_cam[0, 3] = 0.175
            self.T_right_cam = np.eye(4); self.T_right_cam[0, 3] = -0.175
            self.T_left_right = self.T_left_cam @ np.linalg.inv(self.T_right_cam)
            self.T_cam_world = np.eye(4); self.T_cam_world[2, 3] = 1.0
            self.T_left_world = np.eye(4)
            self.T_right_world = np.eye(4)

        # UR-named aliases so the reused PixelBasedPrimitiveImpEnvLogger (which
        # refers to scene.T_ur5e_cam / T_ur16e_cam and 'ur5e'/'ur16e' trajectory
        # keys) works unchanged: left == UR5e analogue, right == UR16e analogue.
        self.T_ur5e_cam = self.T_left_cam
        self.T_ur16e_cam = self.T_right_cam
        self.T_ur5e_ur16e = self.T_left_right

        print('[XArmDualArmScene] Finished init.')

    # ------------------------------------------------------------------
    # Coordinate transforms (world <-> per-arm base)
    # ------------------------------------------------------------------
    def world_to_robot_base(self, world_pose, robot_name='left'):
        T_world = tcp_pose_to_transform(world_pose)
        T_base_world = self.T_left_world if robot_name == 'left' else self.T_right_world
        return matrix_to_pose(T_base_world @ T_world)

    def robot_base_to_world(self, robot_pose, robot_name='left'):
        T_base = tcp_pose_to_transform(robot_pose)
        T_world_base = np.linalg.inv(self.T_left_world if robot_name == 'left' else self.T_right_world)
        return matrix_to_pose(T_world_base @ T_base)

    # ------------------------------------------------------------------
    # Multi-threaded motion (same surface as DualArmScene)
    # ------------------------------------------------------------------
    def both_movel(self, left_pose, right_pose, speed, acc, blocking=True, record=False):
        t1 = ThreadWithResult(target=safe_movel, args=(self.left, left_pose, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_movel, args=(self.right, right_pose, speed, acc, blocking, self.dry_run))
        t1.start(); t2.start()

        self.last_trajectory = None
        if blocking and record and not self.dry_run:
            # Keys mirror the UR scene ('ur5e'=left, 'ur16e'=right) so the reused
            # imp logger's fling-trajectory projection works unchanged.
            traj = {'ur5e': [], 'ur16e': []}
            while t1.is_alive() or t2.is_alive():
                try:
                    traj['ur5e'].append(self.left.get_tcp_pose()[:3])
                    traj['ur16e'].append(self.right.get_tcp_pose()[:3])
                except Exception:
                    pass
                time.sleep(0.1)
            self.last_trajectory = traj

        if blocking:
            t1.join(); t2.join()
            return t1.result and t2.result
        return True

    # JOINT moves: speed is rad/s, not the Cartesian m/s used elsewhere here. None
    # lets XArmLite6.movej apply XARM_JOINT_SPEED / XARM_JOINT_ACC; passing a
    # Cartesian speed made homing crawl at a tenth of the intended acceleration.
    def both_home(self, speed=None, acc=None, blocking=True):
        t1 = ThreadWithResult(target=safe_home, args=(self.left, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_home, args=(self.right, speed, acc, blocking, self.dry_run))
        t1.start(); t2.start()
        if blocking:
            t1.join(); t2.join()
            return t1.result and t2.result
        return True

    def both_out_scene(self, speed=None, acc=None, blocking=True):
        t1 = ThreadWithResult(target=safe_out_scene, args=(self.left, speed, acc, blocking, self.dry_run))
        t2 = ThreadWithResult(target=safe_out_scene, args=(self.right, speed, acc, blocking, self.dry_run))
        t1.start(); t2.start()
        if blocking:
            t1.join(); t2.join()
            return t1.result and t2.result
        return True

    def both_open_gripper(self):
        t1 = ThreadWithResult(target=safe_gripper, args=(self.left, "open", self.dry_run))
        t2 = ThreadWithResult(target=safe_gripper, args=(self.right, "open", self.dry_run))
        t1.start(); t2.start(); t1.join(); t2.join()
        return True

    def both_close_gripper(self):
        t1 = ThreadWithResult(target=safe_gripper, args=(self.left, "close", self.dry_run))
        t2 = ThreadWithResult(target=safe_gripper, args=(self.right, "close", self.dry_run))
        t1.start(); t2.start(); t1.join(); t2.join()
        return True

    def both_fling(self, left_path, right_path, speed, acc, record=False):
        r = self.both_movel(left_path[0], right_path[0], speed=speed, acc=acc)
        if not r:
            return False
        return self.both_movel(left_path[1:], right_path[1:], speed=speed, acc=acc, record=record)

    def get_tcp_distance(self):
        if self.dry_run:
            return 0.5
        left_tcp = self.left.get_tcp_pose()
        right_tcp = transform_pose(self.T_left_right, self.right.get_tcp_pose())
        return np.linalg.norm((right_tcp - left_tcp)[:3])

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def go_camera_pos(self):
        self.both_home()
        self.both_out_scene()

    def take_rgbd(self):
        """The SQUARE window between the arms, not the raw frame.

        Paired with the shifted ``self.intr``, so a pixel in what comes back here
        still inverts to the right point on the table. ``self.camera.take_rgbd()``
        is still the way to get the full frame (e.g. for calibration).
        """
        rgb, depth = self.camera.take_rgbd()
        if self.crop is None:
            return rgb, depth
        return self.crop.apply(rgb), self.crop.apply(depth)

    def restart_camera(self):
        self.camera.restart()

    def get_workspace_masks(self):
        return self.left_mask, self.right_mask

    def get_T_base_cam(self):
        return self.T_left_cam

    def get_camera_intrinsic(self):
        return self.intr

    def _calculate_wrkspace_masks(self, rgb):
        H, W, _ = rgb.shape
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        pixel_points = np.stack([uu.ravel(), vv.ravel()], axis=1)

        left_base = pixels2base_on_table(pixel_points, self.intr, self.T_left_cam, XARM_TABLE_Z)
        right_base = pixels2base_on_table(pixel_points, self.intr, self.T_right_cam, XARM_TABLE_Z)

        left_dist = np.linalg.norm(left_base[:, :2], axis=1)
        right_dist = np.linalg.norm(right_base[:, :2], axis=1)

        left = (left_dist >= self.left_radius[0]) & (left_dist <= self.left_radius[1])
        right = (right_dist >= self.right_radius[0]) & (right_dist <= self.right_radius[1])

        self.left_mask = left.reshape(H, W)
        self.right_mask = right.reshape(H, W)
