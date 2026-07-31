"""Dual-arm pick-and-fling for two xArm Lite 6 arms.

Drop-in for ``PickAndFlingSkill`` (same ``reset()`` / ``step(action, record_debug)``
surface and action layout ``[x0,y0, x1,y1, a0,a1(, valid0,valid1)]`` in pixels;
returns a trajectory dict). Adapted to the Lite 6, which has NO force/torque
sensor, so the UR force-mode stretch and force-mode tension release are replaced
by POSITION-BASED motions:
  * fixed-height grasp (no contact probing);
  * geometric stretch to a capped width along the base-to-base axis;
  * the fling swing path is reused from ``points_to_fling_path`` (pure geometry);
  * release = open grippers.

⚠️ Fling dynamics (hang height, stroke, swing angle, speeds) are hardware- and
fabric-dependent and must be tuned on the real cell.
"""
import time
import numpy as np

from real_robot.utils.transform_utils import (
    point_on_table_base, transform_point, transform_pose,
)
from real_robot.utils.xarm_constants import (
    XARM_TABLE_Z, XARM_MIN_Z, XARM_GRIPPER_OFFSET, XARM_APPROACH_DIST,
    XARM_MOVE_SPEED, XARM_MOVE_ACC, XARM_COLLISION_THRESHOLD, XARM_DOWN_ROTVEC,
    XARM_TABLE_Z_BY_SIDE, XARM_GRIPPER_OFFSET_BY_SIDE, for_side,
)
from .utils import check_trajectories_close, apply_local_z_rotation, points_to_fling_path


# Fling tuning (metres / m·s⁻¹). Conservative defaults; tune on hardware.
HANG_HEIGHT = 0.30
STRETCH_MAX_WIDTH = 0.45
MIN_STRETCH_DIST = 0.20
FLING_SPEED = 1.5
FLING_ACC = 3.0
SWING_STROKE = 0.45
SWING_ANGLE = np.pi / 4
PLACE_HEIGHT = 0.10


class XArmPickAndFlingSkill:
    def __init__(self, scene, config=None):
        self.scene = scene
        if config is None:
            config = {}
        self.min_z = XARM_MIN_Z
        self.approach_dist = XARM_APPROACH_DIST
        self.move_speed = config.get('speed', XARM_MOVE_SPEED)
        self.move_acc = config.get('acc', XARM_MOVE_ACC)
        self.collision_threshold = config.get('collision_threshold', XARM_COLLISION_THRESHOLD)
        self.hang_height = config.get('hang_height', HANG_HEIGHT)
        self.stretch_max_width = config.get('stretch_max_width', STRETCH_MAX_WIDTH)

    def reset(self):
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def _append(self, acc):
        if getattr(self.scene, 'last_trajectory', None):
            t = self.scene.last_trajectory
            acc['ur5e'].extend(t.get('ur5e', []))
            acc['ur16e'].extend(t.get('ur16e', []))

    def step(self, action, record_debug=False):
        # Keys mirror the UR scene ('ur5e'=left, 'ur16e'=right) so the reused imp
        # logger projects the recorded fling trajectory correctly.
        full_traj = {'ur5e': [], 'ur16e': []}
        action = np.asarray(action, dtype=float)

        valid_0 = valid_1 = 1.0
        if len(action) >= 8:
            pick_0_xy, pick_1_xy = action[0:2], action[2:4]
            angle_0, angle_1 = action[4], action[5]
            valid_0, valid_1 = action[6], action[7]
        elif len(action) >= 6:
            pick_0_xy, pick_1_xy = action[0:2], action[2:4]
            angle_0, angle_1 = action[4], action[5]
        else:
            pick_0_xy, pick_1_xy = action[0:2], action[2:4]
            angle_0, angle_1 = 0.0, 0.0

        if valid_0 < 0.5 or valid_1 < 0.5:
            print(f"[XArmPickAndFling] ABORT: invalid pick (flags {valid_0}, {valid_1}).")
            return full_traj

        # Sort so larger-x pick -> LEFT arm.
        pair_0 = {'pick': pick_0_xy, 'angle': angle_0}
        pair_1 = {'pick': pick_1_xy, 'angle': angle_1}
        if pair_0['pick'][0] < pair_1['pick'][0]:
            pair_0, pair_1 = pair_1, pair_0
        pick_l, ang_l = pair_0['pick'], pair_0['angle']
        pick_r, ang_r = pair_1['pick'], pair_1['angle']

        p_pick_l = self._table_point(pick_l, self.scene.T_left_cam, 'left')
        p_pick_r = self._table_point(pick_r, self.scene.T_right_cam, 'right')

        # Collision check in the LEFT base frame.
        p_pick_r_in_l = transform_point(self.scene.T_left_right, p_pick_r)
        traj_l = [p_pick_l + [0, 0, self.approach_dist], p_pick_l]
        traj_r = [p_pick_r_in_l + [0, 0, self.approach_dist], p_pick_r_in_l]
        conflict, dist = check_trajectories_close(traj_l, traj_r, threshold=self.collision_threshold)
        if conflict:
            print(f"[XArmPickAndFling] ABORT: collision predicted (min dist {dist:.3f} m).")
            return full_traj

        rot_l = apply_local_z_rotation(np.array(XARM_DOWN_ROTVEC, dtype=float), ang_l)
        rot_r = apply_local_z_rotation(np.array(XARM_DOWN_ROTVEC, dtype=float), ang_r)

        # 1. Approach + fixed-height grasp
        self.scene.both_open_gripper()
        self.scene.both_home(blocking=True)
        app_l = np.concatenate([p_pick_l + [0, 0, self.approach_dist], rot_l])
        app_r = np.concatenate([p_pick_r + [0, 0, self.approach_dist], rot_r])
        self.scene.both_movel(app_l, app_r, speed=self.move_speed, acc=self.move_acc, blocking=True)
        grasp_l = np.concatenate([p_pick_l, rot_l])
        grasp_r = np.concatenate([p_pick_r, rot_r])
        self.scene.both_movel(grasp_l, grasp_r, speed=self.move_speed * 0.5, acc=self.move_acc * 0.5, blocking=True)
        self.scene.both_close_gripper()
        time.sleep(0.5)

        # 2. Lift to hang height and centre/stretch along the base-to-base axis
        base_to_base = self.scene.T_left_right[:3, 3]
        center = base_to_base / 2.0
        axis = base_to_base / (np.linalg.norm(base_to_base) + 1e-9)
        width = min(self.stretch_max_width, max(MIN_STRETCH_DIST, np.linalg.norm(base_to_base)))

        target_l_in_l = center - axis * width / 2.0
        target_r_in_l = center + axis * width / 2.0
        target_l_in_l[2] = self.hang_height
        target_r_in_l[2] = self.hang_height

        target_r_in_r = transform_point(np.linalg.inv(self.scene.T_left_right), target_r_in_l)
        self.scene.both_movel(
            np.concatenate([target_l_in_l, rot_l]),
            np.concatenate([target_r_in_r, rot_r]),
            speed=self.move_speed, acc=self.move_acc, blocking=True, record=record_debug)
        if record_debug:
            self._append(full_traj)

        # 3. Fling swing path (geometry reused; right path mapped back to right base)
        left_path_full, right_path_full = points_to_fling_path(
            right_point=np.asarray(target_l_in_l),
            left_point=np.asarray(target_r_in_l),
            width=None,
            swing_stroke=SWING_STROKE,
            swing_angle=SWING_ANGLE,
            lift_height=self.hang_height,
            place_height=PLACE_HEIGHT)
        left_path_full[0][:3] = target_l_in_l
        right_path_full[0][:3] = target_r_in_l

        left_fling = left_path_full[:-1]
        right_fling = right_path_full[:-1]
        left_drag = left_path_full[-1]
        right_drag = right_path_full[-1]

        right_fling_base = transform_pose(np.linalg.inv(self.scene.T_left_right), right_fling)
        self.scene.both_fling(left_fling, right_fling_base, FLING_SPEED, FLING_ACC, record=record_debug)
        if record_debug:
            self._append(full_traj)

        right_drag_base = transform_pose(np.linalg.inv(self.scene.T_left_right), right_drag)
        self.scene.both_movel(left_drag.flatten(), right_drag_base.flatten(),
                              speed=self.move_speed, acc=self.move_acc, blocking=True, record=record_debug)
        if record_debug:
            self._append(full_traj)

        # 4. Release (no force tension release on the Lite 6)
        self.scene.both_open_gripper()
        self.scene.both_home()
        return full_traj

    def _table_point(self, pixel, cam_T, side='left'):
        table_z = for_side(XARM_TABLE_Z_BY_SIDE, side, XARM_TABLE_Z)
        offset = for_side(XARM_GRIPPER_OFFSET_BY_SIDE, side, XARM_GRIPPER_OFFSET)
        p = point_on_table_base(pixel[0], pixel[1], self.scene.intr, cam_T, table_z)
        p = np.asarray(p, dtype=float)
        # Clamp the COMMANDED z, not the table plane: clamping first and adding the
        # gripper offset afterwards lifts every grasp by min_z, so the fingertips
        # stop short of the fabric.
        p[2] = max(self.min_z, float(p[2]) + offset)
        return p
