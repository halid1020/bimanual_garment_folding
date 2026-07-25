"""Dual-arm pick-and-place for two xArm Lite 6 arms.

Drop-in for ``PickAndPlaceSkill`` (same ``reset()`` / ``step(action)`` surface,
same 12-element action layout ``[p0,p1,pl0,pl1, a0,a1, active0,active1]`` in
pixel coordinates), but adapted to the Lite 6:
  * NO force/torque sensor -> grasp descends to a FIXED calibrated table height
    (``point_on_table_base`` + ``XARM_GRIPPER_OFFSET``); the controller's
    collision detection is the safety stop.
  * NO force-mode tension release. An OPTIONAL gentle position-based inward nudge
    is available (``inward_release`` config flag, default off).
  * The ``check_trajectories_close`` sequential-vs-simultaneous fallback is kept,
    because at ~35 cm base separation the two arms can meet in the middle.

Arm mapping mirrors the UR skill: the larger-x pick goes to the LEFT arm (the
UR5e analogue), the smaller-x pick to the RIGHT arm.
"""
import time
import numpy as np

from real_robot.utils.transform_utils import point_on_table_base, transform_point
from real_robot.utils.xarm_constants import (
    XARM_TABLE_Z, XARM_MIN_Z, XARM_GRIPPER_OFFSET, XARM_APPROACH_DIST,
    XARM_LIFT_DIST, XARM_MOVE_SPEED, XARM_MOVE_ACC, XARM_COLLISION_THRESHOLD,
    XARM_DOWN_ROTVEC,
)
from real_robot.utils.thread_utils import ThreadWithResult
from .utils import check_trajectories_close, apply_local_z_rotation


class XArmPickAndPlaceSkill:
    def __init__(self, scene, config=None):
        self.scene = scene
        if config is None:
            config = {}
        self.min_z = XARM_MIN_Z
        self.approach_dist = XARM_APPROACH_DIST
        self.lift_dist = XARM_LIFT_DIST
        self.move_speed = config.get('speed', XARM_MOVE_SPEED)
        self.move_acc = config.get('acc', XARM_MOVE_ACC)
        self.collision_threshold = config.get('collision_threshold', XARM_COLLISION_THRESHOLD)
        self.home_after = True
        self.inward_release = config.get('inward_release', False)
        self.inward_release_dist = config.get('inward_release_dist', 0.03)  # m

    def reset(self):
        self.scene.both_open_gripper()
        self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
        time.sleep(0.5)

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=float)
        if len(action) >= 12:
            pick_0_xy, pick_1_xy = action[0:2], action[2:4]
            place_0_xy, place_1_xy = action[4:6], action[6:8]
            angle_0, angle_1 = action[8], action[9]
            active_0, active_1 = bool(action[10]), bool(action[11])
        else:
            pick_0_xy, pick_1_xy = action[0:2], action[2:4]
            place_0_xy, place_1_xy = action[4:6], action[6:8]
            angle_0, angle_1 = 0.0, 0.0
            active_0, active_1 = True, True

        # Sort by pick-x so the larger-x pick -> LEFT arm (UR5e analogue).
        pair_0 = {'pick': pick_0_xy, 'place': place_0_xy, 'angle': angle_0, 'active': active_0}
        pair_1 = {'pick': pick_1_xy, 'place': place_1_xy, 'angle': angle_1, 'active': active_1}
        if pair_0['pick'][0] < pair_1['pick'][0]:
            pair_0, pair_1 = pair_1, pair_0
        pick_l, place_l, ang_l, active_l = pair_0['pick'], pair_0['place'], pair_0['angle'], pair_0['active']
        pick_r, place_r, ang_r, active_r = pair_1['pick'], pair_1['place'], pair_1['angle'], pair_1['active']

        p_pick_l = p_place_l = rot_l = None
        if active_l:
            p_pick_l, p_place_l, rot_l = self._process_coords(pick_l, place_l, ang_l, self.scene.T_left_cam)
        p_pick_r = p_place_r = rot_r = None
        if active_r:
            p_pick_r, p_place_r, rot_r = self._process_coords(pick_r, place_r, ang_r, self.scene.T_right_cam)

        print(f"[XArmPickAndPlace] active -> left: {active_l}, right: {active_r}")

        if active_l and active_r:
            # Collision check in the LEFT base frame.
            p_pick_r_in_l = transform_point(self.scene.T_left_right, p_pick_r)
            p_place_r_in_l = transform_point(self.scene.T_left_right, p_place_r)
            traj_l = [p_pick_l + [0, 0, self.approach_dist], p_pick_l,
                      p_pick_l + [0, 0, self.approach_dist],
                      p_place_l + [0, 0, self.approach_dist], p_place_l]
            traj_r = [p_pick_r_in_l + [0, 0, self.approach_dist], p_pick_r_in_l,
                      p_pick_r_in_l + [0, 0, self.approach_dist],
                      p_place_r_in_l + [0, 0, self.approach_dist], p_place_r_in_l]
            conflict, _ = check_trajectories_close(traj_l, traj_r, threshold=self.collision_threshold)

            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
            self.scene.both_open_gripper()

            if conflict:
                print("[XArmPickAndPlace] Collision predicted; executing sequentially.")
                self._execute_single(self.scene.left, p_pick_l, p_place_l, rot_l)
                self.scene.left.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)
                self._execute_single(self.scene.right, p_pick_r, p_place_r, rot_r)
                self.scene.right.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)
            else:
                self._execute_dual(p_pick_l, p_place_l, rot_l, p_pick_r, p_place_r, rot_r)

        elif active_l:
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
            self.scene.left.open_gripper()
            self._execute_single(self.scene.left, p_pick_l, p_place_l, rot_l)
            if self.home_after:
                self.scene.left.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)

        elif active_r:
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)
            self.scene.right.open_gripper()
            self._execute_single(self.scene.right, p_pick_r, p_place_r, rot_r)
            if self.home_after:
                self.scene.right.home(speed=self.move_speed, acceleration=self.move_acc, blocking=True)
        else:
            print("[XArmPickAndPlace] No active arms; skipping.")

    # ------------------------------------------------------------------
    def _process_coords(self, pick_xy, place_xy, rot_angle, cam_T):
        p_pick = point_on_table_base(pick_xy[0], pick_xy[1], self.scene.intr, cam_T, XARM_TABLE_Z)
        p_place = point_on_table_base(place_xy[0], place_xy[1], self.scene.intr, cam_T, XARM_TABLE_Z)
        p_pick[2] = max(self.min_z, float(p_pick[2])) + XARM_GRIPPER_OFFSET
        p_place[2] = max(self.min_z, float(p_place[2])) + XARM_GRIPPER_OFFSET
        rot = apply_local_z_rotation(np.array(XARM_DOWN_ROTVEC, dtype=float), rot_angle)
        return p_pick, p_place, rot

    def _execute_single(self, robot, pick_pt, place_pt, rot):
        approach_pick = np.concatenate([pick_pt + [0, 0, self.approach_dist], rot])
        robot.movel(approach_pick, speed=self.move_speed, acceleration=self.move_acc, blocking=True)
        robot.open_gripper()
        # Fixed-height descent (no force probing).
        grasp = np.concatenate([pick_pt, rot])
        robot.movel(grasp, speed=self.move_speed * 0.5, acceleration=self.move_acc * 0.5, blocking=True)
        robot.close_gripper()
        time.sleep(0.3)

        lift = np.concatenate([pick_pt + [0, 0, self.lift_dist], rot])
        approach_place = np.concatenate([place_pt + [0, 0, self.approach_dist], rot])
        place = np.concatenate([place_pt + [0, 0, 0.01], rot])
        robot.movel([lift, approach_place, place], speed=self.move_speed, acceleration=self.move_acc, blocking=True)

        robot.open_gripper()
        time.sleep(0.2)
        retract = np.concatenate([place_pt + [0, 0, self.approach_dist], rot])
        robot.movel(retract, speed=self.move_speed, acceleration=self.move_acc, blocking=True)

    def _execute_dual(self, p_pick_l, p_place_l, rot_l, p_pick_r, p_place_r, rot_r):
        app_pick_l = np.concatenate([p_pick_l + [0, 0, self.approach_dist], rot_l])
        app_pick_r = np.concatenate([p_pick_r + [0, 0, self.approach_dist], rot_r])
        self.scene.both_movel(app_pick_l, app_pick_r, speed=self.move_speed, acc=self.move_acc, blocking=True)

        grasp_l = np.concatenate([p_pick_l, rot_l])
        grasp_r = np.concatenate([p_pick_r, rot_r])
        self.scene.both_movel(grasp_l, grasp_r, speed=self.move_speed * 0.5, acc=self.move_acc * 0.5, blocking=True)
        self.scene.both_close_gripper()
        time.sleep(0.3)

        lift_l = np.concatenate([p_pick_l + [0, 0, self.lift_dist], rot_l])
        lift_r = np.concatenate([p_pick_r + [0, 0, self.lift_dist], rot_r])
        app_place_l = np.concatenate([p_place_l + [0, 0, self.approach_dist], rot_l])
        app_place_r = np.concatenate([p_place_r + [0, 0, self.approach_dist], rot_r])
        place_l = np.concatenate([p_place_l + [0, 0, 0.01], rot_l])
        place_r = np.concatenate([p_place_r + [0, 0, 0.01], rot_r])
        self.scene.both_movel([lift_l, app_place_l, place_l], [lift_r, app_place_r, place_r],
                              speed=self.move_speed, acc=self.move_acc, blocking=True)

        if self.inward_release:
            self._inward_release(p_place_l, p_place_r, rot_l, rot_r)

        self.scene.both_open_gripper()
        time.sleep(0.2)
        self.scene.both_movel(app_place_l, app_place_r, speed=self.move_speed, acc=self.move_acc, blocking=True)
        if self.home_after:
            self.scene.both_home(speed=self.move_speed, acc=self.move_acc, blocking=True)

    def _inward_release(self, p_place_l, p_place_r, rot_l, rot_r):
        """Optional gentle position-based nudge of each TCP toward the other arm,
        to relax fabric tension before releasing (no force mode)."""
        p_place_r_in_l = transform_point(self.scene.T_left_right, p_place_r)
        dir_l = np.array(p_place_r_in_l, dtype=float) - np.array(p_place_l, dtype=float)
        dir_l[2] = 0.0
        n = np.linalg.norm(dir_l)
        if n < 1e-6:
            return
        dir_l = dir_l / n
        T_right_left = np.linalg.inv(self.scene.T_left_right)
        p_place_l_in_r = transform_point(T_right_left, p_place_l)
        dir_r = np.array(p_place_l_in_r, dtype=float) - np.array(p_place_r, dtype=float)
        dir_r[2] = 0.0
        nr = np.linalg.norm(dir_r)
        if nr < 1e-6:
            return
        dir_r = dir_r / nr

        step = self.inward_release_dist
        target_l = np.concatenate([p_place_l + [0, 0, 0.01] + dir_l * step, rot_l])
        target_r = np.concatenate([p_place_r + [0, 0, 0.01] + dir_r * step, rot_r])
        self.scene.both_movel(target_l, target_r, speed=0.1, acc=0.1, blocking=True)
