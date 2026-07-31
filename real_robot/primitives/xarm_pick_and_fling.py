"""Dual-arm pick-and-fling for two xArm Lite 6 arms.

Drop-in for ``PickAndFlingSkill`` (``real_robot/primitives/pick_and_fling.py``):
same ``reset()`` / ``step(action, record_debug)`` surface, same action layout
``[x0,y0, x1,y1, a0,a1(, valid0,valid1)]`` in pixels, same returned trajectory
dict. The method decomposition deliberately mirrors the UR file so the two can be
read side by side: ``dual_arm_stretch_and_fling`` -> ``dual_arm_stretch`` ->
``dual_arm_shake`` -> ``dual_arm_release_tension``.

WHAT DIFFERS, AND WHY
---------------------
The Lite 6 has no force/torque sensor, so the UR's three force-mode stages are
replaced by position-controlled motions gated on JOINT EFFORT
(``XArmLite6.get_joint_effort``), which is the only load signal this arm has:

  * ``move_until_contact`` (descend until 5 N)  -> ``_probe_contact``: descends in
    small steps toward the CALIBRATED grasp height, stopping early if effort rises.
    It is bounded below by that height, so it can only ever stop ABOVE the table.
  * force-mode stretch (6 N until taut)         -> ``dual_arm_stretch``: steps
    outward until the width cap, a timeout, or an effort rise -- the same three
    exit conditions as the UR loop, with the effort rise standing in for its
    "TCP speed dropped below 5 mm/s" test.
  * force-mode tension release (-2 N inward)    -> ``dual_arm_release_tension``:
    a short inward position move before the grippers open.

⚠️ The effort signal is weak on a 0.61 kg-payload arm holding light fabric. Every
stage keeps a hard geometric cap, and behaves correctly if the effort never fires.

GEOMETRY. The fling constants live in ``xarm_constants.py`` because they are
derived from the cell (base separation and measured reach), not chosen -- see the
note there. A naive port of the UR numbers puts the wind-up waypoint 0.55 m from
the base, well outside the Lite 6's 0.41 m reach.
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
    XARM_FLING_WIDTH, XARM_FLING_HANG, XARM_FLING_STROKE, XARM_FLING_ANGLE,
    XARM_FLING_PLACE_Z, XARM_FLING_MIN_WIDTH, XARM_FLING_SPEED, XARM_FLING_ACC,
    XARM_SHAKE_COUNT, XARM_SHAKE_AMPLITUDE, XARM_SHAKE_SPEED, XARM_SHAKE_ACC,
    XARM_STRETCH_STEP, XARM_STRETCH_MAX_TIME, XARM_STRETCH_SPEED,
    XARM_RELEASE_DIST, XARM_PROBE_STEP, XARM_EFFORT_THRESHOLD,
)
from real_robot.utils.thread_utils import ThreadWithResult
from .utils import check_trajectories_close, apply_local_z_rotation, points_to_fling_path


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
        self.hang_height = config.get('hang_height', XARM_FLING_HANG)
        self.stretch_max_width = config.get('stretch_max_width', XARM_FLING_WIDTH)
        self.swing_stroke = config.get('swing_stroke', XARM_FLING_STROKE)
        self.place_height = config.get('place_height', XARM_FLING_PLACE_Z)
        # Each stage can be switched off for incremental hardware bring-up.
        self.do_probe = config.get('probe_contact', True)
        self.do_shake = config.get('shake', True)
        self.do_release_tension = config.get('release_tension', True)

    def reset(self):
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def _append(self, acc):
        if acc is None:
            return
        if getattr(self.scene, 'last_trajectory', None):
            t = self.scene.last_trajectory
            acc['ur5e'].extend(t.get('ur5e', []))
            acc['ur16e'].extend(t.get('ur16e', []))

    # ------------------------------------------------------------------
    def step(self, action, record_debug=False):
        # Keys mirror the UR scene ('ur5e' = our LEFT, 'ur16e' = our RIGHT) because
        # the reused imp logger indexes the recorded fling trajectory by those names.
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

        # Sort so the larger-pixel-x pick goes to the LEFT arm, as the UR skill does.
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
        conflict, dist = check_trajectories_close(traj_l, traj_r,
                                                  threshold=self.collision_threshold)
        if conflict:
            print(f"[XArmPickAndFling] ABORT: collision predicted (min dist {dist:.3f} m).")
            return full_traj

        rot_l = apply_local_z_rotation(np.array(XARM_DOWN_ROTVEC, dtype=float), ang_l)
        rot_r = apply_local_z_rotation(np.array(XARM_DOWN_ROTVEC, dtype=float), ang_r)

        # 1. Approach, then descend to the grasp.
        self.scene.both_open_gripper()
        self.scene.both_home(blocking=True)
        app_l = np.concatenate([p_pick_l + [0, 0, self.approach_dist], rot_l])
        app_r = np.concatenate([p_pick_r + [0, 0, self.approach_dist], rot_r])
        self.scene.both_movel(app_l, app_r, speed=self.move_speed, acc=self.move_acc,
                              blocking=True)

        grasp_l = np.concatenate([p_pick_l, rot_l])
        grasp_r = np.concatenate([p_pick_r, rot_r])
        if self.do_probe:
            grasp_l, grasp_r = self._probe_both(app_l, app_r, grasp_l, grasp_r)
        else:
            self.scene.both_movel(grasp_l, grasp_r, speed=self.move_speed * 0.5,
                                  acc=self.move_acc * 0.5, blocking=True)

        self.scene.both_close_gripper()
        time.sleep(0.5)

        # 2. Lift to the hang height, centred on the base-to-base axis.
        base_to_base = self.scene.T_left_right[:3, 3]
        center = base_to_base / 2.0
        axis = base_to_base / (np.linalg.norm(base_to_base) + 1e-9)
        width = min(self.stretch_max_width,
                    max(XARM_FLING_MIN_WIDTH, float(np.linalg.norm(base_to_base))))

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

        # 3. Stretch, shake, fling, drag, release.
        self.dual_arm_stretch_and_fling(rot_l, rot_r, axis,
                                        record_debug=record_debug,
                                        full_trajectory_ref=full_traj)

        self.scene.both_home()
        return full_traj

    # ------------------------------------------------------------------
    def dual_arm_stretch_and_fling(self, rot_l, rot_r, axis,
                                   record_debug=False, full_trajectory_ref=None):
        """Stretch -> shake -> fling -> drag -> release, in the UR's order."""
        if not self.dual_arm_stretch(rot_l, rot_r, axis,
                                     record_debug=record_debug,
                                     full_trajectory_ref=full_trajectory_ref):
            return False

        if self.do_shake:
            self.dual_arm_shake(record_debug=record_debug,
                                full_trajectory_ref=full_trajectory_ref)

        # Re-read where the arms ACTUALLY ended up. The stretch can stop early on
        # effort, so the commanded targets are not where the grippers are, and the
        # swing path has to start from the real poses (the UR does the same).
        p_l = self.scene.left.get_tcp_pose()[:3]
        p_r_in_l = transform_point(self.scene.T_left_right,
                                   self.scene.right.get_tcp_pose()[:3])

        # NOTE: `right_point`/`left_point` name the FLING frame, not our arms.
        # points_to_fling_path puts `right_point` at x = -width/2, which in the left
        # base frame is the smaller x -- our LEFT arm. This pairing is correct.
        left_path_full, right_path_full = points_to_fling_path(
            right_point=np.asarray(p_l, dtype=float),
            left_point=np.asarray(p_r_in_l, dtype=float),
            width=None,
            swing_stroke=self.swing_stroke,
            swing_angle=XARM_FLING_ANGLE,
            lift_height=self.hang_height,
            place_height=self.place_height)
        left_path_full[0][:3] = p_l
        right_path_full[0][:3] = p_r_in_l

        left_fling, right_fling = left_path_full[:-1], right_path_full[:-1]
        left_drag, right_drag = left_path_full[-1], right_path_full[-1]

        right_fling_base = transform_pose(np.linalg.inv(self.scene.T_left_right), right_fling)
        self.scene.both_fling(left_fling, right_fling_base,
                              XARM_FLING_SPEED, XARM_FLING_ACC, record=record_debug)
        if record_debug:
            self._append(full_trajectory_ref)

        right_drag_base = transform_pose(np.linalg.inv(self.scene.T_left_right), right_drag)
        self.scene.both_movel(left_drag.flatten(), right_drag_base.flatten(),
                              speed=self.move_speed, acc=self.move_acc, blocking=True,
                              record=record_debug)
        if record_debug:
            self._append(full_trajectory_ref)

        if self.do_release_tension:
            self.dual_arm_release_tension(axis, record_debug=record_debug,
                                          full_trajectory_ref=full_trajectory_ref)

        self.scene.both_open_gripper()
        return True

    # ------------------------------------------------------------------
    def dual_arm_stretch(self, rot_l, rot_r, axis,
                         max_width=None, max_time=XARM_STRETCH_MAX_TIME,
                         step=XARM_STRETCH_STEP, speed=XARM_STRETCH_SPEED,
                         record_debug=False, full_trajectory_ref=None):
        """Pull outward along the base-to-base axis until the cloth resists.

        Same three exit conditions as the UR's force-mode loop -- width cap,
        timeout, load detected -- but the load test is a rise in joint effort
        rather than the UR's "TCP speed fell below 5 mm/s under constant force".

        The width cap is the SAFETY bound and the effort test is an optimisation:
        if effort is unreadable or never rises, this simply stretches to the cap,
        which is the behaviour this primitive had before.
        """
        max_width = self.stretch_max_width if max_width is None else max_width

        # Baseline while holding still, before pulling -- see XArmLite6's note on
        # why this must not be sampled mid-motion.
        base_l = self.scene.left.effort_baseline()
        base_r = self.scene.right.effort_baseline()
        thr_l = for_side(XARM_EFFORT_THRESHOLD, 'left', 2.0)
        thr_r = for_side(XARM_EFFORT_THRESHOLD, 'right', 2.0)
        if base_l is None or base_r is None:
            print("[XArmPickAndFling] joint effort unavailable; stretching to the "
                  "geometric width cap only.")

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                print(f"[Stretch] timeout after {elapsed:.1f} s.")
                break

            tcp_distance = self.scene.get_tcp_distance()
            if tcp_distance >= max_width:
                print(f"[Stretch] width cap reached: {tcp_distance:.3f} m.")
                break

            loaded_l = self.scene.left.effort_exceeded(base_l, thr_l)
            loaded_r = self.scene.right.effort_exceeded(base_r, thr_r)
            if loaded_l or loaded_r:
                d_l = self.scene.left.effort_delta(base_l)
                d_r = self.scene.right.effort_delta(base_r)
                print("[Stretch] cloth taut at {:.3f} m (effort delta L={}, R={}).".format(
                    tcp_distance,
                    "n/a" if d_l is None else "{:.2f}".format(d_l),
                    "n/a" if d_r is None else "{:.2f}".format(d_r)))
                break

            # One increment outward for each arm, along the base-to-base axis.
            p_l = self.scene.left.get_tcp_pose()[:3] - axis * step
            p_r_in_l = transform_point(self.scene.T_left_right,
                                       self.scene.right.get_tcp_pose()[:3]) + axis * step
            p_r = transform_point(np.linalg.inv(self.scene.T_left_right), p_r_in_l)
            ok = self.scene.both_movel(
                np.concatenate([p_l, rot_l]), np.concatenate([p_r, rot_r]),
                speed=speed, acc=self.move_acc, blocking=True, record=record_debug)
            if record_debug:
                self._append(full_trajectory_ref)
            if not ok:
                # A refused step means the next one is out of reach or off-limits;
                # stop here rather than hammering the controller.
                print("[Stretch] a step was refused; stopping the stretch here.")
                break
        return True

    # ------------------------------------------------------------------
    def dual_arm_shake(self, num_shakes=XARM_SHAKE_COUNT, amplitude=XARM_SHAKE_AMPLITUDE,
                       speed=XARM_SHAKE_SPEED, acc=XARM_SHAKE_ACC,
                       record_debug=False, full_trajectory_ref=None):
        """Rapid vertical bounces to loosen folds. A direct port of the UR stage --
        it is pure position control, so nothing had to be substituted."""
        start_l = self.scene.left.get_tcp_pose()
        start_r = self.scene.right.get_tcp_pose()

        path_l, path_r = [], []
        for _ in range(int(num_shakes)):
            for sign in (+1.0, -1.0):
                up_l = np.array(start_l, dtype=float)
                up_l[2] += sign * amplitude
                path_l.append(up_l)
                up_r = np.array(start_r, dtype=float)
                up_r[2] += sign * amplitude
                path_r.append(up_r)
        path_l.append(np.array(start_l, dtype=float))
        path_r.append(np.array(start_r, dtype=float))

        self.scene.both_movel(path_l, path_r, speed=speed, acc=acc, blocking=True,
                              record=record_debug)
        if record_debug:
            self._append(full_trajectory_ref)
        return True

    # ------------------------------------------------------------------
    def dual_arm_release_tension(self, axis, distance=XARM_RELEASE_DIST,
                                 speed=XARM_STRETCH_SPEED,
                                 record_debug=False, full_trajectory_ref=None):
        """Slacken the cloth before opening, so the release does not drag it.

        Position-based stand-in for the UR's -2 N inward force mode: both arms move
        ``distance`` toward each other along the base-to-base axis.
        """
        p_l = self.scene.left.get_tcp_pose()
        p_r = self.scene.right.get_tcp_pose()
        p_r_in_l = transform_point(self.scene.T_left_right, p_r[:3])

        new_l = np.concatenate([p_l[:3] + axis * distance, p_l[3:6]])
        new_r_in_l = p_r_in_l - axis * distance
        new_r = np.concatenate([
            transform_point(np.linalg.inv(self.scene.T_left_right), new_r_in_l), p_r[3:6]])

        self.scene.both_movel(new_l, new_r, speed=speed, acc=self.move_acc,
                              blocking=True, record=record_debug)
        if record_debug:
            self._append(full_trajectory_ref)
        return True

    # ------------------------------------------------------------------
    def _probe_both(self, app_l, app_r, grasp_l, grasp_r):
        """Descend both arms onto the cloth in parallel; returns the reached poses."""
        results = [grasp_l, grasp_r]

        def run(index, robot, start_pose, target_pose, side):
            results[index] = self._probe_contact(robot, start_pose, target_pose, side)

        t0 = ThreadWithResult(target=run, args=(0, self.scene.left, app_l, grasp_l, 'left'))
        t1 = ThreadWithResult(target=run, args=(1, self.scene.right, app_r, grasp_r, 'right'))
        t0.start(); t1.start()
        t0.join(); t1.join()
        return results[0], results[1]

    def _probe_contact(self, robot, start_pose, target_pose, side,
                       step=XARM_PROBE_STEP, speed=None):
        """Descend toward the CALIBRATED grasp height, stopping early on contact.

        The UR analogue watches a 5 N rise on the F/T sensor. Here the signal is
        joint effort, which is far weaker -- so the calibrated height is a hard
        floor: this can only ever stop the descent EARLY (on a thick fold), never
        drive deeper than the open-loop descent would have. If effort is
        unreadable the behaviour is exactly the previous fixed-height grasp.
        """
        speed = self.move_speed * 0.5 if speed is None else speed
        baseline = robot.effort_baseline()
        threshold = for_side(XARM_EFFORT_THRESHOLD, side, 2.0)

        start_z = float(start_pose[2])
        floor_z = float(target_pose[2])
        if baseline is None or start_z <= floor_z:
            robot.movel(target_pose, speed=speed, acceleration=self.move_acc * 0.5,
                        blocking=True)
            return np.array(target_pose, dtype=float)

        pose = np.array(start_pose, dtype=float)
        n_steps = int(np.ceil((start_z - floor_z) / max(step, 1e-4)))
        for _ in range(n_steps):
            pose[2] = max(floor_z, pose[2] - step)
            if not robot.movel(pose, speed=speed, acceleration=self.move_acc * 0.5,
                               blocking=True):
                break
            if robot.effort_exceeded(baseline, threshold):
                delta = robot.effort_delta(baseline)
                print("[Probe] {} contact at z={:+.4f} m ({:.3f} m above the calibrated "
                      "grasp height, effort delta {:.2f}).".format(
                          side, pose[2], pose[2] - floor_z, delta))
                return np.array(pose, dtype=float)
            if pose[2] <= floor_z + 1e-9:
                break
        return np.array(pose, dtype=float)

    # ------------------------------------------------------------------
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
