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

  * ``move_until_contact`` (descend until 5 N)  -> ``_probe_both``: descends BOTH
    arms in lock step toward the CALIBRATED grasp height, stopping early if effort
    rises. It is bounded below by that height, so it can only ever stop ABOVE the
    table.
  * force-mode stretch (6 N until taut)         -> ``dual_arm_stretch``: steps
    outward until the width cap, a timeout, or an effort rise -- the same three
    exit conditions as the UR loop, with the effort rise standing in for its
    "TCP speed dropped below 5 mm/s" test.
  * force-mode tension release (-2 N inward)    -> ``dual_arm_release_tension``:
    a short inward position move before the grippers open.

⚠️ The effort signal is weak on a 0.61 kg-payload arm holding light fabric. Every
stage keeps a hard geometric cap, behaves correctly if the effort never fires, and
only lets effort STOP anything once ``XARM_EFFORT_VERIFIED`` says the thresholds
have been measured -- see ``_may_act_on_effort``.

GEOMETRY. The fling constants live in ``xarm_constants.py`` because they are
derived from the cell (base separation and measured reach), not chosen -- see the
note there. A naive port of the UR numbers puts the wind-up waypoint 0.55 m from
the base, well outside the Lite 6's 0.41 m reach.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.utils.transform_utils import (
    point_on_table_base, transform_point, transform_pose,
)
from real_robot.utils.xarm_constants import (
    XARM_TABLE_Z, XARM_MIN_Z, XARM_GRIPPER_OFFSET, XARM_APPROACH_DIST,
    XARM_MOVE_SPEED, XARM_MOVE_ACC, XARM_COLLISION_THRESHOLD, XARM_DOWN_ROTVEC,
    XARM_TABLE_Z_BY_SIDE, XARM_GRIPPER_OFFSET_BY_SIDE, for_side,
    XARM_FLING_WIDTH, XARM_FLING_HANG, XARM_FLING_STROKE, XARM_FLING_ANGLE,
    XARM_FLING_FORWARD_Y, XARM_EFFORT_VERIFIED, XARM_EFFORT_CONSECUTIVE,
    XARM_FLING_PLACE_Z, XARM_FLING_MIN_WIDTH, XARM_FLING_SPEED, XARM_FLING_ACC,
    XARM_FLING_WINDUP, XARM_FLING_PLACE_Y, XARM_FLING_LAND_Y, XARM_FLING_MAX_RADIUS,
    XARM_SHAKE_COUNT, XARM_SHAKE_AMPLITUDE, XARM_SHAKE_SPEED, XARM_SHAKE_ACC,
    XARM_STRETCH_STEP, XARM_STRETCH_MAX_TIME, XARM_STRETCH_SPEED,
    XARM_RELEASE_DIST, XARM_PROBE_STEP, XARM_PROBE_BAND, XARM_EFFORT_THRESHOLD,
)
from real_robot.utils.thread_utils import ThreadWithResult
from .utils import (
    check_trajectories_close, apply_local_z_rotation, xarm_points_to_fling_path,
    sort_pairs_by_table_x, retarget_path_to_grasp,
)


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
        # The backward leg, where the hands touch down, and where the cloth is laid
        # down. All three are deliberately NOT derived from swing_stroke -- see
        # xarm_base_fling_poses. land_y in particular USED to be swing_stroke, which
        # is what made the arms land at the far end of the throw.
        self.swing_windup = config.get('swing_windup', XARM_FLING_WINDUP)
        self.land_y = config.get('land_y', XARM_FLING_LAND_Y)
        self.place_y = config.get('place_y', XARM_FLING_PLACE_Y)
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

        # The pick nearer the left base goes to the LEFT arm. Compared on the TABLE,
        # not in pixels -- see sort_pairs_by_table_x for why the pixel sort the UR
        # skill uses cannot be carried over here.
        pair_l, pair_r = sort_pairs_by_table_x(
            {'pick': pick_0_xy, 'angle': angle_0},
            {'pick': pick_1_xy, 'angle': angle_1},
            self.scene.intr, self.scene.T_left_cam,
            for_side(XARM_TABLE_Z_BY_SIDE, 'left', XARM_TABLE_Z))
        pick_l, ang_l = pair_l['pick'], pair_l['angle']
        pick_r, ang_r = pair_r['pick'], pair_r['angle']

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
        #
        # The separation here is the CURRENT distance between the two picked
        # points, not the stretch cap -- the UR does the same (`curr_width`). Using
        # the cap would put the grippers at full stretch before the stretch stage
        # runs, leaving it nothing to do. The upper clamp is ours: the UR applies it
        # later via points_to_gripper_pose, but on this cell the cap is a reach
        # limit, so it has to bind before anything is commanded.
        base_to_base = self.scene.T_left_right[:3, 3]
        center = base_to_base / 2.0
        axis = base_to_base / (np.linalg.norm(base_to_base) + 1e-9)
        curr_width = float(np.linalg.norm(p_pick_r_in_l - p_pick_l))
        width = min(self.stretch_max_width, max(XARM_FLING_MIN_WIDTH, curr_width))

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
        pose_l = self.scene.left.get_tcp_pose()
        pose_r = self.scene.right.get_tcp_pose()
        p_l = pose_l[:3]
        p_r_in_l = transform_point(self.scene.T_left_right, pose_r[:3])
        # The ORIENTATIONS each arm is actually holding the cloth in, both
        # expressed in the LEFT base frame (which is the frame the fling path is
        # built in). The right arm's has to be rotated into it, exactly as its
        # position is -- comparing a right-frame rotation against a left-frame path
        # would be off by the 180 deg between the bases, which is the whole bug
        # this retargeting exists to remove.
        grasp_rot_l = Rotation.from_rotvec(pose_l[3:6])
        grasp_rot_r_in_l = (Rotation.from_matrix(self.scene.T_left_right[:3, :3])
                            * Rotation.from_rotvec(pose_r[3:6]))

        # NOTE: `right_point`/`left_point` name the FLING frame, not our arms.
        #
        # WHICH WAY THE SWING GOES is derived, not chosen: points_to_action_frame
        # takes forward = z x (left_point - right_point), so handing it our arms in
        # their natural order gives base +y -- which is the FRONT of this table, so
        # the natural order is already the right one and this branch is not taken.
        # xarm_base_fling_poses then winds up a LITTLE to y = -XARM_FLING_WINDUP
        # (6 cm into the 0.68 m back -- just enough to load the cloth), STROKES
        # forward to y = +stroke (into the 0.52 m front), comes down PART WAY BACK
        # at y = +XARM_FLING_LAND_Y rather than at the far end, and drags through
        # the base line to XARM_FLING_PLACE_Y just behind it, laying the cloth out
        # flat under the grippers.
        #
        # The swap branch stays because the direction should keep being CHECKED
        # against XARM_FLING_FORWARD_Y rather than left as an unstated assumption:
        # if the cell is ever re-laid so that front is -y, flipping that constant
        # is the whole change. Swapping the two points reverses the cross product
        # and so the whole swing; the returned paths swap with the arguments, hence
        # the swapped unpacking -- points_to_fling_path puts `right_point` at
        # x = -width/2, i.e. the smaller base x, i.e. our LEFT arm. The pairing is
        # what keeps each arm on its own side; it is not what sets the direction.
        toward_front = XARM_FLING_FORWARD_Y < 0
        near, far = (np.asarray(p_r_in_l, dtype=float), np.asarray(p_l, dtype=float)) \
            if toward_front else \
            (np.asarray(p_l, dtype=float), np.asarray(p_r_in_l, dtype=float))
        def build(stroke, hang):
            """The whole path, for a candidate stroke/hang, in each arm's OWN frame."""
            a, b = xarm_points_to_fling_path(
                right_point=near,
                left_point=far,
                width=None,
                swing_stroke=stroke,
                swing_angle=XARM_FLING_ANGLE,
                lift_height=hang,
                place_height=self.place_height,
                windup=self.swing_windup,
                place_y=self.place_y,
                land_y=self.land_y)
            l_path, r_path = (b, a) if toward_front else (a, b)
            # Keep the swing's wrist PITCH, drop its absolute wrist reference: every
            # waypoint becomes a base-frame tilt applied to the grasp this arm is
            # already holding. Waypoint 0 then reproduces the current pose exactly,
            # so entering the fling costs no wrist motion -- see
            # retarget_path_to_grasp for what that was costing before (J4 to
            # -363 deg, servo 4 code 23).
            l_path = retarget_path_to_grasp(l_path, grasp_rot_l)
            r_path = retarget_path_to_grasp(r_path, grasp_rot_r_in_l)
            l_path[0][:3] = p_l
            r_path[0][:3] = p_r_in_l
            r_own = transform_pose(np.linalg.inv(self.scene.T_left_right), r_path)
            return l_path, r_path, r_own

        left_path_full, right_path_full, stroke, hang = self._fit_swing(build)

        left_fling, right_fling = left_path_full[:-1], right_path_full[:-1]
        left_drag, right_drag = left_path_full[-1], right_path_full[-1]

        right_fling_base = transform_pose(np.linalg.inv(self.scene.T_left_right), right_fling)
        flung = self.scene.both_fling(left_fling, right_fling_base,
                                      XARM_FLING_SPEED, XARM_FLING_ACC,
                                      record=record_debug)
        if not flung:
            # ⚠️ NOT a warning to skip past. This return value used to be discarded,
            # so a refused swing was followed straight by the drag and the release:
            # the arms grasped, stretched and then simply put the garment down, and
            # the only symptom was "it is not flinging any more". If the swing did
            # not happen, say so and stop -- dragging a garment that was never
            # thrown just pulls it backwards across the table.
            print("[XArmPickAndFling] ABORT: the controller REFUSED the swing "
                  "(stroke {:.3f} m, hang {:.3f} m). Nothing was flung; skipping "
                  "the drag so the garment is not pulled back.".format(stroke, hang))
            self.scene.both_open_gripper()
            return False
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
    @staticmethod
    def _path_radius(path):
        """Furthest waypoint from the base, for a path in that arm's OWN frame."""
        return float(np.max(np.linalg.norm(np.asarray(path)[:, :3], axis=1)))

    def _fit_swing(self, build):
        """Shrink the swing until BOTH arms can actually reach every waypoint.

        The swing is built at the ACTUAL distance between the two grasped points,
        not at XARM_FLING_WIDTH -- xarm_points_to_fling_path is called with
        width=None. Step 2 clamps that distance to [MIN_WIDTH, cap] but cannot
        widen a pick pair the operator chose close together, and a narrow pair puts
        each gripper FURTHER from its own base (the counter-intuitive direction
        again). At the shipped hang of 0.27 m the arms have to end up 0.280 m apart
        for the stroke waypoint to be reachable at all.

        Before this existed the controller simply refused the move and the refusal
        was discarded, so the fling silently became a grasp-and-put-down. Shrinking
        is better than refusing: a smaller fling is still a fling, and the operator
        is told exactly what was given up.

        Stroke goes first because it is what the throw is FOR; the hang is reduced
        only if a stroke floor is not enough, and it is reduced last because
        lowering it is what makes a long sleeve drag on the table.
        """
        stroke, hang = self.swing_stroke, self.hang_height
        limit = XARM_FLING_MAX_RADIUS
        step, stroke_floor, hang_floor = 0.01, 0.08, 0.20

        while True:
            l_path, r_path, r_own = build(stroke, hang)
            worst = max(self._path_radius(l_path), self._path_radius(r_own))
            if worst <= limit:
                if (stroke, hang) != (self.swing_stroke, self.hang_height):
                    print("[XArmPickAndFling] swing SHRUNK to fit the reach: stroke "
                          "{:.3f} -> {:.3f} m, hang {:.3f} -> {:.3f} m (furthest "
                          "waypoint {:.3f} m, limit {:.3f} m). The picks are close "
                          "together, which pushes both grippers away from their "
                          "bases.".format(self.swing_stroke, stroke,
                                          self.hang_height, hang, worst, limit))
                return l_path, r_path, stroke, hang

            if stroke - step >= stroke_floor:
                stroke = round(stroke - step, 4)
            elif hang - step >= hang_floor:
                hang = round(hang - step, 4)
            else:
                print("[XArmPickAndFling] !! even the smallest swing (stroke "
                      "{:.3f}, hang {:.3f}) needs {:.3f} m of reach against a "
                      "{:.3f} m limit. The two picks are too close together -- "
                      "pick them further apart across the garment."
                      .format(stroke, hang, worst, limit))
                return l_path, r_path, stroke, hang

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
        act = self._may_act_on_effort([base_l, base_r], 'Stretch')
        # Consecutive samples over the line, and whether the ignored crossing has
        # already been mentioned -- there is no value in printing it every 10 mm.
        run, reported = 0, False

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                print(f"[Stretch] timeout after {elapsed:.1f} s.")
                break

            tcp_distance = self.scene.get_tcp_distance()
            remaining = max_width - tcp_distance
            if remaining <= 1e-4:
                print(f"[Stretch] width cap reached: {tcp_distance:.3f} m.")
                break
            # Shrink the last step so the width lands ON the cap instead of one
            # increment past it. Overshooting is not cosmetic here: the cap is set
            # by the reach limit, so 2 cm past it is 2 cm of margin spent.
            this_step = min(step, remaining / 2.0)

            d_l = self.scene.left.effort_delta(base_l)
            d_r = self.scene.right.effort_delta(base_r)
            over = ((d_l is not None and d_l > thr_l)
                    or (d_r is not None and d_r > thr_r))
            # One sample over the line is a noise spike; XARM_EFFORT_CONSECUTIVE of
            # them in a row is a load.
            run = run + 1 if over else 0
            if run >= XARM_EFFORT_CONSECUTIVE:
                msg = ("[Stretch] cloth taut at {:.3f} m (effort delta L={}, R={}, "
                       "{} consecutive).".format(tcp_distance, self._fmt_delta(d_l),
                                                 self._fmt_delta(d_r), run))
                if act:
                    print(msg)
                    break
                if not reported:
                    print(msg + " IGNORED -- continuing to the width cap.")
                    reported = True

            # One increment outward for each arm, along the base-to-base axis.
            p_l = self.scene.left.get_tcp_pose()[:3] - axis * this_step
            p_r_in_l = transform_point(self.scene.T_left_right,
                                       self.scene.right.get_tcp_pose()[:3]) + axis * this_step
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
    # Effort is a HINT until it has been measured
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_delta(d):
        return "n/a" if d is None else "{:.2f}".format(d)

    def _may_act_on_effort(self, baselines, where):
        """May an effort reading STOP a motion here? Says why either way.

        Two conditions, and both have to hold for BOTH arms:

        * ``XARM_EFFORT_VERIFIED`` -- the thresholds have been measured against
          each arm's own noise floor. While it is False the rule written in the
          constants file applies: report every delta, act on none of them. That
          rule is there because of a real failure. On 2026-08-04 the placeholder
          threshold of 2.0 sat BELOW the right arm's unloaded noise (delta 2.16),
          so the probe "found" the cloth 55 mm above the table and the right
          gripper closed on air while the quieter left arm grasped normally -- and
          both arms reported success.
        * a baseline on both arms. Gating one arm and not the other is worse than
          gating neither: the two then descend to different heights, which is the
          same silent failure as above wearing a different hat.
        """
        if not XARM_EFFORT_VERIFIED:
            print("[{}] effort is REPORTED, never acted on -- XARM_EFFORT_VERIFIED "
                  "is False, so the thresholds are still placeholders.".format(where))
            return False
        if any(b is None for b in baselines):
            print("[{}] joint effort is unreadable on at least one arm; BOTH arms "
                  "run to the geometric limit.".format(where))
            return False
        return True

    # ------------------------------------------------------------------
    def _probe_both(self, app_l, app_r, grasp_l, grasp_r):
        """Descend both arms onto the cloth TOGETHER; returns the poses reached.

        LOCK-STEP, and that is the whole point of the function. The previous
        version ran one descent loop per arm in its own thread, each issuing ~16
        sequential blocking 5 mm moves. Every blocking waypoint costs 0.1-0.5 s
        inside the SDK's ``wait_move`` (see XARM_PROBE_BAND for the arithmetic),
        that cost is larger than the 5 mm step itself, and it is independently
        random per arm -- so the two descents drifted apart by seconds and the
        right arm visibly trailed the left. Nothing re-synchronised them until
        both had finished.

        Here every step goes through ``scene.both_movel``, which threads and then
        JOINS, so the arms are level again at each waypoint and the worst visible
        skew is one XARM_PROBE_STEP. The descent is also one coarse synchronised
        move down to XARM_PROBE_BAND above the grasp plus a few fine steps inside
        that band, rather than stepping the whole approach: no contact is possible
        up there, and each step not taken is up to half a second of dead time.

        The calibrated grasp height stays a hard floor, so this can only ever stop
        the descent EARLY, never drive deeper than an open-loop descent would.
        """
        sides = ('left', 'right')
        robots = (self.scene.left, self.scene.right)
        targets = [np.array(grasp_l, dtype=float), np.array(grasp_r, dtype=float)]
        floors = [float(grasp_l[2]), float(grasp_r[2])]
        z = [float(app_l[2]), float(app_r[2])]

        # Baselines in PARALLEL. effort_baseline is 10 samples with a sleep after
        # each, so taking them one arm after the other spends half a second before
        # either arm has moved -- and spends it asymmetrically.
        baselines = [None, None]

        def sample_baseline(i):
            try:
                baselines[i] = robots[i].effort_baseline()
            except Exception as exc:
                # ThreadWithResult leaves .result unset when its target raises, and
                # this caller never reads .result, so without catching it here the
                # failure would vanish and the arm would descend anyway.
                print("[Probe] {} effort baseline failed: {}".format(sides[i], exc))

        threads = [ThreadWithResult(target=sample_baseline, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        act = self._may_act_on_effort(baselines, 'Probe')
        thresholds = [for_side(XARM_EFFORT_THRESHOLD, s, 2.0) for s in sides]
        latched = [None, None]     # z at which contact was ACCEPTED
        would = [None, None]       # (z, n) where it would have been, when not acting
        runs = [0, 0]              # consecutive samples over the threshold
        peak = [None, None]        # largest delta seen, the number to set thresholds from

        def command(zs):
            for i in (0, 1):
                targets[i][2] = zs[i]
            return self.scene.both_movel(
                targets[0], targets[1], speed=self.move_speed * 0.5,
                acc=self.move_acc * 0.5, blocking=True)

        def sample(i):
            """One effort reading: track the peak, the run, and where it crossed."""
            try:
                d = robots[i].effort_delta(baselines[i])
            except Exception as exc:
                print("[Probe] {} effort read failed: {}".format(sides[i], exc))
                return
            if d is None:
                runs[i] = 0
                return
            peak[i] = d if peak[i] is None else max(peak[i], d)
            if d > thresholds[i]:
                runs[i] += 1
                if would[i] is None:
                    would[i] = (z[i], runs[i])
            else:
                runs[i] = 0
            if act and latched[i] is None and runs[i] >= XARM_EFFORT_CONSECUTIVE:
                latched[i] = z[i]

        # Coarse leg: ONE synchronised move. Straight to the grasp height when
        # effort is not allowed to stop anything, which makes this identical to
        # what --skip-probe does -- as it should be, since nothing can intervene.
        band = [f + XARM_PROBE_BAND for f in floors]
        z = ([max(f, min(zi, b)) for zi, b, f in zip(z, band, floors)] if act
             else list(floors))
        if not command(z):
            print("[Probe] the descent was refused; grasping where the arms are.")
        else:
            for i in (0, 1):
                sample(i)

            # Fine leg, inside the band, both arms stepping together. An arm that
            # has latched holds its pose while the other keeps going.
            while act and not all(latched[i] is not None or z[i] <= floors[i] + 1e-9
                                  for i in (0, 1)):
                for i in (0, 1):
                    if latched[i] is None:
                        z[i] = max(floors[i], z[i] - XARM_PROBE_STEP)
                if not command(z):
                    print("[Probe] a step was refused; stopping the descent here.")
                    break
                for i in (0, 1):
                    sample(i)

        # The summary is the reason for sampling effort we are not allowed to use:
        # one fling gives the per-arm numbers XARM_EFFORT_THRESHOLD has to be set
        # from before XARM_EFFORT_VERIFIED can honestly be flipped.
        for i in (0, 1):
            print("[Probe] {:<5}: reached z={:+.4f} m ({:+.4f} m above the calibrated "
                  "grasp height), max effort delta {} (threshold {:.2f})".format(
                      sides[i], z[i], z[i] - floors[i],
                      self._fmt_delta(peak[i]), thresholds[i]))
            if latched[i] is not None:
                print("[Probe]        contact accepted at z={:+.4f} m.".format(latched[i]))
            elif would[i] is not None and act:
                zz, _ = would[i]
                print("[Probe]        crossed the threshold at z={:+.4f} m but never "
                      "for {} samples running, so it was read as noise.".format(
                          zz, XARM_EFFORT_CONSECUTIVE))
            elif would[i] is not None:
                zz, n = would[i]
                print("[Probe]        -> WOULD have stopped at z={:+.4f} m after {} "
                      "sample(s) over the threshold; ignored because "
                      "XARM_EFFORT_VERIFIED is False.".format(zz, n))

        return targets[0].copy(), targets[1].copy()

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
