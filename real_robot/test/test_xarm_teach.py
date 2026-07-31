"""Teach / calibrate the dual xArm Lite 6 cell.

PREREQUISITE for the primitive tests. The primitives descend to ``XARM_TABLE_Z``
and call ``home()`` in ``reset()``, but ``XARM_TABLE_Z``, ``XARM_GRIPPER_OFFSET``,
``XARM_HOME_JOINT`` and ``XARM_WORKSPACE_RADIUS`` in ``real_robot/utils/xarm_constants.py``
are unverified guesses. This script measures them on the real cell and writes
``real_robot/calibration/xarm-cell.yaml``.

Hand-guiding uses the controller's joint-teaching mode (``set_mode(2)``), which
makes the arm BACK-DRIVABLE: support its weight before enabling it. Position mode
is always restored in a ``finally``.

Usage
    source ./setup.sh xarm

    # per arm -- measure the table, a safe home pose, and the usable reach:
    python real_robot/test/test_xarm_teach.py --arm left  --table-z --home --reach
    python real_robot/test/test_xarm_teach.py --arm right --table-z --home --reach

    # both arms -- measure the true base-to-base distance. By default both arms
    # touch ONE shared mark near the table midline; pass --mark-dx if you would
    # rather use two marks a tape-measured distance apart.
    python real_robot/test/test_xarm_teach.py --arm both --separation

    # reach sweep only, no hand-guiding and no motion at all:
    python real_robot/test/test_xarm_teach.py --arm left --reach
"""
import argparse
import datetime
import os
import sys
from contextlib import contextmanager

import numpy as np

from real_robot.robot.xarm_lite6 import XArmLite6
from real_robot.utils import xarm_constants as C
from real_robot.test.test_xarm_lite6_bringup import (
    ERROR_HINTS, preflight, print_network_help, confirm, _fmt,
    report_controller_settings,
)
from real_robot.test.xarm_test_scene import CELL_YAML, load_cell, save_cell


def _read_yaml(path):
    import yaml
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _merge_cell(updates_cell=None, updates_arm=None, arm_name=None, path=CELL_YAML):
    """Merge measurements into xarm-cell.yaml without dropping the other arm."""
    data = _read_yaml(path)
    data.setdefault('cell', {})
    data.setdefault('arms', {})
    if updates_cell:
        data['cell'].update(updates_cell)
    if updates_arm and arm_name:
        data['arms'].setdefault(arm_name, {})
        data['arms'][arm_name].update(updates_arm)
    data['cell']['measured_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    save_cell(data, path)
    print("  written -> {}".format(path))
    return data


@contextmanager
def teach_mode(driver, name):
    """Joint-teaching (free-drive) mode, restoring position mode unconditionally.

    If the virtual walls are on, hand-guiding past one trips controller error 35
    and the arm goes to state 4. We clear it on the way out so the session can
    continue, and say what to do about it.
    """
    arm = driver.arm
    print("\n  [{}] enabling FREE-DRIVE (set_mode(2)). Take the weight of the arm NOW."
          .format(name))
    arm.set_mode(2)
    arm.set_state(0)
    try:
        yield
    finally:
        if arm.error_code == 35:
            print("  [{}] !! controller error 35 (safety boundary): you guided the TCP "
                  "past a virtual wall.".format(name))
            print("     Clearing it. If this keeps happening, re-run with --no-walls -- "
                  "the walls depend on XARM_TABLE_Z, which is what you are measuring.")
            arm.clean_error()
            arm.clean_warn()
        arm.set_mode(0)
        arm.set_state(0)
        print("  [{}] free-drive off, position mode restored.".format(name))


def _healthy(driver, name):
    err = driver.arm.error_code
    if err:
        hint = ERROR_HINTS.get(err, "look it up in UFACTORY Studio")
        print("  !! [{}] controller error {}: {}".format(name, err, hint))
        return False
    if driver.arm.state == 4:
        print("  !! [{}] state = 4 (STOP): the arm is not enabled.".format(name))
        return False
    return True


def _connect(ip, name, walls='auto', allow_offsets=False):
    ok, err = preflight(ip)
    if not ok:
        print_network_help(ip, err)
        return None
    # gripper='none' keeps the pneumatic init out of the way; teaching needs no gripper.
    driver = XArmLite6(ip, gripper='none', side=name, walls=walls)
    if not _healthy(driver, name):
        driver.disconnect()
        return None

    # HARD GATE. Every number this script records is a coordinate read back from the
    # controller, so a non-zero tcp_offset or world_offset does not add noise -- it
    # measures a different frame entirely, and the result looks like data. A right
    # arm with a leftover 374 mm tool once recorded its gripper offset as -0.153 m
    # against the left arm's +0.087 m for the same physical table contact.
    print("\n--- [{}] controller check (no motion) ---".format(name))
    if not report_controller_settings(driver) and not allow_offsets:
        print("\n  !! [{}] REFUSING to measure: the controller state above is not clean."
              .format(name))
        print("     Nothing has been written. Fix it first:")
        print("       python real_robot/test/test_xarm_lite6_bringup.py --arm {} "
              "--clear-tcp-offset".format(name))
        print("     then re-run this script. --allow-offsets overrides, but whatever you")
        print("     measure will be expressed in that shifted frame.")
        driver.disconnect()
        return None
    return driver


# ----------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------
def measure_table(driver, name, table_z, samples, auto_yes):
    """Hand-guide the CLOSED fingertips onto the table at a few spots.

    The commanded TCP is the flange (no tcp_offset is set), so the flange z with
    the fingertips touching the table is ``table_z + gripper_offset`` -- exactly
    the z the primitives command for a grasp. That gives GRIPPER_OFFSET directly,
    and the spread across spots says whether the table is level in the base frame.
    """
    print("\n=== [{}] TABLE HEIGHT ===".format(name))
    print("  Assuming the table surface sits at z = {:+.4f} m in this arm's base"
          " frame (--table-z to change).".format(table_z))
    pts = []
    with teach_mode(driver, name):
        for i in range(samples):
            if not confirm("Rest the CLOSED fingertips flat on the table (spot {}/{}), then Enter"
                           .format(i + 1, samples), auto_yes):
                continue
            pose = driver.get_tcp_pose()
            pts.append(pose[:3])
            print("     flange xyz = {}".format(_fmt(pose[:3])))
    if not pts:
        print("  no samples taken.")
        return None

    pts = np.asarray(pts)
    z_touch = float(np.mean(pts[:, 2]))
    gripper_offset = z_touch - table_z
    print("\n  mean flange z on the table : {:+.4f} m".format(z_touch))
    print("  => XARM_GRIPPER_OFFSET     : {:+.4f} m".format(gripper_offset))
    if len(pts) > 1:
        spread = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))
        span = float(np.max(np.linalg.norm(pts[:, :2] - pts[0, :2], axis=1)))
        print("  z spread over {} spots      : {:.4f} m over a {:.3f} m span".format(
            len(pts), spread, span))
        if spread > 0.005:
            print("  !! >5 mm of spread: the table is not level in the base frame, or the "
                  "arm was pressed into it. A fixed-height grasp will miss on one side.")
    if not (0.0 <= gripper_offset <= 0.30):
        print("  !! implausible offset -- NOTHING WAS WRITTEN.")
        print("     A closed Lite6 gripper is a few centimetres long, so this is a frame")
        print("     error, not a measurement. Check tcp_offset / world_offset on this")
        print("     controller, and that --table-z matches where the table actually is")
        print("     relative to this arm's base origin.")
        return None

    # Per arm, not per cell: the two controllers are configured independently, and
    # writing these cell-wide let one arm's run overwrite the other's measurement.
    _merge_cell(updates_arm={'table_z': round(table_z, 5),
                             'gripper_offset': round(gripper_offset, 5),
                             'table_touch_z': round(z_touch, 5),
                             'table_samples': [[round(float(v), 5) for v in p] for p in pts]},
                arm_name=name)
    print("\n  Paste into real_robot/utils/xarm_constants.py:")
    print("      XARM_GRIPPER_OFFSET_BY_SIDE['{}'] = {:.4f}".format(name, gripper_offset))
    print("      XARM_TABLE_Z_BY_SIDE['{}']        = {:.4f}".format(name, table_z))
    return gripper_offset


def measure_home(driver, name, auto_yes):
    """Hand-guide to a safe ready pose and record the joint vector."""
    print("\n=== [{}] HOME POSE ===".format(name))
    print("  Guide the arm to a ready pose: TCP well above the table, pointing down,")
    print("  clear of the camera view and of the other arm.")
    with teach_mode(driver, name):
        if not confirm("Hold the arm in the home pose, then Enter", auto_yes):
            return None
        code, q = driver.arm.get_servo_angle(is_radian=True)
        pose = driver.get_tcp_pose()
    if code != 0:
        print("  !! get_servo_angle returned code {}".format(code))
        return None

    q = list(np.asarray(q, dtype=float)[:6])
    deg = np.degrees(q)
    r_xy = float(np.linalg.norm(pose[:2]))
    print("\n  joints (deg) : {}".format(_fmt(deg, 2)))
    print("  TCP xyz      : {}   XY radius {:.3f} m".format(_fmt(pose[:3]), r_xy))
    if pose[2] < 0.10:
        print("  !! TCP is less than 10 cm up -- homing from a grasp would scrape the table.")
    radius = C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, name)
    if not (radius[0] <= r_xy <= radius[1]):
        print("  !! home XY radius is outside this arm's XARM_WORKSPACE_RADIUS "
              "({:.3f} - {:.3f}).".format(*radius))

    _merge_cell(updates_arm={'home_joint': [round(float(v), 6) for v in q],
                             'home_tcp': [round(float(v), 5) for v in pose]},
                arm_name=name)
    print("\n  Paste into real_robot/utils/xarm_constants.py:")
    print("      XARM_HOME_JOINT_BY_SIDE['{}'] = np.deg2rad([{}]).tolist()".format(
        name, ", ".join("{:.1f}".format(v) for v in deg)))
    return q


def sweep_reach(driver, name, table_z, gripper_offset):
    """Ask the controller's IK where this arm can actually work. No motion.

    Sweeps radius outward along several directions at the grasp height and at the
    fling hang height, and reports the reachable annulus at each. This is what
    decides whether the fling's stretch targets are commandable at all.
    """
    from real_robot.primitives.xarm_pick_and_fling import (
        HANG_HEIGHT, STRETCH_MAX_WIDTH, MIN_STRETCH_DIST,
    )

    print("\n=== [{}] REACH SWEEP (inverse kinematics only -- the arm does not move) ==="
          .format(name))
    rot = np.array(C.XARM_DOWN_ROTVEC, dtype=float)
    heights = [('grasp', table_z + gripper_offset),
               ('lift  (+{:.2f})'.format(C.XARM_LIFT_DIST), table_z + gripper_offset + C.XARM_LIFT_DIST),
               ('hang  ({:.2f})'.format(HANG_HEIGHT), table_z + HANG_HEIGHT)]
    yaws = np.deg2rad([-60.0, -30.0, 0.0, 30.0, 60.0])
    radii = np.arange(0.05, 0.65, 0.005)

    results = {}
    for label, z in heights:
        per_yaw = []
        for yaw in yaws:
            reachable = []
            for r in radii:
                pose = np.concatenate([[r * np.cos(yaw), r * np.sin(yaw), z], rot])
                code, _ = driver.arm.get_inverse_kinematics(
                    driver._pose_to_xarm(pose), input_is_radian=True)
                if code == 0:
                    reachable.append(r)
            per_yaw.append((float(min(reachable)), float(max(reachable))) if reachable else None)
        ok = [p for p in per_yaw if p is not None]
        if not ok:
            print("  z={:+.3f} ({:<14s}) : NOTHING reachable at this height".format(z, label))
            results[label.split()[0]] = None
            continue
        # Intersection across directions: the annulus that works in every direction.
        lo = max(p[0] for p in ok)
        hi = min(p[1] for p in ok)
        results[label.split()[0]] = (lo, hi)
        note = "" if hi > lo else "   !! empty intersection across directions"
        print("  z={:+.3f} ({:<14s}) : r = {:.3f} - {:.3f} m{}".format(z, label, lo, hi, note))
        if len(ok) < len(yaws):
            print("      ({} of {} directions unreachable at any radius)".format(
                len(yaws) - len(ok), len(yaws)))

    grasp = results.get('grasp')
    if grasp and grasp[1] > grasp[0]:
        # Keep a safety margin off the outer edge, where IK solutions are singular.
        lo = max(C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, name)[0],
                 round(grasp[0] + 0.01, 3))
        hi = round(grasp[1] - 0.02, 3)
        # Check AFTER the margins, not before: a measured annulus can be wide enough
        # to pass `grasp[1] > grasp[0]` and still invert once the margins are applied
        # -- which is how (0.120, 0.105), a range with no interior, once got written.
        if hi <= lo:
            print("\n  !! the usable annulus collapses to ({:.3f}, {:.3f}) once the safety"
                  .format(lo, hi))
            print("     margins are applied -- that is not a workspace. NOTHING WAS WRITTEN.")
            print("     A reach this small means the IK is being asked in the wrong frame;")
            print("     check tcp_offset / world_offset before re-measuring.")
        else:
            print("\n  => XARM_WORKSPACE_RADIUS = ({:.3f}, {:.3f})   "
                  "(1 cm / 2 cm margin off the measured edges)".format(lo, hi))
            _merge_cell(updates_arm={'workspace_radius': [lo, hi],
                                     'reach': {k: (list(v) if v else None)
                                               for k, v in results.items()}},
                        arm_name=name)
            print("\n  Paste into real_robot/utils/xarm_constants.py:")
            print("      XARM_WORKSPACE_RADIUS_BY_SIDE['{}'] = ({:.3f}, {:.3f})".format(
                name, lo, hi))

    hang = results.get('hang')
    if hang:
        # XArmPickAndFlingSkill stretches to center +/- width/2 about the base
        # midpoint, at HANG_HEIGHT. Check that target against the sweep.
        S = C.XARM_BASE_SEPARATION
        width = min(STRETCH_MAX_WIDTH, max(MIN_STRETCH_DIST, S))
        r_target = abs(S / 2.0 - width / 2.0)
        print("\n  Fling check: the stretch step targets r = |{:.2f}/2 - {:.2f}/2| = {:.3f} m"
              .format(S, width, r_target))
        print("  from each base at the {:.2f} m hang height; reachable there over r = "
              "{:.3f} - {:.3f} m.".format(HANG_HEIGHT, hang[0], hang[1]))
        if r_target < hang[0] or r_target > hang[1]:
            print("  !! The stretch step as written is NOT commandable -- retune "
                  "HANG_HEIGHT / STRETCH_MAX_WIDTH in xarm_pick_and_fling.py.")
        else:
            print("  ok: the stretch target is reachable.")
    return results


def fit_yaw_and_offset(p_right, p_left):
    """2D rigid fit: find yaw and translation with p_left = Rz(yaw) @ p_right + t.

    ``p_right``/``p_left`` are (N, 3) arrays of the SAME physical marks, each
    measured in its own arm's base frame. Needs N >= 2: a single shared point
    gives 2 equations for 3 unknowns, so it cannot determine the yaw at all --
    which is why the yaw was an assumption until now.
    """
    A = np.asarray(p_right, dtype=float)[:, :2]
    B = np.asarray(p_left, dtype=float)[:, :2]
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    Ac, Bc = A - ca, B - cb
    # Closed-form least-squares yaw (2D Kabsch).
    num = float(np.sum(Ac[:, 0] * Bc[:, 1] - Ac[:, 1] * Bc[:, 0]))
    den = float(np.sum(Ac[:, 0] * Bc[:, 0] + Ac[:, 1] * Bc[:, 1]))
    yaw = float(np.arctan2(num, den))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    t = cb - R @ ca
    residuals = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    return yaw, t, residuals


def measure_geometry(left, right, marks, auto_yes):
    """Measure the base-to-base transform by touching N shared marks with BOTH arms.

    This replaces the assumed 180 deg yaw. Both arms touch the same physical
    point, so no tape measure is involved and no mounting assumption is made.
    """
    print("\n=== BASE GEOMETRY (yaw + separation) ===")
    print("  Put {} marks on the table where BOTH arms can touch them -- spread them".format(marks))
    print("  out along the arm line, not in a cluster, or the yaw fit will be noisy.")
    if marks < 2:
        print("  !! need at least 2 marks to determine the yaw.")
        return None

    p_L, p_R = [], []
    for i in range(marks):
        with teach_mode(left, 'left'):
            if not confirm("Touch the LEFT fingertips to mark {}/{}, then Enter"
                           .format(i + 1, marks), auto_yes):
                return None
            p_L.append(left.get_tcp_pose()[:3])
        with teach_mode(right, 'right'):
            if not confirm("Touch the RIGHT fingertips to the SAME mark {}/{}, then Enter"
                           .format(i + 1, marks), auto_yes):
                return None
            p_R.append(right.get_tcp_pose()[:3])
        print("  mark {}: left {}   right {}".format(i + 1, _fmt(p_L[-1]), _fmt(p_R[-1])))

    p_L, p_R = np.asarray(p_L), np.asarray(p_R)
    yaw, t, residuals = fit_yaw_and_offset(p_R, p_L)
    separation = float(np.linalg.norm(t))
    dz = float(np.mean(p_L[:, 2] - p_R[:, 2]))

    print("\n  => XARM_BASE_YAW        : {:+.4f} rad  ({:+.1f} deg)".format(yaw, np.degrees(yaw)))
    print("     base-to-base offset  : ({:+.4f}, {:+.4f}) m, |t| = {:.4f} m".format(
        t[0], t[1], separation))
    print("     fit residuals        : max {:.1f} mm, mean {:.1f} mm".format(
        residuals.max() * 1000, residuals.mean() * 1000))
    print("     height difference    : {:+.4f} m".format(dz))

    assumed = np.degrees(C.XARM_BASE_YAW)
    got = np.degrees(yaw)
    if abs((got - assumed + 180) % 360 - 180) > 5.0:
        print("\n  !! The measured yaw differs from the assumed {:+.0f} deg by more than 5 deg."
              .format(assumed))
        print("     Every right-arm target and every virtual wall depends on this, so the")
        print("     assumed value was placing them wrong.")
    if residuals.max() > 0.01:
        print("  !! >10 mm fit residual: the marks were probably touched imprecisely, or the")
        print("     two arms disagree about where the same point is. Re-measure before"
              " trusting this.")
    if abs(dz) > 0.01:
        print("  !! >10 mm height difference: one XARM_TABLE_Z cannot serve both arms.")

    _merge_cell(updates_cell={'base_yaw': round(yaw, 6),
                              'base_separation': round(separation, 4),
                              'base_offset': [round(float(t[0]), 4), round(float(t[1]), 4)],
                              'fit_residual_mm': round(float(residuals.max() * 1000), 2),
                              'height_difference': round(dz, 4)})
    print("\n  Paste into real_robot/utils/xarm_constants.py:")
    print("      XARM_BASE_YAW           = {:.6f}   # {:+.1f} deg".format(yaw, np.degrees(yaw)))
    print("      XARM_BASE_SEPARATION    = {:.3f}".format(separation))
    print("      XARM_GEOMETRY_VERIFIED  = True     # <-- this switches the virtual walls ON")
    return yaw, separation


def measure_separation(left, right, mark_dx, mark_dy, auto_yes):
    """True base-to-base geometry, from marks on the table.

    With ``mark_dx = mark_dy = 0`` both arms touch the SAME mark, which the
    workspace overlap makes possible; otherwise each arm touches its own mark and
    the tape-measured offset between them closes the loop::

        p_left(B) = p_left(A) + (mark_dx, mark_dy, 0)
        p_left(B) = Rz(180) @ p_right(B) + (S, 0, 0)
      => S            = p_left(B).x + p_right(B).x
         y residual   = p_left(B).y + p_right(B).y   (0 if the arms are aligned)
         z residual   = p_left(B).z - p_right(B).z   (0 if mounted at equal height)
    """
    print("\n=== BASE SEPARATION ===")
    sep_marks = float(np.hypot(mark_dx, mark_dy))
    if sep_marks < 1e-6:
        print("  Put ONE mark on the table where both arms can touch it.")
    else:
        print("  Put two marks on the table along the base-to-base axis, {:.3f} m apart"
              .format(sep_marks))
        print("  (mark A nearer the LEFT arm, mark B nearer the RIGHT arm; "
              "--mark-dx/--mark-dy).")

    with teach_mode(left, 'left'):
        if not confirm("Touch the LEFT fingertips to mark A, then Enter", auto_yes):
            return None
        p_LA = left.get_tcp_pose()[:3]
    print("  left  @ A (left base frame)  : {}".format(_fmt(p_LA)))

    with teach_mode(right, 'right'):
        if not confirm("Touch the RIGHT fingertips to mark B, then Enter", auto_yes):
            return None
        p_RB = right.get_tcp_pose()[:3]
    print("  right @ B (right base frame) : {}".format(_fmt(p_RB)))

    p_LB = np.array(p_LA) + np.array([mark_dx, mark_dy, 0.0])
    separation = float(p_LB[0] + p_RB[0])
    res_y = float(p_LB[1] + p_RB[1])
    res_z = float(p_LB[2] - p_RB[2])

    print("\n  => base separation : {:.4f} m   (assumed {:.3f} m)".format(
        separation, C.XARM_BASE_SEPARATION))
    print("     lateral residual: {:+.4f} m   (0 if the bases are laterally aligned)".format(res_y))
    print("     height residual : {:+.4f} m   (0 if both bases are at the same height)".format(res_z))
    if abs(res_y) > 0.02:
        print("  !! >2 cm lateral offset: the arms are not exactly face-to-face, so the "
              "180 deg yaw in T_left_right is an approximation.")
    if abs(res_z) > 0.01:
        print("  !! >1 cm height difference: a single XARM_TABLE_Z cannot be right for both "
              "arms. Shim the bases, or give each arm its own table_z.")
    if abs(separation - C.XARM_BASE_SEPARATION) > 0.02:
        print("  !! differs from XARM_BASE_SEPARATION by more than 2 cm -- update it.")

    _merge_cell(updates_cell={'base_separation': round(separation, 4),
                              'lateral_residual': round(res_y, 4),
                              'height_residual': round(res_z, 4)})
    print("\n  Paste into real_robot/utils/xarm_constants.py:")
    print("      XARM_BASE_SEPARATION = {:.3f}".format(separation))
    return separation


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Teach/measure the dual xArm Lite 6 cell into xarm-cell.yaml.")
    ap.add_argument('--left-ip', default=os.environ.get('XARM_LEFT_IP', '192.168.1.155'))
    ap.add_argument('--right-ip', default=os.environ.get('XARM_RIGHT_IP', '192.168.1.170'))
    ap.add_argument('--arm', choices=['left', 'right', 'both'], default='left')
    ap.add_argument('--table-z', dest='do_table', action='store_true',
                    help="hand-guide the fingertips to the table -> XARM_GRIPPER_OFFSET")
    ap.add_argument('--home', dest='do_home', action='store_true',
                    help="hand-guide to a ready pose -> XARM_HOME_JOINT")
    ap.add_argument('--reach', dest='do_reach', action='store_true',
                    help="IK reach sweep (no motion) -> XARM_WORKSPACE_RADIUS")
    ap.add_argument('--separation', dest='do_sep', action='store_true',
                    help="measure the base-to-base YAW and distance by touching shared "
                         "marks with both arms (needs --arm both)")
    ap.add_argument('--marks', type=int, default=3,
                    help="shared marks for the yaw fit (>=2; 3 is a good default)")
    ap.add_argument('--all', action='store_true', help="table-z + home + reach")
    ap.add_argument('--table-height', type=float, default=None,
                    help="table surface z in the arm's base frame (default: XARM_TABLE_Z)")
    ap.add_argument('--samples', type=int, default=3, help="table touch spots")
    ap.add_argument('--mark-dx', type=float, default=0.0,
                    help="mark A -> mark B offset along the base-to-base axis, m. "
                         "0 (default) = one shared mark, which the 0.14 m workspace "
                         "overlap at 0.66 m separation now makes reachable by both arms")
    ap.add_argument('--mark-dy', type=float, default=0.0, help="mark A -> B lateral offset, m")
    ap.add_argument('--yes', action='store_true', help="don't prompt between steps")
    ap.add_argument('--allow-offsets', action='store_true',
                    help="measure even though a non-zero tcp_offset / world_offset is set "
                         "on the controller. Everything recorded is then expressed in that "
                         "shifted frame -- normally you want --clear-tcp-offset instead")
    ap.add_argument('--no-walls', action='store_true',
                    help="disable the virtual walls while teaching. The floor wall sits "
                         "at XARM_TABLE_Z, which --table-z is measuring, so the first run "
                         "on a new cell may need this")
    args = ap.parse_args()

    if args.all:
        args.do_table = args.do_home = args.do_reach = True
    if not any([args.do_table, args.do_home, args.do_reach, args.do_sep]):
        ap.error("nothing to do: pass --table-z / --home / --reach / --separation / --all")
    if args.do_sep and args.arm != 'both':
        ap.error("--separation needs --arm both")

    cell = load_cell()
    names = ['left', 'right'] if args.arm == 'both' else [args.arm]
    ips = {'left': args.left_ip, 'right': args.right_ip}

    print("=" * 72)
    print("xArm Lite 6 cell teaching")
    print("  arms  : " + ", ".join("{} @ {}".format(n, ips[n]) for n in names))
    print("  output: {}".format(CELL_YAML))
    print("\n  SAFETY: free-drive makes the arm back-drivable -- HOLD IT before each")
    print("  prompt, and keep clear of the table edge.")
    print("=" * 72)

    drivers = {}
    try:
        for n in names:
            # 'auto': off until the geometry this script is measuring is verified.
            drivers[n] = _connect(ips[n], n, walls=False if args.no_walls else 'auto',
                                  allow_offsets=args.allow_offsets)
            if drivers[n] is None:
                return 1

        for n in names:
            # Every value here is per arm. Deliberately NOT carried across the loop:
            # doing so let a failed measurement on one arm silently fall back to the
            # other arm's number, which is how a bad right-arm run ended up sweeping
            # its reach against the left arm's gripper offset.
            table_z = (C.for_side(C.XARM_TABLE_Z_BY_SIDE, n)
                       if args.table_height is None else args.table_height)
            gripper_offset = cell.gripper_offset(n)
            can_sweep = True
            if args.do_table:
                got = measure_table(drivers[n], n, table_z, args.samples, args.yes)
                if got is not None:
                    gripper_offset = got
                else:
                    can_sweep = False
            if args.do_home:
                measure_home(drivers[n], n, args.yes)
            if args.do_reach and not can_sweep:
                print("\n  [{}] skipping the reach sweep: it probes at the grasp height, which"
                      .format(n))
                print("      needs a valid gripper offset, and this arm's table measurement")
                print("      was rejected.")
            elif args.do_reach:
                sweep_reach(drivers[n], n, table_z, gripper_offset)

        if args.do_sep:
            if abs(args.mark_dx) > 1e-9 or abs(args.mark_dy) > 1e-9:
                # Two offset marks: only valid if you already trust the yaw.
                measure_separation(drivers['left'], drivers['right'],
                                   args.mark_dx, args.mark_dy, args.yes)
            else:
                measure_geometry(drivers['left'], drivers['right'], args.marks, args.yes)

        print("\nDone. Paste the printed constants into real_robot/utils/xarm_constants.py,")
        print("then run:  python real_robot/test/test_xarm_primitives.py "
              "--primitive single-pnp --dry-run")
        return 0

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        return 130
    finally:
        for n, d in drivers.items():
            if d is None:
                continue
            try:
                d.arm.set_mode(0)
                d.arm.set_state(0)
            except Exception:
                pass
            try:
                d.disconnect()
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())
