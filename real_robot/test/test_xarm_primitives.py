"""Run the xArm Lite 6 primitives on the real cell, one primitive at a time.

Drives the SHIPPED skill classes unmodified --
``XArmSingleArmPickAndPlaceSkill``, ``XArmPickAndPlaceSkill``,
``XArmPickAndFlingSkill`` -- through camera-free test scenes
(``xarm_test_scene.py``) whose geometry comes from a tape measure rather than the
still-placeholder hand-eye calibration. The primitives' own pixel code path is
therefore exercised end to end, but the targets are written here in METRES on the
table and converted with ``table_xy_to_pixel``.

Every commanded waypoint goes through the controller's own inverse kinematics
first (``IKGuardedArm``): nothing is sent that the arm cannot solve. In
``--dry-run`` (the default) NOTHING MOVES -- the whole primitive is executed for
its geometry only, printing a reachability table. Always dry-run first.

Table coordinates = the LEFT arm's base frame. x runs from 0 at the left base
across the table's short (80 cm) axis to ``separation`` at the right base; y runs
along the table's long (120 cm) axis; z is up.

Usage
    source ./setup.sh xarm

    # geometry only, no hardware needed at all:
    python real_robot/test/test_xarm_primitives.py --primitive single-pnp --offline
    python real_robot/test/test_xarm_primitives.py --primitive fling --offline --case all

    # connected, real IK checks, still no motion:
    python real_robot/test/test_xarm_primitives.py --primitive single-pnp --dry-run

    # for real, one stage at a time:
    python real_robot/test/test_xarm_primitives.py --primitive single-pnp --execute --arm left
    python real_robot/test/test_xarm_primitives.py --primitive dual-pnp   --execute --case both
    python real_robot/test/test_xarm_primitives.py --primitive fling      --execute
"""
import argparse
import os
import sys
import traceback

import numpy as np

from real_robot.primitives.xarm_single_arm_pick_and_place import XArmSingleArmPickAndPlaceSkill
from real_robot.primitives.xarm_pick_and_place import XArmPickAndPlaceSkill
from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
from real_robot.utils import xarm_constants as C
from real_robot.utils.term import green, red
from real_robot.utils.transform_utils import point_on_table_base
from real_robot.test.test_xarm_lite6_bringup import (
    ERROR_HINTS, preflight, print_network_help, confirm, _fmt,
)
from real_robot.test.xarm_test_scene import (
    CELL_YAML, Unreachable, XArmTestDualScene, XArmTestSingleScene,
    load_cell, print_reach_map, selfcheck_camera, table_xy_to_pixel,
)


# Targets in table metres for the measured 0.66 m separation. (x, y): x from the
# LEFT base across toward the right base, y along the arm line (front is +y).
# Right-arm targets are given in TABLE coordinates too, so x_right_base = 0.66 - x.
SINGLE_CASES = {
    'basic':        dict(pick=(0.25, -0.10), place=(0.25, 0.10), rot=0.0),
    'rotated':      dict(pick=(0.24, -0.08), place=(0.24, 0.08), rot=np.pi / 4),
    # Straddles the base so the transit segment crosses the keepout circle and
    # XArmSingleArmPickAndPlaceSkill._get_collision_free_path has to route around.
    'route-around': dict(pick=(0.10, -0.25), place=(0.10, 0.25), rot=0.0),
}

DUAL_CASES = {
    # Picks on opposite halves: each lands 0.23 m from its own base.
    'both':      dict(pick_l=(0.20, -0.12), place_l=(0.20, 0.12),
                      pick_r=(0.46, 0.12), place_r=(0.46, -0.12),
                      active=(1.0, 1.0)),
    'left-only': dict(pick_l=(0.20, -0.12), place_l=(0.20, 0.12),
                      pick_r=(0.46, 0.12), place_r=(0.46, -0.12),
                      active=(1.0, 0.0)),
    'right-only': dict(pick_l=(0.20, -0.12), place_l=(0.20, 0.12),
                       pick_r=(0.46, 0.12), place_r=(0.46, -0.12),
                       active=(0.0, 1.0)),
    # Picks 6 cm apart in the shared overlap band, inside XARM_COLLISION_THRESHOLD
    # (0.12 m), so check_trajectories_close should fire and force the SEQUENTIAL
    # path. Only possible now that the workspaces actually overlap.
    'collision': dict(pick_l=(0.30, 0.0), place_l=(0.24, 0.15),
                      pick_r=(0.36, 0.0), place_r=(0.42, -0.15),
                      active=(1.0, 1.0)),
}

FLING_CASES = {
    'basic':   dict(pick_l=(0.22, 0.0), pick_r=(0.44, 0.0), valid=(1.0, 1.0)),
    'wide':    dict(pick_l=(0.20, -0.18), pick_r=(0.46, 0.18), valid=(1.0, 1.0)),
    # Both flags low: the skill must abort before touching the arms.
    'invalid': dict(pick_l=(0.22, 0.0), pick_r=(0.44, 0.0), valid=(0.0, 0.0)),
}

CASES = {'single-pnp': SINGLE_CASES, 'dual-pnp': DUAL_CASES, 'fling': FLING_CASES}


# ----------------------------------------------------------------------
def preview_point(label, xy, cell, T_cam, intr, arm_name):
    """Print one target as table metres -> pixel -> the base point the primitive
    will actually derive, so a mapping error is visible before anything moves."""
    # Table coordinates are the LEFT base frame, which is the frame the synthetic
    # camera is built in -- so this is the left arm's table_z for either arm.
    # Pass the scene's own intrinsic: if it is a cropped one, the pixel has to be
    # expressed in the cropped frame, which is what the primitive will invert.
    px = table_xy_to_pixel(xy[0], xy[1], cell.separation, cell.table_z('left'),
                           intr=intr)
    base = point_on_table_base(px[0], px[1], intr, T_cam, cell.table_z('left'))
    r_xy = float(np.linalg.norm(base[:2]))
    print("    {:<12s} table ({:+.3f}, {:+.3f}) -> pixel ({:7.1f}, {:7.1f}) -> "
          "{} base {}  r={:.3f} m".format(
              label, xy[0], xy[1], px[0], px[1], arm_name, _fmt(base, 3), r_xy))
    return px


def check_cell(cell, force):
    """The primitives bind XARM_TABLE_Z etc. at import, so a measured cell only
    takes effect once it is pasted into xarm_constants.py. Refuse to move until it
    is, rather than silently using stale numbers."""
    if not cell.calibrated:
        print("\n  !! No {} -- the cell has never been measured.".format(CELL_YAML))
        print("     XARM_TABLE_Z ({:+.3f}), XARM_GRIPPER_OFFSET ({:+.3f}) and "
              "XARM_HOME_JOINT".format(C.XARM_TABLE_Z, C.XARM_GRIPPER_OFFSET))
        print("     are unverified guesses. Run:  python real_robot/test/test_xarm_teach.py"
              " --arm left --all")
        return force

    # Both arms sit at the same table. A per-arm value that differs by more than a
    # centimetre is a symptom -- most often a tcp_offset or world_offset set on one
    # controller and not the other -- not two honest measurements.
    disagree = cell.arms_disagree()
    if disagree:
        print("\n  !! the two arms disagree about values they should share:")
        for name, lo, hi in disagree:
            print("       {:<22s} left {:+.4f}  right {:+.4f}   (delta {:+.4f} m)".format(
                name, lo, hi, hi - lo))
        print("     They are at the same table, so this is a frame error, not data. Check")
        print("     tcp_offset / world_offset on both controllers:")
        print("       python real_robot/test/test_xarm_lite6_bringup.py --arm both --info-only")
        return force

    mismatch = []
    for side in ('left', 'right'):
        for name, have, want in cell.constants_mismatch(side):
            mismatch.append((side, name, have, want))
    if mismatch:
        print("\n  !! xarm-cell.yaml disagrees with xarm_constants.py:")
        for side, name, have, want in mismatch:
            print("       [{:<5s}] {:<22s} constants {:+.4f}  measured {:+.4f}".format(
                side, name, have, want))
        print("     The primitives import these constants directly, so paste the measured")
        print("     values into real_robot/utils/xarm_constants.py before running for real.")
        return force
    print("  cell calibration: {} (matches xarm_constants.py)".format(green("OK")))
    return True


def shutdown(scene):
    """Stop and disconnect a scene's arms. Safe to call on a half-built scene."""
    for arm in [getattr(scene, 'left', None), getattr(scene, 'right', None),
                getattr(scene, 'arm', None)]:
        if arm is None or getattr(arm, 'driver', None) is None:
            continue
        try:
            err = arm.driver.arm.error_code
            if err:
                print("  [{}] controller error {}: {}".format(
                    arm.name, err, ERROR_HINTS.get(err, "see UFACTORY Studio")))
            # Park the gripper BEFORE stopping the arm: open/close hold their
            # solenoid driven, so something must release it or the coil stays
            # powered after the process exits. disconnect() parks too, but doing it
            # here means it happens while the controller is still in a ready state.
            arm.driver.park_gripper()
            arm.driver.arm.set_state(4)      # stop
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass


def summarise(scene, label):
    checked = scene.checked_poses()
    bad = [c for c in checked if not c[3]]
    count = "{} waypoints checked, {} unreachable.".format(len(checked), len(bad))
    print("\n  {} -- {}".format(label, red(count) if bad else green(count)))
    for name, tag, pose, _, reason in bad:
        print(red("    !! {:<6s} {:<12s} xyz=({:+.3f}, {:+.3f}, {:+.3f})  {}".format(
            name, tag, pose[0], pose[1], pose[2], reason)))
    return not bad


# ----------------------------------------------------------------------
def run_single(args, cell, case_name, holder):
    case = SINGLE_CASES[case_name]
    ip = args.left_ip if args.arm == 'left' else args.right_ip
    side = args.arm if args.arm in ('left', 'right') else 'left'

    scene = XArmTestSingleScene(ip, cell, side=side, gripper=args.gripper,
                                execute=args.execute, offline=args.offline)
    holder.append(scene)   # registered immediately, so a later raise still shuts it down
    print("\n  case '{}':".format(case_name))
    pick_px = preview_point("pick", case['pick'], cell, scene.T_cam, scene.intr, side)
    place_px = preview_point("place", case['place'], cell, scene.T_cam, scene.intr, side)
    print("    rotation     {:+.3f} rad".format(case['rot']))

    skill = XArmSingleArmPickAndPlaceSkill(scene, {'speed': args.speed, 'acc': args.acc})
    action = np.array([pick_px[0], pick_px[1], place_px[0], place_px[1], case['rot']])
    print("    action (5)   {}".format(_fmt(action, 1)))

    if confirm("Run single-arm pick-and-place on the {} arm".format(side), args.yes):
        skill.reset()
        skill.step(action)
    return scene


def run_dual(args, cell, case_name, holder):
    case = DUAL_CASES[case_name]
    scene = XArmTestDualScene(args.left_ip, args.right_ip, cell, gripper=args.gripper,
                              execute=args.execute, offline=args.offline)
    holder.append(scene)
    print("\n  case '{}':".format(case_name))
    pl = preview_point("pick L", case['pick_l'], cell, scene.T_left_cam, scene.intr, 'left')
    ll = preview_point("place L", case['place_l'], cell, scene.T_left_cam, scene.intr, 'left')
    pr = preview_point("pick R", case['pick_r'], cell, scene.T_right_cam, scene.intr, 'right')
    lr = preview_point("place R", case['place_r'], cell, scene.T_right_cam, scene.intr, 'right')
    a0, a1 = case['active']
    print("    active       left={} right={}".format(bool(a0), bool(a1)))
    # The skill assigns arms on the TABLE, not by pixel column, so a pick's pixel x
    # says nothing about which arm gets it -- only its base x does. Check that
    # instead: it is the property the case names claim.
    if case['pick_l'][0] > case['pick_r'][0]:
        print("    !! the 'left' pick is FARTHER from the left base than the 'right' one,")
        print("       so the skill will swap them. The case is mis-named, not the camera.")

    skill = XArmPickAndPlaceSkill(scene, {'speed': args.speed, 'acc': args.acc})
    action = np.array([pl[0], pl[1], pr[0], pr[1],
                       ll[0], ll[1], lr[0], lr[1],
                       0.0, 0.0, a0, a1])
    print("    action (12)  {}".format(_fmt(action, 1)))
    if case_name == 'collision':
        print("    expecting check_trajectories_close to fire -> SEQUENTIAL execution")

    if confirm("Run dual-arm pick-and-place, case '{}'".format(case_name), args.yes):
        skill.reset()
        skill.step(action)
    return scene


def run_fling(args, cell, case_name, holder):
    case = FLING_CASES[case_name]
    scene = XArmTestDualScene(args.left_ip, args.right_ip, cell, gripper=args.gripper,
                              execute=args.execute, offline=args.offline)
    holder.append(scene)
    print("\n  case '{}':".format(case_name))
    pl = preview_point("pick L", case['pick_l'], cell, scene.T_left_cam, scene.intr, 'left')
    pr = preview_point("pick R", case['pick_r'], cell, scene.T_right_cam, scene.intr, 'right')
    v0, v1 = case['valid']
    print("    valid flags  {} {}".format(v0, v1))

    # Each stage can be switched off, so the fling can be brought up on hardware a
    # piece at a time rather than committing to the whole swing on the first run.
    skill = XArmPickAndFlingSkill(scene, {
        'speed': args.speed, 'acc': args.acc,
        'probe_contact': not args.skip_probe,
        'shake': not args.skip_shake,
        'release_tension': not args.skip_release,
    })
    stages = [name for name, on in (('probe', not args.skip_probe),
                                    ('shake', not args.skip_shake),
                                    ('release', not args.skip_release)) if not on]
    if stages:
        print("    stages OFF   {}".format(", ".join(stages)))
    action = np.array([pl[0], pl[1], pr[0], pr[1], 0.0, 0.0, v0, v1])
    print("    action (8)   {}".format(_fmt(action, 1)))

    if not confirm("Run pick-and-fling, case '{}'".format(case_name), args.yes):
        return scene
    skill.reset()
    traj = skill.step(action, record_debug=True)

    n_l, n_r = len(traj.get('ur5e', [])), len(traj.get('ur16e', []))
    print("\n    recorded trajectory: {} left samples, {} right samples".format(n_l, n_r))
    if case_name == 'invalid':
        if n_l or n_r:
            print("    !! the invalid case should have aborted before moving.")
    elif args.execute and not (n_l and n_r):
        print("    !! empty trajectory -- the imp logger reuses the 'ur5e'/'ur16e' keys, so")
        print("       an empty dict means the fling recording path is broken.")
    return scene


RUNNERS = {'single-pnp': run_single, 'dual-pnp': run_dual, 'fling': run_fling}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run one xArm Lite 6 primitive on the real cell, geometry-checked.")
    ap.add_argument('--primitive', choices=list(RUNNERS), required=True)
    ap.add_argument('--case', default='basic',
                    help="case name for the chosen primitive, or 'all'")
    ap.add_argument('--left-ip', default=os.environ.get('XARM_LEFT_IP', '192.168.1.155'))
    ap.add_argument('--right-ip', default=os.environ.get('XARM_RIGHT_IP', '192.168.1.170'))
    ap.add_argument('--arm', choices=['left', 'right'], default='left',
                    help="which arm for single-pnp (ignored by the dual primitives)")
    ap.add_argument('--execute', action='store_true',
                    help="actually move the arms (default is a dry run)")
    ap.add_argument('--dry-run', action='store_true',
                    help="connect and run the real IK checks, but do not move (default)")
    ap.add_argument('--offline', action='store_true',
                    help="no hardware at all; geometric reach estimate only")
    ap.add_argument('--gripper', default='lite6', choices=['lite6', 'none'])
    ap.add_argument('--speed', type=float, default=0.10, help="m/s (default is slow)")
    ap.add_argument('--acc', type=float, default=0.3, help="m/s^2")
    ap.add_argument('--yes', action='store_true', help="don't prompt before running")
    ap.add_argument('--force-uncalibrated', action='store_true',
                    help="run even though the cell has not been measured")
    # Fling stages, for incremental hardware bring-up.
    ap.add_argument('--skip-probe', action='store_true',
                    help="fling: skip the effort-gated contact probe and descend "
                         "straight to the calibrated grasp height")
    ap.add_argument('--skip-shake', action='store_true',
                    help="fling: skip the vertical shake after the stretch")
    ap.add_argument('--skip-release', action='store_true',
                    help="fling: open the grippers without the inward tension release")
    args = ap.parse_args()

    # Either safety flag wins over --execute, so an accidental combination of the
    # two never ends up moving an arm.
    if args.offline or args.dry_run:
        args.execute = False
    cases = CASES[args.primitive]
    names = list(cases) if args.case == 'all' else [args.case]
    for n in names:
        if n not in cases:
            ap.error("unknown case '{}' for {}; choose from: {}".format(
                n, args.primitive, ", ".join(cases)))

    cell = load_cell()

    print("=" * 72)
    print("xArm Lite 6 primitive test: {}".format(args.primitive))
    print("  cases      : {}".format(", ".join(names)))
    print("  mode       : {}".format(
        "OFFLINE (no hardware)" if args.offline else
        ("EXECUTE -- THE ARMS WILL MOVE" if args.execute else "dry run (connected, no motion)")))
    print("  separation : {:.3f} m".format(cell.separation))
    for side in ('left', 'right'):
        print("  {:<10s} : table_z {:+.3f} m   gripper offset {:+.3f} m   "
              "reach {:.2f} - {:.2f} m".format(
                  side, cell.table_z(side), cell.gripper_offset(side),
                  *cell.workspace_radius(side)))
    print("=" * 72)

    selfcheck_camera(cell.separation, cell.table_z('left'))
    print_reach_map(cell)

    # Offline is a pure geometry check, so an unmeasured cell is only a warning
    # there; anything that touches hardware needs the real numbers.
    ok = check_cell(cell, args.force_uncalibrated or args.offline)
    if not ok:
        print(red("\n  Refusing to continue. Pass --force-uncalibrated to override, or "
                  "--offline to check geometry only."))
        return 1

    if not args.offline:
        for ip in {args.left_ip, args.right_ip}:
            reachable, err = preflight(ip)
            if not reachable:
                print_network_help(ip, err)
                return 1

    if args.execute:
        print("\n  !! THE ARMS WILL MOVE. Clear the table, keep a hand on the e-stop.")

    failed = []
    try:
        for name in names:
            print("\n" + "-" * 72)
            # One scene per case, disconnected before the next: two live XArmAPI
            # connections to the same controller is not worth risking.
            holder = []
            try:
                scene = RUNNERS[args.primitive](args, cell, name, holder)
                if not summarise(scene, "case '{}'".format(name)):
                    failed.append(name)
            finally:
                for s in holder:
                    shutdown(s)
        print("\n" + "=" * 72)
        if failed:
            print(red("UNREACHABLE waypoints in: {}".format(", ".join(failed))))
            print(red("Retune the offending constants, then re-run the dry run."))
            return 1
        print(green("All waypoints reachable in: {}".format(", ".join(names))))
        if not args.execute:
            print("Dry run only -- nothing moved. Re-run with --execute when ready.")
        return 0

    except Unreachable as e:
        print(red("\n  !! ABORTED before moving: {}".format(e)))
        return 1
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
