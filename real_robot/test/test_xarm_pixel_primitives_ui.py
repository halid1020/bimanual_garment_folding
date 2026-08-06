"""Click a pixel, move the REAL arms. The hardware twin of ``sim/run_sim_ui.py``.

Same program shape as the simulator UI on purpose: the image you click is the same
square crop between the arms, the click helpers are the same ``human_utils`` ones the
human policies use, and the primitive that runs is the same **unmodified**
``XArmPickAndFlingSkill`` / ``XArmPickAndPlaceSkill`` /
``XArmSingleArmPickAndPlaceSkill``. Whatever the sim did, this does -- including the
fling's swing shape, which lives in ``xarm_base_fling_poses`` and belongs to the
skill, not to either scene.

    python real_robot/test/test_xarm_pixel_primitives_ui.py --primitive fling --dry-run
    python real_robot/test/test_xarm_pixel_primitives_ui.py --primitive fling

⚠️ THIS MOVES REAL ARMS. ``--dry-run`` exercises the whole click -> pixel -> table ->
IK path with the drivers stubbed out and nothing energised; do that first, every time
you change anything. Without it you are asked to confirm once, before the first
motion.

Usage
    --primitive {fling,dual-pnp,single-pnp}   which skill to drive
    --arm {left,right}                        single-pnp only
    --dry-run                                 no hardware, no motion
    --skip-probe / --skip-shake / --skip-release   fling bring-up stages
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
from real_robot.primitives.xarm_pick_and_place import XArmPickAndPlaceSkill
from real_robot.primitives.xarm_single_arm_pick_and_place import (
    XArmSingleArmPickAndPlaceSkill,
)
from real_robot.robot.xarm_dual_arm_scene import XArmDualArmScene
from real_robot.robot.xarm_single_arm_scene import XArmSingleArmScene
from real_robot.utils import xarm_constants as C
from real_robot.utils.human_utils import (
    click_points_pick_and_fling,
    click_points_pick_and_place,
    click_points_single_pick_and_place,
)
from real_robot.utils.xarm_camera import base_to_pixel


# ----------------------------------------------------------------------
def valid_flag(mask, px):
    """1.0 if this pixel is reachable, 0.0 if not.

    The primitives take these as their ``valid``/``active`` flags and ABORT on a
    zero rather than driving at an unreachable point. In the simulator the mask is
    the cloth; here it is the arms' reach annulus (``get_workspace_masks``), which
    is the question that actually matters at the hardware stage -- a click outside
    it is a move the controller will refuse, and refusing it here costs nothing.
    """
    if mask is None:
        return 1.0
    u, v = int(round(px[0])), int(round(px[1]))
    if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
        return 0.0
    return 1.0 if mask[v, u] else 0.0


def overlay(rgb, scene, masks):
    """The frame the operator clicks on: reach shading, base marks, a scale bar.

    Deliberately NOT imported from ``sim/run_sim_ui.py``. That module imports
    mujoco, and this one has to run on the robot machine.
    """
    img = rgb.copy()
    left_mask, right_mask = masks if masks else (None, None)
    for mask, tint in ((left_mask, (0, 40, 0)), (right_mask, (0, 0, 40))):
        if mask is None:
            continue
        img[mask] = np.clip(img[mask].astype(np.int16) + np.array(tint), 0, 255) \
            .astype(np.uint8)

    T, intr = getattr(scene, 'T_left_cam', None), getattr(scene, 'intr', None)
    sep = getattr(scene, 'separation', None)
    if T is not None and intr is not None and sep is not None \
            and not isinstance(intr, np.ndarray):
        for label, xy in (('left', (0.0, 0.0)), ('right', (sep, 0.0))):
            try:
                px = base_to_pixel([xy[0], xy[1], C.XARM_TABLE_Z], T, intr)
            except Exception:
                continue
            if not np.isfinite(px).all():
                continue
            p = tuple(int(v) for v in px)
            cv2.drawMarker(img, p, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(img, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "front is UP  |  green = left reach, blue = right reach",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def auto_points(scene, n):
    """Scripted, reachable pixels instead of clicks -- for ``--auto``.

    Placed in TABLE metres and projected, not picked in pixels, so they mean the
    same thing at any crop or separation. They sit either side of the midline,
    inside both reach annuli, which is the arrangement the fling wants.
    """
    # `or` would ask a 4x4 array for its truth value.
    dual = hasattr(scene, 'T_left_cam')
    T = scene.T_left_cam if dual else scene.T_cam
    if dual:
        S = scene.separation
        xs = {2: [S / 2 - 0.13, S / 2 + 0.13],
              4: [S / 2 - 0.13, S / 2 - 0.13, S / 2 + 0.13, S / 2 + 0.13]}[n]
        ys = {2: [0.0, 0.0], 4: [0.0, -0.10, 0.0, -0.10]}[n]
    else:
        # One arm, one base frame: both points inside ITS reach annulus, which is
        # centred on its own base rather than on the cell midline.
        xs, ys = [0.25, 0.25], [-0.06, +0.06]
    return [base_to_pixel([x, y, C.XARM_TABLE_Z], T, scene.intr)
            for x, y in zip(xs, ys)]


def show_and_click(title, rgb, scene, masks, clicker):
    """Put the frame up in the operator's colours and take the clicks.

    ⚠️ cv2 windows are BGR. Handing them the RGB straight off the camera swaps red
    and blue, which is a poor way to judge a garment.
    """
    bgr = cv2.cvtColor(overlay(rgb, scene, masks), cv2.COLOR_RGB2BGR)
    combined = None
    if masks and masks[0] is not None and masks[1] is not None:
        combined = masks[0] | masks[1]
    try:
        return clicker(title, bgr, combined)
    except RuntimeError as exc:
        # The helpers RAISE when 'q' cancels rather than returning None.
        print("  cancelled ({}).".format(exc))
        return None


def confirm_motion(dry_run, auto_yes=False):
    """The one gate before anything energises.

    ⚠️ Enter means GO, matching ``test_xarm_lite6_bringup.confirm``. An earlier
    version here demanded the literal word 'go' and treated everything else --
    including a bare Enter -- as an abort, which is the opposite of what the rest
    of the repo trains your fingers to do. Two real-arm scripts must not disagree
    about what the Enter key does.
    """
    if dry_run:
        print("\n[dry-run] Nothing is energised; no arm will move.")
        return True
    if auto_yes:
        print("\n⚠️  Driving the REAL arms  [--yes]")
        return True
    print("\n" + "=" * 68)
    print("⚠️  This drives the REAL arms. Clear the cell, keep the e-stop in hand.")
    print("=" * 68)
    try:
        reply = input("[Enter = go, 'q' = quit] ").strip().lower()
    except EOFError:
        return False
    return reply not in ('q', 'n', 'no', 'quit')


def describe_fling():
    """Print the swing the skill is about to run, in metres, before it runs it.

    The shape is asymmetric -- a short wind-up, a long forward stroke, a touch-down
    part way back and then a drag to just behind the base line -- and the whole
    point of printing it is that the operator can check the numbers against what
    they then watch happen. The touch-down is worth reading in particular: it is
    the one the arms were landing too far forward on before it got its own number.
    """
    from real_robot.primitives.utils import xarm_base_fling_poses
    p = xarm_base_fling_poses(stroke=C.XARM_FLING_STROKE, swing_angle=C.XARM_FLING_ANGLE,
                              lift_height=C.XARM_FLING_HANG,
                              place_height=C.XARM_FLING_PLACE_Z,
                              windup=C.XARM_FLING_WINDUP,
                              place_y=C.XARM_FLING_PLACE_Y,
                              land_y=C.XARM_FLING_LAND_Y)
    names = ['centre', 'wind up (back)', 'STROKE (front)', 'touch down', 'lay down']
    print("\n  the swing, in the action frame (+y = front, toward you):")
    for name, w in zip(names, p):
        print("    {:<16} y {:+.3f}   z {:+.3f}".format(name, w[1], w[2]))
    print("    -> {:.0f} mm back, {:.0f} mm forward, landing {:.0f} mm out, then "
          "{:.0f} mm of drag to {:.0f} mm behind the base line".format(
              -p[1, 1] * 1000, p[2, 1] * 1000, p[3, 1] * 1000,
              (p[3, 1] - p[4, 1]) * 1000, -p[4, 1] * 1000))


# ----------------------------------------------------------------------
def run_fling(args, scene):
    describe_fling()
    skill = XArmPickAndFlingSkill(scene, {
        'speed': args.speed, 'acc': args.acc,
        'probe_contact': not args.skip_probe,
        'shake': not args.skip_shake,
        'release_tension': not args.skip_release,
    })
    skill.reset()

    while True:
        print("\nGoing to the photo pose, then capturing...")
        scene.go_camera_pos()          # arms out of shot, as the arena does
        rgb, _ = scene.take_rgbd()
        if rgb is None:
            print("No camera frame. Retrying...")
            time.sleep(1)
            continue

        if args.auto:
            clicks = auto_points(scene, 2)
        else:
            clicks = show_and_click("xArm -- pick & fling (2 picks)", rgb, scene,
                                    scene.get_workspace_masks(),
                                    click_points_pick_and_fling)
        if clicks is None:
            return
        p0, p1 = clicks
        combined = np.logical_or(*scene.get_workspace_masks())
        v0, v1 = valid_flag(combined, p0), valid_flag(combined, p1)
        print("  picks {} {}   reachable flags {} {}".format(
            np.round(p0, 1), np.round(p1, 1), v0, v1))
        if v0 < 0.5 or v1 < 0.5:
            print("  !! a pick is outside the reach annulus; the skill will abort. "
                  "Click inside the shaded area.")

        # The 8-element layout the skill parses: pick0, pick1, angles, valid flags.
        # ⚠️ The click helpers return raw (u, v) tuples, NOT an action vector.
        skill.step(np.array([p0[0], p0[1], p1[0], p1[1], 0.0, 0.0, v0, v1],
                            dtype=float))
        if args.once:
            return


def run_dual(args, scene):
    skill = XArmPickAndPlaceSkill(scene, {'speed': args.speed, 'acc': args.acc})
    skill.reset()

    while True:
        print("\nGoing to the photo pose, then capturing...")
        scene.go_camera_pos()
        rgb, _ = scene.take_rgbd()
        if rgb is None:
            print("No camera frame. Retrying...")
            time.sleep(1)
            continue

        if args.auto:
            clicks = auto_points(scene, 4)
        else:
            clicks = show_and_click("xArm -- dual pick & place (4 clicks)", rgb,
                                    scene, scene.get_workspace_masks(),
                                    click_points_pick_and_place)
        if clicks is None:
            return
        # ⚠️ The click ORDER is pick0, place0, pick1, place1; the skill's layout is
        # pick0, pick1, place0, place1. They have to be interleaved.
        pk0, pl0, pk1, pl1 = clicks
        combined = np.logical_or(*scene.get_workspace_masks())
        a0, a1 = valid_flag(combined, pk0), valid_flag(combined, pk1)
        print("  picks {} {}   places {} {}   active flags {} {}".format(
            np.round(pk0, 1), np.round(pk1, 1),
            np.round(pl0, 1), np.round(pl1, 1), a0, a1))
        skill.step(np.array([pk0[0], pk0[1], pk1[0], pk1[1],
                             pl0[0], pl0[1], pl1[0], pl1[1],
                             0.0, 0.0, a0, a1], dtype=float))
        if args.once:
            return


def run_single(args, scene):
    skill = XArmSingleArmPickAndPlaceSkill(scene, {'speed': args.speed,
                                                   'acc': args.acc})
    skill.reset()
    mask = scene.get_workspace_mask()

    while True:
        print("\nGoing to the photo pose, then capturing...")
        scene.out_scene()
        rgb, _ = scene.take_rgbd()
        if rgb is None:
            print("No camera frame. Retrying...")
            time.sleep(1)
            continue

        if args.auto:
            clicks = auto_points(scene, 2)
        else:
            clicks = show_and_click("xArm -- single pick & place (pick, place)",
                                    rgb, scene, (mask, None),
                                    click_points_single_pick_and_place)
        if clicks is None:
            return
        pick, place = clicks
        print("  pick {}   place {}   reachable {}".format(
            np.round(pick, 1), np.round(place, 1), valid_flag(mask, pick)))
        # 5 values: pick u,v, place u,v, rotation.
        skill.step(np.array([pick[0], pick[1], place[0], place[1], 0.0], dtype=float))
        if args.once:
            return


# ----------------------------------------------------------------------
class _StubArm:
    """A driver that answers questions and moves nothing. ``--dry-run`` only.

    The scenes' dry-run branch sets ``self.left = self.right = None``, but the
    primitives reach past the scene's ``safe_*`` wrappers and call the driver
    directly for the things a wrapper cannot fake -- ``get_tcp_pose``,
    ``effort_baseline``, ``effort_exceeded``. So a dry run crashes at the first of
    those unless something stands in.

    ⚠️ It reports a FIXED pose. The stretch loop compares TCP distance against the
    width cap, so with a constant pose it exits on its 5 s timeout instead of on
    the cap -- exactly as the MuJoCo sim does, and harmless, because a dry run is
    checking the pixel -> table -> IK -> waypoint arithmetic and the action
    plumbing, not dynamics. The simulator is where motion gets checked.
    """

    def __init__(self, side, separation):
        self.side = side
        # ⚠️ IN THIS ARM'S OWN BASE FRAME. Each arm sits half a stretch in from the
        # midline, so BOTH report the same x -- the right base is yawed 180 deg, so
        # its +x points back across the table and transform_pose(T_left_right, .)
        # lands this at separation/2 + half_width in the left frame.
        #
        # Reporting the right arm's pose already-in-left-frame collapses the two
        # grasp points onto the same x. points_to_action_frame then takes its axis
        # from (left_point - right_point), which is numerical noise, and the whole
        # swing comes out rotated ~90 deg -- the arms sweep across the table instead
        # of front-to-back. It looks like a broken fling and is a broken stub.
        self._pose = np.array([separation / 2.0 - 0.13, 0.0, C.XARM_FLING_HANG,
                               np.pi, 0.0, 0.0], dtype=float)

    def get_tcp_pose(self):
        return self._pose.copy()

    def effort_baseline(self):
        return None                      # "no signal" -> the geometric cap rules

    def effort_exceeded(self, baseline, threshold):
        return False

    def effort_delta(self, baseline):
        # None, not 0.0 -- the real driver returns None when it has no baseline to
        # measure against, and a dry run that answers 0.0 reports a confident
        # "max effort delta 0.00" for a signal it never read.
        return None if baseline is None else 0.0

    def get_joint_effort(self):
        return None

    def home(self, *a, **k):
        return True

    def out_scene(self, *a, **k):
        return True

    def open_gripper(self, *a, **k):
        return True

    def close_gripper(self, *a, **k):
        return True

    def disconnect(self):
        return True


def calibrate_offline(scene):
    """Give a --dry-run scene the REAL camera and a blank frame to click on.

    Without this a dry run proves nothing: the scenes' dry-run branch fabricates
    ``intr = eye(3)`` and ``T_left_cam = I`` with a made-up 0.175 m offset, so every
    pixel inverts to a meaningless place and the IK is asked about a cell that does
    not exist. Loading the same calibration the live scene loads makes the dry run
    test the geometry you are actually going to run.

    Done HERE rather than in the scene on purpose -- ``xarm_dual_arm_scene.py``
    drives hardware and is not the place to grow a simulator.
    """
    from real_robot.utils.scene_utils import load_camera_to_base
    from real_robot.utils.xarm_camera import (
        crop_window, describe_orientation, load_intrinsic,
    )
    d = "{}/real_robot/calibration".format(os.environ['MP_FOLD_PATH'])
    left = os.path.join(d, 'xarm-left-calib.yaml')
    right = os.path.join(d, 'xarm-right-calib.yaml')

    if not hasattr(scene, 'T_left_cam'):
        # Single-arm scene: one camera, and note it does NOT crop -- its take_rgbd
        # hands back the raw frame, unlike the dual scene's square window.
        side_file = left if getattr(scene, 'side', 'left') == 'left' else right
        scene.T_cam = load_camera_to_base(side_file)
        intr = load_intrinsic(side_file)
        scene.intr = intr
        blank = np.full((int(intr.height), int(intr.width), 3), 190, np.uint8)
        scene.take_rgbd = lambda: (blank.copy(),
                                   np.zeros(blank.shape[:2], np.float32))
        scene._calculate_wrkspace_masks(blank)
        scene.arm = _StubArm(getattr(scene, 'side', 'left'), C.XARM_BASE_SEPARATION)
        print("[dry-run] single-arm camera loaded from the calibration "
              "(UNCROPPED -- that is what the hardware scene does).")
        return

    scene.T_left_cam = load_camera_to_base(left)
    scene.T_right_cam = load_camera_to_base(right)
    scene.T_left_right = scene.T_left_cam @ np.linalg.inv(scene.T_right_cam)
    scene.separation = float(np.linalg.norm(scene.T_left_right[:3, 3]))

    full = load_intrinsic(left)
    scene.crop = crop_window(full, scene.T_left_cam, separation=scene.separation,
                             table_z=C.XARM_TABLE_Z, verbose=True)
    scene.intr = scene.crop.intrinsic(full)
    print(describe_orientation(scene.T_left_cam, scene.intr,
                               scene.separation, C.XARM_TABLE_Z)[1])
    # The UR-named aliases were set from the fabricated values in __init__.
    scene.T_ur5e_cam = scene.T_left_cam
    scene.T_ur16e_cam = scene.T_right_cam
    scene.T_ur5e_ur16e = scene.T_left_right

    side = int(scene.intr.width)
    blank = np.full((side, side, 3), 190, np.uint8)
    scene.take_rgbd = lambda: (blank.copy(), np.zeros((side, side), np.float32))
    if hasattr(scene, '_calculate_wrkspace_masks'):
        scene._calculate_wrkspace_masks(blank)
    scene.left = _StubArm('left', scene.separation)
    scene.right = _StubArm('right', scene.separation)
    print("[dry-run] camera, crop and cell geometry loaded from the calibration; "
          "the frame is blank and the drivers are stubs.")


def teardown(scene):
    """Leave the cell safe: grippers open and de-energised, drivers disconnected.

    Neither scene has a close(); the drivers have disconnect(), and it is the thing
    that vents the solenoid. Skipping it leaves a gripper holding under power.
    """
    for name in ('left', 'right', 'arm'):
        arm = getattr(scene, name, None)
        if arm is None:
            continue
        try:
            arm.disconnect()
        except Exception as exc:
            print("  !! {}.disconnect() failed: {}".format(name, exc))
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="Click-to-act UI for the REAL xArm cell.")
    ap.add_argument('--primitive', choices=['single-pnp', 'dual-pnp', 'fling'],
                    required=True)
    ap.add_argument('--arm', choices=['left', 'right'], default='left',
                    help="single-pnp only")
    ap.add_argument('--left-ip', default=os.environ.get('XARM_LEFT_IP', '192.168.1.155'))
    ap.add_argument('--right-ip', default=os.environ.get('XARM_RIGHT_IP', '192.168.1.170'))
    ap.add_argument('--speed', type=float, default=0.10, help="m/s")
    ap.add_argument('--acc', type=float, default=0.3, help="m/s^2")
    ap.add_argument('--dry-run', action='store_true',
                    help="no hardware, no motion -- run this first")
    ap.add_argument('--auto', action='store_true',
                    help="scripted pixels instead of clicking; needs no display, "
                         "so a --dry-run --auto pass works over ssh")
    ap.add_argument('--once', action='store_true',
                    help="one action then exit, instead of looping")
    ap.add_argument('--yes', action='store_true',
                    help="skip the are-you-sure prompt (same flag name as "
                         "test_xarm_primitives.py)")
    ap.add_argument('--skip-probe', action='store_true',
                    help="fling: skip the effort-gated contact probe")
    ap.add_argument('--skip-shake', action='store_true',
                    help="fling: skip the vertical shake")
    ap.add_argument('--skip-release', action='store_true',
                    help="fling: skip the inward tension release")
    args = ap.parse_args()

    if not confirm_motion(args.dry_run, args.yes):
        print("Aborted.")
        return 1

    if args.primitive == 'single-pnp':
        scene = XArmSingleArmScene(
            args.left_ip if args.arm == 'left' else args.right_ip,
            dry_run=args.dry_run, side=args.arm)
        run = run_single
    else:
        # ⚠️ The third positional is dry_run, NOT the cell calibration. Passing a
        # CellCalibration here puts the scene into dry-run silently -- you click,
        # the console looks healthy, and the arms never move. The scene loads the
        # calibration itself.
        scene = XArmDualArmScene(args.left_ip, args.right_ip, dry_run=args.dry_run)
        run = run_fling if args.primitive == 'fling' else run_dual

    if args.dry_run:
        calibrate_offline(scene)

    try:
        run(args, scene)
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        return 130
    finally:
        teardown(scene)
    return 0


if __name__ == '__main__':
    sys.exit(main())
