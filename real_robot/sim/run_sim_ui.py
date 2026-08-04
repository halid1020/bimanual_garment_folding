"""Click-to-act UI for the simulated xArm Lite 6 cell.

Renders the top-down camera, lets you click the action points, then runs the REAL
primitive against the simulated arms and shows what it did.

    python real_robot/sim/run_sim_ui.py --primitive fling
    python real_robot/sim/run_sim_ui.py --primitive dual-pnp
    python real_robot/sim/run_sim_ui.py --primitive single-pnp --arm left

The click helpers come from ``real_robot/utils/human_utils.py`` -- the same ones the
real-robot human policies use -- so the interface here is the interface you will
drive the robots with.

Headless self-check (no window, scripted picks):

    python real_robot/sim/run_sim_ui.py --headless --primitive fling --auto

which asserts PyBullet IK reached every commanded pose, no joint limit was
exceeded, and the arms never touched. The offline checker in
``test_xarm_primitives.py`` only tests a reach SPHERE; this runs a real kinematic
chain, so it catches joint-limit and self-collision failures the sphere cannot.
"""
import argparse
import os
import sys

import numpy as np

from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
from real_robot.primitives.xarm_pick_and_place import XArmPickAndPlaceSkill
from real_robot.primitives.xarm_single_arm_pick_and_place import (
    XArmSingleArmPickAndPlaceSkill,
)
from real_robot.sim.xarm_cell_sim import XArmSimScene, XArmSimSingleScene
from real_robot.test.xarm_test_scene import base_to_pixel, table_xy_to_pixel
from real_robot.utils import xarm_constants as C
from real_robot.utils.term import green, red, yellow
from real_robot.utils.human_utils import (
    click_points_pick_and_fling, click_points_pick_and_place,
    click_points_single_pick_and_place,
)

TABLE_COLOUR = (90, 90, 90)
KEEPOUT_COLOUR = (60, 60, 220)
REACH_COLOUR = (60, 190, 60)
WALL_COLOUR = (0, 190, 220)
PATH_L, PATH_R = (255, 160, 40), (40, 160, 255)


def _circle_px(centre_xy, radius, T_cam, intr, table_z, n=180):
    a = np.linspace(0, 2 * np.pi, n)
    pts = np.stack([centre_xy[0] + radius * np.cos(a),
                    centre_xy[1] + radius * np.sin(a),
                    np.full(n, table_z)], axis=1)
    return base_to_pixel(pts, T_cam, intr)


def _polyline(img, px, colour, thickness=1, closed=False):
    import cv2
    pts = np.asarray(px, dtype=float)
    good = np.isfinite(pts).all(axis=1)
    pts = pts[good]
    if len(pts) < 2:
        return
    cv2.polylines(img, [pts.astype(np.int32)], closed, colour, thickness, cv2.LINE_AA)


def annotate(rgb, scene):
    """Draw the cell geometry on the camera frame, so a click has context."""
    import cv2
    img = rgb.copy()
    S, tz = scene.separation, scene.table_z
    intr, T = scene.intr, scene.T_left_cam

    rect = C.XARM_TABLE_RECT
    corners = [(rect['x'][0], rect['y'][0]), (rect['x'][1], rect['y'][0]),
               (rect['x'][1], rect['y'][1]), (rect['x'][0], rect['y'][1])]
    _polyline(img, [table_xy_to_pixel(x, y, S, tz, intr) for x, y in corners],
              TABLE_COLOUR, 2, closed=True)

    walls = C.XARM_WALLS
    wcorners = [(walls['x'][0], walls['y'][0]), (walls['x'][1], walls['y'][0]),
                (walls['x'][1], walls['y'][1]), (walls['x'][0], walls['y'][1])]
    _polyline(img, [table_xy_to_pixel(x, y, S, tz, intr) for x, y in wcorners],
              WALL_COLOUR, 1, closed=True)

    for side, base in (('left', (0.0, 0.0)), ('right', (S, 0.0))):
        r_min, r_max = C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, side)
        _polyline(img, _circle_px(base, r_min, T, intr, tz), KEEPOUT_COLOUR, 1, True)
        _polyline(img, _circle_px(base, r_max, T, intr, tz), REACH_COLOUR, 1, True)
        bpx = base_to_pixel([base[0], base[1], tz], T, intr)
        if np.isfinite(bpx).all():
            cv2.drawMarker(img, tuple(bpx.astype(int)), (0, 0, 0),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.putText(img, side, tuple((bpx + [8, -8]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # The frame is the crop now (330 px in the default cell), so the legend is
    # scaled to it rather than to the 1280 px full frame it used to be drawn on.
    fs = max(0.32, img.shape[1] / 1280.0 * 0.5)
    cv2.putText(img, "green=reach  blue=keepout  cyan=walls  grey=table",
                (8, int(18 * max(fs / 0.5, 0.7))), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (30, 30, 30), 1, cv2.LINE_AA)
    return img


def path_to_pixels(scene, pts):
    """TCP samples (LEFT base frame) -> pixels, dropped VERTICALLY to the table.

    Deliberately NOT the true camera projection, which is what this used to draw.
    The question the overlay answers is "did the arm go where I clicked", and the
    perspective projection answers a different one. With the camera 1.0 m up, a TCP
    at the 0.25 m hang height is magnified ~40%: measured on the --auto fling, the
    swing moves by up to 51 px in a 330 px window (15.6%) and the wind-up lands at
    v = -2, off the top edge. Even the grasp comes out 4.5 px from the clicked
    pixel, because the gripper sits XARM_GRIPPER_OFFSET above the table. Dropping
    to the table plane puts the grasp exactly on the click and keeps the path in
    frame; height is not shown, which is the honest trade.
    """
    pts = np.asarray(pts, dtype=float)
    return np.asarray([table_xy_to_pixel(p[0], p[1], scene.separation,
                                         scene.table_z, scene.intr)
                       for p in pts[:, :3]])


def draw_path(img, scene, traj):
    """Overlay the executed TCP paths as their shadow on the table."""
    import cv2
    out = img.copy()
    from real_robot.utils.transform_utils import transform_pose
    for key, colour in (('ur5e', PATH_L), ('ur16e', PATH_R)):
        pts = traj.get(key) or []
        if len(pts) < 2:
            continue
        pts = np.asarray(pts, dtype=float)[:, :3]
        if key == 'ur16e':
            pts = np.asarray([transform_pose(scene.T_left_right,
                                             np.concatenate([p, [0, 0, 0]]))[:3]
                              for p in pts])
        _polyline(out, path_to_pixels(scene, pts), colour, 2)
    fs = max(0.32, out.shape[1] / 1280.0 * 0.5)
    cv2.putText(out, "orange = left TCP,  blue = right TCP  (on the table)",
                (8, int(38 * max(fs / 0.5, 0.7))), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (30, 30, 30), 1, cv2.LINE_AA)
    return out


def valid_flag(mask, px):
    u, v = int(round(px[0])), int(round(px[1]))
    if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
        return 0.0
    return 1.0 if mask[v, u] > 0 else 0.0


def auto_picks(scene):
    """Two scripted grasp points on the garment, for --auto.

    ``scene.intr`` is the CROPPED intrinsic, so these come back as pixels in the
    frame the primitive will be given -- the same frame a click lands in.
    """
    gx, gy, _ = scene.garment_pose
    half = scene.garment_size[0] / 2 * 0.8
    S, tz = scene.separation, scene.table_z
    return (table_xy_to_pixel(gx - half, gy, S, tz, scene.intr),
            table_xy_to_pixel(gx + half, gy, S, tz, scene.intr))


def auto_pnp_points(scene, side=None):
    """Scripted pick/place pairs inside each arm's measured reach.

    Placed by reach annulus rather than by eye, so --auto keeps working if the
    separation or the measured radii change.
    """
    S, tz, intr = scene.separation, scene.table_z, scene.intr
    gx, gy, _ = scene.garment_pose

    def midr(which):
        lo, hi = C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, which)
        return (lo + hi) / 2.0

    if side is not None:
        r = midr(side)
        base_x = 0.0 if side == 'left' else S
        sign = 1.0 if side == 'left' else -1.0
        pick = (base_x + sign * r * 0.8, gy)
        place = (base_x + sign * r * 0.8, gy + 0.12)
        return (table_xy_to_pixel(*pick, S, tz, intr),
                table_xy_to_pixel(*place, S, tz, intr))

    rl, rr = midr('left'), midr('right')
    return (table_xy_to_pixel(rl * 0.8, gy, S, tz, intr),            # pick0 (left)
            table_xy_to_pixel(rl * 0.8, gy + 0.12, S, tz, intr),     # place0
            table_xy_to_pixel(S - rr * 0.8, gy, S, tz, intr),        # pick1 (right)
            table_xy_to_pixel(S - rr * 0.8, gy + 0.12, S, tz, intr))  # place1


def report(scene, traj, label):
    q = scene.quality_report()
    contacts = scene.contacts_between_arms()
    print("\n  {} -- {} TCP samples along the executed path".format(
        label, q.get('path_samples', 0)))
    worst = max(q['max_ik_error_left'], q['max_ik_error_right'])
    # Millimetres: at 4 decimal places in metres a perfectly good 1e-5 m residual
    # prints as 0.0000 and reads like nothing ran.
    line = "  max IK error: left {:.2f} mm, right {:.2f} mm".format(
        q['max_ik_error_left'] * 1000.0, q['max_ik_error_right'] * 1000.0)
    print(green(line) if worst < 0.005 else yellow(line))
    if q.get('ik_failures'):
        print(yellow("  {} waypoint(s) needed an IK restart or did not converge"
                     .format(q['ik_failures'])))
    if q['limit_violations']:
        print(red("  !! joint limit exceeded:"))
        for lbl, j, val in q['limit_violations'][:8]:
            print(red("       {} joint {} at {:+.3f} rad".format(lbl, j, val)))
    else:
        print(green("  no joint-limit violations"))
    if contacts:
        print(red("  !! the arms are in contact ({} points)".format(len(contacts))))
    else:
        print(green("  no arm-arm contact"))
    return worst < 0.005 and not q['limit_violations'] and not contacts


def run_once(args):
    import cv2
    scene = XArmSimScene(gui=not args.headless, gui_delay=args.delay, seed=args.seed)
    try:
        # Out of shot first, exactly as the arena's _process_info does before every
        # frame -- so what you click on is what the robots will actually photograph.
        scene.go_camera_pos()
        rgb, _, mask = scene.take_rgbd()
        frame = annotate(rgb, scene)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.primitive == 'fling':
            if args.auto:
                p0, p1 = auto_picks(scene)
            else:
                p0, p1 = click_points_pick_and_fling("xArm sim -- pick & fling", bgr, mask)
            v0, v1 = valid_flag(mask, p0), valid_flag(mask, p1)
            print("  picks {} {}   valid flags {} {}".format(
                np.round(p0, 1), np.round(p1, 1), v0, v1))
            skill = XArmPickAndFlingSkill(scene, {'speed': args.speed, 'acc': args.acc})
            skill.reset()
            traj = skill.step(np.array([p0[0], p0[1], p1[0], p1[1], 0.0, 0.0, v0, v1]),
                              record_debug=True)

        elif args.primitive == 'dual-pnp':
            if args.auto:
                pk0, pl0, pk1, pl1 = auto_pnp_points(scene)
            else:
                pk0, pl0, pk1, pl1 = click_points_pick_and_place(
                    "xArm sim -- dual pick & place", bgr, mask)
            a0, a1 = valid_flag(mask, pk0), valid_flag(mask, pk1)
            skill = XArmPickAndPlaceSkill(scene, {'speed': args.speed, 'acc': args.acc})
            skill.reset()
            # The skill's layout is pick0, pick1, place0, place1 -- the click order
            # is pick0, place0, pick1, place1, so it has to be interleaved.
            skill.step(np.array([pk0[0], pk0[1], pk1[0], pk1[1],
                                 pl0[0], pl0[1], pl1[0], pl1[1], 0.0, 0.0, a0, a1]))
            traj = scene.last_trajectory or {'ur5e': [], 'ur16e': []}

        else:
            single = XArmSimSingleScene(scene, side=args.arm)
            if args.auto:
                pick, place = auto_pnp_points(scene, side=args.arm)
            else:
                pick, place = click_points_single_pick_and_place(
                    "xArm sim -- single pick & place", bgr, mask)
            skill = XArmSingleArmPickAndPlaceSkill(
                single, {'speed': args.speed, 'acc': args.acc})
            skill.reset()
            skill.step(np.array([pick[0], pick[1], place[0], place[1], 0.0]))
            traj = {'ur5e': [], 'ur16e': []}

        ok = report(scene, traj, args.primitive)

        if not args.headless and (scene.left.path or scene.right.path):
            scene.go_camera_pos()
            after, _, _ = scene.take_rgbd()
            shown = draw_path(annotate(after, scene), scene,
                              {'ur5e': scene.left.path, 'ur16e': scene.right.path})
            cv2.imshow("xArm sim -- executed path (any key to close)",
                       cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0 if ok else 1
    finally:
        scene.close()


def main():
    ap = argparse.ArgumentParser(description="Click-to-act UI for the simulated xArm cell.")
    ap.add_argument('--primitive', choices=['fling', 'dual-pnp', 'single-pnp'],
                    default='fling')
    ap.add_argument('--arm', choices=['left', 'right'], default='left',
                    help="single-pnp only")
    ap.add_argument('--headless', action='store_true', help="no PyBullet window")
    ap.add_argument('--auto', action='store_true',
                    help="scripted picks instead of clicking (fling only)")
    ap.add_argument('--seed', type=int, default=0, help="garment placement")
    ap.add_argument('--speed', type=float, default=0.2)
    ap.add_argument('--acc', type=float, default=0.5)
    ap.add_argument('--delay', type=float, default=0.0,
                    help="seconds per animation frame; 0.002 makes the swing watchable")
    ap.add_argument('--loop', action='store_true', help="keep going until interrupted")
    args = ap.parse_args()

    if not args.headless and not os.environ.get('DISPLAY'):
        print(yellow("No DISPLAY; falling back to --headless."))
        args.headless = True
        args.auto = True

    rc = 0
    try:
        while True:
            rc = run_once(args)
            if not args.loop:
                break
            args.seed += 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return rc


if __name__ == '__main__':
    sys.exit(main())
