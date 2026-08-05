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

which asserts the arm reached every commanded pose CONTINUOUSLY, no joint limit
was exceeded, and the arms never touched. The offline checker in
``test_xarm_primitives.py`` only tests a reach SPHERE; this runs a real kinematic
chain, so it catches joint-limit and self-collision failures the sphere cannot.

The cell is MuJoCo, built from hand-eye calibration: real arm bases, the real
camera pose and intrinsic, and a real flex-cloth garment.
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
from real_robot.sim.xarm_mujoco_cell import XArmMujocoScene, XArmMujocoSingleScene
from real_robot.utils import xarm_constants as C
from real_robot.utils.xarm_camera import base_to_pixel
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


def table_px(scene, x, y):
    """A point on the table, in LEFT-base metres -> a pixel in the scene's frame.

    ⚠️ NOT ``xarm_test_scene.table_xy_to_pixel``, which this used to call. That one
    projects through ``synthetic_T_left_cam`` -- the fabricated camera 1.0 m above
    the arm midpoint. The cell's camera is now the CALIBRATED one, 0.839 m up and
    a little off the midline, so the two disagree by tens of pixels: every overlay
    would be drawn beside the thing it is annotating, and ``--auto`` would aim its
    scripted picks off the cloth.
    """
    return base_to_pixel([x, y, scene.table_z], scene.T_left_cam, scene.intr)


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
    _polyline(img, [table_px(scene, x, y) for x, y in corners],
              TABLE_COLOUR, 2, closed=True)

    walls = C.XARM_WALLS
    wcorners = [(walls['x'][0], walls['y'][0]), (walls['x'][1], walls['y'][0]),
                (walls['x'][1], walls['y'][1]), (walls['x'][0], walls['y'][1])]
    _polyline(img, [table_px(scene, x, y) for x, y in wcorners],
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
    return np.asarray([table_px(scene, p[0], p[1]) for p in pts[:, :3]])


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


def snap_to_mask(px, mask, radius=25):
    """Nudge a scripted pixel onto the cloth, or leave it alone if it already is.

    Only for ``--auto``. A human click is never adjusted -- clicking off the
    garment is a legitimate thing to do and the primitive's abort path is worth
    exercising -- but a self-check that silently skips because its scripted pick
    missed the sheet by two pixels is a test that always passes.
    """
    u, v = int(round(px[0])), int(round(px[1]))
    H, W = mask.shape[:2]
    if 0 <= v < H and 0 <= u < W and mask[v, u] > 0:
        return np.asarray(px, dtype=float)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.asarray(px, dtype=float)
    d = (xs - u) ** 2 + (ys - v) ** 2
    i = int(np.argmin(d))
    if d[i] > radius ** 2:
        return np.asarray(px, dtype=float)          # too far: let it fail honestly
    return np.array([float(xs[i]), float(ys[i])])


def valid_flag(mask, px):
    u, v = int(round(px[0])), int(round(px[1]))
    if not (0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]):
        return 0.0
    return 1.0 if mask[v, u] > 0 else 0.0


def _cloth_grasp_xy(scene, which):
    """A point ON the cloth, as far toward ``which`` arm as the cloth reaches.

    Taken from the cloth's ACTUAL vertex positions rather than from
    ``garment_pose``: the sheet is simulated now, so by the time anything is
    clicked it has fallen, spread and possibly been moved by an earlier
    primitive, and a scripted pick derived from where it was SPAWNED is a pick
    at bare table. That is not a hypothetical -- placing the --auto picks by
    reach annulus put both of them off the sheet, both validity flags came back
    0, and dual-pnp "passed" having moved nothing at all.

    Reachability is still checked, because a point on the cloth that no arm can
    reach is no better: the vertex is pulled toward that arm's base until it is
    inside the measured annulus.
    """
    verts = scene.cloth_vertices()
    if len(verts) == 0:                              # no cloth: fall back to spawn
        gx, gy, _ = scene.garment_pose
        return np.array([gx, gy])
    base = np.array([0.0, 0.0] if which == 'left' else [scene.separation, 0.0])
    order = np.argsort(np.linalg.norm(verts[:, :2] - base, axis=1))
    r_min, r_max = C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, which)
    centre = verts[:, :2].mean(axis=0)
    chosen = None
    for i in order:
        r = float(np.linalg.norm(verts[i, :2] - base))
        if r_min < r < r_max:
            chosen = verts[i, :2]
            break
    if chosen is None:
        # Nothing on the cloth is reachable; clamp the nearest vertex into the annulus.
        d = verts[order[0], :2] - base
        chosen = base + d / max(np.linalg.norm(d), 1e-9) * (r_min + r_max) / 2.0
    # Step in from the edge. The nearest reachable vertex is by construction on the
    # cloth's BOUNDARY, and the boundary is exactly where the segmentation mask
    # stops -- so the projected pixel lands a pixel or two off the mask, the
    # primitive reads a validity flag of 0, and the whole skill skips while
    # reporting success. Both the fling and dual-pnp did precisely that.
    return chosen + 0.2 * (centre - chosen)


def auto_picks(scene):
    """Two scripted grasp points on the garment, for --auto.

    ``scene.intr`` is the CROPPED intrinsic, so these come back as pixels in the
    frame the primitive will be given -- the same frame a click lands in.
    """
    left_xy = _cloth_grasp_xy(scene, 'left')
    right_xy = _cloth_grasp_xy(scene, 'right')
    return (table_px(scene, left_xy[0], left_xy[1]),
            table_px(scene, right_xy[0], right_xy[1]))


def auto_pnp_points(scene, side=None):
    """Scripted pick/place pairs: on the cloth, inside each arm's measured reach.

    The place is 0.12 m toward the FRONT (+y) of the pick, so a successful run
    moves cloth in the direction the whole cell is oriented around.
    """
    if side is not None:
        pick = _cloth_grasp_xy(scene, side)
        return (table_px(scene, pick[0], pick[1]),
                table_px(scene, pick[0], pick[1] + 0.12))

    pick0 = _cloth_grasp_xy(scene, 'left')
    pick1 = _cloth_grasp_xy(scene, 'right')
    return (table_px(scene, pick0[0], pick0[1]),          # pick0 (left)
            table_px(scene, pick0[0], pick0[1] + 0.12),   # place0
            table_px(scene, pick1[0], pick1[1]),          # pick1 (right)
            table_px(scene, pick1[0], pick1[1] + 0.12))   # place1


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

    # RECONFIGURATIONS ARE FAILURES, not warnings. Every one is a waypoint the
    # warm-started (continuity-preserving) solve could not reach, so the sim had
    # to flip the elbow or the wrist to get there. The xArm controller plans a
    # straight Cartesian line and cannot do that: on 2026-08-04 a path full of
    # them drove J4 to -363 deg and e-stopped both arms with servo 4, code 23,
    # while this check printed "no joint-limit violations" and passed.
    recon = q.get('reconfigurations_left', []) + q.get('reconfigurations_right', [])
    if recon:
        print(red("  !! {} waypoint(s) needed the arm RECONFIGURED to be reached."
                  .format(len(recon))))
        print(red("     The real controller cannot do this inside a linear move."))
        for xyz, err in recon[:5]:
            print(red("       ({:+.3f}, {:+.3f}, {:+.3f})  warm-start residual {:.1f} mm"
                      .format(xyz[0], xyz[1], xyz[2], err * 1000.0)))
        if len(recon) > 5:
            print(red("       ... and {} more".format(len(recon) - 5)))
    else:
        print(green("  no reconfigurations -- every pose reachable continuously"))

    # Worst distance to a joint limit. A path can clear every limit and still be
    # one seed away from error 23; this says by how much it cleared them.
    for side in ('left', 'right'):
        m = q.get('joint_margins_{}'.format(side))
        if m is None:
            continue
        line = "  {:<5s} joint margin: {}".format(side, "  ".join(
            "J{}{:+6.0f}".format(j + 1, np.degrees(m[j])) for j in range(len(m))))
        print(green(line) if m.min() > 0 else red(line + "   <- OUTSIDE ITS LIMIT"))

    # How far a joint moved in ONE interpolation step. A path can be smooth in
    # Cartesian space and still lurch, because at a wrist singularity J4 and J6
    # turn about the same axis and the solver is free to slide between them --
    # measured at 54 deg in a single step before the null-space term went in.
    # On hardware that is a judder under load, not a numerical curiosity.
    JUMP_LIMIT = 5.0        # deg per interpolation step
    for side in ('left', 'right'):
        j = q.get('max_joint_jump_{}'.format(side))
        if j is None or not np.any(j):
            continue
        line = "  {:<5s} worst step:   {}".format(side, "  ".join(
            "J{}{:+6.1f}".format(k + 1, np.degrees(j[k])) for k in range(len(j))))
        print(green(line) if np.degrees(j).max() < JUMP_LIMIT
              else red(line + "   <- JUMPS"))

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
    return (worst < 0.005 and not q['limit_violations'] and not contacts
            and not recon)


def report_cloth(scene, before):
    """Where the garment ended up, in metres and in the frame.

    The fling's whole job is to move cloth, and every other number in the report
    is about the arms. Front is +y, so a fling that worked moves the centroid
    toward +y -- and toward the TOP of the frame, which is the same statement in
    the other coordinate system and is worth printing next to it, because the two
    disagreeing is exactly the class of bug that put the garment at the back of
    the table on 2026-08-04.
    """
    after = scene.cloth_centroid()
    if not np.isfinite(after).all() or not np.isfinite(before).all():
        return
    d = after - before
    v_before = table_px(scene, before[0], before[1])[1]
    v_after = table_px(scene, after[0], after[1])[1]
    print("  cloth centroid ({:+.3f}, {:+.3f}) -> ({:+.3f}, {:+.3f}) m   "
          "moved {:+.3f} m in y ({}), {:+.0f} px in v ({})".format(
              before[0], before[1], after[0], after[1], d[1],
              "toward the front" if d[1] > 0 else "toward the back",
              v_after - v_before,
              "up the frame" if v_after < v_before else "down the frame"))


def run_once(args):
    import cv2
    scene = XArmMujocoScene(gui=not args.headless, gui_delay=args.delay,
                            seed=args.seed)
    cloth_before = scene.cloth_centroid()
    try:
        # Out of shot first, exactly as the arena's _process_info does before every
        # frame -- so what you click on is what the robots will actually photograph.
        scene.go_camera_pos()
        rgb, _, mask = scene.take_rgbd()
        frame = annotate(rgb, scene)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if args.primitive == 'fling':
            if args.auto:
                p0, p1 = (snap_to_mask(px, mask) for px in auto_picks(scene))
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
                pk0, pk1 = snap_to_mask(pk0, mask), snap_to_mask(pk1, mask)
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
            single = XArmMujocoSingleScene(scene, side=args.arm)
            if args.auto:
                pick, place = auto_pnp_points(scene, side=args.arm)
                pick = snap_to_mask(pick, mask)
            else:
                pick, place = click_points_single_pick_and_place(
                    "xArm sim -- single pick & place", bgr, mask)
            skill = XArmSingleArmPickAndPlaceSkill(
                single, {'speed': args.speed, 'acc': args.acc})
            skill.reset()
            skill.step(np.array([pick[0], pick[1], place[0], place[1], 0.0]))
            traj = {'ur5e': [], 'ur16e': []}

        ok = report(scene, traj, args.primitive)
        report_cloth(scene, cloth_before)

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
    ap.add_argument('--headless', action='store_true', help="no viewer window")
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

    # Pick a GL backend BEFORE anything imports mujoco: the offscreen renderer
    # binds its context at import time, and on a headless box the windowed
    # default cannot create one.
    if args.headless:
        os.environ.setdefault('MUJOCO_GL', 'egl')

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
