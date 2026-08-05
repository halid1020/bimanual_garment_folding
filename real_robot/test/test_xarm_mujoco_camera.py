"""Offline checks for the MuJoCo cell's CAMERA. NO HARDWARE NEEDED.

Everything a primitive does starts by turning a pixel into a place on the table,
so the one thing the simulator has to get right is that its camera and the camera
``base_to_pixel`` models are the same camera. These checks assert that against a
real render, plus the two framing properties the cell is built around.

They are worth having because both of the conversions involved fail QUIETLY:

  * get the OpenCV -> MuJoCo camera rotation wrong and the camera looks at the
    ceiling. The render comes back uniformly black, which reads like a broken GL
    context rather than a sign error, and cost an hour the first time.
  * get the principal-point sign wrong and nothing at all happens until someone
    uses a camera whose principal point is not exactly centred -- which is every
    real one. The D435i NOMINAL we fall back to has ppx/ppy exactly at W/2, H/2, so
    against that intrinsic the sign is multiplying zero and a mutation test walks
    straight past it. t_projection_round_trip therefore renders TWICE, the second
    time through a deliberately off-centre principal point. Off-centre MARKERS do
    not help here and it is easy to think they do -- what has to be off-centre is
    the principal point itself.

Usage
    python real_robot/test/test_xarm_mujoco_camera.py
"""
import os
import sys

import numpy as np

from real_robot.utils import xarm_constants as C
from real_robot.utils.term import green, red, yellow

# The renderer needs a GL context and these run headless in CI and over ssh.
os.environ.setdefault('MUJOCO_GL', 'egl')

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ----------------------------------------------------------------------
def _calib_dir():
    return "{}/real_robot/calibration".format(os.environ.get('MP_FOLD_PATH', '.'))


def _calibration():
    from real_robot.utils.scene_utils import (
        cell_geometry_from_calibration, load_camera_to_base,
    )
    from real_robot.utils.xarm_camera import load_intrinsic
    d = _calib_dir()
    left = os.path.join(d, 'xarm-left-calib.yaml')
    right = os.path.join(d, 'xarm-right-calib.yaml')
    T_left_right, separation, yaw = cell_geometry_from_calibration(left, right)
    return {'T_left_right': T_left_right, 'separation': separation, 'yaw': yaw,
            'T_left_cam': load_camera_to_base(left),
            'intr': load_intrinsic(left, verbose=False)}


# ----------------------------------------------------------------------
MARKS = [(0.20, -0.15), (0.55, 0.18), (0.37, 0.25),
         (0.10, 0.05), (0.37, 0.00), (0.65, -0.20)]


def _round_trip_error(T_cam, intr):
    """Worst |rendered - base_to_pixel| over MARKS, in pixels.

    The markers are deliberately spread to the corners of the reachable area: a
    camera model that is wrong in focal length or principal point can still be
    right at the image centre, so a centred marker proves almost nothing.
    """
    import mujoco
    import real_robot.sim.xarm_mjcf as M
    from real_robot.utils.xarm_camera import base_to_pixel

    cal = _calibration()
    extra = "".join(
        '<geom name="m{}" type="cylinder" size="0.004 0.0005" pos="{} {} 0.003" '
        'rgba="1 0 0 1"/>'.format(i, x, y) for i, (x, y) in enumerate(MARKS))

    orig = M._table_xml
    M._table_xml = lambda tz, thickness=0.02: orig(tz, thickness) + extra
    try:
        model, info = M.build_cell(cal['T_left_right'], T_cam, intr, table_z=0.0,
                                   garment_pose=(cal['separation'] / 2, -0.45, 0.0))
    finally:
        M._table_xml = orig

    data = mujoco.MjData(model)
    # Park the arms clear of the table so nothing occludes a marker.
    for side in ('left', 'right'):
        adr = [model.jnt_qposadr[j] for j in info['joint_qpos'][side]]
        data.qpos[adr] = [0.0, -1.4, 0.2, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    W, H = info['image_size']
    r = mujoco.Renderer(model, H, W)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=info['camera_name'])
        seg = r.render()
    finally:
        r.close()

    worst = 0.0
    for i, (x, y) in enumerate(MARKS):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'm{}'.format(i))
        ys, xs = np.where((seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                          & (seg[:, :, 0] == gid))
        assert len(xs) > 20, (
            "marker at ({:+.2f}, {:+.2f}) rendered {} pixels -- it is occluded, off "
            "frame, or the camera is not looking at the table".format(x, y, len(xs)))
        # A tight blob. Scattered pixels mean the segmentation ids are being blended
        # by multisampling -- see offsamples="0" in xarm_mjcf.
        spread = max(xs.max() - xs.min(), ys.max() - ys.min())
        assert spread < 25, (
            "marker at ({:+.2f}, {:+.2f}) is smeared over {} px: segmentation ids "
            "are being antialiased. Check <quality offsamples=\"0\">".format(
                x, y, spread))
        rendered = np.array([xs.mean(), ys.mean()])
        predicted = base_to_pixel([x, y, 0.003], T_cam, intr)
        worst = max(worst, float(np.linalg.norm(predicted - rendered)))
    return worst


@check("the MuJoCo camera and base_to_pixel are the same camera")
def t_projection_round_trip():
    """Render markers at known table coordinates; assert the projection lands on them.

    This is the check every pixel-to-metre claim downstream rests on.
    """
    from real_robot.utils.xarm_camera import PinholeIntrinsic

    cal = _calibration()
    worst = _round_trip_error(cal['T_left_cam'], cal['intr'])
    assert worst < 2.0, (
        "the rendered camera and base_to_pixel disagree by up to {:.1f} px. Check "
        "the OpenCV->MuJoCo rotation (R @ diag(1,-1,-1)) in "
        "xarm_mjcf._camera_xml".format(worst))

    # Again with the principal point moved OFF CENTRE. Without this the
    # principalpixel sign is untested whenever the intrinsic in use is centred --
    # which the D435i nominal fallback is, exactly. Verified by mutation: flipping
    # the sign leaves the check above at 0.8 px and fails this one by ~120 px.
    intr = cal['intr']
    off = PinholeIntrinsic(intr.fx, intr.fy, intr.ppx + 60.0, intr.ppy - 40.0,
                           intr.width, intr.height, source='off-centre probe')
    worst_off = _round_trip_error(cal['T_left_cam'], off)
    assert worst_off < 2.0, (
        "with the principal point 60 px right and 40 px up, the render and "
        "base_to_pixel disagree by {:.1f} px. principalpixel is an OFFSET from the "
        "image centre with a FLIPPED y -- (ppx - W/2, -(ppy - H/2)) -- and one of "
        "those two is wrong in xarm_mjcf._camera_xml.".format(worst_off))


@check("the OpenCV -> MuJoCo camera conversion is a rotation, and is its own inverse")
def t_camera_frame_conversion():
    from real_robot.sim.xarm_mjcf import opencv_to_mujoco_rotation
    cal = _calibration()

    R_cv = cal['T_left_cam'][:3, :3]
    R_mj = opencv_to_mujoco_rotation(R_cv)
    assert abs(np.linalg.det(R_mj) - 1.0) < 1e-9, (
        "the converted camera frame has det {:+.6f} -- it is a REFLECTION, not a "
        "rotation".format(np.linalg.det(R_mj)))
    assert np.allclose(opencv_to_mujoco_rotation(R_mj), R_cv, atol=1e-12), \
        "the conversion is not an involution; one of the two directions is wrong"

    # Our camera hangs level over the table, so the MuJoCo frame should come out
    # near the identity. If this ever fails by ~180 deg, the render will be black.
    ang = np.degrees(np.arccos(np.clip((np.trace(R_mj) - 1) / 2, -1, 1)))
    assert ang < 5.0, (
        "the calibrated camera's MuJoCo frame is {:.1f} deg from the identity; a "
        "down-looking camera should be near 0. A ~180 deg result means the camera "
        "is pointed at the ceiling and every render will be black.".format(ang))


@check("the crop puts the two arm bases on its left and right edges")
def t_crop_bases_on_the_edges():
    """The framing the cell is built around, stated as a test rather than a constant.

    ``crop_window`` defaults its side to the base separation precisely so this
    holds at whatever separation the cell is measured to have.
    """
    from real_robot.utils.xarm_camera import base_to_pixel, crop_window
    cal = _calibration()
    S = cal['separation']
    win = crop_window(cal['intr'], cal['T_left_cam'], separation=S, table_z=0.0)
    if win.clamped:
        raise AssertionError(
            "the window was clamped, so the bases cannot be on its edges -- see "
            "the crop-fits check for what to do about it")
    intr = win.intrinsic(cal['intr'])
    left = base_to_pixel([0.0, 0.0, 0.0], cal['T_left_cam'], intr)
    right = base_to_pixel([S, 0.0, 0.0], cal['T_left_cam'], intr)
    assert abs(left[0]) < 2.0, \
        "the left base is at u={:+.1f}, not on the left edge".format(left[0])
    assert abs(right[0] - (win.side - 1)) < 2.0, \
        "the right base is at u={:+.1f}, not on the right edge (side {})".format(
            right[0], win.side)


@check("the base-to-base crop fits the sensor at the calibrated camera height")
def t_crop_fits():
    """Whether the camera can actually SEE the window the cell asks for.

    The camera height is fixed by hand-eye calibration and is not a knob, so when
    this fails the answer is a wider field of view, NOT a higher bracket.
    """
    from real_robot.utils.xarm_camera import crop_window
    cal = _calibration()
    S, intr = cal['separation'], cal['intr']
    height = float(cal['T_left_cam'][2, 3])
    win = crop_window(intr, cal['T_left_cam'], separation=S, table_z=0.0)
    need = S * min(abs(intr.fx), abs(intr.fy)) / height
    have = min(intr.width, intr.height)
    covered = have * height / min(abs(intr.fx), abs(intr.fy))
    assert not win.clamped, (
        "a {:.3f} m base-to-base window needs {:.0f} px but the frame is only "
        "{} px, so it covers {:.3f} m of table -- {:.0f} mm short. The camera sits "
        "at the CALIBRATED {:.3f} m and does not move; use a wider stream (the "
        "D435i depth FOV is 87x58 deg against the colour stream's 69x42) or accept "
        "a smaller crop with the bases outside it. Intrinsic in use: {}".format(
            S, need, have, covered, (S - covered) * 1000.0, height, intr))


@check("the calibrated frame has the left arm on the left and the front at the top")
def t_frame_layout():
    """The layout, checked against the REAL camera rather than a fabricated one.

    This is the check that would have settled the 2026-08-04 sign mistake on the
    spot. The front edge is at +y and the calibration's image v runs toward -y, so
    the front lands at the top -- and no software mirror is needed to get it there.
    """
    from real_robot.utils.xarm_camera import describe_orientation
    cal = _calibration()
    ok, msg = describe_orientation(cal['T_left_cam'], cal['intr'],
                                   cal['separation'], 0.0)
    assert ok, msg
    assert C.XARM_TABLE_RECT['y'][1] > 0, \
        "the front edge must be at +y for the mounted camera to show it at the top"


@check("the measured cell geometry is a rigid transform and roughly face-to-face")
def t_cell_geometry():
    cal = _calibration()
    T = cal['T_left_right']
    R = T[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, \
        "T_left_right has det {:+.6f}; it is not a rotation".format(np.linalg.det(R))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), "T_left_right is not orthonormal"
    yaw_deg = np.degrees(cal['yaw'])
    assert abs(abs(yaw_deg) - 180.0) < 5.0, (
        "the arms are {:.1f} deg apart in yaw, not ~180 -- they are supposed to "
        "face each other across the table".format(yaw_deg))
    assert 0.5 < cal['separation'] < 1.0, \
        "base separation {:.3f} m is not plausible for this cell".format(cal['separation'])


@check("the fling moves the cloth toward the front, in metres AND up the frame")
def t_fling_moves_cloth_forward():
    """The direct regression for the wrong-direction fling.

    Stated in BOTH coordinate systems on purpose. On 2026-08-04 the sign of base y
    was flipped on the strength of a fling that e-stopped during its wind-up, and
    the metre claim and the pixel claim quietly stopped meaning the same thing.
    Asserting them together is what stops that happening again: with the front edge
    at +y and image v running toward -y, "the cloth went forward" and "the cloth
    went up the frame" are the same sentence, and if they ever disagree one of the
    two frames is wrong.

    Runs the real XArmPickAndFlingSkill against the MuJoCo cell -- so it also
    covers the whole pixel -> table -> IK -> swing path, not just the constants.
    """
    from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
    from real_robot.sim.run_sim_ui import (
        auto_picks, snap_to_mask, table_px, valid_flag,
    )
    from real_robot.sim.xarm_mujoco_cell import XArmMujocoScene

    scene = XArmMujocoScene(gui=False, seed=0)
    try:
        before = scene.cloth_centroid()
        scene.go_camera_pos()
        _, _, mask = scene.take_rgbd()
        p0, p1 = (snap_to_mask(px, mask) for px in auto_picks(scene))
        skill = XArmPickAndFlingSkill(scene, {'speed': 0.2, 'acc': 0.5})
        skill.reset()
        skill.step(np.array([p0[0], p0[1], p1[0], p1[1], 0.0, 0.0,
                             valid_flag(mask, p0), valid_flag(mask, p1)]))
        after = scene.cloth_centroid()
        v_before = float(table_px(scene, before[0], before[1])[1])
        v_after = float(table_px(scene, after[0], after[1])[1])
    finally:
        scene.close()

    dy = float(after[1] - before[1])
    assert dy > 0.05, (
        "the fling moved the cloth {:+.3f} m in y. Front is +y, so it was thrown "
        "toward the BACK. Check XARM_FLING_FORWARD_Y and the waypoint order in "
        "xarm_base_fling_poses (wind up to -y, stroke to +y).".format(dy))
    assert v_after < v_before - 20.0, (
        "the cloth moved {:+.3f} m toward the front but only {:+.0f} px in v. The "
        "metres and the pixels disagree, so the camera model and the table frame "
        "have drifted apart -- see t_frame_layout.".format(dy, v_after - v_before))


# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print("MuJoCo cell camera -- offline checks (no hardware)")
    try:
        cal = _calibration()
        print("  separation {:.4f} m, yaw {:.2f} deg, camera {:.3f} m up".format(
            cal['separation'], np.degrees(cal['yaw']), cal['T_left_cam'][2, 3]))
        print("  intrinsic {}".format(cal['intr']))
    except Exception as exc:
        print(red("  cannot load the calibration: {}".format(exc)))
        return 1
    print("=" * 72)

    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print("  {}  {}".format(green("PASS"), name))
        except AssertionError as e:
            failures += 1
            print("  {}  {}\n          {}".format(red("FAIL"), name, red(str(e))))
        except Exception as e:
            failures += 1
            print("  {} {}\n          {}".format(red("ERROR"), name, red(repr(e))))

    print("=" * 72)
    passed = len(CHECKS) - failures
    summary = "{}/{} checks passed.".format(passed, len(CHECKS))
    print(green(summary) if not failures else red(summary))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
