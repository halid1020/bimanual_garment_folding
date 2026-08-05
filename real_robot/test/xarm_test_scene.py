"""Camera-free test scenes for the dual xArm Lite 6 cell, plus a reachability gate.

Lets the SHIPPED primitives (``XArmSingleArmPickAndPlaceSkill``,
``XArmPickAndPlaceSkill``, ``XArmPickAndFlingSkill``) run on the real arms before
the RealSense is mounted and before ``hand_to_eye_calib.py`` has been run.

Two problems this solves:

1. The primitives take PIXEL actions and map them to base coordinates through
   ``point_on_table_base(u, v, intr, T_cam, table_z)``. The real hand-eye files
   (``calibration/xarm-{left,right}-calib.yaml``) are still identity placeholders,
   so that mapping currently sends the arms to arbitrary points. Here we fabricate
   a SYNTHETIC top-down camera from tape-measure geometry, giving an exact,
   invertible pixel <-> table-metres mapping. The primitives' own pixel code path
   is therefore exercised unchanged, but the numbers mean something.

2. At the measured 1.20 m base separation, a lot of the table is out of reach
   (see ``XARM_BASE_SEPARATION`` in xarm_constants.py). ``IKGuardedArm`` runs every
   commanded waypoint through the controller's own inverse kinematics BEFORE
   sending it, and refuses to move if the controller cannot solve it.

Frames
------
"Table coordinates" here ARE the LEFT arm's base frame: x runs from 0 at the left
base along the base-to-base axis (the table's SHORT, 80 cm direction) to
``separation`` at the right base; y runs along the table's LONG, 120 cm direction;
z is up with the table at ``table_z``. The arms are mounted face-to-face, so the
right base is rotated 180 deg about z, and ``T_left_right`` carries that yaw (the
placeholder calibration files do not).
"""
import os

import numpy as np
import yaml

from real_robot.robot.xarm_lite6 import XArmLite6
from real_robot.robot.xarm_dual_arm_scene import XArmDualArmScene
from real_robot.robot.xarm_single_arm_scene import XArmSingleArmScene
from real_robot.utils import xarm_constants as C
from real_robot.utils.term import green, red, yellow, mark
from real_robot.utils.xarm_camera import base_to_pixel as _base_to_pixel
from real_robot.utils.xarm_walls import walls_for_side, check_pose, describe


# --- Synthetic top-down camera -------------------------------------------------
# THE VIEW. The frame is laid out the way someone standing at the front of the cell
# would draw it: the LEFT arm on the left, the RIGHT arm on the right, the FRONT
# edge at the top. That means image u runs toward base +x and image v toward
# base -y (front is +y).
#
# Those two are ONE choice, not two. For a camera looking straight down the frame
# is right-handed with the view direction as camera +z, so u -> +x forces v -> -y;
# "right arm on the right AND front at the top" is only consistent because the
# front edge is at base +y (see the sign note on XARM_TABLE_RECT).
#
# Nothing in the primitives depends on this any more -- they assign arms by
# position on the TABLE (``sort_pairs_by_table_x``), so a camera mounted rolled is
# a view you dislike rather than a swapped grasp. It used to matter: the dual-arm
# skills sorted on pixel column, and a roll silently handed each arm the other
# one's target.
#
# At f = 500 px and 1.0 m height that is 2.0 mm/px, covering 2.56 m along u (only
# 0.80 m is needed) and 1.44 m along v (>= the 1.20 m table length).
IMAGE_W, IMAGE_H = 1280, 720
FOCAL_PX = 500.0
CAM_HEIGHT = C.XARM_CAM_HEIGHT      # m above the table plane


def _cam_centre_y():
    """y of the camera's optical axis in the LEFT base frame.

    The camera looks straight down at the midpoint of the two arm BASES, i.e. the
    arm line, y = 0. That is not the table's centre: the arm line sits 0.52 m from
    the front edge (+y) and 0.68 m from the back, so the table midline is y = -0.08.

    ``synthetic_T_left_cam`` and ``table_xy_to_pixel`` are inverses of each other
    and BOTH reference this, so they cannot drift apart -- moving the camera by
    editing one of them silently breaks the pixel mapping.
    """
    return C.XARM_CAM_CENTRE_Y

# Lite 6 published reach, used only by the OFFLINE geometric fallback check.
# On hardware the controller's own IK is authoritative.
XARM_MAX_REACH = 0.44   # m

CELL_YAML = "{}/real_robot/calibration/xarm-cell.yaml".format(
    os.environ.get('MP_FOLD_PATH', '.'))


class SyntheticIntrinsic:
    """Duck-types the pyrealsense2 intrinsics object that the production scenes
    put in ``scene.intr``: ``intrinsic_to_params`` reads ``.fx/.fy/.ppx/.ppy``.
    """

    def __init__(self, fx, fy, ppx, ppy, width, height):
        self.fx, self.fy = fx, fy
        self.ppx, self.ppy = ppx, ppy
        self.width, self.height = width, height

    def __repr__(self):
        return ("SyntheticIntrinsic(fx={:.1f}, fy={:.1f}, ppx={:.1f}, ppy={:.1f}, "
                "{}x{})".format(self.fx, self.fy, self.ppx, self.ppy,
                                self.width, self.height))


def synthetic_intrinsic():
    """Pinhole intrinsic for the fabricated top-down camera."""
    return SyntheticIntrinsic(FOCAL_PX, FOCAL_PX, IMAGE_W / 2.0, IMAGE_H / 2.0,
                              IMAGE_W, IMAGE_H)


def synthetic_T_left_cam(separation, table_z):
    """Camera-to-LEFT-base transform: the camera hovers over the midpoint of the
    two arm bases, looking straight down.

    Columns of R are the camera's axes expressed in the base frame:
    camera +x (image u) -> base +x  (toward the RIGHT arm),
    camera +y (image v) -> base -y  (toward the BACK; front is +y, so it is at the
                                     top of the frame),
    camera +z (view direction) -> base -z.
    det(R) = +1 -- it is a ROTATION. Writing it as a reflection (e.g. flipping only
    one axis) would still project the corners where you expect and then quietly
    break every handedness-dependent quantity downstream.
    """
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0]])
    T[:3, 3] = [separation / 2.0, _cam_centre_y(), table_z + CAM_HEIGHT]
    return T


def base_to_pixel(points, T_cam, intr=None):
    """Project 3D base-frame points to pixels -- the FULL pinhole projection.

    The implementation now lives in ``real_robot/utils/xarm_camera.py`` so the
    production scenes can use it without importing this test module; this wrapper
    keeps the name and the "default to the synthetic camera" convenience that the
    test tooling was written against.
    """
    return _base_to_pixel(points, T_cam,
                          synthetic_intrinsic() if intr is None else intr)


def face_to_face_T_left_right(separation):
    """Right base expressed in the LEFT base frame: translated along +x by
    ``separation`` and yawed 180 deg (the arms face each other)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    T[0, 3] = separation
    return T


def table_xy_to_pixel(x, y, separation, table_z, intr=None):
    """Inverse of ``point_on_table_base`` for the synthetic camera.

    Takes a point on the table in LEFT-base metres and returns the (u, v) pixel a
    primitive should be given so that it derives that same point back.

    Pass ``intr`` when the scene is CROPPED (``scene.intr``): the answer must be a
    pixel in the same frame the scene will hand the primitive, and a crop moves the
    principal point. With no ``intr`` this is the full-frame synthetic camera, as
    before -- for that case it is the closed form
    ``u = W/2 + f(x - S/2)/h``, ``v = H/2 - f(y - y_cam)/h``, which is what
    projecting through ``synthetic_T_left_cam`` reduces to.
    """
    return base_to_pixel([x, y, table_z], synthetic_T_left_cam(separation, table_z),
                         intr)


# --- Cell calibration ----------------------------------------------------------
SIDES = ('left', 'right')


def _valid_radius(value, where):
    """A usable-reach annulus with no interior is not a measurement.

    An earlier version of the reach sweep applied its safety margins after checking
    the raw span, and wrote ``(0.120, 0.105)``. Loading that silently makes every
    pose in the cell unreachable, which looks like a geometry bug anywhere but here.
    """
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        print(yellow("[xarm-test] !! ignoring malformed workspace_radius {!r} in {}".format(
            value, where)))
        return None
    if not (0.0 <= lo < hi):
        print(yellow("[xarm-test] !! ignoring workspace_radius ({:.3f}, {:.3f}) in {}: the "
                     "annulus has no interior.".format(lo, hi, where)))
        print("[xarm-test]    Re-run test_xarm_teach.py --reach for that arm; falling back "
              "to xarm_constants.py.")
        return None
    return (lo, hi)


class CellCalibration:
    """Values measured by ``test_xarm_teach.py`` and stored in xarm-cell.yaml.

    Falls back to the (unverified) defaults in ``xarm_constants.py`` so the module
    is usable offline, but ``calibrated`` then reports False.

    ``table_z`` / ``gripper_offset`` / ``workspace_radius`` / ``home_joint`` are
    PER ARM, and are accessed as ``cal.gripper_offset(side)``. They used to live in
    the shared ``cell:`` block, where teaching the second arm silently overwrote
    the first arm's measurement -- which is exactly how a right-arm run wiped a good
    left-arm gripper offset before it could be pasted. Legacy top-level values are
    still read, as a default for both sides, so an old file keeps working.
    """

    def __init__(self, data=None):
        data = data or {}
        cell = data.get('cell', {})
        arms = data.get('arms', {}) or {}
        self.calibrated = bool(data)
        self.separation = float(cell.get('base_separation', C.XARM_BASE_SEPARATION))
        self.base_yaw = float(cell.get('base_yaw', C.XARM_BASE_YAW))
        # The FULL measured transform when hand-eye has provided one, because
        # separation + yaw alone throw away the parts that are not zero: the two
        # calibrations put the right base 26 mm off the x axis and 9 mm higher
        # than the left. Rebuilding it from two scalars would quietly square the
        # cell up again, which is the assumption this is here to replace.
        T = cell.get('T_left_right')
        if T is not None:
            self.T_left_right = np.array(T, dtype=float).reshape(4, 4)
        else:
            self.T_left_right = face_to_face_T_left_right(self.separation)
            if abs(self.base_yaw - np.pi) > 1e-9:
                rot = np.eye(4)
                c, s = np.cos(self.base_yaw), np.sin(self.base_yaw)
                rot[:2, :2] = [[c, -s], [s, c]]
                self.T_left_right[:3, :3] = rot[:3, :3]

        self._table_z, self._gripper_offset = {}, {}
        self._workspace_radius, self._home_joint = {}, {}
        for side in SIDES:
            a = arms.get(side, {}) or {}
            self._table_z[side] = float(a.get(
                'table_z', cell.get('table_z', C.for_side(C.XARM_TABLE_Z_BY_SIDE, side))))
            self._gripper_offset[side] = float(a.get(
                'gripper_offset', cell.get('gripper_offset',
                                           C.for_side(C.XARM_GRIPPER_OFFSET_BY_SIDE, side))))
            radius = a.get('workspace_radius', cell.get('workspace_radius'))
            radius = _valid_radius(radius, "arms.{}".format(side)) if radius is not None else None
            self._workspace_radius[side] = tuple(
                radius if radius is not None
                else C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, side))
            self._home_joint[side] = list(a.get(
                'home_joint', C.for_side(C.XARM_HOME_JOINT_BY_SIDE, side)))

    @staticmethod
    def _side(side):
        return side if side in SIDES else 'left'

    def table_z(self, side='left'):
        return self._table_z[self._side(side)]

    def gripper_offset(self, side='left'):
        return self._gripper_offset[self._side(side)]

    def workspace_radius(self, side='left'):
        return self._workspace_radius[self._side(side)]

    def home_joint(self, side='left'):
        return list(self._home_joint[self._side(side)])

    def constants_mismatch(self, side='left'):
        """Which xarm_constants.py values disagree with the measured cell?

        The primitives do ``from ...xarm_constants import XARM_TABLE_Z``, binding
        the value at import time, so a YAML measurement only takes effect once it
        is pasted into xarm_constants.py. Report the difference rather than
        monkey-patching a shipped module behind the user's back.
        """
        side = self._side(side)
        out = []
        checks = [('XARM_TABLE_Z', C.for_side(C.XARM_TABLE_Z_BY_SIDE, side),
                   self.table_z(side), 1e-4),
                  ('XARM_GRIPPER_OFFSET', C.for_side(C.XARM_GRIPPER_OFFSET_BY_SIDE, side),
                   self.gripper_offset(side), 1e-4),
                  ('XARM_BASE_SEPARATION', C.XARM_BASE_SEPARATION, self.separation, 5e-3),
                  ('XARM_BASE_YAW', C.XARM_BASE_YAW, self.base_yaw, np.deg2rad(1.0))]
        for name, have, want, tol in checks:
            if abs(float(have) - float(want)) > tol:
                out.append((name, float(have), float(want)))
        return out

    def arms_disagree(self, tol=0.01):
        """Per-arm values the two arms disagree about by more than ``tol``.

        Both arms sit at the same table, so a measured gripper offset or table
        height that differs by more than a centimetre is a symptom, not data --
        typically a tcp_offset or world_offset set on one controller and not the
        other. Returns ``[(name, left, right), ...]``.
        """
        out = []
        for name, getter in (('gripper_offset', self.gripper_offset),
                             ('table_z', self.table_z)):
            lo, hi = float(getter('left')), float(getter('right'))
            if abs(lo - hi) > tol:
                out.append((name, lo, hi))
        return out


def load_cell(path=CELL_YAML):
    if not os.path.exists(path):
        return CellCalibration(None)
    with open(path, 'r') as f:
        return CellCalibration(yaml.safe_load(f) or {})


def save_cell(data, path=CELL_YAML):
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return path


# --- Reachability gate ---------------------------------------------------------
class Unreachable(RuntimeError):
    pass


class IKGuardedArm:
    """Wraps ``XArmLite6`` and checks every Cartesian waypoint before sending it.

    On hardware the check is the controller's own ``get_inverse_kinematics`` (plus
    ``is_tcp_limit``); offline it degrades to a geometric annulus + reach-sphere
    test. Any attribute this class does not define is delegated to the driver, so
    it is a drop-in wherever an ``XArmLite6`` is expected.

    ``execute=False`` reports on every waypoint but commands nothing, which is how
    a whole primitive can be checked end-to-end without motion.
    """

    def __init__(self, driver, name, radius, execute=True, strict=True, verbose=True,
                 bounds=None):
        self.driver = driver
        self.name = name
        self.radius = tuple(radius)
        self.execute = execute
        self.strict = strict          # raise on unreachable rather than warn
        self.verbose = verbose
        # Virtual walls, in this arm's base frame. Reachability and containment
        # are different questions: a pose can be perfectly reachable and still be
        # off the table, so both are reported.
        self.bounds = bounds if bounds is not None else walls_for_side(
            name if name in ('left', 'right') else 'left')
        self.checked = []             # [(pose, ok, reason)] for the summary table
        # Where the arm WOULD be if every checked waypoint had executed. Primitives
        # that read their own pose back mid-sequence -- the fling's stretch, shake,
        # release and its re-read before the swing -- need an answer even when
        # nothing is moving, or the whole tail of the sequence goes unchecked.
        self._sim_pose = None

    def __getattr__(self, item):
        # Only reached for attributes not defined on this class.
        if self.driver is None:
            raise AttributeError(
                "{} has no driver (offline mode); '{}' is unavailable".format(self.name, item))
        return getattr(self.driver, item)

    # Effort feedback degrades to "no signal" without a driver, which is exactly
    # how the primitives are required to behave when the arm cannot report it.
    def effort_baseline(self, *a, **kw):
        return None if self.driver is None else self.driver.effort_baseline(*a, **kw)

    def effort_delta(self, *a, **kw):
        return None if self.driver is None else self.driver.effort_delta(*a, **kw)

    def effort_exceeded(self, *a, **kw):
        return False if self.driver is None else self.driver.effort_exceeded(*a, **kw)

    # -- checking ---------------------------------------------------------
    def _check_pose(self, pose):
        """-> (ok, reason). ``pose`` is a UR-convention pose in metres + rotvec."""
        pose = np.asarray(pose, dtype=float)
        r_xy = float(np.linalg.norm(pose[:2]))
        r_3d = float(np.linalg.norm(pose[:3]))

        # Walls first: the driver refuses these before it even reaches the
        # controller, so an unreachable-and-out-of-bounds pose should be reported
        # by the reason that would actually stop it.
        if self.bounds is not None:
            inside, violations = check_pose(pose, self.bounds)
            if not inside:
                return False, "outside the virtual walls -- " + "; ".join(violations)

        if pose[2] < C.XARM_MIN_Z:
            return False, "z={:.3f} below the {:.3f} m safety floor".format(pose[2], C.XARM_MIN_Z)
        if r_xy < self.radius[0]:
            return False, "XY radius {:.3f} m inside the {:.3f} m base keepout".format(
                r_xy, self.radius[0])

        if self.driver is None:
            # Offline: no controller to ask, so approximate.
            if r_3d > XARM_MAX_REACH:
                return False, "3D distance {:.3f} m exceeds the ~{:.2f} m reach (offline estimate)".format(
                    r_3d, XARM_MAX_REACH)
            if r_xy > self.radius[1]:
                return False, "XY radius {:.3f} m beyond the {:.3f} m usable reach".format(
                    r_xy, self.radius[1])
            return True, "ok (offline estimate, r_xy={:.3f} r_3d={:.3f})".format(r_xy, r_3d)

        xarm_pose = self.driver._pose_to_xarm(pose)
        code, _ = self.driver.arm.get_inverse_kinematics(xarm_pose, input_is_radian=True)
        if code != 0:
            return False, ("controller IK failed (code {}); r_xy={:.3f} r_3d={:.3f} m"
                           .format(code, r_xy, r_3d))

        # IK is the authoritative reachability test: if the controller solved the
        # pose, the arm can be commanded there. is_tcp_limit is reported alongside
        # it but does NOT veto -- it once failed all six waypoints of a pick-and-place
        # whose IK had already succeeded, which blocks the dry run without saying
        # what limit it is enforcing.
        note = ""
        try:
            code_l, limited = self.driver.arm.is_tcp_limit(xarm_pose, is_radian=True)
        except Exception:
            code_l, limited = -1, None
        if code_l == 0 and limited:
            note = "; NOTE is_tcp_limit says out-of-limit despite IK solving it"
            boundary = self._live_boundary_mm()
            if boundary is not None:
                note += " (controller boundary {} mm)".format(boundary)
        return True, "ok (controller IK, r_xy={:.3f} r_3d={:.3f}){}".format(r_xy, r_3d, note)

    def _live_boundary_mm(self):
        """The controller's currently programmed safety boundary, for diagnostics."""
        try:
            _, boundary, _ = self.driver._live_fence()
        except Exception:
            return None
        return boundary or None

    def check(self, poses, label=""):
        """Check one pose or a list of poses; returns True if all are reachable."""
        poses = np.asarray(poses, dtype=float)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        all_ok = True
        for i, p in enumerate(poses):
            ok, reason = self._check_pose(p)
            tag = "{}[{}]".format(label or "waypoint", i)
            self.checked.append((self.name, tag, np.array(p), ok, reason))
            if self.verbose:
                print("{}{:<10s} {:<8s} xyz=({:+.3f}, {:+.3f}, {:+.3f})  {}".format(
                    mark(ok), self.name, tag, p[0], p[1], p[2],
                    reason if ok else red(reason)))
            all_ok = all_ok and ok
        if not all_ok and self.strict:
            raise Unreachable(
                "[{}] {} contains a pose the arm cannot reach (see above).".format(
                    self.name, label or "trajectory"))
        return all_ok

    # -- intercepted motion ----------------------------------------------
    def _advance_sim(self, p):
        """Remember the last waypoint as where the arm would now be."""
        p = np.asarray(p, dtype=float)
        last = p if p.ndim == 1 else p[-1]
        if len(last) >= 6:
            self._sim_pose = np.array(last[:6], dtype=float)

    def movel(self, p, speed=1.5, acceleration=1.0, blocking=True,
              avoid_singularity=False, **kwargs):
        self.check(p, label="movel")
        self._advance_sim(p)
        if not self.execute:
            return True
        return self.driver.movel(p, speed=speed, acceleration=acceleration,
                                 blocking=blocking, avoid_singularity=avoid_singularity,
                                 **kwargs)

    def movej_ik(self, p, speed=1.5, acceleration=1.0, blocking=True):
        return self.movel(p, speed=speed, acceleration=acceleration, blocking=blocking)

    def movej(self, q, speed=None, acceleration=None, blocking=True):
        if self.verbose:
            print("  -> {} movej {}".format(
                self.name, np.round(np.degrees(np.asarray(q, dtype=float)), 1)))
        if not self.execute:
            return True
        return self.driver.movej(q, speed=speed, acceleration=acceleration, blocking=blocking)

    def home(self, speed=None, acceleration=None, blocking=True):
        if self.verbose:
            print("  -> {} home".format(self.name))
        # Joint move: we cannot cheaply say where the TCP lands, so drop the
        # simulated pose and let it be re-derived on the next readback.
        self._sim_pose = None
        if not self.execute:
            return True
        return self.driver.home(speed=speed, acceleration=acceleration, blocking=blocking)

    def out_scene(self, speed=None, acceleration=None, blocking=True):
        if self.verbose:
            print("  -> {} out_scene".format(self.name))
        if not self.execute:
            return True
        return self.driver.out_scene(speed=speed, acceleration=acceleration, blocking=blocking)

    def open_gripper(self, *a, **kw):
        if self.verbose:
            print("  -> {} open_gripper".format(self.name))
        if not self.execute:
            return True
        return self.driver.open_gripper(*a, **kw)

    def close_gripper(self, *a, **kw):
        if self.verbose:
            print("  -> {} close_gripper".format(self.name))
        if not self.execute:
            return True
        return self.driver.close_gripper(*a, **kw)

    def get_tcp_pose(self):
        """The real pose when executing; the SIMULATED one otherwise.

        A --dry-run is connected but stationary, so the controller reports the
        parked pose rather than where the primitive believes the arm is. Handing
        that back would send every waypoint of a pose-relative stage -- the fling's
        stretch, shake, release, and its re-read before the swing -- to the wrong
        place, and check nothing useful. ``_advance_sim`` tracks the last commanded
        waypoint instead, so those stages are walked and validated.
        """
        if self.execute and self.driver is not None:
            return self.driver.get_tcp_pose()
        if self._sim_pose is None:
            if self.driver is not None:
                self._sim_pose = np.asarray(self.driver.get_tcp_pose(), dtype=float)
            else:
                # Offline with no controller: start from a plausible pose inside the
                # annulus. Waypoints derived from it are still checked, so a bad one
                # is still caught.
                self._sim_pose = np.array(
                    [self.radius[0] + 0.1, 0.0, 0.2, np.pi, 0.0, 0.0], dtype=float)
        return np.array(self._sim_pose, dtype=float)

    def disconnect(self):
        if self.driver is not None:
            self.driver.disconnect()


def _connect(ip, name, cell, gripper, radius, execute, offline):
    # strict=execute: when about to move, the first unreachable pose aborts before
    # anything is commanded; when only checking, keep going so the whole primitive
    # is reported rather than just its first failure.
    bounds = walls_for_side(name, separation=cell.separation)
    if offline:
        print("[xarm-test] {}: OFFLINE (no controller; geometric reach estimate only)".format(name))
        return IKGuardedArm(None, name, radius, execute=False, strict=False, bounds=bounds)
    driver = XArmLite6(ip, gripper=gripper, home_joint=cell.home_joint(name),
                       side=name, separation=cell.separation)
    return IKGuardedArm(driver, name, radius, execute=execute, strict=execute,
                        bounds=bounds)


# --- Scenes --------------------------------------------------------------------
class XArmTestDualScene(XArmDualArmScene):
    """``XArmDualArmScene`` with the camera and hand-eye calibration replaced by
    measured geometry. Only ``__init__`` is overridden -- every motion method
    (``both_movel``, ``both_home``, ``both_fling``, ``both_*_gripper``) is the
    inherited production implementation, so that is what gets tested.
    """

    def __init__(self, left_ip, right_ip, cell, gripper='lite6',
                 execute=True, offline=False):
        self.cell = cell
        self.dry_run = False        # motion gating lives in IKGuardedArm, not here
        self.gripper_type = gripper
        self.last_trajectory = None
        self.camera = None

        # The synthetic camera and the base-to-base transform live in the LEFT base
        # frame, so the table plane they use is the left arm's table_z.
        S, table_z = cell.separation, cell.table_z('left')
        self.left_radius = cell.workspace_radius('left')
        self.right_radius = cell.workspace_radius('right')

        self.left = _connect(left_ip, 'left', cell, gripper,
                             self.left_radius, execute, offline)
        self.right = _connect(right_ip, 'right', cell, gripper,
                              self.right_radius, execute, offline)

        self.intr = synthetic_intrinsic()
        self.T_left_cam = synthetic_T_left_cam(S, table_z)
        self.T_left_right = face_to_face_T_left_right(S)
        # Derived, not hand-written, so the two cameras cannot disagree.
        self.T_right_cam = np.linalg.inv(self.T_left_right) @ self.T_left_cam

        self.T_left_world = np.eye(4)
        self.T_left_world[:3, 3] = self.T_left_right[:3, 3] / 2.0
        self.T_left_world[2, 3] = table_z
        self.T_right_world = np.linalg.inv(self.T_left_right) @ self.T_left_world
        self.T_world_cam = np.linalg.inv(self.T_left_world) @ self.T_left_cam
        self.T_cam_world = np.linalg.inv(self.T_world_cam)

        # Reuse the parent's mask maths; it only reads rgb.shape.
        self._calculate_wrkspace_masks(np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8))

        # UR-named aliases, as in the production scene.
        self.T_ur5e_cam = self.T_left_cam
        self.T_ur16e_cam = self.T_right_cam
        self.T_ur5e_ur16e = self.T_left_right

        print("[XArmTestDualScene] separation {:.3f} m, table_z {:.3f} m, "
              "synthetic camera {:.2f} mm/px.".format(S, table_z, 1000.0 / FOCAL_PX))

    def take_rgbd(self):
        raise NotImplementedError("The test scene has no camera.")

    def restart_camera(self):
        raise NotImplementedError("The test scene has no camera.")

    def checked_poses(self):
        return self.left.checked + self.right.checked


class XArmTestSingleScene(XArmSingleArmScene):
    """``XArmSingleArmScene`` with measured geometry instead of camera calibration."""

    def __init__(self, ip, cell, side='left', gripper='lite6',
                 execute=True, offline=False):
        self.cell = cell
        self.side = side
        self.dry_run = False
        self.gripper_type = gripper
        self.camera = None
        self.radius = cell.workspace_radius(side)

        # The synthetic camera is built in the LEFT base frame and then transformed,
        # so its table plane is the left arm's table_z regardless of which arm this is.
        S, table_z = cell.separation, cell.table_z('left')
        self.arm = _connect(ip, side, cell, gripper, self.radius, execute, offline)

        self.intr = synthetic_intrinsic()
        T_left_cam = synthetic_T_left_cam(S, table_z)
        if side == 'left':
            self.T_cam = T_left_cam
        else:
            self.T_cam = np.linalg.inv(face_to_face_T_left_right(S)) @ T_left_cam
        self.T_world = np.eye(4)

        self._calculate_wrkspace_masks(np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8))
        print("[XArmTestSingleScene] arm '{}', table_z {:.3f} m.".format(side, table_z))

    def take_rgbd(self):
        raise NotImplementedError("The test scene has no camera.")

    def restart_camera(self):
        raise NotImplementedError("The test scene has no camera.")

    def checked_poses(self):
        return self.arm.checked


# --- Self-check ----------------------------------------------------------------
def selfcheck_camera(separation=C.XARM_BASE_SEPARATION, table_z=C.XARM_TABLE_Z, verbose=True):
    """The synthetic camera must round-trip: table metres -> pixel -> (through the
    primitives' own ``point_on_table_base``) -> the same table metres. Also checks
    that the left and right cameras agree through ``T_left_right``.
    """
    from real_robot.utils.transform_utils import point_on_table_base, transform_point

    intr = synthetic_intrinsic()
    T_left_cam = synthetic_T_left_cam(separation, table_z)
    T_left_right = face_to_face_T_left_right(separation)
    T_right_cam = np.linalg.inv(T_left_right) @ T_left_cam

    worst_rt = worst_agree = 0.0
    for x in np.linspace(*C.XARM_TABLE_RECT['x'], num=7):
        for y in np.linspace(*C.XARM_TABLE_RECT['y'], num=7):
            px = table_xy_to_pixel(x, y, separation, table_z)
            back_l = point_on_table_base(px[0], px[1], intr, T_left_cam, table_z)
            worst_rt = max(worst_rt, float(np.linalg.norm(back_l - np.array([x, y, table_z]))))
            back_r = point_on_table_base(px[0], px[1], intr, T_right_cam, table_z)
            worst_agree = max(worst_agree, float(np.linalg.norm(
                transform_point(T_left_right, back_r) - back_l)))
    if verbose:
        print("[xarm-test] synthetic camera round-trip error : {:.3e} m".format(worst_rt))
        print("[xarm-test] left/right camera agreement error : {:.3e} m".format(worst_agree))
    assert worst_rt < 1e-9, "synthetic camera does not round-trip"
    assert worst_agree < 1e-9, "left and right synthetic cameras disagree"
    return worst_rt, worst_agree


def print_reach_map(cell):
    """Print where on the table each arm can actually work."""
    lo_l, hi_l = cell.workspace_radius('left')
    lo_r, hi_r = cell.workspace_radius('right')
    S = cell.separation
    # Both spans in the LEFT base frame: the right arm reaches x in [S-hi_r, S-lo_r].
    r_near, r_far = S - hi_r, S - lo_r
    print("\nReach along the base-to-base axis (x measured from the LEFT base):")
    print("  left arm  : {:.2f} - {:.2f} m".format(lo_l, hi_l))
    ov_lo, ov_hi = max(lo_l, r_near), min(hi_l, r_far)
    if ov_hi - ov_lo > 1e-6:
        print("  overlap   : {:.2f} - {:.2f} m  ({:.2f} m deep, both arms can reach)".format(
            ov_lo, ov_hi, ov_hi - ov_lo))
    elif r_near > hi_l:
        print("  DEAD BAND : {:.2f} - {:.2f} m  ({:.2f} m neither arm can reach)".format(
            hi_l, r_near, r_near - hi_l))
    else:
        print("  overlap   : none -- the reach envelopes meet at a single point.")
        print("              A larger measured usable radius would open a lens here;"
              " run --reach.")
    print("  right arm : {:.2f} - {:.2f} m".format(r_near, r_far))
    if cell.workspace_radius('left') != cell.workspace_radius('right'):
        print("              (the two arms measured DIFFERENT usable radii)")
    print("  table     : x [{:+.2f}, {:+.2f}]  y [{:+.2f}, {:+.2f}] m  (front is +y)".format(
        C.XARM_TABLE_RECT['x'][0], C.XARM_TABLE_RECT['x'][1],
        C.XARM_TABLE_RECT['y'][0], C.XARM_TABLE_RECT['y'][1]))
    print("  walls     : {}   (table frame)".format(describe(C.XARM_WALLS)))


def selfcheck_calibration(verbose=True):
    """Per-arm calibration must not let one arm read the other's numbers.

    No hardware and no files: builds CellCalibration from dicts directly.
    """
    # A legacy file kept these cell-wide. It must still load, as a default for both.
    legacy = CellCalibration({'cell': {'table_z': 0.01, 'gripper_offset': 0.087,
                                       'workspace_radius': [0.12, 0.41]}})
    for side in SIDES:
        assert abs(legacy.gripper_offset(side) - 0.087) < 1e-9, side
        assert abs(legacy.table_z(side) - 0.01) < 1e-9, side
        assert legacy.workspace_radius(side) == (0.12, 0.41), side
    assert legacy.arms_disagree() == [], "a legacy file applies to both arms equally"

    # Per-arm values must stay separate -- this is the bug that wiped a good left-arm
    # measurement when the right arm was taught afterwards.
    per_arm = CellCalibration({'arms': {
        'left': {'gripper_offset': 0.087, 'table_z': 0.0,
                 'workspace_radius': [0.12, 0.41], 'home_joint': [0.1] * 6},
        'right': {'gripper_offset': 0.085, 'table_z': 0.0,
                  'workspace_radius': [0.13, 0.40], 'home_joint': [0.2] * 6},
    }})
    assert abs(per_arm.gripper_offset('left') - 0.087) < 1e-9
    assert abs(per_arm.gripper_offset('right') - 0.085) < 1e-9
    assert per_arm.workspace_radius('left') != per_arm.workspace_radius('right')
    assert per_arm.home_joint('left') != per_arm.home_joint('right')
    assert per_arm.arms_disagree() == [], "2 mm apart is agreement"

    # The symptom that started this: a phantom 374 mm tool on one controller.
    skewed = CellCalibration({'arms': {'left': {'gripper_offset': 0.0869},
                                       'right': {'gripper_offset': -0.1530}}})
    bad = skewed.arms_disagree()
    assert [n for n, _, _ in bad] == ['gripper_offset'], bad

    # An annulus with no interior must be rejected, not handed out: loading it makes
    # every pose in the cell look unreachable.
    inverted = CellCalibration({'cell': {'workspace_radius': [0.120, 0.105]}})
    for side in SIDES:
        lo, hi = inverted.workspace_radius(side)
        assert lo < hi, "inverted radius was accepted for {}: {}".format(side, (lo, hi))

    if verbose:
        print(green("[xarm-test] calibration self-check: per-arm isolation, legacy fallback, "
                    "disagreement detection and range validation all OK."))
    return True


if __name__ == '__main__':
    selfcheck_camera()
    selfcheck_calibration()
    print_reach_map(load_cell())
