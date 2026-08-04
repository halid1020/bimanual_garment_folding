"""PyBullet model of the dual xArm Lite 6 cell, with a top-down camera.

The point of this module is that the SHIPPED primitives run against it unmodified:
``XArmSimScene`` duck-types ``XArmDualArmScene`` (``both_movel``, ``both_fling``,
``both_home``, the grippers, ``get_tcp_distance``, ``intr``, ``T_left_cam``,
``T_left_right``, ``.left`` / ``.right``), so ``XArmPickAndFlingSkill`` and friends
drive simulated arms through exactly the code path they use on hardware. A bug in a
primitive shows up here rather than on the robot.

FRAMES. The PyBullet world IS the LEFT arm's base frame: the left base sits at the
origin, the right base at ``(separation, 0, 0)`` yawed 180 deg, and the table top is
the z = ``table_z`` plane -- which is also the arms' mounting plane, since the
measured ``table_z`` is 0 in each base frame.

WHAT IS REAL AND WHAT IS NOT
  * Kinematics: real. The URDF is UFACTORY's own (via hygradme/lite6_urdf) and its
    forward kinematics reproduce poses measured on the hardware to 2-4 mm.
  * The camera: real projection, built FROM ``synthetic_intrinsic()`` so the
    rendered image and ``point_on_table_base`` agree by construction.
  * The cloth: NOT real. It is a drawn catenary sheet hanging between the grippers,
    with no physics. It shows where cloth would be, not what it would do.
  * IK: PyBullet's damped least-squares solver, NOT the xArm controller's. Good for
    sequencing, framing and gross reachability; the controller's own
    ``get_inverse_kinematics`` stays the authority before any real motion.
  * Effort: not simulated. ``effort_*`` report "no signal", so the primitives take
    their documented fallback and the geometric caps drive the stretch.
"""
import math
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from real_robot.sim.fetch_lite6_assets import asset_dir, assets_present, urdf_path
from real_robot.test.xarm_test_scene import (
    IMAGE_H, IMAGE_W, base_to_pixel, face_to_face_T_left_right, synthetic_T_left_cam,
    synthetic_intrinsic,
)
from real_robot.primitives.utils import snap_path_wrist
from real_robot.utils import xarm_constants as C
from real_robot.utils.transform_utils import transform_pose
from real_robot.utils.xarm_camera import crop_window, describe_orientation

EEF_LINK = 6            # link_eef -- the flange, which is what the controller reports
N_JOINTS = 6


def _rotvec_to_quat(rotvec):
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_quat()


def _quat_to_rotvec(quat):
    return Rotation.from_quat(np.asarray(quat, dtype=float)).as_rotvec()


def _interp_pose(start, target, a):
    """Pose at fraction ``a`` along start -> target: lerp position, SLERP rotation.

    The rotation has to be interpolated too. Snapping straight to the target
    orientation on the first substep makes the wrist twist before the arm has
    moved, which invents IK failures that the real controller would never see.
    """
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    out = np.empty(6)
    out[:3] = start[:3] * (1.0 - a) + target[:3] * a
    key = Rotation.from_rotvec(np.stack([start[3:6], target[3:6]]))
    out[3:6] = Slerp([0.0, 1.0], key)(a).as_rotvec()
    return out


def _pose_to_mat(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(pose[3:6], dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(pose[:3], dtype=float)
    return T


def _mat_to_pose(T):
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


# ----------------------------------------------------------------------
class XArmSimArm:
    """One simulated Lite 6, presented with the same surface as ``XArmLite6``.

    Poses in and out are in THIS arm's base frame (metres + rotvec), exactly as the
    real driver does; the conversion to the PyBullet world frame happens here.
    """

    def __init__(self, bullet, uid, name, T_world_base, home_joint, gui_delay=0.0):
        self.p = bullet
        self.uid = uid
        self.name = name
        self.T_world_base = np.asarray(T_world_base, dtype=float)
        self.T_base_world = np.linalg.inv(self.T_world_base)
        self.home_joint = list(home_joint)
        self.gui_delay = gui_delay
        self.gripper_closed = False
        self.last_ik_error = 0.0
        self.max_ik_error = 0.0
        self.ik_failures = 0
        self.limit_violations = []
        # Every pose this arm passed through, in ITS OWN base frame. The
        # pick-and-place skills never call the scene's record= path, so without
        # this only the fling could be drawn.
        self.path = []

        self.lower, self.upper, self.ranges = [], [], []
        for j in range(N_JOINTS):
            info = self.p.getJointInfo(uid, j)
            self.lower.append(info[8])
            self.upper.append(info[9])
            self.ranges.append(info[9] - info[8])
        self.set_joints(self.home_joint)

    # -- state ---------------------------------------------------------
    def get_joints(self):
        return np.array([self.p.getJointState(self.uid, j)[0] for j in range(N_JOINTS)])

    def set_joints(self, q):
        for j in range(N_JOINTS):
            self.p.resetJointState(self.uid, j, float(q[j]))

    def log_pose(self):
        self.path.append(self.get_tcp_pose()[:3])

    def _eef_world(self):
        ls = self.p.getLinkState(self.uid, EEF_LINK, computeForwardKinematics=True)
        return np.asarray(ls[4], dtype=float), np.asarray(ls[5], dtype=float)

    def get_tcp_pose(self):
        """UR convention, in THIS arm's base frame."""
        pos, quat = self._eef_world()
        T_world_eef = np.eye(4)
        T_world_eef[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T_world_eef[:3, 3] = pos
        return _mat_to_pose(self.T_base_world @ T_world_eef)

    # Effort is not simulated. Reporting "no signal" is not a stub: it is what the
    # primitives must cope with on hardware when the reading is unusable, so this
    # exercises their fallback path rather than a path that only exists in sim.
    def effort_baseline(self, *a, **kw):
        return None

    def effort_delta(self, *a, **kw):
        return None

    def effort_exceeded(self, *a, **kw):
        return False

    # -- motion --------------------------------------------------------
    def _ik_once(self, pos, quat, iters):
        q = self.p.calculateInverseKinematics(
            self.uid, EEF_LINK, pos, quat,
            maxNumIterations=iters, residualThreshold=1e-6)
        return np.asarray(q[:N_JOINTS], dtype=float)

    def _residual(self, q, pose_base):
        """Set ``q`` and measure how far the flange lands from the request."""
        self.set_joints(q)
        got = self.get_tcp_pose()
        return float(np.linalg.norm(got[:3] - np.asarray(pose_base)[:3]))

    def _within_limits(self, q):
        return all(self.lower[j] - 1e-6 <= q[j] <= self.upper[j] + 1e-6
                   for j in range(N_JOINTS))

    def solve_ik(self, pose_base, tol=2e-3):
        """Joint solution for a pose in this arm's base frame.

        Deliberately NOT PyBullet's null-space form (``lowerLimits``/``restPoses``):
        measured against these fling waypoints it is the worst of the three options,
        leaving up to 0.20 m of residual on a pose that is comfortably reachable.
        Plain IK seeded from the current configuration is much better and preserves
        continuity between substeps; random restarts are the last resort, and they
        do solve every waypoint here to ~1e-4 m.

        Reporting a solver artefact as an unreachable pose would be worse than
        useless -- it would send someone tuning constants that were never wrong.
        """
        T_world = self.T_world_base @ _pose_to_mat(pose_base)
        pos = T_world[:3, 3].tolist()
        quat = Rotation.from_matrix(T_world[:3, :3]).as_quat().tolist()
        seed = self.get_joints()

        best_q, best_err = None, np.inf
        for iters in (200, 600):
            self.set_joints(seed)                 # warm start = continuity
            q = self._ik_once(pos, quat, iters)
            err = self._residual(q, pose_base)
            if err < best_err and self._within_limits(q):
                best_q, best_err = q, err
            if best_err <= tol:
                self.set_joints(best_q)
                return best_q

        # Restarts. These can flip the elbow, so only once the warm start has failed.
        rng = np.random.default_rng(0)
        for _ in range(8):
            self.set_joints([rng.uniform(l, u) for l, u in zip(self.lower, self.upper)])
            q = self._ik_once(pos, quat, 400)
            err = self._residual(q, pose_base)
            if err < best_err and self._within_limits(q):
                best_q, best_err = q, err
            if best_err <= tol:
                break

        if best_q is None:                        # nothing within limits: keep the seed
            self.ik_failures += 1
            self.set_joints(seed)
            return seed
        if best_err > tol:
            self.ik_failures += 1
        self.set_joints(best_q)
        return best_q

    def _record_quality(self, target_pose_base, label):
        """How far the solved configuration actually landed from what was asked."""
        got = self.get_tcp_pose()
        err = float(np.linalg.norm(got[:3] - np.asarray(target_pose_base)[:3]))
        self.last_ik_error = err
        self.max_ik_error = max(self.max_ik_error, err)
        q = self.get_joints()
        for j in range(N_JOINTS):
            if q[j] < self.lower[j] - 1e-6 or q[j] > self.upper[j] + 1e-6:
                self.limit_violations.append((label, j + 1, float(q[j])))
        return err

    def movel(self, p_target, speed=1.5, acceleration=1.0, blocking=True,
              avoid_singularity=False, steps=None, **kwargs):
        """Interpolate to one pose or along a list of poses, in this arm's frame."""
        poses = np.asarray(p_target, dtype=float)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        for i, target in enumerate(poses):
            self._interpolate_to(target, speed=speed, steps=steps,
                                 label="movel[{}]".format(i))
        return True

    def _interpolate_to(self, target_pose, speed=0.2, steps=None, label=""):
        start = self.get_tcp_pose()
        dist = float(np.linalg.norm(np.asarray(target_pose)[:3] - start[:3]))
        if steps is None:
            # ~5 mm per frame, so a long swing is visibly a swing rather than a jump.
            steps = int(np.clip(np.ceil(dist / 0.005), 1, 400))
        for s in range(1, steps + 1):
            a = s / float(steps)
            self.set_joints(self.solve_ik(_interp_pose(start, target_pose, a)))
            self.log_pose()
            if self.gui_delay:
                time.sleep(self.gui_delay)
        self._record_quality(target_pose, label)

    def movej(self, q, speed=None, acceleration=None, blocking=True):
        self.set_joints(q)
        return True

    def home(self, speed=None, acceleration=None, blocking=True):
        self.set_joints(self.home_joint)
        return True

    def photo_joint(self):
        return C.photo_pose_from_home(self.home_joint)

    def out_scene(self, speed=None, acceleration=None, blocking=True, steps=40):
        """Swing to the photo pose -- home with joint 1 turned to this arm's left.

        Interpolated in JOINT space (it is a joint move on hardware too), and
        deliberately NOT logged to ``self.path``: getting out of the camera's way
        is not part of the primitive, and drawing it would put a phantom arc across
        every executed-path overlay.
        """
        start = self.get_joints()
        target = np.asarray(self.photo_joint(), dtype=float)
        for s in range(1, int(steps) + 1):
            a = s / float(steps)
            self.set_joints(start * (1.0 - a) + target * a)
            if self.gui_delay:
                time.sleep(self.gui_delay)
        return True

    def open_gripper(self, *a, **kw):
        self.gripper_closed = False
        return True

    def close_gripper(self, *a, **kw):
        self.gripper_closed = True
        return True

    def disconnect(self):
        return True


# ----------------------------------------------------------------------
class XArmSimScene:
    """Dual-arm cell. Duck-types ``XArmDualArmScene`` for the primitives."""

    def __init__(self, cell=None, gui=True, gui_delay=0.0, garment_pose=None,
                 seed=0, verbose=True):
        import pybullet as pb
        import pybullet_data
        self.p = pb

        from real_robot.test.xarm_test_scene import load_cell
        self.cell = load_cell() if cell is None else cell
        self.separation = self.cell.separation
        self.table_z = self.cell.table_z('left')
        self.gui = gui
        self.verbose = verbose
        self.dry_run = False
        self.gripper_type = 'lite6'
        self.last_trajectory = None
        self._cloth_ids = []

        self.client = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        if gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            pb.resetDebugVisualizerCamera(1.6, 50, -35,
                                          [self.separation / 2.0, 0.05, 0.1])

        self._build_table()
        self._build_arms(gui_delay)
        self._build_garment(garment_pose, seed)

        # Camera: derived from the same intrinsic the primitives invert, so the
        # rendered pixels and point_on_table_base cannot disagree.
        self._full_intr = synthetic_intrinsic()
        self.T_left_cam = synthetic_T_left_cam(self.separation, self.table_z)
        self.T_left_right = face_to_face_T_left_right(self.separation)
        self.T_right_cam = np.linalg.inv(self.T_left_right) @ self.T_left_cam
        self._view, self._proj = self._camera_matrices()

        # The square window between the arms, exactly as the production scene
        # does it: PyBullet still renders the full frame (the projection matrix is
        # the full camera's), and take_rgbd() crops what it hands out.
        self.crop = crop_window(self._full_intr, self.T_left_cam,
                                separation=self.separation, table_z=self.table_z)
        self.intr = self.crop.intrinsic(self._full_intr)

        # UR-named aliases, as the production scene exposes.
        self.T_ur5e_cam = self.T_left_cam
        self.T_ur16e_cam = self.T_right_cam
        self.T_ur5e_ur16e = self.T_left_right

        if verbose:
            print("[sim] cell: bases {:.3f} m apart, table_z {:+.3f} m".format(
                self.separation, self.table_z))
            print("[sim] camera at ({:.3f}, {:.3f}, {:.3f}) m -- above the arm midpoint, "
                  "{:.2f} m up".format(self.T_left_cam[0, 3], self.T_left_cam[1, 3],
                                       self.T_left_cam[2, 3], C.XARM_CAM_HEIGHT))
            self._report_fov()
            print("[sim] crop {} -- centred between the arms, x [{:+.3f}, {:+.3f}] "
                  "y [{:+.3f}, {:+.3f}] m".format(
                      self.crop, self.separation / 2 - C.XARM_CROP_SIZE / 2,
                      self.separation / 2 + C.XARM_CROP_SIZE / 2,
                      -C.XARM_CROP_SIZE / 2, C.XARM_CROP_SIZE / 2))
            print(describe_orientation(self.T_left_cam, self.intr,
                                       self.separation, self.table_z)[1])

    # -- construction --------------------------------------------------
    def _build_table(self):
        pb = self.p
        sx, sy = C.XARM_TABLE_SIZE
        cx = (C.XARM_TABLE_RECT['x'][0] + C.XARM_TABLE_RECT['x'][1]) / 2.0
        cy = (C.XARM_TABLE_RECT['y'][0] + C.XARM_TABLE_RECT['y'][1]) / 2.0
        thick = 0.02
        col = pb.createCollisionShape(pb.GEOM_BOX,
                                      halfExtents=[sx / 2, sy / 2, thick / 2])
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[sx / 2, sy / 2, thick / 2],
                                   rgbaColor=[0.82, 0.80, 0.76, 1.0])
        self.table_uid = pb.createMultiBody(0, col, vis,
                                            [cx, cy, self.table_z - thick / 2])

    def _build_arms(self, gui_delay):
        pb = self.p
        self.assets_ok = assets_present()
        if not self.assets_ok:
            raise RuntimeError(
                "The Lite 6 description is missing. Run:\n"
                "    python real_robot/sim/fetch_lite6_assets.py")
        pb.setAdditionalSearchPath(asset_dir())

        left_uid = pb.loadURDF(urdf_path(), [0, 0, self.table_z],
                               pb.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        right_uid = pb.loadURDF(
            urdf_path(), [self.separation, 0, self.table_z],
            pb.getQuaternionFromEuler([0, 0, C.XARM_BASE_YAW]), useFixedBase=True)

        T_left = np.eye(4)
        T_left[2, 3] = self.table_z
        T_right = face_to_face_T_left_right(self.separation)
        T_right[2, 3] = self.table_z

        self.left = XArmSimArm(pb, left_uid, 'left', T_left,
                               C.for_side(C.XARM_HOME_JOINT_BY_SIDE, 'left'), gui_delay)
        self.right = XArmSimArm(pb, right_uid, 'right', T_right,
                                C.for_side(C.XARM_HOME_JOINT_BY_SIDE, 'right'), gui_delay)

    def _build_garment(self, garment_pose, seed):
        """A thin slab on the table. Its segmentation mask is the cloth mask, so
        clicking off it exercises the primitives' validity-flag abort."""
        pb = self.p
        rng = np.random.default_rng(seed)
        if garment_pose is None:
            gx = self.separation / 2.0 + float(rng.uniform(-0.05, 0.05))
            gy = float(rng.uniform(-0.10, 0.10))
            yaw = float(rng.uniform(-0.4, 0.4))
        else:
            gx, gy, yaw = garment_pose
        self.garment_size = (0.26, 0.30)
        half = [self.garment_size[0] / 2, self.garment_size[1] / 2, 0.001]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half)
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half,
                                   rgbaColor=[0.20, 0.45, 0.85, 1.0])
        self.garment_uid = pb.createMultiBody(
            0, col, vis, [gx, gy, self.table_z + 0.001],
            pb.getQuaternionFromEuler([0, 0, yaw]))
        self.garment_pose = (gx, gy, yaw)

    # -- camera --------------------------------------------------------
    def _camera_matrices(self):
        pb = self.p
        eye = self.T_left_cam[:3, 3]
        forward = self.T_left_cam[:3, 2]        # camera +z is the view direction
        up = -self.T_left_cam[:3, 1]            # image v points down, so up is -y
        view = pb.computeViewMatrix(eye.tolist(), (eye + forward).tolist(), up.tolist())
        fov_deg = math.degrees(2.0 * math.atan(IMAGE_H / (2.0 * self._full_intr.fy)))
        proj = pb.computeProjectionMatrixFOV(fov_deg, IMAGE_W / float(IMAGE_H), 0.05, 5.0)
        return view, proj

    def _report_fov(self):
        h = C.XARM_CAM_HEIGHT
        w_m, h_m = IMAGE_W / self._full_intr.fx * h, IMAGE_H / self._full_intr.fy * h
        tw = C.XARM_TABLE_SIZE[0]
        tl = C.XARM_TABLE_SIZE[1]
        fits = w_m >= tw and h_m >= tl
        print("[sim] placeholder intrinsic f={:.0f} px sees {:.2f} x {:.2f} m at {:.1f} m "
              "-- table {:.2f} x {:.2f} m {}".format(
                  self.intr.fx, w_m, h_m, h, tw, tl, "FITS" if fits else "DOES NOT FIT"))
        need = tl * 900.0 / IMAGE_H
        print("[sim] note: a RealSense colour stream (fx ~ 900 px at {}x{}) would need "
              "~{:.2f} m of height to see the {:.2f} m table length.".format(
                  IMAGE_W, IMAGE_H, need, tl))

    def take_rgbd(self):
        """(rgb uint8 NxNx3, depth, cloth mask uint8) -- the SQUARE window between
        the arms, paired with the shifted ``self.intr``, as the production scene
        returns."""
        pb = self.p
        w, h, rgba, depth, seg = pb.getCameraImage(
            IMAGE_W, IMAGE_H, self._view, self._proj,
            renderer=pb.ER_TINY_RENDERER)
        rgb = np.reshape(np.asarray(rgba, dtype=np.uint8), (h, w, 4))[:, :, :3]
        seg = np.reshape(np.asarray(seg, dtype=np.int32), (h, w))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[seg == self.garment_uid] = 255
        depth = np.reshape(np.asarray(depth), (h, w))
        return (np.ascontiguousarray(self.crop.apply(rgb)),
                self.crop.apply(depth),
                np.ascontiguousarray(self.crop.apply(mask)))

    def get_workspace_masks(self):
        return None, None

    def get_T_base_cam(self):
        return self.T_left_cam

    # -- motion (the XArmDualArmScene surface) -------------------------
    def both_movel(self, left_pose, right_pose, speed, acc, blocking=True, record=False):
        """Both arms in lockstep, so a dual move looks like a dual move.

        The real scene threads the two ``movel`` calls; here they are interleaved
        step by step instead, which is closer to what actually happens on hardware
        and keeps the render deterministic.
        """
        lp = np.asarray(left_pose, dtype=float)
        rp = np.asarray(right_pose, dtype=float)
        lp = lp.reshape(1, -1) if lp.ndim == 1 else lp
        rp = rp.reshape(1, -1) if rp.ndim == 1 else rp
        n = max(len(lp), len(rp))
        traj = {'ur5e': [], 'ur16e': []}
        for i in range(n):
            lt = lp[min(i, len(lp) - 1)]
            rt = rp[min(i, len(rp) - 1)]
            self._step_pair(lt, rt, traj if record else None, "movel[{}]".format(i))
        self.last_trajectory = traj if record else None
        return True

    def _step_pair(self, left_target, right_target, traj, label):
        ls, rs = self.left.get_tcp_pose(), self.right.get_tcp_pose()
        dist = max(float(np.linalg.norm(left_target[:3] - ls[:3])),
                   float(np.linalg.norm(right_target[:3] - rs[:3])))
        steps = int(np.clip(np.ceil(dist / 0.005), 1, 400))
        for s in range(1, steps + 1):
            a = s / float(steps)
            for arm, start, target in ((self.left, ls, left_target),
                                       (self.right, rs, right_target)):
                arm.set_joints(arm.solve_ik(_interp_pose(start, target, a)))
                arm.log_pose()
            self._update_cloth()
            if traj is not None:
                traj['ur5e'].append(self.left.get_tcp_pose()[:3])
                traj['ur16e'].append(self.right.get_tcp_pose()[:3])
            if self.gui and self.left.gui_delay:
                time.sleep(self.left.gui_delay)
        self.left._record_quality(left_target, label)
        self.right._record_quality(right_target, label)

    def both_fling(self, left_path, right_path, speed, acc, record=False):
        r = self.both_movel(left_path[0], right_path[0], speed=speed, acc=acc)
        if not r:
            return False
        return self.both_movel(left_path[1:], right_path[1:], speed=speed, acc=acc,
                               record=record)

    def both_home(self, speed=None, acc=None, blocking=True):
        self.left.home()
        self.right.home()
        self._update_cloth()
        return True

    def both_out_scene(self, speed=None, acc=None, blocking=True, steps=40):
        """Both arms swing to their own left, in step, so the photo is unobstructed."""
        starts = [arm.get_joints() for arm in (self.left, self.right)]
        targets = [np.asarray(arm.photo_joint(), dtype=float)
                   for arm in (self.left, self.right)]
        for s in range(1, int(steps) + 1):
            a = s / float(steps)
            for arm, start, target in zip((self.left, self.right), starts, targets):
                arm.set_joints(start * (1.0 - a) + target * a)
            self._update_cloth()
            if self.gui and self.left.gui_delay:
                time.sleep(self.left.gui_delay)
        return True

    def go_camera_pos(self):
        """Home, then out of shot -- the production scene's capture sequence.

        Home first is not decoration: out_scene only turns joint 1, so the arm must
        be in the (taught, table-clearing) home configuration before it rotates.
        """
        self.both_home()
        return self.both_out_scene()

    def both_open_gripper(self):
        self.left.open_gripper()
        self.right.open_gripper()
        self._update_cloth()
        return True

    def both_close_gripper(self):
        self.left.close_gripper()
        self.right.close_gripper()
        self._update_cloth()
        return True

    def get_tcp_distance(self):
        left_tcp = self.left.get_tcp_pose()
        right_tcp = transform_pose(self.T_left_right, self.right.get_tcp_pose())
        return float(np.linalg.norm((right_tcp - left_tcp)[:3]))

    # -- cloth proxy ---------------------------------------------------
    def _update_cloth(self):
        """Draw a sheet hanging between the grippers. NOT physics -- a catenary
        whose sag comes from the slack between the grasp width and the garment
        size, so you can see roughly where cloth would be during the swing."""
        pb = self.p
        held = self.left.gripper_closed and self.right.gripper_closed
        if not held:
            for uid in self._cloth_ids:
                pb.removeUserDebugItem(uid)
            self._cloth_ids = []
            return
        a = self.left.get_tcp_pose()[:3]
        b = transform_pose(self.T_left_right, self.right.get_tcp_pose())[:3]
        span = float(np.linalg.norm(b - a))
        slack = max(self.garment_size[0] - span, 0.0)
        sag = min(0.35 * math.sqrt(max(slack, 1e-6) * self.garment_size[0]),
                  self.garment_size[1])
        strands, nodes = 5, 9
        new_ids = []
        for k in range(strands):
            depth = (k / max(strands - 1.0, 1.0)) * self.garment_size[1]
            pts = []
            for i in range(nodes):
                t = i / (nodes - 1.0)
                pos = a * (1 - t) + b * t
                drop = sag * 4.0 * t * (1 - t)
                pts.append([pos[0], pos[1] + depth * 0.0, pos[2] - drop - depth])
            for i in range(nodes - 1):
                uid = pb.addUserDebugLine(
                    pts[i], pts[i + 1], lineColorRGB=[0.20, 0.45, 0.85], lineWidth=2,
                    replaceItemUniqueId=(self._cloth_ids[len(new_ids)]
                                         if len(new_ids) < len(self._cloth_ids) else -1))
                new_ids.append(uid)
        for stale in self._cloth_ids[len(new_ids):]:
            pb.removeUserDebugItem(stale)
        self._cloth_ids = new_ids

    # -- diagnostics ---------------------------------------------------
    def contacts_between_arms(self):
        self.p.performCollisionDetection()
        return self.p.getContactPoints(bodyA=self.left.uid, bodyB=self.right.uid)

    def quality_report(self):
        return {
            'max_ik_error_left': self.left.max_ik_error,
            'max_ik_error_right': self.right.max_ik_error,
            'ik_failures': self.left.ik_failures + self.right.ik_failures,
            'path_samples': len(self.left.path) + len(self.right.path),
            'limit_violations': self.left.limit_violations + self.right.limit_violations,
        }

    def close(self):
        try:
            self.p.disconnect(self.client)
        except Exception:
            pass


# ----------------------------------------------------------------------
class XArmSimSingleScene:
    """One arm of the cell, with the surface ``XArmSingleArmScene`` presents."""

    def __init__(self, dual, side='left'):
        self.dual = dual
        self.side = side
        self.arm = dual.left if side == 'left' else dual.right
        self.intr = dual.intr
        self.T_cam = dual.T_left_cam if side == 'left' else dual.T_right_cam
        self.dry_run = False
        self.gripper_type = 'lite6'
        self.last_trajectory = None

    def movel(self, pose, speed, acc, blocking=True, record=False):
        return self.arm.movel(pose, speed=speed, acceleration=acc, blocking=blocking)

    def home(self, speed=None, acc=None, blocking=True):
        return self.arm.home()

    def out_scene(self, speed=None, acc=None, blocking=True):
        return self.arm.out_scene()

    def open_gripper(self):
        r = self.arm.open_gripper()
        self.dual._update_cloth()
        return r

    def close_gripper(self):
        r = self.arm.close_gripper()
        self.dual._update_cloth()
        return r

    def take_rgbd(self):
        return self.dual.take_rgbd()

    def get_T_base_cam(self):
        return self.T_cam


def project_path(points, T_cam, intr):
    """Base-frame path -> pixels, for drawing the executed motion on the frame."""
    return base_to_pixel(np.asarray(points, dtype=float), T_cam, intr)
