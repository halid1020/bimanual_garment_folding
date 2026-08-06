"""MuJoCo model of the dual xArm Lite 6 cell, with the CALIBRATED top-down camera.

The point of this module is that the SHIPPED primitives run against it unmodified:
``XArmMujocoScene`` duck-types ``XArmDualArmScene`` (``both_movel``, ``both_fling``,
``both_home``, the grippers, ``get_tcp_distance``, ``intr``, ``T_left_cam``,
``T_left_right``, ``.left`` / ``.right``), so ``XArmPickAndFlingSkill`` and friends
drive simulated arms through exactly the code path they use on hardware. A bug in a
primitive shows up here rather than on the robot.

FRAMES. The MuJoCo world IS the LEFT arm's base frame: the left base sits at the
origin, the right base at the measured ``T_left_right``, and the table top is the
z = ``table_z`` plane -- which is also the arms' mounting plane, since the measured
``table_z`` is 0 in each base frame.

WHAT IS REAL AND WHAT IS NOT
  * Kinematics: real. The URDF is UFACTORY's own (via hygradme/lite6_urdf) and its
    forward kinematics reproduce poses measured on the hardware to 2-4 mm.
  * The cell geometry: MEASURED. Both arm bases come from hand-eye calibration
    (0.749 m apart, yawed 178.7 deg), not from the assumed constants.
  * The camera: real projection through the CALIBRATED extrinsic and the measured
    intrinsic (a documented D435i nominal until --dump-intrinsics has been run).
    ``t_projection_round_trip`` renders markers at known table coordinates and
    checks ``base_to_pixel`` lands on them; it agrees to under a pixel.
  * The cloth: REAL, a MuJoCo flex sheet with elasticity and contact. It is not a
    validated garment model -- the stiffness is plausible, not measured -- but it
    drapes, folds and is thrown, which the PyBullet cell's drawn catenary did not.
  * Grasping: an equality CONNECT between the flange and the nearest cloth vertex.
    A pinch constrains a point; nothing here models finger geometry or slip.
  * IK: ours, damped least squares, WARM START ONLY -- see solve_ik. Not the xArm
    controller's; its ``get_inverse_kinematics`` stays the authority before real
    motion.
  * Effort: not simulated. ``effort_*`` report "no signal", so the primitives take
    their documented fallback and the geometric caps drive the stretch.
"""
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from real_robot.sim.xarm_mjcf import build_cell, CLOTH_COUNT
from real_robot.utils import xarm_constants as C
from real_robot.utils.scene_utils import (
    cell_geometry_from_calibration, load_camera_to_base,
)
from real_robot.utils.transform_utils import transform_pose
from real_robot.utils.xarm_camera import (
    base_to_pixel, crop_window, describe_orientation, load_intrinsic,
)

N_JOINTS = 6

# How close counts as "the arm got there". Separate from solve_ik's `tol`, which is
# only a stopping criterion for an iterative solver: a DLS run that stopped at
# 2.3 mm has not discovered anything physical about the robot. This is the number
# that decides whether a warm-started solve REACHED the pose, and it is the same
# one run_sim_ui.report uses to call a run good, so the two cannot disagree.
REACH_TOL = 0.005       # m

CALIB_DIR = "{}/real_robot/calibration".format(os.environ.get('MP_FOLD_PATH', '.'))


# ----------------------------------------------------------------------
def _pose_to_mat(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(pose, dtype=float)[3:6]).as_matrix()
    T[:3, 3] = np.asarray(pose, dtype=float)[:3]
    return T


def _mat_to_pose(T):
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def _interp_pose(start, target, a):
    """Straight line in position, SLERP in orientation -- a linear Cartesian move."""
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    rots = Rotation.from_rotvec(np.stack([start[3:6], target[3:6]]))
    out = np.empty(6)
    out[:3] = start[:3] * (1.0 - a) + target[:3] * a
    out[3:6] = Slerp([0.0, 1.0], rots)([a])[0].as_rotvec()
    return out


# ----------------------------------------------------------------------
class XArmMujocoArm:
    """One simulated Lite 6, presented with the same surface as ``XArmLite6``.

    Poses in and out are in THIS arm's base frame (metres + rotvec), exactly as the
    real driver does; the conversion to the MuJoCo world frame happens here.
    """

    def __init__(self, cell, side, T_world_base, home_joint):
        import mujoco
        self._mj = mujoco
        self.cell = cell
        self.name = side
        self.side = side
        self.T_world_base = np.asarray(T_world_base, dtype=float)
        self.T_base_world = np.linalg.inv(self.T_world_base)
        self.home_joint = list(home_joint)
        self.gripper_closed = False
        self.last_ik_error = 0.0
        self.max_ik_error = 0.0
        self.ik_failures = 0
        # Waypoints the warm-started solve could not reach, i.e. poses the arm can
        # only get to by RECONFIGURING (flipping the elbow or the wrist). This is
        # the number that matters: the xArm controller plans a straight Cartesian
        # line and cannot reconfigure mid-move, so anything counted here is a
        # motion the real arm will refuse or fault on. The PyBullet cell repaired
        # these silently with random restarts, which is how the 2026-08-04 fling --
        # J4 to -363 deg, servo 4 code 23, both arms e-stopped -- passed --auto
        # with "no joint-limit violations".
        self.reconfigurations = []
        self.q_log = []
        self.limit_violations = []
        # Worst single-step joint motion, per joint. See the end of solve_ik.
        self.max_joint_jump = np.zeros(N_JOINTS)
        self.worst_jump = None
        # Every pose this arm passed through, in ITS OWN base frame. The
        # pick-and-place skills never call the scene's record= path, so without
        # this only the fling could be drawn.
        self.path = []

        # ⚠️ ONE SCRATCH MjData PER ARM, not one per cell. Every scene both_*
        # method runs the two arms in PARALLEL THREADS, and MjData is not
        # thread-safe -- two arms solving IK against one scratch segfaults the
        # interpreter partway through the contact probe.
        self.ik_data = mujoco.MjData(cell.model)

        m = cell.model
        self.eef_body = cell.info['eef_body'][side]
        self.joint_ids = cell.info['joint_qpos'][side]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.lower = np.array([m.jnt_range[j][0] for j in self.joint_ids])
        self.upper = np.array([m.jnt_range[j][1] for j in self.joint_ids])
        self.set_joints(self.home_joint)

    # -- state ---------------------------------------------------------
    def get_joints(self):
        return np.array(self.cell.data.qpos[self.qpos_adr], dtype=float)

    def set_joints(self, q):
        """Command a joint configuration.

        The arms are driven KINEMATICALLY: this writes qpos, and the cell rewrites
        it after every physics step so gravity cannot pull the arm down. That keeps
        the IK diagnostics meaning what they say -- with position actuators, "the
        arm could not reach this pose" and "the servo lagged" would be the same
        measurement.
        """
        self.cell.command_joints(self.side, np.asarray(q, dtype=float))

    def log_pose(self):
        self.path.append(self.get_tcp_pose()[:3])

    def get_tcp_pose(self):
        """UR convention, in THIS arm's base frame."""
        d = self.cell.data
        T_world_eef = np.eye(4)
        T_world_eef[:3, :3] = d.xmat[self.eef_body].reshape(3, 3)
        T_world_eef[:3, 3] = d.xpos[self.eef_body]
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

    # -- inverse kinematics --------------------------------------------
    def _fk(self, q):
        """Flange pose in the WORLD frame for ``q``, without touching the sim.

        Runs on a scratch MjData so a solver iteration cannot nudge the cloth.
        """
        d = self.ik_data
        d.qpos[self.qpos_adr] = q
        self._mj.mj_kinematics(self.cell.model, d)
        self._mj.mj_comPos(self.cell.model, d)
        return (np.array(d.xpos[self.eef_body]),
                np.array(d.xmat[self.eef_body]).reshape(3, 3))

    def solve_ik(self, pose_base, tol=1e-3, rot_tol=5e-3, max_iters=120,
                 damping=2e-2, null_gain=0.5):
        """Joint solution for a pose in this arm's base frame -- WARM START ONLY.

        Damped least squares, seeded from where the arm currently is, and never
        restarted from anywhere else. That restriction is the whole point.
        PyBullet's solver fell back to 8 random restarts, which teleport the arm
        between IK branches; the xArm controller plans a straight Cartesian line
        and cannot do that, so a path stitched together from different branches
        looks perfectly reachable here and faults on the robot. Refusing to
        reconfigure means an unreachable waypoint is REPORTED as unreachable.

        Both position and orientation are solved -- a fling is mostly wrist pitch,
        so a position-only solve would hide exactly the failure that mattered.

        ⚠️ TWO THINGS KEEP THE WRIST STILL, and both were learned the hard way,
        from a swing that visibly juddered while every summary number looked fine:

        1. STOP WHEN IT HAS CONVERGED, on BOTH position and orientation, and stop
           again when it stops improving. The first version broke only on a 1e-3
           RAD orientation tolerance -- 0.06 deg, tighter than this arm can be
           commanded. Waypoints that met their position target in two iterations
           went on iterating sixty more times, chasing an angle they could not
           reach, and every one of those iterations was free to wander.

        2. PULL THE NULL SPACE BACK TOWARD THE SEED. At a wrist singularity (J5
           near zero) J4 and J6 turn about the same axis, so the tool pose does
           not constrain their difference at all and DLS slides along it. Measured
           on the fling: J4 +54 deg and J6 -53 deg in ONE interpolation step,
           equal and opposite, on both arms. The projector (I - J+ J) is
           numerically zero away from singularities, so this costs nothing where
           it is not needed and is the whole fix where it is.
        """
        T_world = self.T_world_base @ _pose_to_mat(pose_base)
        target_pos, target_R = T_world[:3, 3], T_world[:3, :3]

        m = self.cell.model
        seed = self.get_joints().copy()
        q = seed.copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        best_q, best_err = q.copy(), np.inf
        prev_score = np.inf
        eye6 = np.eye(6)

        for _ in range(max_iters):
            cur_pos, cur_R = self._fk(q)
            e_pos = target_pos - cur_pos
            e_rot = Rotation.from_matrix(target_R @ cur_R.T).as_rotvec()
            err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))
            if err < best_err:
                best_q, best_err = q.copy(), err
            if err < tol and rot_err < rot_tol:
                break
            # Converged as far as it is going to. Iterating past this point does
            # not improve the pose and does move the joints.
            score = err + 0.05 * rot_err
            if prev_score - score < 1e-9:
                break
            prev_score = score

            self._mj.mj_jacBody(m, self.ik_data, jacp, jacr, self.eef_body)
            J = np.vstack([jacp[:, self.dof_adr], jacr[:, self.dof_adr]])
            e = np.concatenate([e_pos, e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * eye6, e)
            # ⚠️ The null-space projector must use a TRUE pseudo-inverse, not the
            # damped one used for the task term. With damping 5e-2, I - J_damped+ J
            # is far from zero even in a perfectly well-conditioned pose, so the
            # pull toward the seed fights the tracking everywhere instead of only
            # at singularities -- measured as a steady-state error that grew along
            # the path to 10 mm and was reported as 14 unreachable waypoints.
            # np.linalg.pinv cuts on singular values, so this is ~0 unless the
            # wrist really is degenerate.
            J_true = np.linalg.pinv(J)
            dq += (np.eye(len(q)) - J_true @ J) @ (seed - q) * null_gain
            step = float(np.linalg.norm(dq))
            if step > 0.15:                    # rad, don't leap across a branch
                dq *= 0.15 / step
            q = np.clip(q + dq, self.lower, self.upper)

        # Judge reachability on REACH_TOL, not on the solver's stopping tolerance:
        # a run that stopped at 2.3 mm has discovered nothing about the robot.
        if best_err > REACH_TOL:
            self.reconfigurations.append(
                (np.asarray(pose_base)[:3].copy(), float(best_err)))
            self.ik_failures += 1
        # How far the joints moved to get here. A real controller interpolates
        # joint space between waypoints, so a large jump is a lurch on hardware
        # even when the Cartesian path either side of it is perfectly smooth --
        # and it is invisible to every other number in the report.
        jump = np.abs(best_q - seed)
        if jump.max() > self.max_joint_jump.max():
            self.worst_jump = (seed.copy(), best_q.copy())
        self.max_joint_jump = np.maximum(self.max_joint_jump, jump)
        self.q_log.append(best_q.copy())
        return best_q

    def joint_margins(self):
        """Smallest distance to a limit on each joint over everything solved.

        A path can stay inside every limit and still be one bad seed away from
        error 23; the margin says which. Returns None until something has moved.
        """
        if not self.q_log:
            return None
        q = np.asarray(self.q_log, dtype=float)
        return np.minimum(q - self.lower, self.upper - q).min(axis=0)

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

    # -- motion --------------------------------------------------------
    def movel(self, p_target, speed=None, acceleration=None, blocking=True,
              avoid_singularity=False, steps=None, **kwargs):
        """Interpolate to one pose or along a list of poses, in this arm's frame."""
        poses = np.asarray(p_target, dtype=float)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        for i, target in enumerate(poses):
            self._interpolate_to(target, speed=speed, steps=steps,
                                 label="movel[{}]".format(i))
        return True

    def _interpolate_to(self, target_pose, speed=None, steps=None, label=""):
        start = self.get_tcp_pose()
        dist = float(np.linalg.norm(np.asarray(target_pose)[:3] - start[:3]))
        steps = self.cell.steps_for(dist, speed) if steps is None else int(steps)
        for s in range(1, steps + 1):
            a = s / float(steps)
            self.set_joints(self.solve_ik(_interp_pose(start, target_pose, a)))
            self.cell.advance()
            self.log_pose()
        self._record_quality(target_pose, label)

    def movej(self, q, speed=None, acceleration=None, blocking=True, steps=40):
        start = self.get_joints()
        target = np.asarray(q, dtype=float)
        for s in range(1, int(steps) + 1):
            a = s / float(steps)
            self.set_joints(start * (1.0 - a) + target * a)
            self.cell.advance()
        return True

    def home(self, speed=None, acceleration=None, blocking=True):
        return self.movej(self.home_joint)

    def photo_joint(self):
        return C.photo_pose_from_home(self.home_joint)

    def out_scene(self, speed=None, acceleration=None, blocking=True, steps=40):
        """Swing to the photo pose -- home with joint 1 turned to this arm's left.

        Interpolated in JOINT space (it is a joint move on hardware too), and
        deliberately NOT logged to ``self.path``: getting out of the camera's way
        is not part of the primitive, and drawing it would put a phantom arc across
        every executed-path overlay.
        """
        return self.movej(self.photo_joint(), steps=steps)

    def open_gripper(self, *a, **kw):
        self.gripper_closed = False
        self.cell.release(self.side)
        return True

    def close_gripper(self, *a, **kw):
        self.gripper_closed = True
        self.cell.grasp(self.side)
        return True

    def disconnect(self):
        return True


# ----------------------------------------------------------------------
class XArmMujocoScene:
    """Dual-arm cell. Duck-types ``XArmDualArmScene`` for the primitives."""

    def __init__(self, cell=None, gui=False, gui_delay=0.0, garment_pose=None,
                 seed=0, verbose=True, calib_dir=CALIB_DIR, settle_steps=400):
        import mujoco
        self._mj = mujoco

        from real_robot.test.xarm_test_scene import load_cell
        self.cell = load_cell() if cell is None else cell
        self.table_z = self.cell.table_z('left')
        self.gui = gui
        self.gui_delay = gui_delay
        self.verbose = verbose
        self.dry_run = False
        self.gripper_type = 'lite6'
        self.last_trajectory = None
        self.viewer = None
        self._renderer = None

        left_calib = os.path.join(calib_dir, 'xarm-left-calib.yaml')
        right_calib = os.path.join(calib_dir, 'xarm-right-calib.yaml')

        # MEASURED, not assumed. Both arm bases and the camera come out of the same
        # two hand-eye files the robot uses, so the simulated cell is the cell.
        self.T_left_right, self.separation, self.base_yaw = \
            cell_geometry_from_calibration(left_calib, right_calib)
        self.T_left_cam = load_camera_to_base(left_calib)
        self.T_right_cam = np.linalg.inv(self.T_left_right) @ self.T_left_cam
        self._full_intr = load_intrinsic(left_calib, verbose=verbose)

        rng = np.random.default_rng(seed)
        if garment_pose is None:
            garment_pose = (self.separation / 2.0 + float(rng.uniform(-0.05, 0.05)),
                            float(rng.uniform(-0.10, 0.10)),
                            float(rng.uniform(-0.4, 0.4)))
        self.garment_pose = tuple(garment_pose)
        self.garment_size = (0.26, 0.30)

        self.model, self.info = build_cell(
            self.T_left_right, self.T_left_cam, self._full_intr,
            table_z=self.table_z, garment_pose=self.garment_pose,
            garment_size=self.garment_size,
            gripper_offset=self.cell.gripper_offset('left'))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        # The primitives thread the two arms (every both_* method), so every
        # mutation of the shared MjData is serialised. Without this the second
        # thread's mj_step lands in the middle of the first one's and takes the
        # interpreter down -- reproducibly, a few hundred steps into the contact
        # probe. An RLock because advance() is reached from grasp() as well.
        self._lock = threading.RLock()

        T_left = np.eye(4)
        T_left[2, 3] = self.table_z
        T_right = self.T_left_right.copy()
        T_right[2, 3] += self.table_z
        self._q_cmd = {}
        self.left = XArmMujocoArm(self, 'left', T_left,
                                  self.cell.home_joint('left'))
        self.right = XArmMujocoArm(self, 'right', T_right,
                                   self.cell.home_joint('right'))

        if gui:
            self._open_viewer()
        # Let the cloth fall onto the table before anyone photographs it.
        self.advance(settle_steps)

        # The square window between the arms, exactly as the production scene does
        # it: MuJoCo still renders the full frame, and take_rgbd() crops what it
        # hands out. size_m defaults to the separation, so the two base centres
        # land on the left and right edges of the crop.
        self.crop = crop_window(self._full_intr, self.T_left_cam,
                                separation=self.separation, table_z=self.table_z,
                                verbose=verbose)
        self.intr = self.crop.intrinsic(self._full_intr)

        # UR-named aliases, as the production scene exposes.
        self.T_ur5e_cam = self.T_left_cam
        self.T_ur16e_cam = self.T_right_cam
        self.T_ur5e_ur16e = self.T_left_right

        if verbose:
            print("[sim] cell: bases {:.3f} m apart (yaw {:.1f} deg), table_z {:+.3f} m "
                  "-- MEASURED by hand-eye".format(
                      self.separation, np.degrees(self.base_yaw), self.table_z))
            print("[sim] camera at ({:+.3f}, {:+.3f}, {:+.3f}) m, {:.3f} m above the "
                  "table -- calibrated".format(
                      self.T_left_cam[0, 3], self.T_left_cam[1, 3],
                      self.T_left_cam[2, 3], self.T_left_cam[2, 3] - self.table_z))
            print("[sim] intrinsic {}".format(self._full_intr))
            print("[sim] crop {} -- centred between the arms, x [{:+.3f}, {:+.3f}] "
                  "y [{:+.3f}, {:+.3f}] m".format(
                      self.crop, self.separation / 2 - self.crop.size_m / 2,
                      self.separation / 2 + self.crop.size_m / 2,
                      -self.crop.size_m / 2, self.crop.size_m / 2))
            print(describe_orientation(self.T_left_cam, self.intr,
                                       self.separation, self.table_z)[1])

    # -- physics -------------------------------------------------------
    def command_joints(self, side, q):
        """Record where an arm should be. The write happens in ``advance``.

        ⚠️ It must NOT write qpos here. ``advance`` derives the arm's qvel from
        the difference between the command and where the arm currently is, so
        applying the command early makes that difference identically zero -- the
        arms then teleport through the swing with a reported velocity of 0, the
        grasp constraint sees a stationary gripper, and the cloth is left hanging
        in mid-air behind the fling.
        """
        self._q_cmd[side] = np.asarray(q, dtype=float).copy()

    def steps_for(self, dist, speed=None):
        """Physics steps for a Cartesian move of ``dist`` metres at ``speed`` m/s.

        Real time, not a fixed number of frames: the cloth's response to a fling
        depends on how fast the fling actually is, so a swing has to take the
        seconds it would take.
        """
        speed = C.XARM_MOVE_SPEED if not speed else float(speed)
        return int(np.clip(np.ceil(dist / max(speed, 1e-3) / self.dt), 1, 4000))

    def advance(self, n=1):
        """Step the physics, holding the arms where they were commanded.

        The arms are kinematic, so their qpos is rewritten after every step and
        their qvel is set from the commanded difference -- without that the grasp
        constraint sees a stationary gripper and the cloth is left behind by the
        swing.
        """
        with self._lock:
            for _ in range(int(n)):
                for side, q in self._q_cmd.items():
                    arm = getattr(self, side)
                    prev = np.array(self.data.qpos[arm.qpos_adr], dtype=float)
                    self.data.qvel[arm.dof_adr] = (q - prev) / self.dt
                    self.data.qpos[arm.qpos_adr] = q
                self._mj.mj_step(self.model, self.data)
                # Put the arms back exactly where they were told to be: the step
                # will have let gravity and the cloth's reaction move them, and a
                # kinematic arm does not sag.
                for side, q in self._q_cmd.items():
                    self.data.qpos[getattr(self, side).qpos_adr] = q
            self._mj.mj_forward(self.model, self.data)
            # ⚠️ ONLY FROM THE MAIN THREAD. launch_passive drives GLFW/OpenGL, and
            # the primitives run the two arms in worker threads (every scene
            # both_* method) -- calling sync() from one of those
            # is a GL call off the context's thread, which takes the process out
            # with a segfault at some arbitrary later point. Worker-thread motion
            # simply catches up at the next main-thread step.
            if (self.viewer is not None
                    and threading.current_thread() is threading.main_thread()):
                self.viewer.sync()
                if self.gui_delay:
                    time.sleep(self.gui_delay)

    # -- grasping ------------------------------------------------------
    def cloth_vertices(self):
        """Cloth vertex positions in the world (= LEFT base) frame, (N, 3)."""
        return np.array(self.data.xpos[self.info['cloth_bodies']], dtype=float)

    def gripper_tip(self, side):
        """Where the fingertips are, in the world frame."""
        arm = getattr(self, side)
        R = self.data.xmat[arm.eef_body].reshape(3, 3)
        return (np.array(self.data.xpos[arm.eef_body])
                + R @ np.array([0.0, 0.0, self.info['gripper_offset']]))

    def grasp(self, side, max_dist=0.06):
        """Pin the nearest cloth vertex to this gripper.

        Nearest rather than "the vertex under the clicked pixel" because by the
        time the gripper closes the cloth has been approached and possibly nudged,
        and what the gripper can hold is what is in front of it. ``max_dist``
        keeps a gripper closing on empty table from grabbing cloth a hand away --
        the same silent failure the effort probe produced on hardware.
        """
        verts = self.cloth_vertices()
        if len(verts) == 0:
            return False
        tip = self.gripper_tip(side)
        d = np.linalg.norm(verts - tip, axis=1)
        i = int(np.argmin(d))
        if d[i] > max_dist:
            if self.verbose:
                print("[sim] {} gripper closed on nothing: nearest cloth vertex is "
                      "{:.3f} m away".format(side, d[i]))
            return False
        eq = self.info['eq_id'][side]
        with self._lock:
            self.model.eq_obj2id[eq] = self.info['cloth_bodies'][i]
            self.data.eq_active[eq] = 1
            self._mj.mj_forward(self.model, self.data)
        return True

    def release(self, side):
        with self._lock:
            self.data.eq_active[self.info['eq_id'][side]] = 0
        return True

    def cloth_centroid(self):
        v = self.cloth_vertices()
        return v.mean(axis=0) if len(v) else np.full(3, np.nan)

    # -- camera --------------------------------------------------------
    def _renderer_for(self):
        if self._renderer is None:
            W, H = self.info['image_size']
            self._renderer = self._mj.Renderer(self.model, H, W)
        return self._renderer

    def take_rgbd(self):
        """(rgb uint8 NxNx3, depth, cloth mask uint8) -- the SQUARE window between
        the arms, paired with the shifted ``self.intr``, as the production scene
        returns."""
        r = self._renderer_for()
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(self.data, camera=self.info['camera_name'])
        rgb = r.render()

        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.info['camera_name'])
        depth = r.render()
        r.disable_depth_rendering()

        r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self.info['camera_name'])
        seg = r.render()
        r.disable_segmentation_rendering()

        # The cloth mask is the flex's own segmentation id. This is exact and free;
        # it is also why the model sets offsamples="0" -- with multisampling the id
        # buffer is blended at every edge and the mask comes back speckled with
        # cloth pixels that are bare table.
        mask = np.zeros(seg.shape[:2], dtype=np.uint8)
        mask[(seg[:, :, 1] == int(self._mj.mjtObj.mjOBJ_FLEX))
             & (seg[:, :, 0] == self.info['cloth_flex_id'])] = 255

        return (np.ascontiguousarray(self.crop.apply(rgb)),
                self.crop.apply(depth),
                np.ascontiguousarray(self.crop.apply(mask)))

    def get_workspace_masks(self):
        return None, None

    def get_T_base_cam(self):
        return self.T_left_cam

    # -- motion (the XArmDualArmScene surface) -------------------------
    def both_movel(self, left_pose, right_pose, speed=None, acc=None, blocking=True,
                   record=False):
        """Both arms in lockstep, so a dual move looks like a dual move."""
        lp = np.asarray(left_pose, dtype=float)
        rp = np.asarray(right_pose, dtype=float)
        lp = lp.reshape(1, -1) if lp.ndim == 1 else lp
        rp = rp.reshape(1, -1) if rp.ndim == 1 else rp
        n = max(len(lp), len(rp))
        traj = {'ur5e': [], 'ur16e': []}
        for i in range(n):
            lt = lp[min(i, len(lp) - 1)]
            rt = rp[min(i, len(rp) - 1)]
            self._step_pair(lt, rt, traj if record else None, speed,
                            "movel[{}]".format(i))
        self.last_trajectory = traj if record else None
        return True

    def _step_pair(self, left_target, right_target, traj, speed, label):
        ls, rs = self.left.get_tcp_pose(), self.right.get_tcp_pose()
        dist = max(float(np.linalg.norm(left_target[:3] - ls[:3])),
                   float(np.linalg.norm(right_target[:3] - rs[:3])))
        steps = self.steps_for(dist, speed)
        for s in range(1, steps + 1):
            a = s / float(steps)
            for arm, start, target in ((self.left, ls, left_target),
                                       (self.right, rs, right_target)):
                arm.set_joints(arm.solve_ik(_interp_pose(start, target, a)))
            self.advance()
            for arm in (self.left, self.right):
                arm.log_pose()
            if traj is not None:
                traj['ur5e'].append(self.left.get_tcp_pose()[:3])
                traj['ur16e'].append(self.right.get_tcp_pose()[:3])
        self.left._record_quality(left_target, label)
        self.right._record_quality(right_target, label)

    def both_fling(self, left_path, right_path, speed=None, acc=None, record=False):
        r = self.both_movel(left_path[0], right_path[0], speed=speed, acc=acc)
        if not r:
            return False
        return self.both_movel(left_path[1:], right_path[1:], speed=speed, acc=acc,
                               record=record)

    def both_home(self, speed=None, acc=None, blocking=True):
        return self._both_movej([np.asarray(self.left.home_joint, dtype=float),
                                 np.asarray(self.right.home_joint, dtype=float)])

    def both_out_scene(self, speed=None, acc=None, blocking=True, steps=40):
        """Both arms swing to their own left, in step, so the photo is unobstructed."""
        return self._both_movej([np.asarray(self.left.photo_joint(), dtype=float),
                                 np.asarray(self.right.photo_joint(), dtype=float)],
                                steps=steps)

    def _both_movej(self, targets, steps=60):
        starts = [self.left.get_joints(), self.right.get_joints()]
        for s in range(1, int(steps) + 1):
            a = s / float(steps)
            for arm, start, target in zip((self.left, self.right), starts, targets):
                arm.set_joints(start * (1.0 - a) + target * a)
            self.advance()
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
        self.advance(5)
        return True

    def both_close_gripper(self):
        self.left.close_gripper()
        self.right.close_gripper()
        self.advance(5)
        return True

    def get_tcp_distance(self):
        left_tcp = self.left.get_tcp_pose()
        right_tcp = transform_pose(self.T_left_right, self.right.get_tcp_pose())
        return float(np.linalg.norm((right_tcp - left_tcp)[:3]))

    # -- diagnostics ---------------------------------------------------
    def contacts_between_arms(self):
        """Contacts whose two geoms belong to DIFFERENT arms."""
        left = set(self.info['arm_geoms']['left'].tolist())
        right = set(self.info['arm_geoms']['right'].tolist())
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in left and g2 in right) or (g1 in right and g2 in left):
                out.append((g1, g2, float(c.dist)))
        return out

    def quality_report(self):
        return {
            'max_ik_error_left': self.left.max_ik_error,
            'max_ik_error_right': self.right.max_ik_error,
            'ik_failures': self.left.ik_failures + self.right.ik_failures,
            'path_samples': len(self.left.path) + len(self.right.path),
            'limit_violations': self.left.limit_violations + self.right.limit_violations,
            'reconfigurations_left': list(self.left.reconfigurations),
            'reconfigurations_right': list(self.right.reconfigurations),
            'joint_margins_left': self.left.joint_margins(),
            'joint_margins_right': self.right.joint_margins(),
            'max_joint_jump_left': self.left.max_joint_jump,
            'max_joint_jump_right': self.right.max_joint_jump,
        }

    # -- lifecycle -----------------------------------------------------
    def _open_viewer(self):
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False)
        except Exception as exc:                # no display, no GLFW, headless box
            print("[sim] no interactive viewer ({}); running without one.".format(exc))
            self.viewer = None

    def close(self):
        if self._renderer is not None:
            # Explicit: leaving it to __del__ makes EGL throw a teardown traceback
            # after the process has otherwise finished cleanly.
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


# ----------------------------------------------------------------------
class XArmMujocoSingleScene:
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

    def movel(self, pose, speed=None, acc=None, blocking=True, record=False):
        return self.arm.movel(pose, speed=speed, acceleration=acc, blocking=blocking)

    def home(self, speed=None, acc=None, blocking=True):
        return self.arm.home()

    def out_scene(self, speed=None, acc=None, blocking=True):
        return self.arm.out_scene()

    def open_gripper(self):
        return self.arm.open_gripper()

    def close_gripper(self):
        return self.arm.close_gripper()

    def take_rgbd(self):
        return self.dual.take_rgbd()

    def get_T_base_cam(self):
        return self.T_cam


def project_path(points, T_cam, intr):
    """Base-frame path -> pixels, for drawing the executed motion on the frame."""
    return base_to_pixel(np.asarray(points, dtype=float), T_cam, intr)
