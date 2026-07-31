"""Driver for a single UFACTORY xArm Lite 6, presented as a drop-in replacement
for ``UR_RTDE`` (real_robot/robot/ur.py).

The scenes and primitives elsewhere in this package speak the UR convention:
poses are ``[x, y, z, rx, ry, rz]`` with translation in METRES and orientation as
an axis-angle (Rodrigues) rotvec, and speeds in m/s. This class exposes exactly
that interface and performs ALL unit/convention conversion internally, so no
downstream geometry code needs to change:

    metres  <-> millimetres           (x1000)
    rotvec  <-> roll/pitch/yaw radians (scipy 'xyz' Euler; see note below)
    m/s     <-> mm/s                   (x1000)

⚠️ EULER-ORDER NOTE: the xArm SDK's ``set_position``/``get_position`` express
orientation as roll/pitch/yaw. We map rotvec <-> RPY with scipy's intrinsic
'xyz' order, which matches UFACTORY's documented convention. The driver is
self-consistent by construction; the *physical* correctness of this mapping must
be confirmed with a ``get_position`` round-trip on hardware during bring-up.

The Lite 6 has NO force/torque sensor, so ``get_tcp_force``/``get_tcp_speed``
return zeros and ``start_force_mode`` is unsupported. The xArm primitives descend
to a fixed calibrated table height instead of probing for contact, and rely on
the controller's collision detection as the safety stop.

VIRTUAL WALLS: every arm is confined to a box around the table (``XARM_WALLS``,
transformed into this arm's base frame by ``side``). Enforced twice:
  * the controller's own safety boundary (``set_reduced_tcp_boundary`` +
    ``set_fence_mode``), programmed at connect. No code path can bypass it --
    not UFACTORY Studio, not free-drive teaching, not a raw ``arm.set_position``.
  * a check here on every waypoint, so a violation is REJECTED with a message
    naming the wall instead of surfacing as a bare controller error 35 with the
    arm halted mid-motion.
Note this deliberately does not touch ``set_reduced_mode``: that is a separate
flag which would also cap TCP and joint speed, throttling the fling.

The controller fence PERSISTS across disconnects, process exits and power cycles,
so this driver reconciles it on every connect rather than only programming it when
walls are enabled -- ``walls=False`` actively clears it. Otherwise a boundary from
an earlier run stays armed and invisible, and while the TCP sits outside it the
controller refuses every move in any direction with error 35.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.utils.xarm_constants import (
    XARM_HOME_JOINT, XARM_OUT_SCENE_JOINT,
    XARM_JOINT_SPEED, XARM_JOINT_ACC, XARM_BLEND_RADIUS,
    XARM_COLLISION_SENSITIVITY, XARM_GEOMETRY_VERIFIED,
)
from real_robot.utils.xarm_walls import (
    AXES, walls_for_side, check_pose, to_sdk_boundary_mm, describe,
)


# SDK return codes (xarm.x3.code.APIState). These are the RETURN value of a call,
# and are a different thing from the controller's latched error_code.
API_CODE_HINTS = {
    -1: "not connected",
    -2: "arm not ready (motion_enable / set_state may have failed)",
    -3: "SDK exception",
    -4: "command does not exist on this firmware",
    -6: "TCP limit: the target is outside the Cartesian limits",
    -7: "joint limit: the target needs a joint beyond its range",
    -8: "out of range",
    -9: "EMERGENCY STOP: the arm went to state 4 (STOP) during the move",
    -10: "servo does not exist",
    -11: "conversion failed",
}

# Controller-latched error codes (a different namespace from the return codes above).
CONTROLLER_ERROR_HINTS = {
    1: "emergency stop button pressed",
    2: "emergency IO of the control box triggered (check the EI terminals)",
    3: "emergency stop on the three-state enabling switch",
    22: "self-collision detected",
    23: "joint angle exceeds limit",
    24: "speed exceeds limit",
    25: "planning error",
    31: "collision detected (abnormal joint current)",
    35: "safety boundary limit",
    40: "no valid inverse-kinematics solution for the commanded pose",
}


def rotvec_to_rpy(rotvec):
    """axis-angle (rad) -> (roll, pitch, yaw) in rad, xArm 'xyz' convention."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_euler('xyz')


def rpy_to_rotvec(rpy):
    """(roll, pitch, yaw) rad -> axis-angle (rad), xArm 'xyz' convention."""
    return Rotation.from_euler('xyz', np.asarray(rpy, dtype=float)).as_rotvec()


class XArmLite6:
    """Single Lite 6 arm + Lite6 pneumatic gripper, UR_RTDE-compatible interface."""

    def __init__(self, ip, gripper='lite6',
                 home_joint=None, out_scene_joint=None,
                 collision_sensitivity=XARM_COLLISION_SENSITIVITY,
                 side='left', walls='auto', separation=None):
        # Lazy import so this module (and registration) imports even if the xArm
        # SDK is not installed; mirrors how ur.py assumes ur_rtde is present only
        # when a UR arm is actually constructed.
        from xarm.wrapper import XArmAPI

        self.ip = ip
        self.gripper_type = gripper
        self.side = side
        # True only when the LAST move was rejected by the virtual walls before
        # anything was sent. A controller rejection or abort leaves it False, so
        # callers can tell "never sent" from "sent and then failed".
        self.last_move_refused = False
        self.home_joint = list(home_joint) if home_joint is not None else list(XARM_HOME_JOINT)
        self.out_scene_joint = list(out_scene_joint) if out_scene_joint is not None else list(XARM_OUT_SCENE_JOINT)

        # walls: 'auto' (default) -> on only once the cell geometry has actually
        # been measured; True -> force on; False -> off; a dict of
        # {'x': (lo, hi), ...} in THIS arm's base frame -> use as given.
        #
        # 'auto' exists because the walls are placed using XARM_BASE_YAW, which is
        # an assumption about how the bases are mounted. Enforcing a box derived
        # from the wrong frame does not protect the table -- it just refuses
        # legitimate moves, and puts the wall somewhere the arm never goes.
        if walls == 'auto':
            if XARM_GEOMETRY_VERIFIED:
                self.bounds = walls_for_side(side, separation=separation)
            else:
                self.bounds = None
                print("[XArmLite6] walls OFF: the cell geometry is not verified yet "
                      "(XARM_GEOMETRY_VERIFIED is False).")
                print("[XArmLite6]   Measure it:  python real_robot/test/test_xarm_teach.py "
                      "--arm both --separation")
                print("[XArmLite6]   then set XARM_BASE_YAW / XARM_BASE_SEPARATION and flip "
                      "the flag in xarm_constants.py.")
        elif walls is True:
            self.bounds = walls_for_side(side, separation=separation)
        elif walls in (False, None):
            self.bounds = None
        else:
            self.bounds = {a: (float(min(v)), float(max(v))) for a, v in walls.items()}

        self.arm = XArmAPI(ip, is_radian=True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)            # 0 = position control
        self.arm.set_state(0)           # 0 = ready
        try:
            self.arm.set_collision_sensitivity(collision_sensitivity)
            self.arm.set_self_collision_detection(True)
        except Exception as e:
            print(f"[XArmLite6] Collision-detection setup warning: {e}")

        # The fence is CONTROLLER-side state: it survives disconnect, process exit
        # and power cycles. Reconcile it to this driver's configuration on every
        # connect, never assume it. Skipping this when walls are off is what left a
        # boundary from an earlier run silently armed, refusing every move with a
        # bare error 35 while the driver reported "walls OFF".
        self._reconcile_fence()
        if self.bounds is not None:
            self._warn_if_outside_walls()

        # Lite6 gripper is a pneumatic open/close end-effector.
        if gripper == 'lite6':
            try:
                self.arm.open_lite6_gripper()
                time.sleep(0.5)
                self.arm.stop_lite6_gripper()
            except Exception as e:
                print(f"[XArmLite6] Lite6 gripper init warning: {e}")

        print(f"[XArmLite6] Connected to {ip} (gripper={gripper}).")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def disconnect(self):
        if getattr(self, 'arm', None) is not None:
            self.arm.disconnect()
            self.arm = None

    # ------------------------------------------------------------------
    # Virtual walls
    # ------------------------------------------------------------------
    @staticmethod
    def _code(ret):
        """set_fence_mode returns the whole ret list, unlike its neighbours."""
        return ret[0] if isinstance(ret, (list, tuple)) else ret

    def _live_fence(self):
        """What the CONTROLLER currently thinks, as ``(on, boundary_mm, reduced_on)``.

        Returns ``(None, None, None)`` if it cannot be read -- which must be
        treated as "unknown", never as "off".
        """
        try:
            code, states = self.arm.get_reduced_states()
        except Exception as e:
            print(f"[XArmLite6] !! cannot query the safety boundary: {e}")
            return None, None, None
        if code != 0:
            print(f"[XArmLite6] !! cannot read the safety boundary (code {code}).")
            return None, None, None
        # states: [reduced_mode_on, boundary, max_tcp_speed, max_joint_speed,
        #          joint_ranges, safety_boundary_on, collision_rebound_on]
        on = bool(states[5]) if len(states) > 5 else False
        boundary = list(states[1])[:6] if len(states) > 1 else []
        reduced_on = bool(states[0]) if states else False
        return on, boundary, reduced_on

    def _reconcile_fence(self):
        """Make the controller's fence agree with this driver's `walls` setting.

        The driver's configuration is authoritative. A boundary programmed by an
        earlier run is invisible from here -- it does not appear in any log, and it
        rejects moves that this process believes are unconstrained -- so leaving it
        latched is never the safe option.
        """
        if self.bounds is not None:
            return self._program_fence(True)

        live_on, live_boundary, _ = self._live_fence()
        if live_on:
            print(f"[XArmLite6] !! a safety boundary is ARMED in the {self.side} controller "
                  f"({self.ip}) from an earlier session:")
            print(f"[XArmLite6]      boundary {live_boundary} mm "
                  f"[x_max, x_min, y_max, y_min, z_max, z_min]")
            try:
                pose = self.get_tcp_pose()
                print(f"[XArmLite6]      the TCP is now at "
                      f"({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}) m")
            except Exception:
                pass
            print("[XArmLite6]    Clearing it, because this driver was constructed with the "
                  "walls off.")
            print("[XArmLite6]    (While the TCP sits outside an armed boundary the controller "
                  "refuses")
            print("[XArmLite6]     EVERY move, in any direction, with error 35.)")
        return self._program_fence(False)

    def _program_fence(self, on):
        """Program the controller's safety boundary and switch it on/off.

        Read back with get_reduced_states() rather than trusting the return
        codes: a boundary that silently failed to take is worse than none,
        because it looks protected. Turning the fence OFF needs no boundary, so
        this works with ``self.bounds`` unset.
        """
        try:
            boundary = to_sdk_boundary_mm(self.bounds) if on else None
            if boundary is not None:
                code_b = self._code(self.arm.set_reduced_tcp_boundary(boundary))
            else:
                code_b = 0
            code_f = self._code(self.arm.set_fence_mode(bool(on)))
            if code_b != 0 or code_f != 0:
                print(f"[XArmLite6] !! fence setup returned codes "
                      f"boundary={code_b} fence={code_f}")

            live, got, reduced_on = self._live_fence()
            if live is None:
                print("[XArmLite6] !! the fence state could not be verified; treat the "
                      "controller boundary as UNKNOWN.")
                return False
            if bool(on) != live:
                print(f"[XArmLite6] !! safety boundary is {'ON' if live else 'OFF'} "
                      f"but {'ON' if on else 'OFF'} was requested.")
                return False
            if on and got != boundary:
                print(f"[XArmLite6] !! boundary read back as {got}, expected {boundary}.")
                return False
            if on:
                print(f"[XArmLite6] walls ON ({self.side}): {describe(self.bounds)}")
                print(f"[XArmLite6]   controller fence {boundary} mm "
                      f"(reduced mode {'ON -- SPEEDS ARE CAPPED' if reduced_on else 'off'})")
            else:
                print(f"[XArmLite6] controller fence OFF ({self.side}), verified.")
            return True
        except Exception as e:
            print(f"[XArmLite6] !! fence setup failed: {e}")
            return False

    def _explain_boundary_error(self):
        """Say WHICH boundary error 35 tripped on, and where the TCP actually is.

        Error 35 on its own is unactionable: the boundary lives in the controller
        and may have been armed by an earlier session, so the numbers it is being
        judged against are nowhere in this process.
        """
        print("[XArmLite6]   -> safety boundary limit (controller-side fence)")
        on, boundary, _ = self._live_fence()
        if not boundary or on is None:
            print("[XArmLite6]      ...but the boundary could not be read back.")
            return
        print(f"[XArmLite6]      fence is {'ON' if on else 'OFF'}, boundary {boundary} mm "
              f"[x_max, x_min, y_max, y_min, z_max, z_min]")
        # Same order as to_sdk_boundary_mm: (max, min) per axis, mm -> m.
        live_bounds = {a: (boundary[2 * i + 1] / 1000.0, boundary[2 * i] / 1000.0)
                       for i, a in enumerate(AXES)}
        try:
            pose = self.get_tcp_pose()
        except Exception:
            return
        print(f"[XArmLite6]      TCP is at ({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}) m")
        inside, violations = check_pose(pose, live_bounds)
        if inside:
            print("[XArmLite6]      The TCP is inside that box, so the TARGET or the path "
                  "left it.")
        else:
            print("[XArmLite6]      The TCP is ALREADY outside it, so every move is refused "
                  "in any direction:")
            for v in violations:
                print(f"[XArmLite6]        {v}")
            if self.bounds is None:
                print("[XArmLite6]      This driver has no walls configured, so that fence is "
                      "NOT ours --")
                print("[XArmLite6]      it is left over from another session or from UFACTORY "
                      "Studio.")
                print("[XArmLite6]      Clear it with driver.disable_walls().")

    def _warn_if_outside_walls(self):
        """The arm may already be parked outside the box, which would trip the
        fence on the first move. Say so at connect, rather than mid-primitive."""
        try:
            pose = self.get_tcp_pose()
        except Exception:
            return
        ok, bad = check_pose(pose, self.bounds)
        if not ok:
            print(f"[XArmLite6] !! {self.side} is currently OUTSIDE the walls:")
            for v in bad:
                print(f"[XArmLite6]      {v}")
            print("[XArmLite6]    The first commanded move will be refused, and the "
                  "controller may report error 35.")
            print("[XArmLite6]    Jog it back inside (UFACTORY Studio or free-drive), or "
                  "construct with walls=False.")

    def enable_walls(self):
        if self.bounds is None:
            print("[XArmLite6] no walls configured; construct with walls=True to set them.")
            return False
        return self._program_fence(True)

    def disable_walls(self):
        """Turn OFF both layers, so the software check and the controller fence
        never disagree about what is allowed.

        Runs even when no walls are configured here: the fence may have been armed
        by an earlier session or by UFACTORY Studio, and clearing it is the whole
        point of calling this.
        """
        self.bounds = None
        return self._program_fence(False)

    def _check_joint_bounds(self, q, what):
        """Wall-check a JOINT target by asking the controller where its TCP lands."""
        if self.bounds is None:
            return True
        try:
            code, pose_mm = self.arm.get_forward_kinematics(
                list(np.asarray(q, dtype=float)), input_is_radian=True)
        except Exception as e:
            print(f"[XArmLite6] forward-kinematics wall check unavailable ({e}); "
                  f"relying on the controller fence for this {what}.")
            return True
        if code != 0:
            print(f"[XArmLite6] forward kinematics returned code {code}; cannot wall-check "
                  f"this {what}. Relying on the controller fence.")
            return True
        pose_m = np.asarray(pose_mm[:3], dtype=float) / 1000.0
        return self._check_bounds(pose_m, what)

    def _check_bounds(self, poses, what):
        """Reject (never clamp) any waypoint outside the walls."""
        if self.bounds is None:
            return True
        poses = np.asarray(poses, dtype=float)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        ok_all = True
        for i, p in enumerate(poses):
            ok, bad = check_pose(p, self.bounds)
            if not ok:
                ok_all = False
                print(f"[XArmLite6] !! {what} waypoint {i} for the {self.side} arm is outside "
                      f"the virtual walls -- REFUSED, nothing sent.")
                print(f"[XArmLite6]      xyz = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")
                for v in bad:
                    print(f"[XArmLite6]      {v}")
        return ok_all

    # ------------------------------------------------------------------
    # Motion (UR-convention I/O)
    # ------------------------------------------------------------------
    def _pose_to_xarm(self, pose):
        """UR pose [x,y,z(m), rx,ry,rz(rotvec)] -> [x,y,z(mm), r,p,y(rad)]."""
        pose = np.asarray(pose, dtype=float)
        xyz_mm = (pose[:3] * 1000.0).tolist()
        rpy = rotvec_to_rpy(pose[3:6]).tolist()
        return xyz_mm + rpy

    def home(self, speed=None, acceleration=None, blocking=True):
        return self.movej(self.home_joint, speed=speed, acceleration=acceleration, blocking=blocking)

    def out_scene(self, speed=None, acceleration=None, blocking=True):
        return self.movej(self.out_scene_joint, speed=speed, acceleration=acceleration, blocking=blocking)

    def movej(self, q, speed=None, acceleration=None, blocking=True):
        speed = XARM_JOINT_SPEED if speed is None else speed
        acceleration = XARM_JOINT_ACC if acceleration is None else acceleration
        # Joint targets are wall-checked through forward kinematics, so home() and
        # out_scene() are covered too. Only the endpoint is checked -- the path
        # between joint configurations is not a straight line in Cartesian space,
        # which is what the controller fence is there to catch.
        self.last_move_refused = False
        if not self._check_joint_bounds(q, 'movej'):
            self.last_move_refused = True
            return False
        code = self.arm.set_servo_angle(
            angle=list(np.asarray(q, dtype=float)), is_radian=True,
            speed=speed, mvacc=acceleration, wait=blocking)
        return self._ok(code, 'movej')

    def movel(self, p, speed=1.5, acceleration=1.0, blocking=True,
              avoid_singularity=False, blend_radius=XARM_BLEND_RADIUS,
              motion_type=None):
        """Linear Cartesian move. ``p`` is a single UR pose or a list/array of
        poses (a trajectory); speeds are m/s and m/s^2 (converted to mm here).
        ``avoid_singularity`` is accepted for signature-compatibility with
        UR_RTDE but is a no-op on the Lite 6 (no shoulder-singularity cylinder).

        ``motion_type`` (firmware >= 1.11.100) controls the controller's planner:
        0 = strictly linear (the default, and what garment manipulation wants --
        a straight line between pick and place), 1 = linear where possible and
        joint planning otherwise, 2 = always joint planning. Use 1 only to get an
        arm out of a pose where a straight-line path cannot be planned; it does
        NOT guarantee a straight line, so the TCP may take an unexpected route.
        """
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p.reshape(1, -1)

        # Check the WHOLE trajectory before sending any of it, so a bad waypoint
        # in the middle does not leave the arm stranded part-way through.
        self.last_move_refused = False
        if not self._check_bounds(p, 'movel'):
            self.last_move_refused = True
            return False

        speed_mm = float(speed) * 1000.0
        acc_mm = float(acceleration) * 1000.0
        radius_mm = float(blend_radius) * 1000.0

        n = p.shape[0]
        ok = True
        for i in range(n):
            xarm_pose = self._pose_to_xarm(p[i])
            is_last = (i == n - 1)
            # Blend through intermediate waypoints; stop exactly on the last one.
            radius = -1 if is_last else radius_mm
            wait = blocking if is_last else False
            kwargs = {}
            if motion_type is not None:
                kwargs['motion_type'] = int(motion_type)
            code = self.arm.set_position(
                x=xarm_pose[0], y=xarm_pose[1], z=xarm_pose[2],
                roll=xarm_pose[3], pitch=xarm_pose[4], yaw=xarm_pose[5],
                speed=speed_mm, mvacc=acc_mm, radius=radius,
                is_radian=True, wait=wait, **kwargs)
            ok = self._ok(code, 'movel') and ok
        return ok

    def movej_ik(self, p, speed=1.5, acceleration=1.0, blocking=True):
        # xArm plans the IK internally for a Cartesian target, so this is movel.
        return self.movel(p, speed=speed, acceleration=acceleration, blocking=blocking)

    # ------------------------------------------------------------------
    # Gripper (Lite6 pneumatic: open/close only)
    # ------------------------------------------------------------------
    def open_gripper(self, sleep_time=0.6):
        try:
            self.arm.open_lite6_gripper()
            time.sleep(sleep_time)
            self.arm.stop_lite6_gripper()
        except Exception as e:
            print(f"[XArmLite6] open_gripper warning: {e}")

    def close_gripper(self, sleep_time=0.6):
        try:
            self.arm.close_lite6_gripper()
            time.sleep(sleep_time)
            self.arm.stop_lite6_gripper()
        except Exception as e:
            print(f"[XArmLite6] close_gripper warning: {e}")

    # ------------------------------------------------------------------
    # State readback (UR-convention)
    # ------------------------------------------------------------------
    def get_tcp_pose(self):
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            print(f"[XArmLite6] get_position error code {code}")
        xyz_m = np.asarray(pose[:3], dtype=float) / 1000.0
        rotvec = rpy_to_rotvec(pose[3:6])
        return np.concatenate([xyz_m, rotvec])

    def get_tcp_speed(self):
        # Lite 6 exposes only a scalar realtime speed; the force-mode loops that
        # need a full 6-vector are UR-only and never run on this arm.
        return np.zeros(6, dtype=float)

    def get_tcp_force(self):
        # No F/T sensor on the Lite 6.
        return np.zeros(6, dtype=float)

    def start_force_mode(self):
        raise NotImplementedError(
            "xArm Lite 6 has no force/torque sensor; force mode is unsupported. "
            "The xArm primitives use fixed-height grasps instead.")

    # ------------------------------------------------------------------
    def _ok(self, code, what):
        if code != 0:
            hint = API_CODE_HINTS.get(code)
            print(f"[XArmLite6] {what} returned code {code}"
                  + (f" -- {hint}" if hint else ""))
            # The command WAS sent and the controller aborted it. Read the latched
            # error FRESH before clean_error() wipes it -- that code is the only
            # thing that says WHY, and self.arm.error_code is a cached value that
            # can lag the report stream.
            try:
                ec, ew = 0, 0
                got = self.arm.get_err_warn_code(show=False)
                if isinstance(got, (list, tuple)) and len(got) >= 2 and got[0] == 0:
                    ec, ew = got[1][0], got[1][1]
                print(f"[XArmLite6]   controller error_code={ec}, warn_code={ew}, "
                      f"state={self.arm.state}, mode={self.arm.mode}")
                if ec == 35:
                    self._explain_boundary_error()
                elif ec:
                    hint = CONTROLLER_ERROR_HINTS.get(ec)
                    print(f"[XArmLite6]   -> {hint}" if hint
                          else "[XArmLite6]   -> look this code up in UFACTORY Studio.")
                elif code == -9:
                    # state 4 with NO latched error: the controller stopped the
                    # motion rather than faulting. Note this branch only runs when
                    # the error really was read as 0 -- a boundary violation
                    # latches 35 and is handled above.
                    print("[XArmLite6]   No error was latched, so this is not an e-stop. The")
                    print("[XArmLite6]   controller stopped rather than faulting. Candidates:")
                    print("[XArmLite6]     * no straight-line path from this pose (wrist")
                    print("[XArmLite6]       singularity, joint 5 near +/-90 deg, or a joint limit)")
                    print("[XArmLite6]     * a safety boundary armed in the controller (check")
                    print("[XArmLite6]       stage 0 of the bring-up, which now prints it)")
                    print("[XArmLite6]   Options: move away with a JOINT move first, or retry")
                    print("[XArmLite6]   with motion_type=1 (linear if possible, else joint).")
            except Exception:
                pass
            try:
                self.arm.clean_error()
                self.arm.clean_warn()
                self.arm.set_state(0)
            except Exception:
                pass
            return False
        return True
