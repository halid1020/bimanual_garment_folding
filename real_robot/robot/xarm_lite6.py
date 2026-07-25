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
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.utils.xarm_constants import (
    XARM_HOME_JOINT, XARM_OUT_SCENE_JOINT,
    XARM_JOINT_SPEED, XARM_JOINT_ACC, XARM_BLEND_RADIUS,
    XARM_COLLISION_SENSITIVITY,
)


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
                 collision_sensitivity=XARM_COLLISION_SENSITIVITY):
        # Lazy import so this module (and registration) imports even if the xArm
        # SDK is not installed; mirrors how ur.py assumes ur_rtde is present only
        # when a UR arm is actually constructed.
        from xarm.wrapper import XArmAPI

        self.ip = ip
        self.gripper_type = gripper
        self.home_joint = list(home_joint) if home_joint is not None else list(XARM_HOME_JOINT)
        self.out_scene_joint = list(out_scene_joint) if out_scene_joint is not None else list(XARM_OUT_SCENE_JOINT)

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
        code = self.arm.set_servo_angle(
            angle=list(np.asarray(q, dtype=float)), is_radian=True,
            speed=speed, mvacc=acceleration, wait=blocking)
        return self._ok(code, 'movej')

    def movel(self, p, speed=1.5, acceleration=1.0, blocking=True,
              avoid_singularity=False, blend_radius=XARM_BLEND_RADIUS):
        """Linear Cartesian move. ``p`` is a single UR pose or a list/array of
        poses (a trajectory); speeds are m/s and m/s^2 (converted to mm here).
        ``avoid_singularity`` is accepted for signature-compatibility with
        UR_RTDE but is a no-op on the Lite 6 (no shoulder-singularity cylinder).
        """
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            p = p.reshape(1, -1)

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
            code = self.arm.set_position(
                x=xarm_pose[0], y=xarm_pose[1], z=xarm_pose[2],
                roll=xarm_pose[3], pitch=xarm_pose[4], yaw=xarm_pose[5],
                speed=speed_mm, mvacc=acc_mm, radius=radius,
                is_radian=True, wait=wait)
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
            print(f"[XArmLite6] {what} returned code {code}")
            try:
                self.arm.clean_error()
                self.arm.clean_warn()
                self.arm.set_state(0)
            except Exception:
                pass
            return False
        return True
