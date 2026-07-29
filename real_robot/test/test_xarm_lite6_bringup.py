"""Hardware bring-up smoke test for the dual xArm Lite 6 cell.

FIRST script to run on the new arms. It makes each arm move SLIGHTLY (a few cm /
a few degrees, always relative to wherever the arm currently is) and checks that
``XArmLite6`` speaks to the hardware correctly.

Its real purpose is stage 1: ``XArmLite6`` converts the UR convention used by all
the geometry code (metres + axis-angle rotvec) to the xArm SDK's (mm + RPY),
assuming scipy intrinsic 'xyz' Euler order. That assumption compiles and runs
either way -- it only shows up as wrong ORIENTATIONS at the cloth. Stage 1
settles it before any garment motion.

SAFETY
  * Never calls home()/out_scene(): XARM_HOME_JOINT / XARM_OUT_SCENE_JOINT in
    xarm_constants.py are unverified guesses and could swing an arm into the table.
  * Every motion is a small delta from the current pose, and returns to it.
  * Slow speeds; each stage waits for you to press Enter (unless --yes).
  * Ctrl-C (or any error) stops the arms and disconnects.

Usage
    source ./setup.sh
    # 1) connectivity only, no motion:
    python real_robot/test/test_xarm_lite6_bringup.py --info-only --arm both
    # 2) one arm at a time:
    python real_robot/test/test_xarm_lite6_bringup.py --arm left
    python real_robot/test/test_xarm_lite6_bringup.py --arm right
    # 3) only once both pass, the threaded dual path:
    python real_robot/test/test_xarm_lite6_bringup.py --arm both --dual

Requires the xArm SDK:  pip install xarm-python-sdk
"""
import argparse
import os
import socket
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.robot.xarm_lite6 import XArmLite6
from real_robot.utils.motion_utils import safe_movel
from real_robot.utils.thread_utils import ThreadWithResult


# Common xArm CONTROLLER error codes. Not exhaustive -- UFACTORY Studio shows the
# authoritative human-readable message for whatever the controller reports.
ERROR_HINTS = {
    1: "emergency stop button is pressed in -- twist to release it",
    2: "emergency IO of the control box is triggered -- check the external e-stop "
       "wired to the control box's EI terminals (if you run without one, those "
       "terminals need the shorting link fitted)",
    3: "emergency stop on the three-state enabling switch is pressed",
    11: "servo joint 1 error", 12: "servo joint 2 error", 13: "servo joint 3 error",
    14: "servo joint 4 error", 15: "servo joint 5 error", 16: "servo joint 6 error",
    21: "kinematic error", 22: "self-collision detected",
    23: "joint angle exceeds limit", 24: "speed exceeds limit",
    31: "collision detected (abnormal joint current)",
    35: "safety boundary limit",
}

# Tolerances for the conversion checks.
ROT_TOL_DEG = 1.0      # stage 1: re-commanding the measured pose must not rotate the wrist
POS_TOL_M = 0.005      # stage 1: ...nor translate it
JOG_TOL_M = 0.008      # stage 2: measured jog must match the commanded delta


def _fmt(v, prec=4):
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in np.asarray(v, dtype=float)) + "]"


def _rot_error_deg(rotvec_a, rotvec_b):
    """Angle of the relative rotation between two rotvecs, in degrees."""
    ra = Rotation.from_rotvec(np.asarray(rotvec_a, dtype=float))
    rb = Rotation.from_rotvec(np.asarray(rotvec_b, dtype=float))
    return float(np.degrees(np.linalg.norm((ra.inv() * rb).as_rotvec())))


def preflight(ip, port=502, timeout=2.0):
    """Can we open a TCP socket to the controller? The SDK raises a bare
    'connect socket failed' with a traceback, which hides the usual cause: the
    host has no route onto the robot subnet. Check first and say so plainly.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        s.close()


def _route_hint(ip):
    """Which interface would traffic to `ip` actually leave by?"""
    try:
        out = subprocess.run(['ip', 'route', 'get', ip], capture_output=True,
                             text=True, timeout=3).stdout.strip().splitlines()
        return out[0] if out else None
    except Exception:
        return None


def print_network_help(ip, err):
    route = _route_hint(ip)
    print(f"\n  !! Cannot reach the controller at {ip}: {err}")
    if route:
        print(f"     route: {route}")
        if 'dev lo' in route or 'via' in route and 'wl' in route:
            print("     ^ traffic is leaving over WiFi/the default gateway, NOT a wired link"
                  " to the arm.")
    print("""
     Usual causes, in order:
       1. The wired NIC has no IP on the robot subnet. Check with `ip -br addr`:
          if the ethernet port shows UP but no address, that's it. The xArm
          controllers do not serve DHCP, so a DHCP-configured port stays IP-less.
          Fix (one-off, keeps WiFi as the default route):
            sudo nmcli connection add type ethernet ifname <NIC> con-name xarm-lan \\
                 ipv4.method manual ipv4.addresses 192.168.1.100/24 \\
                 ipv4.never-default yes ipv6.method ignore
            sudo nmcli connection up xarm-lan
       2. Wrong IP. Read the actual address off the controller in UFACTORY Studio
          or the arm's LED display, then pass --left-ip / --right-ip.
       3. Controller not powered / cable in the wrong port / e-stop engaged.
     Verify with:  ping -c2 {ip}
""".format(ip=ip))


def confirm(msg, auto_yes):
    if auto_yes:
        print(f"\n>>> {msg}  [--yes]")
        return True
    reply = input(f"\n>>> {msg}  [Enter = go, 's' = skip, 'q' = quit] ").strip().lower()
    if reply == 'q':
        raise KeyboardInterrupt("Aborted by user.")
    return reply != 's'


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------
def stage_report(name, driver):
    """Stage 0: connect and report state. No motion."""
    print(f"\n--- [{name}] stage 0: state report (no motion) ---")
    sdk = driver.arm

    code, pos_mm = sdk.get_position(is_radian=False)
    print(f"  SDK get_position  (code {code}): "
          f"xyz = {_fmt(pos_mm[:3], 2)} mm, rpy = {_fmt(pos_mm[3:6], 2)} deg")

    code, joints_deg = sdk.get_servo_angle(is_radian=False)
    print(f"  SDK get_servo_angle (code {code}): {_fmt(joints_deg, 2)} deg")

    pose = driver.get_tcp_pose()
    print(f"  Driver get_tcp_pose: xyz = {_fmt(pose[:3])} m, "
          f"rotvec = {_fmt(pose[3:6])} rad")
    print(f"  (hand-guide the TCP to the table and re-run --info-only to read "
          f"XARM_TABLE_Z; z is now {pose[2]:+.4f} m)")

    print(f"  error_code = {sdk.error_code}, warn_code = {sdk.warn_code}, "
          f"mode = {sdk.mode}, state = {sdk.state}")

    healthy = True
    if sdk.error_code:
        healthy = False
        hint = ERROR_HINTS.get(sdk.error_code)
        print(f"  !! controller error {sdk.error_code}"
              + (f": {hint}" if hint else " -- look it up in UFACTORY Studio."))
        print("     The driver already ran clean_error() at connect, so an error that is")
        print("     still present is PHYSICALLY asserted -- clear the cause, then re-run.")
    if sdk.state == 4:
        healthy = False
        print("  !! state = 4 (STOP): the arm is not enabled, so joint/pose readback above")
        print("     may be stale or zeroed -- do not trust it until the arm is ready.")
    if healthy:
        print("  OK: no errors, arm is ready to move.")
    return pose, healthy


def stage_roundtrip(name, driver, speed, acc, auto_yes):
    """Stage 1: re-command the measured pose. THE critical convention check."""
    if not confirm(f"[{name}] stage 1: re-command the CURRENT pose "
                   f"(the arm should NOT visibly move)", auto_yes):
        return True

    p0 = driver.get_tcp_pose()
    print(f"  measured p0    : {_fmt(p0)}")
    driver.movel(p0, speed=speed, acceleration=acc, blocking=True)
    time.sleep(0.3)
    p1 = driver.get_tcp_pose()
    print(f"  after re-command: {_fmt(p1)}")

    dpos = float(np.linalg.norm(p1[:3] - p0[:3]))
    drot = _rot_error_deg(p0[3:6], p1[3:6])
    print(f"  drift: translation {dpos*1000:.2f} mm, rotation {drot:.3f} deg")

    if dpos > POS_TOL_M or drot > ROT_TOL_DEG:
        print("\n  !! FAILED. Re-commanding the pose the driver just measured moved the arm,")
        print("     so the rotvec <-> RPY mapping is not physically correct.")
        print("     Fix `rotvec_to_rpy` / `rpy_to_rotvec` in real_robot/robot/xarm_lite6.py")
        print("     (currently scipy intrinsic 'xyz'); try 'XYZ', 'zyx', or 'ZYX'.")
        print("     Do NOT run any garment motion until this passes.")
        return False

    print("  OK: rotvec<->RPY and the mm scaling round-trip physically.")
    return True


def stage_cartesian(name, driver, delta, speed, acc, auto_yes):
    """Stage 2: small jogs along each base axis, returning to start each time."""
    axis_names = ['+X', '+Y', '+Z']
    # Jog Z first: it moves away from the table, so it is the safest of the three.
    order = [2, 0, 1]
    ok = True

    for i in order:
        if not confirm(f"[{name}] stage 2: jog {delta*100:.0f} cm along base "
                       f"{axis_names[i]}, then back", auto_yes):
            continue

        p0 = driver.get_tcp_pose()
        target = p0.copy()
        target[i] += delta

        driver.movel(target, speed=speed, acceleration=acc, blocking=True)
        time.sleep(0.3)
        p1 = driver.get_tcp_pose()

        measured = p1[:3] - p0[:3]
        err = abs(measured[i] - delta)
        status = "OK " if err <= JOG_TOL_M else "!! "
        print(f"  {status}{axis_names[i]}: commanded {delta:+.3f} m, "
              f"measured {_fmt(measured, 4)} m (axis error {err*1000:.1f} mm)")
        if err > JOG_TOL_M:
            print(f"     Expected ~{delta:+.3f} m on {axis_names[i]}. A ~1000x error means the "
                  f"m<->mm scaling is wrong; motion on a different axis means the base "
                  f"frame is not what the geometry code assumes.")
            ok = False

        driver.movel(p0, speed=speed, acceleration=acc, blocking=True)
        time.sleep(0.3)

    return ok


def stage_joint(name, driver, joint_delta_deg, auto_yes):
    """Stage 3: nudge joint 1 and come back (exercises the set_servo_angle path)."""
    if not confirm(f"[{name}] stage 3: nudge joint 1 by {joint_delta_deg:.0f} deg, "
                   f"then back", auto_yes):
        return True

    code, q0_deg = driver.arm.get_servo_angle(is_radian=False)
    if code != 0:
        print(f"  !! get_servo_angle returned code {code}; skipping.")
        return False
    q0 = np.deg2rad(np.asarray(q0_deg, dtype=float))
    print(f"  start joints: {_fmt(q0_deg, 2)} deg")

    q1 = q0.copy()
    q1[0] += np.deg2rad(joint_delta_deg)
    driver.movej(q1, blocking=True)
    time.sleep(0.3)
    _, q_mid = driver.arm.get_servo_angle(is_radian=False)
    print(f"  jogged joints: {_fmt(q_mid, 2)} deg")

    driver.movej(q0, blocking=True)
    time.sleep(0.3)
    _, q_end = driver.arm.get_servo_angle(is_radian=False)
    print(f"  returned     : {_fmt(q_end, 2)} deg")

    back_err = float(np.max(np.abs(np.asarray(q_end) - np.asarray(q0_deg))))
    if back_err > 0.5:
        print(f"  !! did not return to the start joints (max error {back_err:.2f} deg).")
        return False
    print("  OK: joint move and return.")
    return True


def stage_gripper(name, driver, auto_yes):
    """Stage 4 (opt-in): Lite6 pneumatic gripper open/close."""
    if not confirm(f"[{name}] stage 4: gripper open/close (needs air supply)", auto_yes):
        return True
    print("  closing...")
    driver.close_gripper()
    time.sleep(1.0)
    print("  opening...")
    driver.open_gripper()
    print("  OK: gripper commands sent (confirm visually -- the Lite6 gripper has no feedback).")
    return True


def stage_dual(drivers, delta, speed, acc, auto_yes):
    """Stage 5 (opt-in): both arms jog at once through the threaded path that
    XArmDualArmScene.both_movel uses (no camera / calibration needed)."""
    if not confirm(f"stage 5: BOTH arms jog {delta*100:.0f} cm in +Z simultaneously, "
                   f"then back", auto_yes):
        return True

    starts = {name: d.get_tcp_pose() for name, d in drivers.items()}
    targets = {}
    for name, p0 in starts.items():
        t = p0.copy()
        t[2] += delta
        targets[name] = t

    def run(poses):
        threads = [
            ThreadWithResult(target=safe_movel,
                             args=(drivers[name], poses[name], speed, acc, True, False))
            for name in drivers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return all(getattr(t, 'result', False) for t in threads)

    ok = run(targets)
    time.sleep(0.3)
    for name, d in drivers.items():
        measured = d.get_tcp_pose()[2] - starts[name][2]
        print(f"  {name}: commanded dz {delta:+.3f} m, measured {measured:+.3f} m")

    ok = run(starts) and ok
    print(f"  {'OK' if ok else '!!'}: threaded dual motion "
          f"({'both moves reported success' if ok else 'at least one move failed'}).")
    return ok


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults come from `source ./setup.sh xarm`, which exports these alongside
    # configuring the wired NIC; the literals are the fallback.
    ap.add_argument('--left-ip', default=os.environ.get('XARM_LEFT_IP', '192.168.1.155'))
    ap.add_argument('--right-ip', default=os.environ.get('XARM_RIGHT_IP', '192.168.1.170'))
    ap.add_argument('--arm', choices=['left', 'right', 'both'], default='left',
                    help="which arm(s) to test; motion stages run one arm at a time")
    ap.add_argument('--delta', type=float, default=0.03, help="Cartesian jog, metres")
    ap.add_argument('--joint-delta', type=float, default=5.0, help="joint jog, degrees")
    ap.add_argument('--speed', type=float, default=0.05, help="m/s (deliberately slow)")
    ap.add_argument('--acc', type=float, default=0.2, help="m/s^2")
    ap.add_argument('--yes', action='store_true', help="don't prompt between stages")
    ap.add_argument('--info-only', action='store_true', help="stage 0 only, no motion")
    ap.add_argument('--test-gripper', action='store_true', help="include the gripper stage")
    ap.add_argument('--dual', action='store_true',
                    help="include the simultaneous both-arm stage (requires --arm both)")
    args = ap.parse_args()

    names = ['left', 'right'] if args.arm == 'both' else [args.arm]
    ips = {'left': args.left_ip, 'right': args.right_ip}

    if args.info_only:
        motion_desc = "NONE (--info-only)"
    else:
        motion_desc = (f"{args.delta*100:.0f} cm / {args.joint_delta:.0f} deg jogs "
                       f"at {args.speed} m/s, relative to the current pose")

    print("=" * 70)
    print("  xArm Lite 6 BRING-UP")
    print(f"  arms   : {', '.join(f'{n} @ {ips[n]}' for n in names)}")
    print(f"  motion : {motion_desc}")
    print("  CLEAR THE TABLE around both arms and keep a hand on the e-stop.")
    print("  This script never commands the (unverified) home pose.")
    print("=" * 70)

    if args.dual and args.arm != 'both':
        print("\n--dual requires --arm both; ignoring --dual.")
        args.dual = False

    drivers = {}
    healthy = {}
    all_ok = True
    try:
        for name in names:
            print(f"\n[{name}] connecting to {ips[name]} ...")
            reachable, err = preflight(ips[name])
            if not reachable:
                print_network_help(ips[name], err)
                return 1
            # gripper='none' -> the driver skips the pneumatic init (no air supply yet).
            drivers[name] = XArmLite6(ips[name],
                                      gripper='lite6' if args.test_gripper else 'none')
            _, healthy[name] = stage_report(name, drivers[name])

        if args.info_only:
            print("\n--info-only: done, nothing was moved.")
            return 0 if all(healthy.values()) else 1

        for name in names:
            d = drivers[name]
            if not healthy[name]:
                # Never command an arm the controller has faulted: motion_enable
                # failed, so commands are either refused or act on stale state.
                print(f"\n[{name}] FAULTED -- skipping all motion stages for this arm.")
                all_ok = False
                continue
            if not stage_roundtrip(name, d, args.speed, args.acc, args.yes):
                print(f"\n[{name}] stage 1 failed -- skipping the remaining motion stages.")
                all_ok = False
                continue
            all_ok &= stage_cartesian(name, d, args.delta, args.speed, args.acc, args.yes)
            all_ok &= stage_joint(name, d, args.joint_delta, args.yes)
            if args.test_gripper:
                all_ok &= stage_gripper(name, d, args.yes)

        if args.dual and all_ok:
            all_ok &= stage_dual(drivers, args.delta, args.speed, args.acc, args.yes)
        elif args.dual:
            print("\nSkipping stage 5 (dual): a single-arm stage failed.")

    except KeyboardInterrupt:
        print("\n\n!! Interrupted -- stopping arms.")
        for d in drivers.values():
            try:
                d.arm.set_state(4)
            except Exception:
                pass
        all_ok = False
    except Exception as e:
        print(f"\n\n!! Error: {e}")
        for d in drivers.values():
            try:
                d.arm.set_state(4)
            except Exception:
                pass
        all_ok = False
        raise
    finally:
        for name, d in drivers.items():
            try:
                d.disconnect()
                print(f"[{name}] disconnected.")
            except Exception:
                pass

    print("\n" + "=" * 70)
    print("  RESULT: " + ("all stages passed." if all_ok else "SOMETHING FAILED -- see above."))
    if all_ok and not args.info_only:
        print("  Next: calibrate XARM_TABLE_Z / XARM_GRIPPER_OFFSET, then tune")
        print("  XARM_HOME_JOINT / XARM_OUT_SCENE_JOINT in real_robot/utils/xarm_constants.py.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
