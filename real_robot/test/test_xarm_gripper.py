"""Test the UFACTORY Gripper for Lite 6 on one or both arms. NO ARM MOTION.

The gripper is PNEUMATIC. In the SDK (``xarm/x3/gripper.py``) the three calls are
nothing but tool-GPIO writes to a solenoid valve::

    open_lite6_gripper()  -> DO0=1, DO1=0
    close_lite6_gripper() -> DO0=0, DO1=1
    stop_lite6_gripper()  -> DO0=0, DO1=0

so it needs air pressure, correct tool-IO wiring, and firmware >= 1.10.0.

``stop`` drives BOTH lines low, and whether the fingers keep holding after that
depends on the valve: a double-solenoid valve latches, a spring-return one vents.
``XArmLite6.close_gripper()`` used to call stop after closing, so on this cell it
released the garment the instant the call returned; it now leaves the solenoid
driven, which is correct either way. Stage 3 still identifies the valve type,
because that decides whether ``stop_gripper()`` is safe while something is held.

Safe to run with the arm parked -- it never commands a motion.

Usage
    source ./setup.sh xarm
    python real_robot/test/test_xarm_gripper.py --arm both
    python real_robot/test/test_xarm_gripper.py --arm left --cycles 10
"""
import argparse
import os
import sys
import time

from real_robot.utils.term import green, red, yellow
from real_robot.robot.xarm_lite6 import XArmLite6
from real_robot.test.test_xarm_lite6_bringup import (
    ERROR_HINTS, preflight, print_network_help, confirm,
)


MIN_FIRMWARE = (1, 10, 0)
DO_STATES = [("open ", 1, 0), ("close", 0, 1), ("stop ", 0, 0)]


def _parse_fw(text):
    """'v2.2.2' / '2.3.0' -> (2, 2, 2); None if it cannot be parsed."""
    if not text:
        return None
    digits = ''.join(c if (c.isdigit() or c == '.') else ' ' for c in str(text))
    for token in digits.split():
        parts = token.split('.')
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            return tuple(int(p) for p in parts[:3])
    return None


def ask(question, auto_yes, default=True):
    """A yes/no the operator must answer by looking at the hardware."""
    if auto_yes:
        print("  ? {}  -> assuming {} [--yes]".format(question, "yes" if default else "no"))
        return default
    while True:
        reply = input("  ? {} [y/n] ".format(question)).strip().lower()
        if reply in ('y', 'yes'):
            return True
        if reply in ('n', 'no'):
            return False


# ----------------------------------------------------------------------
def stage_report(name, driver):
    """Firmware, controller health and the current tool-IO state."""
    print("\n--- [{}] report ---".format(name))
    arm = driver.arm
    fw = arm.version
    print("  firmware        : {}".format(fw))
    parsed = _parse_fw(fw)
    if parsed is None:
        print("  !! could not parse the firmware string; the Lite6 gripper API needs "
              ">= {}.".format('.'.join(map(str, MIN_FIRMWARE))))
    elif parsed < MIN_FIRMWARE:
        print("  !! firmware {} is older than {} -- the Lite6 gripper API is unavailable."
              .format(parsed, MIN_FIRMWARE))
        return False

    if arm.error_code:
        hint = ERROR_HINTS.get(arm.error_code, "look it up in UFACTORY Studio")
        print("  !! controller error {}: {}".format(arm.error_code, hint))
        return False
    print("  error/warn      : {} / {}".format(arm.error_code, arm.warn_code))

    # OUTPUT, not get_tgpio_digital: that reads the tool digital INPUTS, which say
    # nothing about what the solenoids are being driven with.
    code, io = arm.get_tgpio_output_digital()
    print("  tool digital OUT: code={} state={}".format(code, io))
    if code != 0:
        print(red("  !! cannot read the tool IO -- check the end-effector cable."))
        return False
    print("\n  Before continuing: is the air supply ON and connected to the gripper?")
    return True


def stage_raw_io(name, driver, dwell, auto_yes):
    """Drive DO0/DO1 directly, bypassing the driver.

    Separates "the code is wrong" from "there is no air / the tool IO is not
    wired". If nothing moves here, no amount of driver work will help.
    """
    print("\n--- [{}] stage 1: raw tool-IO ---".format(name))
    if not confirm("Watch the fingers; the raw DO lines will be toggled", auto_yes):
        return None
    arm = driver.arm
    seen = []
    for label, d0, d1 in DO_STATES:
        arm.set_tgpio_digital(0, d0)
        arm.set_tgpio_digital(1, d1)
        time.sleep(dwell)
        code, io = arm.get_tgpio_output_digital()
        print("  {} -> DO0={} DO1={}   readback: code={} {}".format(label, d0, d1, code, io))
        seen.append(io)
    arm.set_tgpio_digital(0, 0)
    arm.set_tgpio_digital(1, 0)

    moved = ask("Did the fingers actually move during that sequence?", auto_yes)
    if not moved:
        print(red("  !! No movement with the DO lines confirmed switching means the fault is"))
        print("     downstream of the controller: air supply off, air line not connected,")
        print("     or the solenoid valve not wired to tool DO0/DO1.")
    return moved


def stage_driver_cycle(name, driver, cycles, auto_yes):
    """Time XArmLite6.open_gripper()/close_gripper() so sleep_time can be tuned."""
    print("\n--- [{}] stage 2: driver open/close x{} ---".format(name, cycles))
    if not confirm("Cycle the gripper through the driver", auto_yes):
        return None
    for i in range(cycles):
        t0 = time.time()
        driver.open_gripper()
        t1 = time.time()
        driver.close_gripper()
        t2 = time.time()
        print("  cycle {}/{}: open {:.2f} s, close {:.2f} s".format(
            i + 1, cycles, t1 - t0, t2 - t1))
    full = ask("Did the fingers fully open AND fully close every cycle?", auto_yes)
    if not full:
        print(red("  !! Increase sleep_time in XArmLite6.open_gripper/close_gripper (currently"))
        print("     0.6 s), or raise the air pressure -- the valve is not being given long")
        print("     enough to complete its stroke.")
    return full


def stage_hold(name, driver, hold_seconds, auto_yes):
    """Is the valve bistable (latches) or spring-return (vents when de-energised)?

    The driver no longer depends on the answer -- close_gripper() keeps the solenoid
    driven, which works for either type. (It used to call stop_lite6_gripper()
    afterwards, which on a spring-return valve released the grasp the instant the
    call returned.) This stage still tells you which valve you have, and therefore
    whether stop_gripper() is safe to call while something is held.
    """
    print("\n--- [{}] stage 3: does the grip HOLD after stop? ---".format(name))
    print("  stop_lite6_gripper() drives both DO lines low. A bistable valve latches")
    print("  and keeps holding; a spring-return valve vents and drops the load. The")
    print("  driver holds the coil either way, so this is about stop_gripper() only.")
    if not confirm("Close the gripper, then stop it, and watch for {:.0f} s"
                   .format(hold_seconds), auto_yes):
        return None

    arm = driver.arm
    arm.set_tgpio_digital(0, 0)
    arm.set_tgpio_digital(1, 1)          # close, and keep driving
    time.sleep(1.0)
    closed_driven = ask("Are the fingers closed now (still being driven)?", auto_yes)

    driver.arm.stop_lite6_gripper()      # both lines low
    print("  stop_lite6_gripper() sent; waiting {:.0f} s...".format(hold_seconds))
    time.sleep(hold_seconds)
    still_closed = ask("Are the fingers STILL closed after the stop?", auto_yes)

    print("\n  VERDICT for {}:".format(name))
    if not closed_driven:
        print("    inconclusive -- the gripper never closed under drive. Fix stage 1/2 first.")
        return None
    if still_closed:
        print(green("    HOLDS -- a bistable valve. stop_gripper() is safe even with a garment"))
        print("    held, and the coil need not stay energised during a long carry.")
    else:
        print(yellow("    RELEASES -- a spring-return valve. NEVER call stop_gripper() while"))
        print("    something is held: it vents and drops the load. The driver's")
        print("    close_gripper() keeps the coil driven, so grasps are fine.")
    arm.set_tgpio_digital(0, 1)
    arm.set_tgpio_digital(1, 0)
    time.sleep(0.6)
    arm.stop_lite6_gripper()
    return still_closed


def stage_fabric(name, driver, hold_seconds, auto_yes):
    """Can it actually grip a garment, and keep gripping it?"""
    print("\n--- [{}] stage 4: fabric grip ---".format(name))
    if not confirm("Place a piece of the garment fabric between the fingers", auto_yes):
        return None
    driver.open_gripper()
    time.sleep(0.5)
    if not confirm("Fabric in place? The gripper will close on it", auto_yes):
        return None

    driver.arm.set_tgpio_digital(0, 0)
    driver.arm.set_tgpio_digital(1, 1)
    time.sleep(1.0)
    gripped = ask("Is the fabric held? (tug it gently -- does it stay?)", auto_yes)
    if gripped:
        driver.arm.stop_lite6_gripper()
        time.sleep(hold_seconds)
        still = ask("After stop_lite6_gripper(), is the fabric STILL held?", auto_yes)
        print("    fabric grip survives stop: {}".format("YES" if still else "NO"))
    else:
        still = False
        print(red("  !! The gripper cannot hold this fabric. Check air pressure, and whether"))
        print("     the fingertips suit cloth (flat fingers slide off a single layer).")

    driver.open_gripper()
    return gripped and still


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Test the pneumatic UFACTORY Lite 6 gripper. No arm motion.")
    ap.add_argument('--left-ip', default=os.environ.get('XARM_LEFT_IP', '192.168.1.155'))
    ap.add_argument('--right-ip', default=os.environ.get('XARM_RIGHT_IP', '192.168.1.170'))
    ap.add_argument('--arm', choices=['left', 'right', 'both'], default='both')
    ap.add_argument('--cycles', type=int, default=3, help="driver open/close cycles")
    ap.add_argument('--dwell', type=float, default=1.5, help="seconds per raw IO state")
    ap.add_argument('--hold-seconds', type=float, default=5.0,
                    help="how long to watch for a released grip")
    ap.add_argument('--skip-fabric', action='store_true', help="skip the fabric stage")
    ap.add_argument('--yes', action='store_true', help="don't prompt; assume 'yes' answers")
    args = ap.parse_args()

    names = ['left', 'right'] if args.arm == 'both' else [args.arm]
    ips = {'left': args.left_ip, 'right': args.right_ip}

    print("=" * 72)
    print("UFACTORY Lite 6 gripper test (pneumatic; tool DO0/DO1 -> solenoid valve)")
    print("  arms: " + ", ".join("{} @ {}".format(n, ips[n]) for n in names))
    print("  The arm does NOT move. Keep hands clear of the FINGERS.")
    print("  Air supply must be on and connected.")
    print("=" * 72)

    drivers = {}
    verdicts = {}
    try:
        for n in names:
            ok, err = preflight(ips[n])
            if not ok:
                print_network_help(ips[n], err)
                return 1
            # gripper='lite6' so the driver's own pneumatic init runs and is tested.
            drivers[n] = XArmLite6(ips[n], gripper='lite6')

        for n in names:
            d = drivers[n]
            if not stage_report(n, d):
                print("  [{}] skipping the remaining stages.".format(n))
                verdicts[n] = None
                continue
            if stage_raw_io(n, d, args.dwell, args.yes) is False:
                print("  [{}] raw IO produced no movement; skipping the rest.".format(n))
                verdicts[n] = None
                continue
            stage_driver_cycle(n, d, args.cycles, args.yes)
            verdicts[n] = stage_hold(n, d, args.hold_seconds, args.yes)
            if not args.skip_fabric:
                stage_fabric(n, d, args.hold_seconds, args.yes)

        print("\n" + "=" * 72)
        print("SUMMARY -- does the grip survive stop_lite6_gripper()?")
        for n in names:
            v = verdicts.get(n)
            print("  {:<6s}: {}".format(
                n, "HOLDS" if v else ("RELEASES" if v is False else "not determined")))
        if any(v is False for v in verdicts.values()):
            print("\n  At least one gripper RELEASES. XArmLite6.close_gripper() needs a hold")
            print("  path before any primitive is run with a garment.")
        print("=" * 72)
        return 0

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        return 130
    finally:
        for d in drivers.values():
            try:
                d.arm.stop_lite6_gripper()
            except Exception:
                pass
            try:
                d.disconnect()
            except Exception:
                pass


if __name__ == '__main__':
    sys.exit(main())
