"""Offline checks for the xArm virtual walls. NO HARDWARE NEEDED.

Runs ``XArmLite6`` against a fake controller that records every command, so we
can assert what the driver would actually send. Covers the things that are hard
to verify on hardware without deliberately crashing an arm:

  * the controller fence is programmed and read back at connect;
  * ``set_reduced_mode`` is NOT touched (it would cap speed and throttle the fling);
  * an out-of-box ``movel`` is refused and sends NOTHING;
  * a multi-waypoint ``movel`` with one bad waypoint sends nothing at all,
    rather than stranding the arm part-way;
  * ``movej`` / ``home`` are wall-checked through forward kinematics;
  * ``disable_walls()`` turns off both layers together;
  * a fence left ARMED in the controller by an earlier session is cleared when
    this driver is constructed with the walls off (that boundary is invisible from
    here and refuses every move with error 35 while the TCP sits outside it).

Usage
    python real_robot/test/test_xarm_walls_offline.py
"""
import contextlib
import sys
import types

import numpy as np

from real_robot.utils import xarm_constants as C
from real_robot.utils.term import green, red
from real_robot.utils.xarm_walls import walls_for_side, to_sdk_boundary_mm


class FakeXArmAPI:
    """Records commands; answers state queries with whatever the test set up."""

    # Set to a boundary list to simulate a controller that ALREADY has a fence
    # armed from an earlier session. This is controller-side state: it survives
    # process exit and power cycles, so a fresh driver must never assume it is off.
    STALE_FENCE = None

    def __init__(self, ip, is_radian=True, **kwargs):
        self.ip = ip
        self.positions = []          # set_position calls
        self.servo_angles = []       # set_servo_angle calls
        self.boundary = list(self.STALE_FENCE) if self.STALE_FENCE else None
        self.fence_on = self.STALE_FENCE is not None
        self.reduced_mode_on = 0
        self.tgpio = [0, 0]
        self.gripper_calls = []
        self.error_code = 0
        self.warn_code = 0
        self.state = 0
        self.mode = 0
        self.version = 'v2.3.0'
        self.joints = [0.0] * 6
        # Where the fake arm currently is, and where FK says a joint target lands
        # (mm, as the real SDK reports).
        self.tcp_mm = [300.0, 0.0, 200.0, np.pi, 0.0, 0.0]
        self.fk_result_mm = [300.0, 0.0, 200.0, np.pi, 0.0, 0.0]

    # -- setup calls the driver makes at connect ------------------------
    def clean_error(self): return 0
    def clean_warn(self): return 0
    def motion_enable(self, enable=True): return 0
    def set_mode(self, mode): return 0
    def set_state(self, state): self.state = state; return 0
    def set_collision_sensitivity(self, v): return 0
    def set_self_collision_detection(self, v): return 0
    # Tool DO lines: DO0 = open solenoid, DO1 = close solenoid.
    def open_lite6_gripper(self): self.tgpio = [1, 0]; self.gripper_calls.append('open'); return 0
    def close_lite6_gripper(self): self.tgpio = [0, 1]; self.gripper_calls.append('close'); return 0
    def stop_lite6_gripper(self): self.tgpio = [0, 0]; self.gripper_calls.append('stop'); return 0

    def get_tgpio_output_digital(self, ionum=None):
        return 0, list(self.tgpio)

    def get_tgpio_digital(self, ionum=None):
        # The tool INPUTS -- deliberately different from the outputs, so anything
        # verifying the solenoid drive against these fails the test.
        return 0, [0, 0]
    def disconnect(self): return 0

    # -- fence ----------------------------------------------------------
    def set_reduced_tcp_boundary(self, boundary):
        self.boundary = list(boundary)
        return 0

    def set_fence_mode(self, on):
        self.fence_on = bool(on)
        return [0]      # NB: the real SDK returns the whole list here, not a code

    def set_reduced_mode(self, on):
        raise AssertionError("the driver must not touch set_reduced_mode -- it caps speed")

    def get_reduced_states(self, is_radian=None):
        return 0, [self.reduced_mode_on, list(self.boundary or [0] * 6),
                   1000, 1.0, [0] * 14, int(self.fence_on), 0]

    # -- state / motion --------------------------------------------------
    def get_position(self, is_radian=True):
        return 0, list(self.tcp_mm)

    def get_servo_angle(self, is_radian=True):
        return 0, list(self.joints)

    def get_forward_kinematics(self, angles, input_is_radian=None, return_is_radian=None):
        return 0, list(self.fk_result_mm)

    def set_position(self, **kwargs):
        self.positions.append(kwargs)
        # Pretend the move completed, so a caller that measures the resulting pose
        # (the bring-up jog stages) sees the motion it asked for.
        for i, key in enumerate(('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
            if kwargs.get(key) is not None:
                self.tcp_mm[i] = kwargs[key]
        return 0

    def set_servo_angle(self, **kwargs):
        self.servo_angles.append(kwargs)
        angle = kwargs.get('angle')
        if angle is not None:
            self.joints = list(np.degrees(np.asarray(angle, dtype=float)))
        return 0


def _install_fake_sdk():
    """The driver does a lazy `from xarm.wrapper import XArmAPI`, so replacing
    that module before construction is enough -- no import-order tricks."""
    wrapper = types.ModuleType('xarm.wrapper')
    wrapper.XArmAPI = FakeXArmAPI
    pkg = types.ModuleType('xarm')
    pkg.wrapper = wrapper
    sys.modules['xarm'] = pkg
    sys.modules['xarm.wrapper'] = wrapper


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _driver(side='left', walls=True, tcp_mm=None):
    from real_robot.robot.xarm_lite6 import XArmLite6
    d = XArmLite6('0.0.0.0', gripper='none', side=side, walls=walls)
    if tcp_mm is not None:
        d.arm.tcp_mm = list(tcp_mm)
    return d


@contextlib.contextmanager
def stale_fence(boundary, tcp_mm=None):
    """Pretend the controller was left with a boundary armed by a previous run."""
    FakeXArmAPI.STALE_FENCE = list(boundary)
    original = FakeXArmAPI.__init__
    if tcp_mm is not None:
        def parked(self, ip, is_radian=True, **kwargs):
            original(self, ip, is_radian=is_radian, **kwargs)
            self.tcp_mm = list(tcp_mm)
        FakeXArmAPI.__init__ = parked
    try:
        yield
    finally:
        FakeXArmAPI.STALE_FENCE = None
        FakeXArmAPI.__init__ = original


# ----------------------------------------------------------------------
@check("fence is programmed and verified at connect")
def t_fence_programmed():
    d = _driver('left')
    expected = to_sdk_boundary_mm(walls_for_side('left'))
    assert d.arm.boundary == expected, (d.arm.boundary, expected)
    assert d.arm.fence_on is True
    assert d.arm.reduced_mode_on == 0, "reduced mode must stay off (speed caps)"


@check("right arm gets its own transformed box")
def t_right_box():
    d = _driver('right')
    assert d.arm.boundary == to_sdk_boundary_mm(walls_for_side('right'))

    # The 180 deg yaw mirrors both x and y. x happens to come out symmetric
    # (the bases are equidistant from their edges), but y must NOT: the arm line
    # sits off-centre on the table, so front and back swap between the arms.
    left, right = walls_for_side('left'), walls_for_side('right')
    for axis in ('x', 'y'):
        lo, hi = left[axis]
        span = (hi - lo)
        assert abs((right[axis][1] - right[axis][0]) - span) < 1e-9, \
            "{} span must be preserved by the transform".format(axis)
    assert abs(right['y'][0] + left['y'][1]) < 1e-9 and \
           abs(right['y'][1] + left['y'][0]) < 1e-9, \
           "y limits must be mirrored: left {} vs right {}".format(left['y'], right['y'])
    assert left['z'] == right['z'], "z is unaffected by a yaw"


def _centre(bounds):
    return [(bounds[a][0] + bounds[a][1]) / 2.0 for a in ('x', 'y', 'z')]


@check("in-bounds movel is sent")
def t_movel_ok():
    d = _driver('left')
    ok = d.movel(_centre(walls_for_side('left')) + [np.pi, 0.0, 0.0], blocking=False)
    assert ok is True, "the centre of the box must be accepted"
    assert len(d.arm.positions) == 1


@check("out-of-bounds movel is refused and sends nothing")
def t_movel_refused():
    # Derived from the walls, 10 cm outside each one in turn, so these stay valid
    # if the cell geometry changes.
    bounds = walls_for_side('left')
    for axis, side in [(a, s) for a in ('x', 'y', 'z') for s in (0, 1)]:
        p = _centre(bounds)
        i = ('x', 'y', 'z').index(axis)
        p[i] = bounds[axis][side] + (-0.10 if side == 0 else 0.10)
        why = "{} {} the {:+.3f} m wall".format(
            axis, "below" if side == 0 else "above", bounds[axis][side])
        d = _driver('left')
        ok = d.movel(p + [np.pi, 0.0, 0.0], blocking=False)
        assert ok is False, "should have refused: {}".format(why)
        assert d.arm.positions == [], "nothing may be sent when refused ({})".format(why)


@check("one bad waypoint cancels the WHOLE trajectory")
def t_trajectory_all_or_nothing():
    d = _driver('left')
    traj = [[0.30, 0.00, 0.10, np.pi, 0.0, 0.0],
            [0.30, 0.90, 0.10, np.pi, 0.0, 0.0],     # outside
            [0.30, 0.20, 0.10, np.pi, 0.0, 0.0]]
    ok = d.movel(traj, blocking=False)
    assert ok is False
    assert d.arm.positions == [], "must not strand the arm part-way through"


@check("movej / home are wall-checked through forward kinematics")
def t_movej_checked():
    d = _driver('left')
    d.arm.fk_result_mm = [300.0, 900.0, 100.0, np.pi, 0.0, 0.0]   # y outside
    assert d.movej([0.0] * 6, blocking=False) is False
    assert d.arm.servo_angles == []

    d.arm.fk_result_mm = [300.0, 100.0, 100.0, np.pi, 0.0, 0.0]   # inside
    assert d.movej([0.0] * 6, blocking=False) is True
    assert len(d.arm.servo_angles) == 1

    d.arm.fk_result_mm = [300.0, 900.0, 100.0, np.pi, 0.0, 0.0]
    assert d.home(blocking=False) is False, "home() must be gated too"


@check("walls=False disables the check and the fence")
def t_walls_off():
    # Start from a controller that HAS a fence armed, otherwise this passes
    # vacuously: the fake starts with fence_on False, so "it is off afterwards"
    # would prove nothing about the driver.
    with stale_fence(to_sdk_boundary_mm(walls_for_side('left'))):
        d = _driver('left', walls=False)
        assert d.bounds is None
        assert d.arm.fence_on is False, "walls=False must CLEAR the controller fence"
        assert d.movel([0.30, 0.90, 0.05, np.pi, 0.0, 0.0], blocking=False) is True


@check("a stale controller fence is cleared at connect (walls off)")
def t_stale_fence_cleared():
    from real_robot.robot.xarm_lite6 import XArmLite6

    # Exactly the situation that stopped the right arm: a boundary programmed by
    # an earlier run, with the arm parked OUTSIDE it. Any move -- in any
    # direction, nowhere near a wall -- is refused with error 35 until it is off.
    boundary = to_sdk_boundary_mm(walls_for_side('right'))
    outside = [-239.4, 236.3, 291.1, np.pi, 0.0, 0.0]

    for walls in (False, 'auto'):
        if walls == 'auto' and C.XARM_GEOMETRY_VERIFIED:
            continue        # 'auto' means walls ON once the geometry is measured
        with stale_fence(boundary, tcp_mm=outside):
            d = XArmLite6('0.0.0.0', gripper='none', side='right', walls=walls)
            assert d.bounds is None
            assert d.arm.fence_on is False, (
                f"walls={walls!r} left a stale controller fence armed; the arm "
                f"cannot move while its TCP is outside it")


@check("close_gripper keeps the solenoid DRIVEN (does not vent the grasp)")
def t_gripper_holds():
    from real_robot.robot.xarm_lite6 import XArmLite6
    d = XArmLite6('0.0.0.0', gripper='lite6', side='left', walls=False)
    # Connect parks it open, driven -- not with both lines low, which on a
    # spring-return valve leaves the fingers wherever the spring puts them.
    assert d.arm.tgpio == [1, 0], d.arm.tgpio

    d.arm.gripper_calls.clear()
    d.close_gripper(sleep_time=0.0)
    # THE regression: a trailing stop_lite6_gripper() drops both lines, and a
    # spring-return valve then releases the object the instant the call returns.
    assert 'stop' not in d.arm.gripper_calls, (
        "close_gripper must not de-energise the valve: {}".format(d.arm.gripper_calls))
    assert d.arm.tgpio == [0, 1], d.arm.tgpio

    d.open_gripper(sleep_time=0.0)
    assert 'stop' not in d.arm.gripper_calls, d.arm.gripper_calls
    assert d.arm.tgpio == [1, 0], d.arm.tgpio

    # Releasing the solenoids stays available, but only when asked for explicitly.
    d.stop_gripper()
    assert d.arm.tgpio == [0, 0], d.arm.tgpio


@check("disconnect leaves the gripper open and DE-ENERGISED")
def t_gripper_parked_on_disconnect():
    from real_robot.robot.xarm_lite6 import XArmLite6
    d = XArmLite6('0.0.0.0', gripper='lite6', side='left', walls=False)
    fake = d.arm
    d.close_gripper(sleep_time=0.0)
    assert fake.tgpio == [0, 1]

    d.disconnect()
    # Holding the coil is what makes a grasp work, so something has to release it:
    # the controller keeps the tool DO state after the process exits, and an
    # energised solenoid would sit there powered until the next run.
    assert fake.tgpio == [0, 0], (
        "disconnect left the solenoid driven: {}".format(fake.tgpio))
    # Opened first, so the resting state is known for a bistable valve too.
    assert fake.gripper_calls[-2:] == ['open', 'stop'], fake.gripper_calls


@check("gripper verification reads the tool OUTPUTS, not the inputs")
def t_gripper_verify_uses_outputs():
    from real_robot.robot.xarm_lite6 import XArmLite6
    d = XArmLite6('0.0.0.0', gripper='none', side='left', walls=False)
    # The fake reports inputs as [0, 0] and outputs as whatever was driven, so a
    # verify that reads get_tgpio_digital would fail here.
    assert d.close_gripper(sleep_time=0.0, verify=True) is True
    assert d.open_gripper(sleep_time=0.0, verify=True) is True


@check("a stale fence is REPLACED, not merged, when walls are on")
def t_stale_fence_replaced():
    # A leftover boundary must not survive alongside ours: the controller keeps
    # one box, and it has to be the one this driver believes in.
    with stale_fence([1, 2, 3, 4, 5, 6]):
        d = _driver('right', walls=True)
        assert d.arm.fence_on is True
        assert d.arm.boundary == to_sdk_boundary_mm(walls_for_side('right'))


@check("disable_walls() turns off both layers together")
def t_disable_walls():
    d = _driver('left')
    assert d.arm.fence_on is True
    d.disable_walls()
    assert d.arm.fence_on is False, "controller fence must come off too"
    assert d.bounds is None
    assert d.movel([0.30, 0.90, 0.05, np.pi, 0.0, 0.0], blocking=False) is True


@check("connecting outside the box warns but does not fail")
def t_outside_at_connect():
    from real_robot.robot.xarm_lite6 import XArmLite6
    original = FakeXArmAPI.__init__

    def parked_outside(self, ip, is_radian=True, **kwargs):
        original(self, ip, is_radian=is_radian, **kwargs)
        self.tcp_mm = [300.0, 900.0, 200.0, np.pi, 0.0, 0.0]
    FakeXArmAPI.__init__ = parked_outside
    try:
        d = XArmLite6('0.0.0.0', gripper='none', side='left', walls=True)
        assert d.bounds is not None, "walls stay configured even if parked outside"
    finally:
        FakeXArmAPI.__init__ = original


@check("walls='auto' waits for verified geometry")
def t_auto_gate():
    from real_robot.robot.xarm_lite6 import XArmLite6
    d = XArmLite6('0.0.0.0', gripper='none', side='left')     # default is 'auto'
    if C.XARM_GEOMETRY_VERIFIED:
        assert d.bounds is not None, "verified geometry should switch the walls on"
        assert d.arm.fence_on is True
    else:
        # The whole point: a box placed with an unverified XARM_BASE_YAW would
        # refuse legitimate moves rather than protect the table.
        assert d.bounds is None, "walls must stay off until the geometry is measured"
        assert d.arm.fence_on is False, "the controller fence must stay off too"
        assert d.movel([0.30, 0.90, 0.05, np.pi, 0.0, 0.0], blocking=False) is True


@check("every default primitive target is inside the walls")
def t_primitive_targets_inside():
    from real_robot.utils.xarm_walls import check_pose, table_to_base
    from real_robot.test.test_xarm_primitives import (
        SINGLE_CASES, DUAL_CASES, FLING_CASES,
    )
    bounds = {'left': walls_for_side('left'), 'right': walls_for_side('right')}
    z = C.XARM_TABLE_Z + C.XARM_GRIPPER_OFFSET
    targets = []
    for case in SINGLE_CASES.values():
        targets += [('left', case['pick']), ('left', case['place'])]
    for case in DUAL_CASES.values():
        targets += [('left', case['pick_l']), ('left', case['place_l']),
                    ('right', case['pick_r']), ('right', case['place_r'])]
    for case in FLING_CASES.values():
        targets += [('left', case['pick_l']), ('right', case['pick_r'])]
    for side, (x, y) in targets:
        base = table_to_base([x, y, z], side)
        ok, bad = check_pose(base, bounds[side])
        assert ok, "{} target ({:+.3f}, {:+.3f}) is outside the walls: {}".format(
            side, x, y, bad)


# ----------------------------------------------------------------------
def main():
    _install_fake_sdk()
    print("=" * 72)
    print("xArm virtual walls -- offline checks (fake controller, no hardware)")
    from real_robot.utils.xarm_walls import describe
    print("  table walls: {}".format(describe(C.XARM_WALLS)))
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
