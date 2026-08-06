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
import os
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
        # Scripted joint EFFORT, consumed one value per get_joint_states call, the
        # last value repeating for ever. None (the default) stands for firmware
        # that cannot report effort at all, which is what every other check here
        # wants -- get_joint_effort then returns None and nothing is gated on it.
        self.effort_values = None
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
    # Motion smoothness. The real controller FORGETS these on reboot, so the driver
    # has to set them on every connect -- recorded here so a test can prove it does.
    def set_tcp_jerk(self, jerk): self.tcp_jerk = jerk; return 0
    def set_tcp_maxacc(self, acc): self.tcp_maxacc = acc; return 0
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

    def get_joint_states(self, is_radian=True, num=3):
        # (position, velocity, effort), which is the shape XArmLite6.get_joint_effort
        # reads states[2] out of. A non-zero code is how the real SDK says "this
        # firmware does not have it".
        if self.effort_values is None:
            return -1, None
        v = (self.effort_values[0] if len(self.effort_values) == 1
             else self.effort_values.pop(0))
        return 0, [list(self.joints), [0.0] * 6, [float(v)] + [0.0] * 5]

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


def _measured_cell():
    """The separation and the per-arm reach AS MEASURED, or the constants if not.

    ⚠️ Both, from the same source, or neither. Hand-eye calibration measures the
    separation now (T_left_cam @ inv(T_right_cam)) and it came out 0.749 m against
    the assumed 0.66; test_xarm_teach.py --reach measured r_max at 0.41 m against
    the constant's conservative 0.40. Checking the measured separation against the
    UNmeasured reach mixes the two cells and reports a violation that does not
    exist -- which is exactly what this check did on its first run.

    ⚠️ AND THE REACH DEPENDS ON HEIGHT, which this check originally ignored.
    workspace_radius is derived from the GRASP-height sweep with a 2 cm edge
    margin (0.430 -> 0.410). The fling's binding waypoint is at the HANG height,
    where the same sweep measured 0.425, so the comparable limit is 0.405 -- five
    millimetres TIGHTER, not looser. Checking a hang-height waypoint against a
    grasp-height radius overstates the margin, which is exactly the mistake this
    file already warns about in the other direction.

    Reach on these arms: 0.430 at grasp (z=0.086), 0.440 at lift (z=0.186), 0.425
    at hang (z=0.250). It peaks around 0.19 m and falls away above that, so a
    HIGHER hang has less reach, not more -- and the numbers here were measured at
    whatever XARM_FLING_HANG was at the time. Change the hang height and re-run
    `test_xarm_teach.py --arm both --reach`; sweep_reach probes at
    `table_z + C.XARM_FLING_HANG`, so it follows the constant automatically.

    Returns (separation, {side: (r_min, r_max)}, {side: hang_r_max}, measured?).
    """
    import os
    import yaml
    from real_robot.utils.scene_utils import cell_geometry_from_calibration
    d = "{}/real_robot/calibration".format(os.environ.get('MP_FOLD_PATH', '.'))
    radius = {s: C.for_side(C.XARM_WORKSPACE_RADIUS_BY_SIDE, s)
              for s in ('left', 'right')}
    hang = {s: radius[s][1] for s in ('left', 'right')}
    try:
        _, separation, _ = cell_geometry_from_calibration(
            os.path.join(d, 'xarm-left-calib.yaml'),
            os.path.join(d, 'xarm-right-calib.yaml'))
    except Exception:
        return float(C.XARM_BASE_SEPARATION), radius, hang, False

    try:
        cell = yaml.safe_load(open(os.path.join(d, 'xarm-cell.yaml')))
        for s in ('left', 'right'):
            arm = cell['arms'][s]
            radius[s] = tuple(arm['workspace_radius'])
            r_hang = (arm.get('reach') or {}).get('hang')
            # The same 2 cm margin sweep_reach takes off the measured grasp edge,
            # because IK solutions near the boundary are singular at any height.
            hang[s] = round(r_hang[1] - 0.02, 3) if r_hang else radius[s][1]
    except Exception:
        return float(separation), radius, hang, False
    return float(separation), radius, hang, True


@check("the fling constants satisfy their own geometric derivation")
def t_fling_envelope():
    """The fling constants are DERIVED from the cell, not chosen.

    xarm_base_fling_poses builds the swing in a frame centred between the bases, so
    each gripper sits at x = (S - width)/2 from its own base and the FURTHEST
    waypoint is the forward stroke, at (x, +stroke, hang). Two constraints follow,
    and a naive port of the UR numbers violates both. Assert them here so a later
    edit to the separation or the measured reach fails on a laptop, not on the arm.

    (The furthest waypoint used to be the WIND-UP, at (x, -stroke, hang), back when
    the swing was symmetric. The wind-up is XARM_FLING_WINDUP now -- a third of the
    stroke -- so it is nowhere near binding, but the radius is the same because the
    constraint only sees |y|.)

    ⚠️ S and the reach are both the MEASURED ones, not the constants. The two
    constraints move in OPPOSITE directions, and not the way intuition suggests:
    NARROWING the stretch pushes each gripper further from its own base, so the
    keepout gets easier and the reach gets HARDER. "Hold the garment less taut"
    asks more of the arms, not less. At the measured 0.7489 m with the 0.24 m
    stretch the swing radius is 0.4042 m, clearing the measured 0.410 m reach by
    1.4% -- and only because the stroke came down to 0.19 m in the same edit; at
    the old 0.25 m stroke that same stretch is 0.4356 m, 25.6 mm out of reach. It
    would also FAIL against the constants' conservative 0.400 m, which is why both
    numbers have to come from the same cell.
    """
    S, radius, hang_reach, measured = _measured_cell()
    width, hang = C.XARM_FLING_WIDTH, C.XARM_FLING_HANG
    stroke = C.XARM_FLING_STROKE
    x = (S - width) / 2.0

    for side in ('left', 'right'):
        r_min, r_grasp = radius[side]
        # Hang-height waypoints get the hang-height limit. See _measured_cell.
        r_max = hang_reach[side]
        assert x >= r_min, (
            "{}: stretch puts the gripper {:.3f} m from its base, inside the {:.3f} m "
            "keepout. Reduce XARM_FLING_WIDTH to <= {:.3f} m.".format(
                side, x, r_min, S - 2 * r_min))
        swing = float(np.sqrt(x ** 2 + stroke ** 2 + hang ** 2))
        assert swing <= r_max, (
            "{}: the forward stroke waypoint is {:.3f} m out, beyond the {:.3f} m "
            "reach, at the {} {:.4f} m separation. Either widen the stretch to "
            ">= {:.3f} m (a wider cell wants a wider stretch: it pulls each gripper "
            "back toward its own base) or shorten the stroke to <= {:.3f} m.".format(
                side, swing, r_max, "MEASURED" if measured else "assumed", S,
                S - 2 * np.sqrt(max(r_max ** 2 - stroke ** 2 - hang ** 2, 0.0)),
                np.sqrt(max(r_max ** 2 - x ** 2 - hang ** 2, 0.0))))
        # Every other waypoint of the swing, checked rather than assumed. The
        # wind-up and the lay-down are both small in y now, and the last two are
        # lower, so none should bind -- but "should not" is not a check. Each is
        # measured against the reach at ITS OWN height: the two low waypoints sit
        # near the grasp height, where the arm reaches further than at the hang.
        for label, y, z, limit in (
                ('wind-up', -C.XARM_FLING_WINDUP, hang, r_max),
                ('touch-down', C.XARM_FLING_LAND_Y, C.XARM_FLING_PLACE_Z, r_grasp),
                ('lay-down', C.XARM_FLING_PLACE_Y, C.XARM_FLING_PLACE_Z, r_grasp)):
            r = float(np.sqrt(x ** 2 + y ** 2 + z ** 2))
            assert r <= limit, (
                "{}: the {} waypoint at ({:+.3f}, {:+.3f}) is {:.3f} m out, beyond "
                "the {:.3f} m reach at that height".format(
                    side, label, y, z, r, limit))

    # The asymmetry is the operator's instruction, not a tuning accident: the swing
    # must read as "forward" to someone watching it. Pin the ratio so a later edit
    # cannot quietly restore the symmetric shape.
    assert C.XARM_FLING_WINDUP < 0.5 * stroke, (
        "the {:.3f} m wind-up is more than half the {:.3f} m forward stroke, so "
        "the fling reads as backwards-then-forwards again".format(
            C.XARM_FLING_WINDUP, stroke))
    assert C.XARM_FLING_PLACE_Y < 0.0, (
        "the garment is laid down at y={:+.3f}, in FRONT of the base-to-base line; "
        "it should finish just behind it".format(C.XARM_FLING_PLACE_Y))
    assert abs(C.XARM_FLING_PLACE_Y) < 0.5 * stroke, (
        "the lay-down at y={:+.3f} is a long way behind the line, which drags the "
        "garment back off the flung-out area".format(C.XARM_FLING_PLACE_Y))

    # The four y waypoints have to stay in order, or the swing quietly inverts.
    # Landing BEYOND the stroke means the hands travel further forward while
    # descending -- which is what the shape did before land_y existed, waypoint 3
    # having simply reused the stroke's y. Landing at or behind place_y makes the
    # drag run forwards instead of back, so nothing gets laid out.
    assert C.XARM_FLING_LAND_Y <= stroke, (
        "the hands land at y={:+.3f}, beyond the {:+.3f} m stroke, so they move "
        "FURTHER forward as they come down".format(C.XARM_FLING_LAND_Y, stroke))
    assert C.XARM_FLING_PLACE_Y < C.XARM_FLING_LAND_Y, (
        "the lay-down at y={:+.3f} is not behind the touch-down at y={:+.3f}, so "
        "the drag runs forwards and lays nothing out".format(
            C.XARM_FLING_PLACE_Y, C.XARM_FLING_LAND_Y))

    assert C.XARM_FLING_MIN_WIDTH <= width, "min stretch width exceeds the cap"
    assert C.XARM_FLING_PLACE_Z < hang, "touch-down must be below the hang height"


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


@check("the crop is centred between the arms and still inverts to metres")
def t_crop_centred():
    """The window handed to perception must be centred on the arm midpoint, and a
    pixel in it must invert to the point it was derived from.

    Both halves matter. The centre is the user-visible requirement; the round-trip
    is what stops the crop from silently mis-aiming every grasp, because the whole
    scheme rests on the principal point moving with the image.
    """
    from real_robot.test.xarm_test_scene import (
        synthetic_T_left_cam, synthetic_intrinsic, table_xy_to_pixel,
    )
    from real_robot.utils.transform_utils import point_on_table_base
    from real_robot.utils.xarm_camera import base_to_pixel, crop_window

    S, tz = C.XARM_BASE_SEPARATION, C.XARM_TABLE_Z
    intr, T = synthetic_intrinsic(), synthetic_T_left_cam(S, tz)
    win = crop_window(intr, T, S, tz)
    assert not win.clamped, "the crop had to be clamped: {}".format(win)

    # Where the arm midpoint lands in the WINDOW. Note this must not be phrased as
    # "the principal point inverts to the midpoint": the principal point is the
    # point under the camera whatever the crop, so that version passes for any
    # offset window and tests the camera's placement rather than the crop's.
    cropped = win.intrinsic(intr)
    mid = base_to_pixel([S / 2.0, C.XARM_CAM_CENTRE_Y, tz], T, cropped)
    off = float(np.linalg.norm(mid - np.array([win.side / 2.0, win.side / 2.0])))
    assert off <= 1.0, (
        "the arm midpoint sits at pixel ({:.1f}, {:.1f}) of a {} px window whose "
        "centre is ({:.1f}, {:.1f}) -- {:.1f} px off".format(
            mid[0], mid[1], win.side, win.side / 2.0, win.side / 2.0, off))

    worst = 0.0
    for x in np.linspace(S / 2 - 0.25, S / 2 + 0.25, 5):
        for y in np.linspace(-0.25, 0.25, 5):
            px = table_xy_to_pixel(x, y, S, tz, intr=cropped)
            assert 0 <= px[0] < win.side and 0 <= px[1] < win.side, (
                "({:+.3f}, {:+.3f}) m falls outside the {} px crop at pixel "
                "({:.1f}, {:.1f})".format(x, y, win.side, px[0], px[1]))
            back = point_on_table_base(px[0], px[1], cropped, T, tz)
            worst = max(worst, float(np.linalg.norm(back - np.array([x, y, tz]))))
    assert worst < 1e-9, "cropped pixels do not round-trip: {:.3e} m".format(worst)


@check("the crop window fits the frame for the sim and a real RealSense")
def t_crop_fits():
    """XARM_CROP_SIZE is a length on the TABLE, so whether it fits depends on the
    lens and the mounting height. A clamped window is no longer centred between the
    arms, which is a silent aiming error -- so check the two cameras this cell will
    actually use before either is bolted up.

    ⚠️ These are the two SYNTHETIC cameras, and they exist to check crop_window's
    arithmetic. The verdict for the cell that is actually bolted up -- calibrated
    extrinsic, measured separation, real or nominal RealSense intrinsic -- is
    t_crop_fits in test_xarm_mujoco_camera.py, and it does not currently pass.
    """
    from real_robot.test.xarm_test_scene import (
        SyntheticIntrinsic, synthetic_T_left_cam, synthetic_intrinsic,
    )
    from real_robot.utils.xarm_camera import crop_window

    S, tz = C.XARM_BASE_SEPARATION, C.XARM_TABLE_Z
    cams = [("sim placeholder at {:.1f} m".format(C.XARM_CAM_HEIGHT),
             synthetic_intrinsic(), C.XARM_CAM_HEIGHT),
            ("RealSense colour (fx 900) at 1.50 m",
             SyntheticIntrinsic(900.0, 900.0, 640.0, 360.0, 1280, 720), 1.50)]
    for label, intr, height in cams:
        T = synthetic_T_left_cam(S, tz)
        T[2, 3] = tz + height
        win = crop_window(intr, T, S, tz)
        assert not win.clamped, (
            "{}: a {:.2f} m crop does not fit -- {}. Lower XARM_CROP_SIZE or raise "
            "the camera.".format(label, C.XARM_CROP_SIZE, win))
        assert win.side >= 128, (
            "{}: the crop is only {} px, too coarse once resized".format(
                label, win.side))


@check("the photo pose is home with joint 1 swung to that arm's left")
def t_photo_pose():
    """The pose is derived, never written out.

    Only J1 may change: J1 turns about the base z axis, so the TCP keeps home's
    height and radius and the pose inherits home's clearance over the table. If a
    later edit moves any other joint, that guarantee is gone and this fails.
    """
    for side in ('left', 'right'):
        home = C.for_side(C.XARM_HOME_JOINT_BY_SIDE, side)
        photo = C.for_side(C.XARM_OUT_SCENE_JOINT_BY_SIDE, side)
        assert len(photo) == len(home) == 6
        d = np.asarray(photo) - np.asarray(home)
        assert abs(d[0] - C.XARM_PHOTO_YAW) < 1e-9, (
            "{}: J1 moves {:+.1f} deg, expected {:+.1f}".format(
                side, np.rad2deg(d[0]), np.rad2deg(C.XARM_PHOTO_YAW)))
        assert np.allclose(d[1:], 0.0), (
            "{}: joints 2-6 must be untouched, they moved by {} deg".format(
                side, np.round(np.rad2deg(d[1:]), 3)))

    # And the driver must derive it from the home it was actually given, so a home
    # taught into xarm-cell.yaml carries through instead of being ignored.
    taught = [0.10, -0.20, 0.30, -0.40, 0.50, -0.60]
    d = _driver(side='right', walls=False)
    assert np.allclose(d.out_scene_joint,
                       C.photo_pose_from_home(d.home_joint)), \
        "the driver's out-scene pose is not derived from its own home"
    d2 = _driver_with_home(taught)
    assert np.allclose(d2.home_joint, taught), "an explicit home was overridden"
    assert np.allclose(d2.out_scene_joint, C.photo_pose_from_home(taught)), (
        "a taught home did not carry into the photo pose: {}".format(
            np.round(d2.out_scene_joint, 3)))


def _driver_with_home(home):
    from real_robot.robot.xarm_lite6 import XArmLite6
    return XArmLite6('0.0.0.0', gripper='none', side='left', walls=False,
                     home_joint=home)


@check("the table's front edge is at base +y")
def t_table_front_back():
    """Which physical edge base +y points at is half of the view convention.

    A camera looking straight down cannot put the right arm on the right AND the
    front edge at the top unless the front edge is at +y, so this constant and
    ``synthetic_T_left_cam``'s roll have to agree. Flipping one without the other
    gives a frame that is upside down and walls that protect the wrong half of the
    table.
    """
    lo, hi = C.XARM_TABLE_RECT['y']
    assert abs(hi - C.XARM_BASE_TO_FRONT) < 1e-12, (
        "the front edge is at y={:+.3f}, expected +{:.3f} (front must be +y)".format(
            hi, C.XARM_BASE_TO_FRONT))
    assert abs((hi - lo) - C.XARM_TABLE_SIZE[1]) < 1e-12, (
        "the table rect spans {:.3f} m but XARM_TABLE_SIZE says {:.3f}".format(
            hi - lo, C.XARM_TABLE_SIZE[1]))
    assert lo < 0 < hi, "the arm line must be inside the table: y {}".format((lo, hi))


@check("the frame has the left arm on the left and the front at the top")
def t_image_orientation():
    """The view convention, checked on the synthetic camera.

    Both halves have been wrong: until this change the frame was mirrored, so the
    arm named 'right' was drawn on the left of the image.
    """
    from real_robot.test.xarm_test_scene import (
        synthetic_T_left_cam, synthetic_intrinsic,
    )
    from real_robot.utils.xarm_camera import base_to_pixel, describe_orientation

    S, tz = C.XARM_BASE_SEPARATION, C.XARM_TABLE_Z
    intr, T = synthetic_intrinsic(), synthetic_T_left_cam(S, tz)

    # A rotation, not a reflection. A reflection puts the corners where you expect
    # and then quietly breaks every handedness-dependent quantity downstream.
    det = float(np.linalg.det(T[:3, :3]))
    assert abs(det - 1.0) < 1e-9, "the camera rotation has det {:+.3f}".format(det)

    left = base_to_pixel([0.0, 0.0, tz], T, intr)
    right = base_to_pixel([S, 0.0, tz], T, intr)
    assert right[0] > left[0], (
        "the right arm is drawn at u={:.0f}, LEFT of the left arm at u={:.0f} -- "
        "the frame is mirrored".format(right[0], left[0]))

    front = base_to_pixel([S / 2, C.XARM_TABLE_RECT['y'][1], tz], T, intr)
    back = base_to_pixel([S / 2, C.XARM_TABLE_RECT['y'][0], tz], T, intr)
    assert front[1] < back[1], (
        "the front edge is drawn at v={:.0f}, BELOW the back edge at v={:.0f}".format(
            front[1], back[1]))

    ok, msg = describe_orientation(T, intr, S, tz)
    assert ok, msg


@check("arms are assigned by table position, not by pixel column")
def t_side_assignment():
    """The pick nearer the left base goes to the left arm, under ANY camera roll.

    The old pixel sort ("larger pixel-x -> left arm") was correct only for the
    camera roll it was written against. Rolling the camera 180 deg -- which is
    exactly what this change did, and what a differently-bolted bracket would do on
    hardware -- silently handed each arm the other one's target: both grasps stay
    individually reachable, so nothing complains.
    """
    from real_robot.primitives.utils import sort_pairs_by_table_x
    from real_robot.test.xarm_test_scene import (
        synthetic_T_left_cam, synthetic_intrinsic, base_to_pixel,
    )
    from real_robot.utils.transform_utils import point_on_table_base

    S, tz = C.XARM_BASE_SEPARATION, C.XARM_TABLE_Z
    intr = synthetic_intrinsic()
    T = synthetic_T_left_cam(S, tz)
    # The same camera rolled 180 deg about its optical axis: the frame is mirrored,
    # every table point lands in a different pixel column, the geometry is unchanged.
    T_rolled = T.copy()
    T_rolled[:3, :3] = T[:3, :3] @ np.diag([-1.0, -1.0, 1.0])   # roll pi about +z_cam

    near_left, near_right = (0.20, -0.12), (0.46, 0.12)
    for label, cam in (("as built", T), ("rolled 180 deg", T_rolled)):
        px_l = base_to_pixel([near_left[0], near_left[1], tz], cam, intr)
        px_r = base_to_pixel([near_right[0], near_right[1], tz], cam, intr)
        for order, (a, b) in (("in order", (px_l, px_r)),
                              ("reversed", (px_r, px_l))):
            pair_l, pair_r = sort_pairs_by_table_x(
                {'pick': a, 'tag': 'first'}, {'pick': b, 'tag': 'second'},
                intr, cam, tz)
            got_l = point_on_table_base(pair_l['pick'][0], pair_l['pick'][1],
                                        intr, cam, tz)
            got_r = point_on_table_base(pair_r['pick'][0], pair_r['pick'][1],
                                        intr, cam, tz)
            assert abs(got_l[0] - near_left[0]) < 1e-6, (
                "{}, {}: the left arm was given x={:+.3f}, expected {:+.3f}".format(
                    label, order, got_l[0], near_left[0]))
            assert got_l[0] < got_r[0], "{}, {}: the arms are crossed".format(
                label, order)


@check("the executed-path overlay is drawn on the table plane")
def t_overlay_drops_to_table():
    """The overlay must answer "did the arm go where I clicked".

    A lifted sample has to be drawn at its shadow, not where the camera would see
    it: at the 0.25 m hang height under a 1.0 m camera the two differ by tens of
    pixels in a 330 px window, and the wind-up leaves the frame entirely.
    """
    from real_robot.sim.run_sim_ui import path_to_pixels
    from real_robot.test.xarm_test_scene import (
        synthetic_T_left_cam, synthetic_intrinsic, table_xy_to_pixel,
    )
    from real_robot.utils.xarm_camera import base_to_pixel, crop_window

    S, tz = C.XARM_BASE_SEPARATION, C.XARM_TABLE_Z

    class _Scene:                       # the attributes path_to_pixels reads
        separation, table_z = S, tz
        # table_px projects through the scene's OWN camera now (the cell's is the
        # calibrated one), so the fake has to carry a camera as well as an
        # intrinsic. With the synthetic camera this reduces to table_xy_to_pixel,
        # which is what `want` below computes independently.
        T_left_cam = synthetic_T_left_cam(S, tz)
        intr = crop_window(synthetic_intrinsic(), synthetic_T_left_cam(S, tz),
                           S, tz).intrinsic(synthetic_intrinsic())

    scene = _Scene()
    T = synthetic_T_left_cam(S, tz)

    # A grasp and the wind-up above it: both must draw at the same table pixel.
    x, y = 0.15, -C.XARM_FLING_STROKE
    grasp = [x, y, tz + C.XARM_GRIPPER_OFFSET]
    hang = [x, y, tz + C.XARM_FLING_HANG]
    px = path_to_pixels(scene, [grasp, hang])
    want = table_xy_to_pixel(x, y, S, tz, scene.intr)
    for i, (name, p) in enumerate((('grasp', grasp), ('hang', hang))):
        assert np.linalg.norm(px[i] - want) < 1e-6, (
            "the {} sample is drawn at {} instead of its table shadow {}".format(
                name, np.round(px[i], 2), np.round(want, 2)))
        assert (0 <= px[i]).all() and (px[i] < scene.intr.width).all(), (
            "the {} sample falls outside the {} px window".format(
                name, scene.intr.width))

    # And confirm the difference this avoids is real, not a rounding argument.
    persp = base_to_pixel(hang, T, scene.intr)
    assert np.linalg.norm(persp - want) > 20.0, (
        "perspective and table projections differ by only {:.1f} px here, so this "
        "check is not testing what it claims".format(np.linalg.norm(persp - want)))


# ----------------------------------------------------------------------
# The contact probe: does it act on effort it is not allowed to act on, and do
# the two arms descend TOGETHER?
# ----------------------------------------------------------------------
class _ProbeScene:
    """The minimum ``_probe_both`` touches: two drivers and a recording both_movel.

    Recording is the assertion vehicle for ``t_probe_is_synchronised``: if the
    probe ever went back to driving one arm at a time, the pairs would not be here
    to inspect.
    """

    def __init__(self, left, right):
        self.left, self.right = left, right
        self.commands = []           # [(z_left, z_right)] -- one entry per step

    def both_movel(self, left_pose, right_pose, speed, acc, blocking=True,
                   record=False):
        self.commands.append((float(left_pose[2]), float(right_pose[2])))
        a = self.left.movel(left_pose, speed=speed, acceleration=acc,
                            blocking=blocking)
        b = self.right.movel(right_pose, speed=speed, acceleration=acc,
                             blocking=blocking)
        return a and b


@contextlib.contextmanager
def _effort_verified(value):
    """Flip the gate. The primitive imports the constant BY VALUE, so patching
    xarm_constants would change nothing -- the module's own global is the one that
    is read."""
    from real_robot.primitives import xarm_pick_and_fling as F
    was = F.XARM_EFFORT_VERIFIED
    F.XARM_EFFORT_VERIFIED = value
    try:
        yield F
    finally:
        F.XARM_EFFORT_VERIFIED = was


# Floors deliberately a fraction of a millimetre apart, as the measured per-side
# gripper offsets are (0.0860 vs 0.0857), so "the arms track each other" is being
# checked rather than "the two numbers are literally identical".
_FLOOR_L, _FLOOR_R = 0.0860, 0.0857


def _probe(effort_l, effort_r, verified, baseline=0.0):
    """Run one descent and hand back (skill, scene, reached poses)."""
    from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
    left = _driver('left', walls=False)
    right = _driver('right', walls=False)
    # effort_baseline() consumes 10 samples before the descent starts.
    left.arm.effort_values = [baseline] * 10 + list(effort_l)
    right.arm.effort_values = [baseline] * 10 + list(effort_r)

    scene = _ProbeScene(left, right)
    skill = XArmPickAndFlingSkill(scene, {'speed': 0.10, 'acc': 0.30})
    grasp_l = np.array([0.30, 0.00, _FLOOR_L, np.pi, 0.0, 0.0])
    grasp_r = np.array([0.30, 0.00, _FLOOR_R, np.pi, 0.0, 0.0])
    app_l, app_r = grasp_l.copy(), grasp_r.copy()
    app_l[2] += C.XARM_APPROACH_DIST
    app_r[2] += C.XARM_APPROACH_DIST

    with _effort_verified(verified):
        reached = skill._probe_both(app_l, app_r, grasp_l, grasp_r)
    return skill, scene, reached


@check("contact probe ignores effort while XARM_EFFORT_VERIFIED is False")
def t_probe_ignores_unverified_effort():
    # The 2026-08-04 fault, reproduced: the right arm's noise floor sits above the
    # placeholder threshold and the left arm's does not. Acting on that stops the
    # right arm high, the left arm grasps normally, and both report success.
    spike = [C.XARM_EFFORT_THRESHOLD['right'] + 5.0] * 20
    _, scene, (reached_l, reached_r) = _probe(
        effort_l=[0.0] * 20, effort_r=spike, verified=False)

    assert abs(reached_l[2] - _FLOOR_L) < 1e-9 and abs(reached_r[2] - _FLOOR_R) < 1e-9, (
        "an unverified effort reading stopped the descent early: left reached "
        "{:.4f} (floor {:.4f}), right {:.4f} (floor {:.4f})".format(
            reached_l[2], _FLOOR_L, reached_r[2], _FLOOR_R))
    # And it should have taken ONE synchronised move to get there: with nothing
    # able to interrupt, stepping would only burn controller round trips.
    assert len(scene.commands) == 1, (
        "the descent took {} moves; with effort inert it is a single "
        "synchronised move".format(len(scene.commands)))


@check("contact probe needs XARM_EFFORT_CONSECUTIVE samples, not one spike")
def t_probe_needs_consecutive_samples():
    over = C.XARM_EFFORT_THRESHOLD['left'] + 5.0
    # One sample over the line, then quiet. A noise spike must not be a grasp.
    _, _, (reached_l, reached_r) = _probe(
        effort_l=[over] + [0.0] * 20, effort_r=[over] + [0.0] * 20, verified=True)
    assert abs(reached_l[2] - _FLOOR_L) < 1e-9 and abs(reached_r[2] - _FLOOR_R) < 1e-9, (
        "a single effort spike latched contact: left stopped at {:.4f}, right at "
        "{:.4f}".format(reached_l[2], reached_r[2]))

    # XARM_EFFORT_CONSECUTIVE in a row IS a load, and must stop the descent above
    # the calibrated floor.
    n = C.XARM_EFFORT_CONSECUTIVE
    _, _, (reached_l, reached_r) = _probe(
        effort_l=[over] * 20, effort_r=[over] * 20, verified=True)
    assert reached_l[2] > _FLOOR_L + 1e-6 and reached_r[2] > _FLOOR_R + 1e-6, (
        "{} consecutive over-threshold samples did not stop the descent (left "
        "reached the floor at {:.4f})".format(n, reached_l[2]))
    # First sample is taken at the top of the band, so the nth lands n-1 steps in.
    want = C.XARM_PROBE_BAND - (n - 1) * C.XARM_PROBE_STEP
    assert abs((reached_l[2] - _FLOOR_L) - want) < 1e-6, (
        "latched {:.4f} m above the floor, expected {:.4f} m".format(
            reached_l[2] - _FLOOR_L, want))


@check("contact probe descends both arms in lock step")
def t_probe_is_synchronised():
    _, scene, _ = _probe(effort_l=[0.0] * 40, effort_r=[0.0] * 40, verified=True)

    # Every command is a PAIR -- the probe never drives one arm on its own. This
    # is what stops the two descents drifting apart: both_movel joins, so the arms
    # are level again at every waypoint.
    assert len(scene.commands) == 1 + int(round(C.XARM_PROBE_BAND / C.XARM_PROBE_STEP)), (
        "expected one coarse move plus {} steps inside the band, got {}".format(
            int(round(C.XARM_PROBE_BAND / C.XARM_PROBE_STEP)), len(scene.commands)))
    for i, (z_l, z_r) in enumerate(scene.commands):
        gap = abs((z_l - _FLOOR_L) - (z_r - _FLOOR_R))
        assert gap <= C.XARM_PROBE_STEP + 1e-9, (
            "step {}: the arms are {:.4f} m apart, more than one probe step "
            "({:.4f} m)".format(i, gap, C.XARM_PROBE_STEP))
    # Both controllers got the same number of waypoints, which is the other half
    # of "together" -- equal spacing is no use if one arm gets extra moves.
    assert len(scene.left.arm.positions) == len(scene.right.arm.positions), (
        "left got {} waypoints, right got {}".format(
            len(scene.left.arm.positions), len(scene.right.arm.positions)))


class _RecordingSkill:
    def __init__(self):
        self.action = None

    def reset(self):
        pass

    def step(self, action, record_debug=False):
        self.action = np.asarray(action, dtype=float)
        return {'ur5e': [], 'ur16e': []}


def _bare_arena(reach_masks, crop_size):
    """An XArmDualArmArena with only the attributes step() touches.

    Built with object.__new__ rather than the constructor on purpose: the real
    __init__ connects two controllers and a RealSense. What is under test is the
    action pipeline -- click -> pixels -> skill payload -- and that is pure
    arithmetic over the calibration.
    """
    from real_robot.robot.xarm_dual_arm_arena import XArmDualArmArena

    class _Scene:
        def get_workspace_masks(self):
            return reach_masks

        def restart_camera(self):
            pass

    a = object.__new__(XArmDualArmArena)
    a.measure_time = False
    a.crop_size, a.x1, a.y1 = crop_size, 0, 0
    a.snap_to_cloth_mask = False
    a.mask_generator = None                 # masking off -> angles are 0.0
    a.cloth_mask = np.ones((crop_size, crop_size), np.uint8)
    a.dual_arm = _Scene()
    a.pick_and_fling_skill = _RecordingSkill()
    a.pick_and_place_skill = _RecordingSkill()
    a.track_trajectory = False
    a.action_step = 0
    a.all_infos = []
    a._process_info = lambda info, **kw: info
    return a


@check("the arena hands the skill the picks that were CLICKED")
def t_arena_keeps_the_clicked_picks():
    """The regression for "every fling aborts on a collision".

    DualArmArena.step sorts the two picks by PIXEL x and snaps each into one arm's
    reach annulus. That rule encodes the UR camera roll; on this cell the left arm
    sits at SMALLER pixel x, so each pick was snapped into the FAR arm's annulus.
    The annuli overlap in a band 0.07 m deep, so both picks landed in it: clicks
    0.25 m apart arrived 0.07 m apart, the 0.12 m collision check fired, and the
    fling aborted every time (the pick-and-place skill instead fell back to
    "executing sequentially", with both arms grasping in the middle).

    So: whatever the operator clicks is what the skill must receive. Assignment is
    the skill's job, from TABLE x, which is roll-independent.
    """
    from real_robot.utils.transform_utils import pixels2base_on_table
    from real_robot.utils.xarm_camera import base_to_pixel, crop_window, load_intrinsic
    from real_robot.utils.scene_utils import load_camera_to_base

    d = "{}/real_robot/calibration".format(os.environ['MP_FOLD_PATH'])
    T_l = load_camera_to_base(os.path.join(d, 'xarm-left-calib.yaml'))
    T_r = load_camera_to_base(os.path.join(d, 'xarm-right-calib.yaml'))
    S = float(np.linalg.norm((T_l @ np.linalg.inv(T_r))[:3, 3]))
    full = load_intrinsic(os.path.join(d, 'xarm-left-calib.yaml'))
    intr = crop_window(full, T_l, separation=S, table_z=C.XARM_TABLE_Z).intrinsic(full)
    side = int(intr.width)

    uu, vv = np.meshgrid(np.arange(side), np.arange(side))
    flat = np.stack([uu.ravel(), vv.ravel()], axis=1)
    masks = []
    for T in (T_l, T_r):
        r = np.linalg.norm(pixels2base_on_table(flat, intr, T, C.XARM_TABLE_Z)[:, :2], axis=1)
        masks.append(((r >= 0.12) & (r <= 0.41)).reshape(side, side))

    # Two picks a garment apart, either side of the midline, both reachable.
    want = [base_to_pixel([x, 0.0, C.XARM_TABLE_Z], T_l, intr) for x in (0.30, 0.45)]
    norm = np.array([[(p[1] / side) * 2 - 1, (p[0] / side) * 2 - 1] for p in want])

    arena = _bare_arena(tuple(masks), side)
    arena.step({'norm-pixel-pick-and-fling': norm.flatten()})
    got = arena.pick_and_fling_skill.action

    assert got is not None and len(got) == 8, (
        "the fling skill got {} values, expected 8".format(
            None if got is None else len(got)))
    for i, w in enumerate(want):
        assert np.allclose(got[2 * i:2 * i + 2], np.round(w), atol=1.5), (
            "pick {} was clicked at {} but reached the skill as {}".format(
                i, np.round(w, 1), got[2 * i:2 * i + 2]))
    assert got[6] == 1.0 and got[7] == 1.0, (
        "both picks are inside the reach annuli but the flags say {}".format(got[6:8]))

    # And the separation must survive: this is the number the 0.12 m collision
    # check sees, and the whole failure was it arriving as 0.07.
    base = [pixels2base_on_table(np.array([got[2 * i:2 * i + 2]]), intr, T_l,
                                 C.XARM_TABLE_Z)[0] for i in (0, 1)]
    sep = float(np.linalg.norm(base[0][:2] - base[1][:2]))
    assert sep > C.XARM_COLLISION_THRESHOLD, (
        "picks 0.15 m apart on the table reached the skill {:.3f} m apart, under "
        "the {:.2f} m collision threshold -- they have been collapsed again".format(
            sep, C.XARM_COLLISION_THRESHOLD))


@check("the driver sets TCP jerk and max acceleration at connect")
def t_motion_limits_applied():
    """The knob that actually makes the swing fast, and it was never turned.

    The SDK's default TCP jerk is 1000 mm/s^3 = 1 m/s^3, so the arm takes seconds
    to ramp up to a commanded acceleration and a 0.25 m swing ends long before it
    gets there. Commanded SPEED is separately clamped at 1000 mm/s inside the SDK
    (xarm/x3/xarm.py), so raising that past 1 m/s does nothing at all -- which is
    why two rounds of raising XARM_FLING_SPEED changed nothing visible.

    The controller forgets both on reboot, so "set once by hand in UFACTORY
    Studio" is not a fix; the driver has to do it on every connect.
    """
    d = _driver('left', walls=False)
    assert getattr(d.arm, 'tcp_jerk', None) == C.XARM_TCP_JERK, (
        "TCP jerk was not set at connect (controller has {}, expected {})".format(
            getattr(d.arm, 'tcp_jerk', None), C.XARM_TCP_JERK))
    assert getattr(d.arm, 'tcp_maxacc', None) == C.XARM_TCP_MAXACC, (
        "TCP max acceleration was not set at connect (controller has {}, "
        "expected {})".format(getattr(d.arm, 'tcp_maxacc', None), C.XARM_TCP_MAXACC))

    # And the commanded speed must not be a fiction: the SDK clamps at 1000 mm/s.
    assert C.XARM_FLING_SPEED <= 1.0, (
        "XARM_FLING_SPEED is {} m/s, but the SDK clamps commanded TCP speed at "
        "1.0 m/s -- the excess never reaches the controller".format(
            C.XARM_FLING_SPEED))


@check("dual pick-and-place keeps picks and places in their roles")
def t_arena_dual_pnp_ordering():
    """pick0, pick1, place0, place1 -- in, and out.

    The raw CLICK order is pick0, place0, pick1, place1, but the human policy
    reorders before building the action, so the arena receives picks-then-places.
    Reading it in click order sends place0 as pick1: every point is a real,
    reachable place on the table, so nothing errors -- the arms just grasp where
    they should have released. That is a transposition you can only catch by
    asserting the roles, which is what this does.
    """
    side = 512
    reach = np.ones((side, side), bool)
    arena = _bare_arena((reach, reach), side)

    # Four distinct points, so a transposition cannot hide behind symmetry.
    pts = np.array([[100, 120], [400, 130], [150, 380], [430, 390]], float)
    norm = np.stack([(pts[:, 1] / side) * 2 - 1, (pts[:, 0] / side) * 2 - 1], axis=1)
    arena.step({'norm-pixel-dual-pick-and-place': norm.flatten()})
    got = arena.pick_and_place_skill.action

    assert got is not None and len(got) == 12, (
        "the pick-and-place skill got {} values, expected 12".format(
            None if got is None else len(got)))
    names = ['pick0', 'pick1', 'place0', 'place1']
    for i, name in enumerate(names):
        assert np.allclose(got[2 * i:2 * i + 2], pts[i], atol=1.5), (
            "{} should be {} but the skill got {} -- picks and places have been "
            "transposed".format(name, pts[i], got[2 * i:2 * i + 2]))


@check("single pick-and-place reaches the arm on the pick's side")
def t_arena_single_pnp_arm_choice():
    """A click on the right of the table must move the RIGHT arm.

    The skill picks the arm by sorting the two pairs on table x, whole dicts
    travelling, active flag included. The arena used to hand it the SAME pick in
    both slots, which makes that comparison a tie -- and `base_x(p0) <= base_x(p1)`
    is True on a tie, so the active pair always landed on the left arm however far
    right you clicked. Duplicated points do not carry the answer, so the arena has
    to choose the arm and then make the sort agree with it.
    """
    from real_robot.primitives.utils import sort_pairs_by_table_x
    from real_robot.utils.transform_utils import pixels2base_on_table
    from real_robot.utils.xarm_camera import base_to_pixel, crop_window, load_intrinsic
    from real_robot.utils.scene_utils import load_camera_to_base

    d = "{}/real_robot/calibration".format(os.environ['MP_FOLD_PATH'])
    T_l = load_camera_to_base(os.path.join(d, 'xarm-left-calib.yaml'))
    T_r = load_camera_to_base(os.path.join(d, 'xarm-right-calib.yaml'))
    T_lr = T_l @ np.linalg.inv(T_r)
    S = float(np.linalg.norm(T_lr[:3, 3]))
    full = load_intrinsic(os.path.join(d, 'xarm-left-calib.yaml'))
    intr = crop_window(full, T_l, separation=S, table_z=C.XARM_TABLE_Z).intrinsic(full)
    side = int(intr.width)

    uu, vv = np.meshgrid(np.arange(side), np.arange(side))
    flat = np.stack([uu.ravel(), vv.ravel()], axis=1)
    masks = []
    for T in (T_l, T_r):
        r = np.linalg.norm(pixels2base_on_table(flat, intr, T, C.XARM_TABLE_Z)[:, :2], axis=1)
        masks.append(((r >= 0.12) & (r <= 0.41)).reshape(side, side))

    # x = 0.20 m is deep in the left arm's half; x = 0.55 m is deep in the right's.
    for x_pick, want_left in ((0.20, True), (0.55, False)):
        arena = _bare_arena(tuple(masks), side)
        arena.dual_arm.intr = intr
        arena.dual_arm.T_left_cam = T_l
        arena.dual_arm.T_left_right = T_lr
        pts = [base_to_pixel([x, 0.0, C.XARM_TABLE_Z], T_l, intr)
               for x in (x_pick, x_pick + 0.05)]
        norm = np.array([[(p[1] / side) * 2 - 1, (p[0] / side) * 2 - 1] for p in pts])
        arena.step({'norm-pixel-single-pick-and-place': norm.flatten()})
        got = arena.pick_and_place_skill.action
        assert got is not None and len(got) == 12, "single-pnp payload is not 12 long"

        # Now ask the SKILL's own sorter which arm ends up active -- the arena's
        # intent is only correct if the sort agrees with it.
        pair_l, pair_r = sort_pairs_by_table_x(
            {'pick': got[0:2], 'active': got[10]},
            {'pick': got[2:4], 'active': got[11]},
            intr, T_l, C.XARM_TABLE_Z)
        active_left = bool(pair_l['active']) and not bool(pair_r['active'])
        active_right = bool(pair_r['active']) and not bool(pair_l['active'])
        assert active_left or active_right, (
            "exactly one arm must be active; flags sorted to L={} R={}".format(
                pair_l['active'], pair_r['active']))
        assert active_left == want_left, (
            "a pick at table x={:.2f} m (midline {:.3f}) should drive the {} arm, "
            "but the {} arm came out active".format(
                x_pick, S / 2, "LEFT" if want_left else "RIGHT",
                "LEFT" if active_left else "RIGHT"))


@check("a narrow pick pair shrinks the swing instead of losing it")
def t_fling_fits_narrow_picks():
    """The swing is built at the ACTUAL grasp width, not XARM_FLING_WIDTH.

    xarm_points_to_fling_path is called with width=None, so the two points the
    operator picked set the geometry. Close picks push BOTH grippers further from
    their own bases -- the counter-intuitive direction -- and at hang 0.27 the arms
    must end up 0.280 m apart for the stroke waypoint to be reachable at all.

    Before _fit_swing, a closer pair made the controller refuse the swing, and the
    refusal was discarded: the arms grasped, stretched, and then just put the
    garment down. "It is not flinging any more", with nothing in the log.
    """
    from real_robot.primitives.utils import (
        retarget_path_to_grasp, xarm_points_to_fling_path)
    from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
    from real_robot.utils.transform_utils import transform_pose
    from scipy.spatial.transform import Rotation

    S, _, _, _ = _measured_cell()
    T_lr = np.eye(4)
    T_lr[:3, :3] = Rotation.from_euler('z', np.pi).as_matrix()
    T_lr[0, 3] = S

    skill = object.__new__(XArmPickAndFlingSkill)
    skill.swing_stroke = C.XARM_FLING_STROKE
    skill.hang_height = C.XARM_FLING_HANG
    skill.place_height = C.XARM_FLING_PLACE_Z
    skill.swing_windup = C.XARM_FLING_WINDUP
    skill.place_y = C.XARM_FLING_PLACE_Y
    skill.land_y = C.XARM_FLING_LAND_Y

    def builder(width):
        """The same path build dual_arm_stretch_and_fling does, at a given width."""
        centre = S / 2.0
        p_l = np.array([centre - width / 2.0, 0.0, C.XARM_FLING_HANG])
        p_r_in_l = np.array([centre + width / 2.0, 0.0, C.XARM_FLING_HANG])
        ident = Rotation.identity()

        def build(stroke, hang):
            a, b = xarm_points_to_fling_path(
                right_point=p_l, left_point=p_r_in_l, width=None,
                swing_stroke=stroke, swing_angle=C.XARM_FLING_ANGLE,
                lift_height=hang, place_height=skill.place_height,
                windup=skill.swing_windup, place_y=skill.place_y,
                land_y=skill.land_y)
            l_path = retarget_path_to_grasp(a, ident)
            r_path = retarget_path_to_grasp(b, ident)
            l_path[0][:3] = p_l
            r_path[0][:3] = p_r_in_l
            return l_path, r_path, transform_pose(np.linalg.inv(T_lr), r_path)
        return build

    # Wide enough: the configured swing must survive untouched.
    _, _, stroke, hang = skill._fit_swing(builder(C.XARM_FLING_WIDTH))
    assert (stroke, hang) == (C.XARM_FLING_STROKE, C.XARM_FLING_HANG), (
        "at the configured {:.2f} m stretch the swing was shrunk to stroke {:.3f} "
        "hang {:.3f} -- the constants no longer fit their own derivation".format(
            C.XARM_FLING_WIDTH, stroke, hang))

    # Narrow pair: it must shrink, and the result must actually fit.
    narrow = 0.22
    l_path, r_path, stroke, hang = skill._fit_swing(builder(narrow))
    assert stroke < C.XARM_FLING_STROKE or hang < C.XARM_FLING_HANG, (
        "a {:.2f} m grasp width needs more reach than {:.3f} m but the swing was "
        "not shrunk".format(narrow, C.XARM_FLING_MAX_RADIUS))
    r_own = transform_pose(np.linalg.inv(T_lr), r_path)
    worst = max(skill._path_radius(l_path), skill._path_radius(r_own))
    assert worst <= C.XARM_FLING_MAX_RADIUS + 1e-9, (
        "after shrinking, the furthest waypoint is still {:.4f} m out against a "
        "{:.3f} m limit".format(worst, C.XARM_FLING_MAX_RADIUS))
    # And it must still be a fling: forwards, and not a token one.
    assert stroke >= 0.08, "the swing was shrunk to a {:.3f} m stroke".format(stroke)


@check("get_mask_v2(None, rgb) is an all-ones placeholder of the right shape")
def t_mask_passthrough():
    """Masking off must not change the SHAPE of anything downstream.

    The arenas hand this straight to cv2.resize, cv2.erode and calculate_iou.
    A bool array, a float array or a (w, h) instead of (h, w) would all sail
    through construction and fail on hardware, mid-episode, with the arms live.
    """
    from real_robot.utils.mask_utils import get_mask_v2
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    m = get_mask_v2(None, rgb)

    assert m.shape == rgb.shape[:2], (
        "placeholder mask is {}, expected {} -- cv2.resize and calculate_iou both "
        "assume it matches the image".format(m.shape, rgb.shape[:2]))
    assert m.dtype == np.uint8, (
        "placeholder mask dtype is {}, expected uint8 -- cv2.erode rejects bool "
        "arrays".format(m.dtype))
    assert m.min() == 1 and m.max() == 1, "the placeholder must be all ones"

    # Put it through exactly what the arenas do with it, in order, rather than
    # asserting properties and hoping those were the right ones.
    import cv2
    from real_robot.utils.mask_utils import calculate_iou
    resized = cv2.resize(m, (512, 512))                       # _process_info
    eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=5)   # snap_to_cloth_mask
    iou = calculate_iou(m, m)                                 # the task's evaluate()
    assert resized.shape == (512, 512), "resize of the placeholder went wrong"
    assert np.sum(eroded) > 0, (
        "eroding the placeholder empties it, which sends snap_to_cloth_mask down "
        "its 'erosion removed entire mask' branch on every action")
    assert abs(iou - 1.0) < 1e-9, (
        "IoU of the placeholder against itself is {:.3f}, not 1.0 -- the reason "
        "success() is not to be trusted while masking is off".format(iou))


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
