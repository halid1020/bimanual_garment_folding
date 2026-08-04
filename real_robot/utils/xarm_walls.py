"""Virtual walls for the xArm Lite 6 cell: geometry, checking, and SDK encoding.

The walls are defined once in TABLE coordinates (``XARM_WALLS`` in
``xarm_constants.py``), which are the LEFT arm's base frame: x runs across the
arms toward the right base, y runs along the arm line with the FRONT of the setup
at -y, z is up. They are derived from the table rectangle
(``XARM_TABLE_RECT``) inset by ``XARM_WALL_MARGIN``, so they follow the real table
edges -- which are NOT symmetric about the arm line (0.52 m of table in front,
0.68 m behind). Each arm enforces them in its OWN base frame, so the box has to be
transformed per arm, and the y limits come out mirrored between the two.

Because the two arms are mounted face-to-face, the right base is yawed 180 deg
about z relative to the left::

    x_right = separation - x_table
    y_right = -y_table
    z_right =  z_table

A 180 deg yaw maps an axis-aligned box to an axis-aligned box, which is what
makes the controller's own safety boundary (``set_reduced_tcp_boundary``, which
takes an axis-aligned min/max per axis) usable for both arms. Any other mounting
angle would need a different mechanism.

This module is pure geometry -- it neither imports nor touches the SDK. Run it
directly for a self-check:  ``python real_robot/utils/xarm_walls.py``
"""
import numpy as np

from real_robot.utils.xarm_constants import (
    XARM_WALLS, XARM_BASE_SEPARATION, XARM_TABLE_Z, XARM_TABLE_SIZE,
    XARM_TABLE_RECT, XARM_BASE_TO_FRONT,
)

AXES = ('x', 'y', 'z')


def walls_for_side(side='left', separation=None, walls=None):
    """Table-frame walls expressed in one arm's base frame.

    ``side`` is 'left' or 'right'. Returns ``{'x': (lo, hi), 'y': ..., 'z': ...}``
    with ``lo < hi`` on every axis.
    """
    if side not in ('left', 'right'):
        raise ValueError("side must be 'left' or 'right', got {!r}".format(side))
    walls = XARM_WALLS if walls is None else walls
    separation = XARM_BASE_SEPARATION if separation is None else float(separation)

    out = {a: (float(min(walls[a])), float(max(walls[a]))) for a in AXES}
    if side == 'right':
        x_lo, x_hi = out['x']
        y_lo, y_hi = out['y']
        # Both axes flip, so the pair order reverses too.
        out['x'] = (separation - x_hi, separation - x_lo)
        out['y'] = (-y_hi, -y_lo)
    return out


def check_pose(pose, bounds):
    """Is a TCP pose inside the walls?

    ``pose`` is a UR-convention pose (metres + rotvec) or any sequence whose
    first three elements are xyz in metres. Returns ``(ok, violations)`` where
    each violation is a ready-to-print sentence.
    """
    p = np.asarray(pose, dtype=float)[:3]
    violations = []
    for i, axis in enumerate(AXES):
        lo, hi = bounds[axis]
        if p[i] < lo:
            violations.append(
                "{} = {:+.3f} m is {:.0f} mm beyond the {:+.3f} m wall".format(
                    axis, p[i], (lo - p[i]) * 1000.0, lo))
        elif p[i] > hi:
            violations.append(
                "{} = {:+.3f} m is {:.0f} mm beyond the {:+.3f} m wall".format(
                    axis, p[i], (p[i] - hi) * 1000.0, hi))
    return (not violations), violations


def to_sdk_boundary_mm(bounds):
    """Encode for ``XArmAPI.set_reduced_tcp_boundary``.

    That call wants ``[x_max, x_min, y_max, y_min, z_max, z_min]`` and casts every
    element with ``int()``, which TRUNCATES toward zero -- so rounding outward
    here would widen the box. Round each limit INWARD instead, to the mm, so the
    programmed fence is never looser than the configured walls.

    The ``round(..., 6)`` snaps to the micrometre first: the walls are derived by
    subtraction (0.6 - 0.05 = 0.5499999999999999), and rounding that inward
    without snapping would cost a whole millimetre and make the left and right
    boxes differ by 1 mm despite being geometrically identical.
    """
    out = []
    for axis in AXES:
        lo, hi = bounds[axis]
        out.append(int(np.floor(round(hi * 1000.0, 6))))   # max: round down
        out.append(int(np.ceil(round(lo * 1000.0, 6))))    # min: round up
    return out


def describe(bounds):
    return "x [{:+.3f}, {:+.3f}]  y [{:+.3f}, {:+.3f}]  z [{:+.3f}, {:+.3f}] m".format(
        bounds['x'][0], bounds['x'][1], bounds['y'][0], bounds['y'][1],
        bounds['z'][0], bounds['z'][1])


def table_to_base(point, side='left', separation=None):
    """A point in table coordinates -> the given arm's base frame."""
    separation = XARM_BASE_SEPARATION if separation is None else float(separation)
    p = np.asarray(point, dtype=float).copy()
    if side == 'right':
        p[0] = separation - p[0]
        p[1] = -p[1]
    return p


# ----------------------------------------------------------------------
def _selfcheck(verbose=True):
    """The table box and the two per-arm boxes must describe the same region."""
    S = XARM_BASE_SEPARATION
    left = walls_for_side('left')
    right = walls_for_side('right')

    for name, b in (('left', left), ('right', right)):
        for axis in AXES:
            lo, hi = b[axis]
            assert lo < hi, "{} {} wall is inverted: {}".format(name, axis, b[axis])

    # A grid of table points must be judged identically by both arms, once each
    # point is expressed in that arm's own frame.
    xs = np.linspace(XARM_WALLS['x'][0] - 0.10, XARM_WALLS['x'][1] + 0.10, 13)
    ys = np.linspace(XARM_WALLS['y'][0] - 0.10, XARM_WALLS['y'][1] + 0.10, 13)
    zs = np.linspace(XARM_WALLS['z'][0] - 0.05, XARM_WALLS['z'][1] + 0.05, 5)
    disagreements = 0
    for x in xs:
        for y in ys:
            for z in zs:
                pt = np.array([x, y, z])
                ok_l, _ = check_pose(table_to_base(pt, 'left'), left)
                ok_r, _ = check_pose(table_to_base(pt, 'right'), right)
                disagreements += int(ok_l != ok_r)
    assert disagreements == 0, "{} points judged differently by the two arms".format(disagreements)

    # Corners of the table box sit exactly ON the walls, from either side.
    for sx in (0, 1):
        for sy in (0, 1):
            corner = np.array([XARM_WALLS['x'][sx], XARM_WALLS['y'][sy], XARM_WALLS['z'][0]])
            for side, b in (('left', left), ('right', right)):
                ok, bad = check_pose(table_to_base(corner, side), b)
                assert ok, "corner {} outside the {} box: {}".format(corner, side, bad)

    mm_l, mm_r = to_sdk_boundary_mm(left), to_sdk_boundary_mm(right)
    if verbose:
        print("[xarm-walls] cell   : bases {:.3f} m apart, table {:.2f} x {:.2f} m, "
              "arm line {:.2f} m from the front edge, table_z {:+.3f} m".format(
                  S, XARM_TABLE_SIZE[0], XARM_TABLE_SIZE[1],
                  XARM_BASE_TO_FRONT, XARM_TABLE_Z))
        print("[xarm-walls] table  : x [{:+.3f}, {:+.3f}]  y [{:+.3f}, {:+.3f}] m  "
              "(edges, left base frame; front is +y)".format(
                  XARM_TABLE_RECT['x'][0], XARM_TABLE_RECT['x'][1],
                  XARM_TABLE_RECT['y'][0], XARM_TABLE_RECT['y'][1]))
        print("[xarm-walls] walls  : {}".format(describe(
            {a: tuple(XARM_WALLS[a]) for a in AXES})))
        print("[xarm-walls] left   : {}".format(describe(left)))
        print("[xarm-walls] right  : {}".format(describe(right)))
        print("[xarm-walls] SDK mm : left  {}".format(mm_l))
        print("[xarm-walls]          right {}".format(mm_r))
        print("[xarm-walls] {} grid points, both arms agree on every one."
              .format(len(xs) * len(ys) * len(zs)))
    return left, right


if __name__ == '__main__':
    _selfcheck()
