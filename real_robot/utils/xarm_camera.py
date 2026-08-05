"""Top-down camera geometry for the xArm Lite 6 cell: projection, and the square
crop that every consumer of the camera sees.

THE CROP CONTRACT. The scene hands out a square window of the table centred on the
midpoint of the two arm bases, together with an intrinsic whose principal point has
been shifted into that window (``ppx -= x1``, ``ppy -= y1``). The extrinsics are
untouched -- cropping moves the image origin, not the camera. Because the shifted
intrinsic travels with the cropped image, ``point_on_table_base`` and friends invert
crop pixels straight back to the same metres, so nothing downstream -- the
primitives, the workspace masks, the trajectory overlay -- needs to know a crop
happened. The alternative (crop for display, remap every click) puts an offset at
every call site, and one missed site is a silently mis-aimed grasp.

Kept out of ``xarm_test_scene.py`` so the production scenes do not import a test
module; that file re-exports ``base_to_pixel`` under its original name.
"""
import numpy as np
import yaml

from real_robot.utils import xarm_constants as C


class PinholeIntrinsic:
    """A pinhole intrinsic that duck-types the pyrealsense2 intrinsics object.

    The production scenes put the RealSense object straight into ``scene.intr``,
    and every helper here (plus ``intrinsic_to_params``) reads only
    ``fx/fy/ppx/ppy/width/height``. This is the same shape, for the cases where
    there is no camera plugged in: the simulator, and the offline tests.

    ``xarm_test_scene.SyntheticIntrinsic`` is the same idea, but lives in a test
    module -- and the whole reason this file exists is so production code does not
    import test modules to do its projection.
    """

    def __init__(self, fx, fy, ppx, ppy, width, height, source='unspecified'):
        self.fx, self.fy = float(fx), float(fy)
        self.ppx, self.ppy = float(ppx), float(ppy)
        self.width, self.height = int(width), int(height)
        self.source = source

    def __repr__(self):
        return ("PinholeIntrinsic(fx={:.1f}, fy={:.1f}, ppx={:.1f}, ppy={:.1f}, "
                "{}x{}, from {})".format(self.fx, self.fy, self.ppx, self.ppy,
                                         self.width, self.height, self.source))


# Nominal RealSense D435i COLOUR intrinsic at 1280x720, derived from the
# datasheet field of view (69.4 x 42.5 deg): fx = (W/2)/tan(69.4/2 deg) = 924.5,
# fy = (H/2)/tan(42.5/2 deg) = 926.0.
#
# ⚠️ This is a NOMINAL, not a measurement. Per-device values differ by a percent
# or so, and the number decides whether the base-to-base crop fits the sensor at
# all (see XARM_CROP_SIZE), so it is worth a minute with the real camera:
#     python real_robot/calibration/xarm_hand_to_eye_calib.py --dump-intrinsics
# writes the real one into the calib YAML, and load_intrinsic() then prefers it.
XARM_CAM_INTRINSIC_D435I = dict(fx=924.5, fy=926.0, ppx=640.0, ppy=360.0,
                                width=1280, height=720)


def load_intrinsic(yaml_path=None, verbose=True):
    """The camera intrinsic, measured if we have it -> ``PinholeIntrinsic``.

    Reads an ``intrinsics:`` block out of a hand-eye calibration YAML. Hand-eye
    only ever saved the EXTRINSIC, so files written before this existed have no
    such block; those fall back to the D435i nominal above, loudly, because a
    guessed focal length silently changes how many metres a pixel is worth.
    """
    if yaml_path is not None:
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except OSError:
            data = {}
        block = data.get('intrinsics')
        if block:
            return PinholeIntrinsic(source=yaml_path, **{
                k: block[k] for k in ('fx', 'fy', 'ppx', 'ppy', 'width', 'height')})

    if verbose:
        print("[xarm-camera] !! no measured intrinsic{} -- using the D435i NOMINAL "
              "({fx:.0f} px at {width}x{height}). Run xarm_hand_to_eye_calib.py "
              "--dump-intrinsics to replace it.".format(
                  " in {}".format(yaml_path) if yaml_path else "",
                  **XARM_CAM_INTRINSIC_D435I))
    return PinholeIntrinsic(source='D435i nominal', **XARM_CAM_INTRINSIC_D435I)


def base_to_pixel(points, T_cam, intr):
    """Project 3D base-frame points to pixels -- the FULL pinhole projection.

    ``table_xy_to_pixel`` only inverts ``point_on_table_base`` for points ON the
    table plane. Overlaying a trajectory needs lifted waypoints projected too, and
    using the table-plane inverse for those would draw them in the wrong place.
    Returns (N, 2), or (2,) for a single point. Points behind the camera come back
    as NaN rather than mirrored in front of it.
    """
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = pts.reshape(1, -1)[:, :3] if single else pts[:, :3]

    R, t = T_cam[:3, :3], T_cam[:3, 3]
    p_cam = (R.T @ (pts - t).T).T          # base -> camera frame
    z = p_cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = intr.ppx + intr.fx * p_cam[:, 0] / z
        v = intr.ppy + intr.fy * p_cam[:, 1] / z
    out = np.stack([u, v], axis=1)
    out[z <= 1e-9] = np.nan
    return out[0] if single else out


class CroppedIntrinsic:
    """``intr`` shifted into a crop window.

    Duck-types the pyrealsense2 intrinsics object the production scenes put in
    ``scene.intr`` -- ``intrinsic_to_params`` and every projection helper here read
    only ``fx/fy/ppx/ppy`` -- exactly as ``SyntheticIntrinsic`` does. Focal lengths
    are unchanged: a crop is a change of image origin, not of the lens.
    """

    def __init__(self, intr, x1, y1, width, height=None):
        self.fx, self.fy = float(intr.fx), float(intr.fy)
        self.ppx = float(intr.ppx) - float(x1)
        self.ppy = float(intr.ppy) - float(y1)
        self.width = int(width)
        self.height = int(width if height is None else height)

    def __repr__(self):
        return ("CroppedIntrinsic(fx={:.1f}, fy={:.1f}, ppx={:.1f}, ppy={:.1f}, "
                "{}x{})".format(self.fx, self.fy, self.ppx, self.ppy,
                                self.width, self.height))


class CropWindow:
    """The square window: origin ``(x1, y1)`` and side ``side`` pixels."""

    def __init__(self, x1, y1, side, size_m, clamped=False):
        self.x1, self.y1, self.side = int(x1), int(y1), int(side)
        self.size_m = float(size_m)
        self.clamped = bool(clamped)

    def apply(self, img):
        """Crop an image (or a mask) to this window."""
        if img is None:
            return None
        return img[self.y1:self.y1 + self.side, self.x1:self.x1 + self.side]

    def intrinsic(self, intr):
        return CroppedIntrinsic(intr, self.x1, self.y1, self.side)

    def __repr__(self):
        return "CropWindow(x1={}, y1={}, side={} px, {:.3f} m{})".format(
            self.x1, self.y1, self.side, self.size_m,
            ", CLAMPED" if self.clamped else "")


def _image_size(intr, image_shape=None):
    if image_shape is not None:
        return int(image_shape[1]), int(image_shape[0])
    return int(getattr(intr, 'width')), int(getattr(intr, 'height'))


def crop_window(intr, T_cam, separation=None, table_z=0.0,
                size_m=None, image_shape=None, verbose=False):
    """The square crop centred between the two arm bases.

    ``T_cam`` is the camera-to-LEFT-base transform, so the centre is the projection
    of ``(separation/2, 0, table_z)``. The side defaults to ``separation`` itself,
    which puts the two arm BASE CENTRES exactly on the left and right edges of the
    window -- the property the crop is for, expressed as geometry rather than as a
    number that has to be kept in step with the cell. Pass ``size_m`` to override.

    The metric side assumes the camera looks straight DOWN: it converts metres to
    pixels at the table plane through a single scale. That is exact for the
    placeholder camera and stays a good approximation for a small mounting tilt,
    but after hand-eye calibration a genuinely tilted camera makes the window
    slightly non-square on the table. The centre is exact either way -- it is a
    real projection.
    """
    separation = C.XARM_BASE_SEPARATION if separation is None else separation
    size_m = separation if size_m is None else size_m
    width, height = _image_size(intr, image_shape)

    centre = base_to_pixel([separation / 2.0, C.XARM_CAM_CENTRE_Y, table_z],
                           T_cam, intr)
    if not np.isfinite(centre).all():
        raise ValueError("the arm midpoint does not project into the image; "
                         "check T_cam ({}), it may be inverted".format(T_cam[:3, 3]))

    cam_height = float(T_cam[2, 3]) - float(table_z)
    if cam_height <= 0:
        raise ValueError("camera is at or below the table plane (height "
                         "{:.3f} m)".format(cam_height))
    side = int(round(size_m * min(abs(intr.fx), abs(intr.fy)) / cam_height))

    clamped = False
    if side > min(width, height):
        side = int(min(width, height))
        clamped = True
    x1 = int(round(centre[0] - side / 2.0))
    y1 = int(round(centre[1] - side / 2.0))
    cx1, cy1 = int(np.clip(x1, 0, width - side)), int(np.clip(y1, 0, height - side))
    clamped = clamped or (cx1, cy1) != (x1, y1)

    window = CropWindow(cx1, cy1, side, size_m, clamped)
    if clamped:
        # Not cosmetic: a clamped window is no longer centred on the arm midpoint,
        # so every pixel handed to a primitive is offset from where it was clicked.
        #
        # The camera HEIGHT is not the knob: it is fixed by hand-eye calibration.
        # The knob is the field of view. A D435i colour stream at 1280x720 sees
        # only ~0.65 m along the arm line from the calibrated 0.84 m, which is less
        # than the 0.75 m base separation, so the base-to-base window cannot fit it;
        # the depth stream is far wider (~0.93 m) and does.
        print("[xarm-camera] !! crop {} was clamped to fit {}x{} -- it is NO LONGER "
              "centred between the arms. It needs {} px of a {} px frame. Use a "
              "wider stream, or pass a smaller size_m and accept that the arm bases "
              "sit outside the window.".format(
                  window, width, height,
                  int(round(size_m * min(abs(intr.fx), abs(intr.fy)) / cam_height)),
                  min(width, height)))
    elif verbose:
        print("[xarm-camera] crop {} centred on the arm midpoint at pixel "
              "({:.1f}, {:.1f})".format(window, centre[0], centre[1]))
    return window


def crop_image(img, window):
    return window.apply(img)


# --- Which way up is the frame? ----------------------------------------------
# The intended layout, stated once: the LEFT arm on the left of the frame, the
# RIGHT arm on the right, the FRONT edge of the table at the top. For a camera
# looking straight down those are one choice rather than two -- u -> +x forces
# v -> -y -- which is why the front edge is at base +y (see XARM_TABLE_RECT).
def describe_orientation(T_cam, intr, separation=None, table_z=0.0,
                         base_to_front=None):
    """How the table is laid out in the frame -> ``(ok, message)``.

    ``ok`` is True when the frame matches the intended layout above. On hardware
    the layout comes from how the camera is physically bolted on, via hand-eye
    calibration -- this repo cannot choose it, only report it.

    Nothing depends on ``ok``: the primitives assign arms by table position
    (``sort_pairs_by_table_x``), so a rotated mount gives a view you dislike, not a
    swapped grasp. It is printed at scene init because a frame that comes up
    mirrored is otherwise found by watching a garment get folded the wrong way
    round, which is a slow and expensive way to learn how a bracket is oriented.
    """
    separation = C.XARM_BASE_SEPARATION if separation is None else separation
    base_to_front = C.XARM_BASE_TO_FRONT if base_to_front is None else base_to_front

    left = base_to_pixel([0.0, 0.0, table_z], T_cam, intr)
    right = base_to_pixel([separation, 0.0, table_z], T_cam, intr)
    front = base_to_pixel([separation / 2.0, base_to_front, table_z], T_cam, intr)
    back = base_to_pixel([separation / 2.0, base_to_front - C.XARM_TABLE_SIZE[1],
                          table_z], T_cam, intr)
    if not all(np.isfinite(p).all() for p in (left, right, front, back)):
        return False, ("[xarm-camera] orientation UNKNOWN: part of the table does not "
                       "project into the image (check T_cam)")

    arms_ok = right[0] > left[0]
    front_ok = front[1] < back[1]
    msg = "[xarm-camera] frame: {} arm on the left, {} at the top".format(
        "left" if arms_ok else "RIGHT", "front" if front_ok else "BACK")
    if arms_ok and front_ok:
        return True, msg + " -- as intended"
    return False, (msg + " -- NOT the intended layout (left arm left, front at top). "
                   "Grasps are still correct (arms are assigned on the table, not by "
                   "pixel column); the view is mirrored/rotated w.r.t. the convention "
                   "in xarm_constants.py.")
