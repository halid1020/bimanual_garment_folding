"""The dual xArm Lite 6 cell as a MuJoCo model, built from the CALIBRATED cell.

Nothing here is invented geometry. The two arm bases come from hand-eye
calibration (``T_left_right = T_left_cam @ inv(T_right_cam)``), the camera pose is
the left arm's calibration matrix as-is, and the camera's intrinsic is whatever
``load_intrinsic`` found -- the real one once it has been dumped off the RealSense,
a documented nominal until then. The point is that a pixel in the simulator means
the same patch of table as a pixel on the robot, so ``point_on_table_base`` and
friends can be exercised for real rather than against a fabricated camera.

FRAMES. The MuJoCo world IS the LEFT arm's base frame, exactly as the PyBullet
cell had it: left base at the origin, right base at the measured ``T_left_right``,
table top at ``z = table_z``.

⚠️ TWO CONVERSIONS ARE EASY TO GET WRONG HERE, and both fail silently-ish:

  1. THE CAMERA FRAME. OpenCV (and therefore every calibration file, and
     ``base_to_pixel``) looks down camera +z with image +y pointing DOWN. MuJoCo
     looks down camera -z with image +y pointing UP. So
         R_mujoco = R_opencv @ diag(1, -1, -1).
     Get it wrong by the obvious guess and the camera looks at the ceiling: the
     render comes back a uniform black frame, which reads like a broken GL context
     rather than a wrong sign. For our calibrated camera the correct MuJoCo
     rotation is (very nearly) the IDENTITY.

  2. THE PRINCIPAL POINT. MuJoCo's ``principalpixel`` is an OFFSET FROM THE IMAGE
     CENTRE, not an absolute pixel, and BOTH components run the other way from
     OpenCV's, because conversion 1 flips the camera's y and z:
         principalpixel = (-(ppx - W/2), -(ppy - H/2)).
     ⚠️ The x negation is easy to miss -- it is natural to flip only y, since that
     is the axis conversion 1 is about, and the measured error for a 60 px offset
     is then 120 px, i.e. exactly twice the offset. Worse, a camera whose principal
     point happens to be centred hides ALL of this: the D435i nominal we fall back
     to has ppx/ppy exactly at W/2, H/2, so the sim looked perfect and would have
     gone wrong the day the real intrinsics were dumped. That is why
     ``t_projection_round_trip`` renders a second time through a deliberately
     OFF-CENTRE principal point rather than trusting the calibrated one.

Both are asserted by ``t_projection_round_trip``, which renders markers at known
table coordinates and checks that ``base_to_pixel`` lands on them.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from real_robot.sim.fetch_lite6_assets import assets_present, urdf_path
from real_robot.utils import xarm_constants as C

# The gripper's finger tips, as an offset down the tool z axis from link_eef. The
# Lite 6 description ends at the flange -- there is no gripper in the URDF -- and
# the hardware measures the same distance as XARM_GRIPPER_OFFSET, so grasps are
# taken at this point rather than at the flange.
TCP_SITE_OFFSET = 'gripper_offset'

# Cloth resolution. 9x9 is 81 vertices, which steps at roughly 6000 Hz on this
# machine -- fast enough that the fling is not worth coarsening for.
CLOTH_COUNT = 9


def opencv_to_mujoco_rotation(R_cv):
    """Camera rotation, OpenCV convention -> MuJoCo convention. See the ⚠️ above."""
    return np.asarray(R_cv, dtype=float) @ np.diag([1.0, -1.0, -1.0])


def _quat_wxyz(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _camera_xml(T_left_cam, intr, name='top'):
    """A <camera> carrying the calibrated pose AND the calibrated intrinsic.

    MuJoCo takes focal length in pixels only alongside a physical sensor size, so
    a pixel pitch is chosen arbitrarily (1e-5 m) and both are scaled by it. Only
    the RATIO matters to the projection, so the choice is free; what is not free
    is that fx and fy must be scaled by the SAME pitch, or the image comes out
    stretched in a way that looks like a bad calibration.
    """
    pitch = 1e-5
    R_mj = opencv_to_mujoco_rotation(T_left_cam[:3, :3])
    quat = _quat_wxyz(R_mj)
    pos = T_left_cam[:3, 3]
    W, H = int(intr.width), int(intr.height)
    return (
        '<camera name="{name}" mode="fixed" pos="{px} {py} {pz}" '
        'quat="{qw} {qx} {qy} {qz}" resolution="{W} {H}" '
        'sensorsize="{sw} {sh}" focalpixel="{fx} {fy}" '
        'principalpixel="{cx} {cy}"/>'.format(
            name=name, px=pos[0], py=pos[1], pz=pos[2],
            qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3],
            W=W, H=H, sw=W * pitch, sh=H * pitch,
            fx=intr.fx, fy=intr.fy,
            cx=-(intr.ppx - W / 2.0), cy=-(intr.ppy - H / 2.0)))


def _table_xml(table_z, thickness=0.02):
    x0, x1 = C.XARM_TABLE_RECT['x']
    y0, y1 = C.XARM_TABLE_RECT['y']
    return ('<geom name="table" type="box" size="{sx} {sy} {sz}" '
            'pos="{cx} {cy} {cz}" rgba="0.82 0.80 0.76 1" '
            'friction="0.7 0.005 0.0001"/>'.format(
                sx=(x1 - x0) / 2.0, sy=(y1 - y0) / 2.0, sz=thickness / 2.0,
                cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0,
                cz=table_z - thickness / 2.0))


def _grasp_equality_xml(gripper_offset):
    """One CONNECT per arm, compiled in and INACTIVE.

    At runtime the scene retargets ``eq_obj2id`` to whichever cloth vertex the
    gripper closed on and flips ``eq_active``; the constraint itself never has to
    be rebuilt. A connect rather than a weld because a two-finger pinch on fabric
    constrains a POINT, not an orientation -- a weld would hold the cloth rigid
    about the fingertip and stiffen the whole swing.

    ``solref`` is deliberately soft. The arms are driven kinematically, so they
    are effectively infinitely heavy; asking the solver to tie 0.12 kg of cloth
    rigidly to one of them at fling speed is asking for an impulse it can only
    answer with a spike.
    """
    rows = "".join(
        '<connect name="{s}_grasp" body1="{s}/link_eef" body2="cloth_0" '
        'anchor="0 0 {off}" active="false" solref="0.01 1"/>'.format(
            s=side, off=gripper_offset)
        for side in ('left', 'right'))
    return '<equality>{}</equality>'.format(rows)


def _cloth_xml(garment_pose, garment_size, table_z, count=CLOTH_COUNT):
    gx, gy, yaw = garment_pose
    sx, sy = garment_size
    # Spacing must exceed twice the vertex radius or the compiler refuses the
    # model ("Spacing must be larger than geometry size"), and the z spacing is
    # used even for a 2D sheet, so it cannot be left at zero.
    dx, dy = sx / (count - 1), sy / (count - 1)
    spacing = max(dx, dy)
    return (
        '<flexcomp name="cloth" type="grid" dim="2" count="{n} {n} 1" '
        'spacing="{dx} {dy} {sp}" pos="{gx} {gy} {gz}" quat="{qw} 0 0 {qz}" '
        'radius="0.002" mass="0.12" rgba="0.20 0.45 0.85 1">'
        '<contact selfcollide="none" internal="false" solref="0.005 1"/>'
        '<elasticity young="2e3" poisson="0.2" thickness="1e-3"/>'
        '</flexcomp>'.format(
            n=count, dx=dx, dy=dy, sp=spacing,
            gx=gx, gy=gy, gz=table_z + 0.004,
            qw=np.cos(yaw / 2.0), qz=np.sin(yaw / 2.0)))


def build_cell(T_left_right, T_left_cam, intr, table_z=0.0,
               garment_pose=(0.37, 0.0, 0.0), garment_size=(0.26, 0.30),
               cloth_count=CLOTH_COUNT, gripper_offset=None):
    """Compile the cell -> ``(model, info)``.

    ``info`` carries the names and ids the scene needs to drive it, so no caller
    has to re-derive a name from a prefix convention.
    """
    if not assets_present():
        raise RuntimeError(
            "The Lite 6 description is missing. Run:\n"
            "    python real_robot/sim/fetch_lite6_assets.py")

    import mujoco

    gripper_offset = (C.XARM_GRIPPER_OFFSET if gripper_offset is None
                      else float(gripper_offset))
    W, H = int(intr.width), int(intr.height)

    world_xml = """<mujoco model="xarm_cell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" azimuth="-90" elevation="-25"/>
    <!-- ⚠️ offsamples="0" is REQUIRED, not a quality preference. Segmentation
         rendering encodes the object id in the colour channels, so multisampling
         BLENDS TWO IDS at every silhouette edge and hands back a third object
         that is not there. Measured with the default offsamples="4": marker discs
         that should be a tight 9x9 blob came back with 40% of their pixels
         scattered over 400 px of the frame, and the cloth mask picks up the same
         speckle -- which downstream is a grasp validity flag saying "cloth" about
         a pixel of bare table. -->
    <quality offsamples="0"/>
    <!-- ⚠️ WHERE THE VIEWER STANDS. MuJoCo's default free camera is azimuth 90,
         which looks along +y -- i.e. it puts you at -y, the BACK of this table,
         watching the cell from behind. Every direction then reads inverted: a
         fling stroking toward the front (+y) travels AWAY from you and looks like
         it is being thrown at the back wall. That is not a hypothetical; it is
         what the operator reported on first opening the viewer, about a swing
         that the camera frame and the cloth centroid both confirmed was going
         forward.
         azimuth -90 puts you at +y, at the FRONT edge, standing where the
         operator actually stands. It is set on the <global> tag above. -->
    <headlight ambient="0.5 0.5 0.5" diffuse="0.5 0.5 0.5"/>
  </visual>
  <worldbody>
    <light name="key" pos="{lx} {ly} 2.0" dir="0 0 -1" directional="true"/>
    {table}
    {camera}
  </worldbody>
</mujoco>""".format(
        W=W, H=H,
        lx=T_left_right[0, 3] / 2.0, ly=0.0,
        table=_table_xml(table_z),
        camera=_camera_xml(T_left_cam, intr))

    spec = mujoco.MjSpec.from_string(world_xml)

    # ⚠️ A FRESH spec per arm. Attaching one spec twice fails with "incompatible
    # id in body array" -- attach renames the child in place, so the second attach
    # is handed something already called left/..., and asks for right/left/... .
    poses = {'left': np.eye(4), 'right': np.asarray(T_left_right, dtype=float)}
    for side in ('left', 'right'):
        T = poses[side].copy()
        T[2, 3] += table_z
        arm = mujoco.MjSpec.from_file(urdf_path())
        base = arm.bodies[1]              # link_base; bodies[0] is the world
        frame = spec.worldbody.add_frame(pos=T[:3, 3].tolist(),
                                         quat=_quat_wxyz(T[:3, :3]))
        frame.attach_body(base, side + '/', '')

    # ⚠️ WHY THE MODEL IS FINISHED AS TEXT RATHER THAN IN THE SPEC.
    #
    # <elasticity> is an engine PLUGIN, and in MuJoCo 3.9 it does not survive
    # MjSpec: put a <flexcomp> in a spec and spec.compile() refuses the model with
    #     "flex 'cloth' is not rigid and has no equality constraints or passive
    #      forces"
    # because the flex arrives without its plugin. The same XML compiles fine
    # through MjModel.from_xml_string. But attaching two copies of a URDF is a
    # SPEC-only operation -- MJCF has no way to include a model twice under
    # different names.
    #
    # So each half is done where it works: the arms are attached in the spec, the
    # spec is emitted back to XML, and the cloth and the grasp constraints are
    # appended to that text before the plugin-aware compile. The alternative is a
    # cloth with no bending or stretch model at all (<edge equality="true">), which
    # flings like chain mail.
    #
    # The surgery is three insertions, and the assertions below check all three
    # landed -- a silently dropped flex would otherwise look like a cloth that
    # never appears in frame.
    xml = spec.to_xml()
    meshdir = os.path.dirname(os.path.abspath(urdf_path()))
    xml = xml.replace('<compiler ', '<compiler meshdir="{}" '.format(meshdir), 1)
    xml = xml.replace(
        '</worldbody>',
        _cloth_xml(garment_pose, garment_size, table_z, cloth_count)
        + '</worldbody>', 1)
    xml = xml.replace('</mujoco>', _grasp_equality_xml(gripper_offset) + '</mujoco>', 1)

    model = mujoco.MjModel.from_xml_string(xml)
    if model.nflex != 1:
        raise RuntimeError("the cloth did not survive model assembly (nflex={}); "
                           "the <flexcomp> insertion missed".format(model.nflex))
    if model.neq != 2:
        raise RuntimeError("expected 2 grasp equalities, got {}".format(model.neq))
    if meshdir not in xml:
        raise RuntimeError("meshdir was not injected; the arm meshes will not load")

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    def geoms_of(side):
        """Every geom belonging to one arm, found by BODY ownership.

        Not by geom name: the Lite 6 URDF does not name its geoms, so the
        attach prefix never reaches them and a name-prefix filter silently
        returns nothing -- and a contact check over nothing always says the arms
        never touched.
        """
        return np.array([g for g in range(model.ngeom)
                         if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                               int(model.geom_bodyid[g])) or ''
                             ).startswith(side + '/')], dtype=int)

    info = {
        'camera_name': 'top',
        'eef_body': {s: bid('{}/link_eef'.format(s)) for s in ('left', 'right')},
        'base_body': {s: bid('{}/link_base'.format(s)) for s in ('left', 'right')},
        'joint_qpos': {s: np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                              '{}/joint{}'.format(s, j + 1))
            for j in range(6)]) for s in ('left', 'right')},
        'cloth_bodies': np.array(
            [bid('cloth_{}'.format(i)) for i in range(cloth_count ** 2)]),
        'cloth_flex_id': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, 'cloth'),
        'eq_id': {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                       '{}_grasp'.format(s))
                  for s in ('left', 'right')},
        'gripper_offset': gripper_offset,
        'image_size': (W, H),
        'arm_geoms': {s: geoms_of(s) for s in ('left', 'right')},
    }
    return model, info
