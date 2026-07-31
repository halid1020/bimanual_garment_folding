"""Single-arm pick-and-place for one xArm Lite 6.

Drop-in for ``SingleArmPickAndPlaceSkill``: same ``reset()`` / ``step(action)``
surface and the same 5-element action ``[pick_x, pick_y, place_x, place_y, rot]``
in pixel coordinates. Adapted to the Lite 6: fixed-height grasp (no force
probing), controller collision detection as the safety stop. A small base-keepout
route-around is retained for safety.
"""
import time
import numpy as np

from real_robot.utils.transform_utils import point_on_table_base
from real_robot.utils.xarm_constants import (
    XARM_TABLE_Z, XARM_MIN_Z, XARM_GRIPPER_OFFSET, XARM_APPROACH_DIST,
    XARM_LIFT_DIST, XARM_MOVE_SPEED, XARM_MOVE_ACC, XARM_WORKSPACE_RADIUS,
    XARM_DOWN_ROTVEC, XARM_TABLE_Z_BY_SIDE, XARM_GRIPPER_OFFSET_BY_SIDE,
    XARM_WORKSPACE_RADIUS_BY_SIDE, for_side,
)
from .utils import apply_local_z_rotation


class XArmSingleArmPickAndPlaceSkill:
    def __init__(self, scene, config=None):
        self.scene = scene
        if config is None:
            config = {}
        self.min_z = XARM_MIN_Z
        self.approach_dist = XARM_APPROACH_DIST
        self.lift_height = XARM_LIFT_DIST
        self.move_speed = config.get('speed', XARM_MOVE_SPEED)
        self.move_acc = config.get('acc', XARM_MOVE_ACC)
        self.home_after = True
        # The table height and the gripper length are measured PER ARM: the two
        # controllers are configured independently, so one arm's calibration is not
        # evidence about the other's.
        self.side = getattr(scene, 'side', 'left')
        self.table_z = config.get('table_z', for_side(XARM_TABLE_Z_BY_SIDE, self.side,
                                                      XARM_TABLE_Z))
        self.gripper_offset = config.get(
            'gripper_offset', for_side(XARM_GRIPPER_OFFSET_BY_SIDE, self.side,
                                       XARM_GRIPPER_OFFSET))
        self.base_safe_radius = config.get(
            'base_safe_radius',
            for_side(XARM_WORKSPACE_RADIUS_BY_SIDE, self.side, XARM_WORKSPACE_RADIUS)[0])

    def reset(self):
        self.scene.open_gripper()
        self.scene.home(blocking=True)
        time.sleep(0.5)

    def _get_collision_free_path(self, start_pose, end_pose):
        """Route around the base keepout circle if the XY segment crosses it."""
        A = start_pose[:2]
        B = end_pose[:2]
        D = B - A
        len_sq = float(np.dot(D, D))
        if len_sq == 0:
            return [end_pose]
        t = -np.dot(A, D) / len_sq
        if t <= 0 or t >= 1:
            return [end_pose]
        closest = A + t * D
        dist = np.linalg.norm(closest)
        if dist >= self.base_safe_radius:
            return [end_pose]
        if dist > 1e-3:
            dir_safe = closest / dist
        else:
            dir_safe = np.array([-D[1], D[0]])
            dir_safe = dir_safe / np.linalg.norm(dir_safe)
        waypoint = np.copy(start_pose)
        waypoint[:2] = dir_safe * self.base_safe_radius
        waypoint[2] = max(start_pose[2], end_pose[2])
        waypoint[3:6] = end_pose[3:6]
        return [waypoint, end_pose]

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if len(action) != 5:
            raise ValueError(f"Expected 5 action values (px,py,lx,ly,rot), got {len(action)}")
        pick_pixel = action[0:2]
        place_pixel = action[2:4]
        rotation_angle = action[4]

        target_rot = apply_local_z_rotation(np.array(XARM_DOWN_ROTVEC, dtype=float), rotation_angle)

        p_pick = point_on_table_base(pick_pixel[0], pick_pixel[1], self.scene.intr, self.scene.T_cam, self.table_z)
        p_place = point_on_table_base(place_pixel[0], place_pixel[1], self.scene.intr, self.scene.T_cam, self.table_z)
        # Clamp the COMMANDED z, not the table plane. Clamping first and then adding
        # the gripper offset lifts every grasp by min_z (5 mm with the table at
        # z = 0), so the fingertips stop short of the fabric -- and the floor still
        # does its job here, because it is the commanded height it needs to bound.
        p_pick[2] = max(self.min_z, float(p_pick[2]) + self.gripper_offset)
        p_place[2] = max(self.min_z, float(p_place[2]) + self.gripper_offset)

        print(f"[XArmSingleArmPnP] pick {pick_pixel} -> place {place_pixel} | rot {rotation_angle:.2f}")

        # A. Approach & fixed-height grasp
        approach_pick = np.concatenate([p_pick + [0, 0, self.approach_dist], target_rot])
        self.scene.movel(approach_pick, speed=self.move_speed, acc=self.move_acc)
        self.scene.open_gripper()
        grasp = np.concatenate([p_pick, target_rot])
        self.scene.movel(grasp, speed=self.move_speed * 0.5, acc=self.move_acc * 0.5)
        self.scene.close_gripper()
        time.sleep(0.3)

        # B. Lift, transit (route around base if needed), place
        lift_pt = np.concatenate([p_pick + [0, 0, self.lift_height], target_rot])
        approach_place = np.concatenate([p_place + [0, 0, self.approach_dist], target_rot])
        final_place = np.concatenate([p_place + [0, 0, 0.01], target_rot])
        transit = self._get_collision_free_path(lift_pt, approach_place)
        full_path = [lift_pt] + transit + [final_place]
        self.scene.movel(full_path, speed=self.move_speed, acc=self.move_acc)

        # C. Release & retract
        self.scene.open_gripper()
        time.sleep(0.2)
        retract = np.concatenate([p_place + [0, 0, self.approach_dist], target_rot])
        self.scene.movel(retract, speed=self.move_speed, acc=self.move_acc)

        if self.home_after:
            self.scene.home()
