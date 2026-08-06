"""Real dual-arm arena for two xArm Lite 6 arms.

Subclasses ``DualArmArena`` and overrides ``__init__`` (to build the xArm scene
and xArm skills under the same attribute names the parent uses --
``self.dual_arm``, ``self.pick_and_place_skill``, ``self.pick_and_fling_skill``)
and ``step`` (see below). ``reset``, ``_process_info``, ``evaluate`` and
``success`` are robot-agnostic and inherited unchanged.

We deliberately do NOT call ``super().__init__`` (the parent constructor connects
the UR5e+UR16e); the ~config-plumbing below is replicated so the UR file stays
untouched.

⚠️ WHY ``step`` IS OVERRIDDEN. The parent's ``step`` decides which arm gets which
pick from PIXEL x ("larger pixel-x -> robot 0") and then snaps each pick into that
arm's reach annulus. Both halves are wrong on this cell:

  * the pixel-x rule encodes one particular camera roll. Hand-eye says our left
    arm is at SMALLER pixel x, so the rule hands each arm the other one's target;
  * snapping then drags each pick across the table into the far arm's annulus. The
    two annuli overlap in a band only 0.07 m deep, so BOTH picks land in it. That
    is not theory -- it is why every fling aborted with "collision predicted
    (min dist 0.017-0.035 m)" against the 0.12 m threshold. Clicks 0.25 m apart
    came out 0.07 m apart, and the abort was correct: the arms really would have
    collided.

The xArm skills already assign picks by TABLE x (``sort_pairs_by_table_x``), which
is roll-independent, so the arena must not pre-assign at all. And a click is a
grasp the operator chose: snapping it into a reach ring silently grasps empty
table, so unreachable clicks are REJECTED via the validity flags instead, which
makes the skill abort loudly. That is the same contract
``test/test_xarm_pixel_primitives_ui.py`` runs.
"""
import os
import time

import cv2
import numpy as np

from real_robot.robot.dual_arm_arena import DualArmArena
from real_robot.robot.utils import get_grasp_rotation, snap_to_mask
from real_robot.robot.xarm_dual_arm_scene import XArmDualArmScene
from real_robot.primitives.xarm_pick_and_place import XArmPickAndPlaceSkill
from real_robot.primitives.xarm_pick_and_fling import XArmPickAndFlingSkill
from real_robot.loggers.pixel_based_primitive_env_imp_logger import PixelBasedPrimitiveImpEnvLogger
from real_robot.utils.mask_utils import get_mask_generator
from real_robot.utils.xarm_constants import XARM_WORKSPACE_RADIUS, XARM_TABLE_Z

# Same fallback as every xArm test script, and `source ./setup.sh xarm` exports
# both. The config normally supplies them; these only cover a caller that builds
# the arena by hand.
DEFAULT_LEFT_IP = os.environ.get('XARM_LEFT_IP', '192.168.1.155')
DEFAULT_RIGHT_IP = os.environ.get('XARM_RIGHT_IP', '192.168.1.170')


def _warn_no_cloth_mask():
    """Say it once, at the top, where the operator will actually see it.

    Running without a segmenter is fine for driving the arms and wrong for
    reading the numbers, and the difference is easy to forget an hour into a
    session -- so it gets a banner rather than a log line.
    """
    print("=" * 72)
    print("!! CLOTH MASKING IS OFF (use_cloth_mask: False).")
    print("   The mask is an all-ones placeholder, so the ARMS behave normally but")
    print("   coverage, the IoUs and success() are NOT real numbers -- success can")
    print("   read True from the very first step. Do not report anything from this run.")
    print("   To fix: put sam_vit_h_4b8939.pth in $REAL_ROBOT_PATH/models/ and set")
    print("   use_cloth_mask: True in the arena config.")
    print("=" * 72)


class XArmDualArmArena(DualArmArena):
    def __init__(self, config):
        self.name = "xarm_dual_arm_garment_arena"
        self.config = config
        self.draw_fatten_contour = False
        self.measure_time = config.get('measure_time', False)
        # 0, not the UR cell's 60. XArmDualArmScene.take_rgbd() already returns the
        # square window centred between the two arms, so the parent's own
        # crop_size = min(h, w) crop is deliberately an IDENTITY here and needs no
        # nudge. The 60 px inherited from the UR arena was a hand-tuned offset for
        # a camera that was not above the midpoint; applying it on top of a square
        # frame would shift the window off the end of the image.
        self.roi_off_x = config.get('roi_off_x', 0)

        dry_run = config.get("dry_run", False)
        self.dual_arm = XArmDualArmScene(
            left_robot_ip=config.get("left_ip", DEFAULT_LEFT_IP),
            right_robot_ip=config.get("right_ip", DEFAULT_RIGHT_IP),
            dry_run=dry_run,
            workspace_radius=config.get("workspace_radius", XARM_WORKSPACE_RADIUS),
        )

        self.pick_and_place_skill = XArmPickAndPlaceSkill(self.dual_arm)
        self.pick_and_fling_skill = XArmPickAndFlingSkill(self.dual_arm)
        self.logger = PixelBasedPrimitiveImpEnvLogger()

        # None means "no segmenter", which get_mask_v2 answers with an all-ones
        # placeholder. Default True, so a cell that HAS the checkpoint behaves
        # exactly as before; this exists only so the cell can be driven before the
        # 2.4 GB SAM download is in place.
        self.use_cloth_mask = config.get("use_cloth_mask", True)
        self.mask_generator = get_mask_generator() if self.use_cloth_mask else None
        if self.mask_generator is None:
            _warn_no_cloth_mask()

        self.num_train_trials = config.get("num_train_trials", 100)
        self.num_val_trials = config.get("num_val_trials", 10)
        self.num_eval_trials = config.get("num_eval_trials", 30)
        self.action_horizon = config.get("action_horizon", 20)
        self.snap_to_cloth_mask = config.get("snap_to_cloth_mask", False)
        self.init_from = config.get("init_from", "crumpled")
        self.maskout_background = config.get("maskout_background", False)
        self.use_sim_workspace = config.get("use_sim_workspace", False)
        self.asset_dir = f"{os.environ['MP_FOLD_PATH']}/assets"
        self.track_trajectory = config.get("track_trajectory", False)

        self.current_episode = None
        self.frames = []
        self.all_infos = []
        self.goal = None
        self.debug = config.get("debug", False)

        self.resolution = (512, 512)
        self.action_step = 0
        self.evaluate_result = None
        self.last_flattened_step = -1
        self.id = 0
        self.init_coverage = None
        self.flattened_obs = None

        print('[XArmDualArmArena] Finished init.')

    # ------------------------------------------------------------------
    # Action -> skill
    # ------------------------------------------------------------------
    def _reach_mask(self):
        """Where EITHER arm can reach, in crop pixels.

        The union, not per-arm, because at this point we do not yet know which arm
        gets which pick -- the skill decides that from table x. So the arena's job
        is only "could anybody reach this?", and the skill's own valid-flag abort
        catches the rest.
        """
        left, right = self.dual_arm.get_workspace_masks()
        return np.logical_or(np.asarray(left), np.asarray(right))

    def _table_point(self, px):
        """Crop pixel -> point on the table, in the LEFT base frame."""
        from real_robot.utils.transform_utils import point_on_table_base
        return np.asarray(point_on_table_base(
            px[0], px[1], self.dual_arm.intr, self.dual_arm.T_left_cam,
            XARM_TABLE_Z), dtype=float)

    def _table_pixel(self, p):
        """The inverse: a left-base-frame table point -> crop pixel."""
        from real_robot.utils.xarm_camera import base_to_pixel
        return np.asarray(base_to_pixel([p[0], p[1], XARM_TABLE_Z],
                                        self.dual_arm.T_left_cam,
                                        self.dual_arm.intr), dtype=float)

    def _choose_arm(self, pick_crop, place_crop):
        """Which arm should do a one-armed pick-and-place. Returns (use_left, why).

        Reachability first -- an arm that cannot reach both ends is not a
        candidate, whatever the geometry says. Only when both can reach does the
        cell midline decide, which is the same "nearer base wins" rule the dual
        skills use.
        """
        left, right = (np.asarray(m) for m in self.dual_arm.get_workspace_masks())

        def covers(mask):
            for pt in (pick_crop, place_crop):
                x, y = int(pt[0]), int(pt[1])
                if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]
                        and mask[y, x]):
                    return False
            return True

        in_l, in_r = covers(left), covers(right)
        if in_l and not in_r:
            return True, "only the left arm reaches both points"
        if in_r and not in_l:
            return False, "only the right arm reaches both points"
        mid = float(np.linalg.norm(self.dual_arm.T_left_right[:3, 3])) / 2.0
        x = self._table_point(pick_crop)[0]
        near_left = x < mid
        if in_l and in_r:
            return near_left, "both reach; pick is at x={:.3f} m, {} the {:.3f} m " \
                              "midline".format(x, "before" if near_left else "after", mid)
        return near_left, "NEITHER arm reaches both points -- the skill will abort"

    def _pick_angle(self, pt_crop):
        """Grasp rotation from the cloth outline, or 0 when there is no outline.

        With masking off the cloth mask is an all-ones placeholder, and
        get_grasp_rotation on a rectangle returns the rectangle's orientation --
        a confident number about nothing. Zero is the honest answer.
        """
        if self.mask_generator is None:
            return 0.0
        return get_grasp_rotation(self.cloth_mask, pt_crop)

    def step(self, action):
        """xArm action pipeline. See the module docstring for why this is not
        inherited."""
        if self.measure_time:
            start_time = time.time()

        norm_pixels = np.array(list(action.values())[0]).reshape(-1, 2)
        norm_pixels[:, [0, 1]] = norm_pixels[:, [1, 0]]
        action_type = list(action.keys())[0]

        full_action = None
        points_executed = None

        if action_type in ('norm-pixel-dual-pick-and-place',
                           'norm-pixel-single-pick-and-place',
                           'norm-pixel-pick-and-fling'):
            points_crop = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.int32)

            # Cloth snapping only, and only when there is a real cloth mask. The
            # parent also snaps to the arms' reach annuli; we do not -- see the
            # module docstring.
            if self.snap_to_cloth_mask and self.mask_generator is not None:
                is_pick = {
                    'norm-pixel-pick-and-fling': lambda i: True,
                    'norm-pixel-dual-pick-and-place': lambda i: i < 2,
                    'norm-pixel-single-pick-and-place': lambda i: i == 0,
                }[action_type]
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(self.cloth_mask, kernel, iterations=5)
                target = eroded if np.sum(eroded) else self.cloth_mask
                points_crop = np.array([
                    snap_to_mask(pt, target) if is_pick(i) else pt
                    for i, pt in enumerate(points_crop)])

            reach = self._reach_mask()

            def reachable(pt_crop):
                x, y = int(pt_crop[0]), int(pt_crop[1])
                if 0 <= y < reach.shape[0] and 0 <= x < reach.shape[1]:
                    return 1.0 if reach[y, x] else 0.0
                return 0.0

            points_orig = points_crop + np.array([self.x1, self.y1])
            points_executed = points_orig.flatten()

            if action_type == 'norm-pixel-pick-and-fling' and len(points_orig) == 2:
                p0, p1 = points_orig[0], points_orig[1]
                # NO pixel-x sort and NO snapping: the skill assigns by table x.
                angles = [self._pick_angle(points_crop[0]),
                          self._pick_angle(points_crop[1])]
                flags = [reachable(points_crop[0]), reachable(points_crop[1])]
                full_action = np.concatenate([p0, p1, angles, flags])

            elif action_type == 'norm-pixel-dual-pick-and-place' and len(points_orig) == 4:
                # ⚠️ The ACTION is already pick0, pick1, place0, place1 -- the human
                # policy reorders the clicks before building it
                # (real_world_human_policy.py: concatenate([pick_0, pick_1,
                # place_0, place_1])). The raw CLICK order is pick0, place0, pick1,
                # place1, and reading the action in click order sends place0 as
                # pick1: the arms go to the right places in the wrong roles, which
                # looks like a broken primitive rather than a transposition.
                pk0, pk1, pl0, pl1 = points_orig
                angles = [self._pick_angle(points_crop[0]),
                          self._pick_angle(points_crop[1])]
                flags = [reachable(points_crop[0]), reachable(points_crop[1])]
                points_executed = np.concatenate([pk0, pk1, pl0, pl1])
                full_action = np.concatenate([pk0, pk1, pl0, pl1, angles, flags])

            elif action_type == 'norm-pixel-single-pick-and-place' and len(points_orig) == 2:
                # One arm only, driven through the DUAL skill with the other arm's
                # active flag cleared.
                #
                # ⚠️ THE OTHER SLOT MUST NOT BE A COPY OF THIS PICK. The skill sorts
                # the two pairs with `base_x(pair_0) <= base_x(pair_1)`, and whole
                # dicts travel, active flag included. Duplicate the pick into both
                # slots and that comparison is a tie -- `<=` is always True, so the
                # active pair always lands on the LEFT arm no matter where you
                # clicked. So: choose the arm here, then place a dummy on the far
                # side of the real pick so the sort has a real question to answer.
                pick, place = points_orig
                use_left, why = self._choose_arm(points_crop[0], points_crop[1])
                p = self._table_point(pick)
                dummy = p.copy()
                dummy[0] += 0.15 if use_left else -0.15
                dummy_px = self._table_pixel(dummy)

                angle = self._pick_angle(points_crop[0])
                ok = reachable(points_crop[0])
                points_executed = np.concatenate([pick, place])
                pairs = ([pick, dummy_px, place, dummy_px, [angle, 0.0], [ok, 0.0]]
                         if use_left else
                         [dummy_px, pick, dummy_px, place, [0.0, angle], [0.0, ok]])
                full_action = np.concatenate(pairs)
                print("[XArmDualArmArena] single pick-and-place -> {} arm ({})"
                      .format('LEFT' if use_left else 'RIGHT', why))

            if full_action is not None:
                print("[XArmDualArmArena] {}: picks {} reachable {}".format(
                    action_type, points_crop[:2].tolist(),
                    full_action[-2:].tolist()))

        if self.measure_time:
            self.process_action_time.append(time.time() - start_time)
            start_time = time.time()

        self.info = {}
        if action_type in ('norm-pixel-dual-pick-and-place',
                           'norm-pixel-single-pick-and-place'):
            self.pick_and_place_skill.reset()
            self.pick_and_place_skill.step(full_action)
        elif action_type == 'norm-pixel-pick-and-fling':
            self.pick_and_fling_skill.reset()
            self.info['debug_trajectory'] = self.pick_and_fling_skill.step(
                full_action, record_debug=self.track_trajectory)
        elif action_type == 'no-operation':
            pass
        else:
            raise ValueError(action_type)

        self.action_step += 1

        if self.measure_time:
            self.primitive_time.append(time.time() - start_time)
            start_time = time.time()

        if self.action_step % 5 == 0:
            self.dual_arm.restart_camera()

        self.all_infos.append(self.info)
        self.info = self._process_info(self.info)

        if points_executed is not None:
            applied = (1.0 * points_executed.reshape(-1, 2)
                       - np.array([self.x1, self.y1])) / self.crop_size * 2 - 1
            applied[:, [0, 1]] = applied[:, [1, 0]]
            self.info['applied_action'] = {action_type: applied.flatten()}

        if self.measure_time:
            self.perception_time.append(time.time() - start_time)

        return self.info
