import numpy as np
from typing import Dict, Any, List, Optional
import cv2
import time
import os

from real_robot.robot.single_arm_scene import SingleArmScene
from real_robot.robot.utils import get_grasp_rotation, snap_to_mask, process_depth
from real_robot.utils.mask_utils import get_mask_generator, get_mask_v2
from real_robot.utils.transform_utils import MOVE_ACC, MOVE_SPEED
from real_robot.primitives.single_arm_pick_and_place import SingleArmPickAndPlaceSkill
from real_robot.loggers.single_arm_pixel_logger import SingleArmPixelLogger
from actoris_harena import Arena

class SingleArmPickAndPlaceArena(Arena):
    """
    Real Single-Arm Arena implementation using only UR5e with only Pick and Place Primitive
    """

    def __init__(self, config):
        self.name = "single_arm_garment_pick_and_place_arena"
        self.config = config
        self.measure_time = config.get('measure_time', False)

        # Robot initialization
        dry_run = config.get("dry_run", False)
        
        self.single_arm = SingleArmScene(
            ur5e_robot_ip=config.get("ur5e_ip", "192.168.1.10"),
            dry_run=dry_run
        )
        
        # Use the new Single Arm Skill
        self.pick_and_place_skill = SingleArmPickAndPlaceSkill(self.single_arm)
        
        self.logger = SingleArmPixelLogger()
        self.mask_generator = get_mask_generator()

        # Arena parameters
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
        
        # Initialize flattened storage
        self.flattened_obs = None
        
        print('Finished init SingleArmPickAndPlaceArena')

    def reset(self, episode_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reset the arena (move both arms home, open grippers, capture initial observation).
        """
        self.eid = episode_config.get("eid", np.random.randint(0, 9999)) if episode_config else 0
        print(f"[Arena] Resetting episode {self.eid}")

        # Ensure we have the flattened observation (loaded or captured) before starting
        self.flattened_obs = None
        self.get_flattened_obs()
        
        self.last_info = None
        self.action_step = 0
        
        # Reset robot to safe state
        self.init_coverage = None
        self.task.reset(self)
        if self.init_from == 'crumpled':
            input("Press [Enter] to finish resetting cloth state to a crumpled state...")
        
        self.info = {}
        self.all_infos = [self.info]
        self.info = self._process_info(self.info)
        
        self.init_coverage = self.coverage
        self.clear_frames()
        self.primitive_time = []
        self.perception_time = []
        self.process_action_time = []
        self.all_infos = [self.info]
        return self.info

    def get_flattened_obs(self):
        if self.flattened_obs is None:
            self.garment_id = input("\n[Arena] Enter garment name (e.g. shirt_01): ").strip()
            base_asset_dir = f"{os.environ.get('MP_FOLD_PATH', '.')}/assets"
            save_dir = os.path.join(base_asset_dir, 'real_garments', self.garment_id)
            
            fn_rgb = os.path.join(save_dir, "rgb.png")
            fn_raw_rgb = os.path.join(save_dir, "raw_rgb.png")
            fn_depth = os.path.join(save_dir, "depth.png")
            fn_mask = os.path.join(save_dir, "mask.png")
            fn_r0_mask = os.path.join(save_dir, "robot0_mask.png")
            fn_r1_mask = os.path.join(save_dir, "robot1_mask.png")
            fn_info = os.path.join(save_dir, "info.json")

            if os.path.exists(save_dir) and os.path.exists(fn_info):
                print(f"[Arena] Found cached observation folder for '{self.garment_id}'. Loading images...")
                try:
                    rgb = cv2.cvtColor(cv2.imread(fn_rgb), cv2.COLOR_BGR2RGB)
                    raw_rgb = cv2.cvtColor(cv2.imread(fn_raw_rgb), cv2.COLOR_BGR2RGB)
                    depth = cv2.imread(fn_depth, cv2.IMREAD_UNCHANGED)
                    mask_img = cv2.imread(fn_mask, cv2.IMREAD_GRAYSCALE)
                    mask = (mask_img > 127).astype(np.bool_)
                    r0_mask = (cv2.imread(fn_r0_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.bool_)
                    r1_mask = (cv2.imread(fn_r1_mask, cv2.IMREAD_GRAYSCALE) > 127).astype(np.bool_)

                    with open(fn_info, 'r') as f:
                        meta_info = json.load(f)

                    self.flattened_obs = {
                        'observation': {
                            "rgb": rgb,
                            "depth": depth,
                            "mask": mask,
                            "raw_rgb": raw_rgb,
                            "action_step": meta_info.get("action_step", 0),
                            "robot0_mask": r1_mask,
                            "robot1_mask": r0_mask,
                        },
                        "eid": meta_info.get("eid", 0),
                        "arena_id": 0,
                        "arena": self
                    }
                    
                    self.flatten_coverage = np.sum(mask)
                    print("[Arena] Successfully loaded flattened state from PNGs/JSON.")

                except Exception as e:
                    print(f"[Arena] Error loading data: {e}. Will recapture manually.")
                    self.flattened_obs = None

            if self.flattened_obs is None:
                print("\n" + "=" * 60)
                print(f"No valid data found in '{save_dir}'.")
                print("Please prepare the flattened garment position manually.")
                print("=" * 60)
                input("Press [Enter] to capture the flattened cloth state...")
                self.flattened_obs = {}
                self.flattened_obs = self._process_info(self.flattened_obs, task_related=False, flattened_obs=False)
                self.flatten_coverage = self.coverage
                
                obs = self.flattened_obs['observation']
                print(f"[Arena] Saving human-readable observation to {save_dir}...")
                os.makedirs(save_dir, exist_ok=True)
                
                try:
                    cv2.imwrite(fn_rgb, cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_raw_rgb, cv2.cvtColor(obs['raw_rgb'], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(fn_depth, obs['depth'])
                    cv2.imwrite(fn_mask, (obs['mask'] * 255).astype(np.uint8))
                    cv2.imwrite(fn_r0_mask, (obs['robot0_mask'] * 255).astype(np.uint8))
                    cv2.imwrite(fn_r1_mask, (obs['robot1_mask'] * 255).astype(np.uint8))
                    
                    meta_info = {
                        "eid": int(self.flattened_obs.get("eid", 0)),
                        "action_step": int(obs.get("action_step", 0)),
                        "note": "Flattened state capture"
                    }
                    
                    with open(fn_info, 'w') as f:
                        json.dump(meta_info, f, indent=4)
                        
                    print("[Arena] Saved successfully.")
                except Exception as e:
                    print(f"[Arena] Warning: Could not save flattened obs: {e}")

                print("=" * 60 + "\n")
        
        return self.flattened_obs
    
    def step(self, action):
        if self.measure_time: start_time = time.time()

        norm_pixels = action.reshape(-1, 2)

        # 1. Denormalize
        points_crop = ((norm_pixels + 1) / 2 * self.crop_size).astype(np.int32)
        
        # 2. Snap to Mask
        if self.snap_to_cloth_mask:
            mask = self.cloth_mask
            kernel = np.ones((3, 3), np.uint8) 
            eroded_mask = cv2.erode(mask, kernel, iterations=10)
            target_mask = mask if np.sum(eroded_mask) == 0 else eroded_mask
            
            snapped_pick = snap_to_mask(points_crop[0], target_mask)
            points_crop[0] = snapped_pick
        
        points_orig = points_crop + np.array([self.x1, self.y1])
        
        # 3. Snap to Robot Workspace
        full_mask_0 = self.single_arm.get_workspace_mask()

        pick_angle = 0
        valid_flags = 1 # Default True

        # Helper to check if a crop-space point is on the cloth mask
        def check_validity(pt_crop):
            x, y = int(pt_crop[0]), int(pt_crop[1])
            h, w = self.cloth_mask.shape
            if 0 <= x < w and 0 <= y < h:
                return 1.0 if self.cloth_mask[y, x] > 0 else 0.0
            return 0.0
        
        final_pick = snap_to_mask(points_orig[0], full_mask_0)
        final_place = snap_to_mask(points_orig[1], full_mask_0)
        
        # 4. Calculate Rotation
        pt_crop = final_pick - np.array([self.x1, self.y1])
        pick_angle = get_grasp_rotation(self.cloth_mask, pt_crop)

        valid_0 = check_validity(pt_crop)

        
        # 5. Construct Single Arm Action [px, py, lx, ly, rot]
        skill_action = np.concatenate([final_pick, final_place, [pick_angle]])
        
        if self.measure_time:
            self.process_action_time.append(time.time() - start_time)
            start_time = time.time()

        self.info = {}
        self.pick_and_place_skill.reset()
        self.pick_and_place_skill.step(skill_action)
        
        executed_points = np.concatenate([final_pick, final_place])

        if self.measure_time:
            self.primitive_time.append(time.time() - start_time)
            start_time = time.time()
        
        
        self.action_step += 1
        
        if self.action_step % 5 == 0:
            self.single_arm.restart_camera()
        
        
        self.all_infos.append(self.info)    
        self.info = self._process_info(self.info)
        
        
        applied_action = (1.0 * executed_points.reshape(-1, 2) - np.array([self.x1, self.y1])) / self.crop_size * 2 - 1
        self.info['applied_action'] = applied_action.flatten()
            
        if self.measure_time:
            self.perception_time.append(time.time() - start_time)

        return self.info

    def _process_info(self, info, task_related=True, flattened_obs=True):
        # Changed from both_home to standard home/out_scene
        self.single_arm.open_gripper()
        self.single_arm.home(MOVE_SPEED, MOVE_ACC)
        self.single_arm.out_scene(MOVE_SPEED, MOVE_ACC)
        time.sleep(1.0)

        raw_rgb, raw_depth = self.single_arm.take_rgbd()
        workspace_mask_0 = self.single_arm.get_workspace_mask()
        
        # Handle dry run where mask might be None
        if workspace_mask_0 is None:
            workspace_mask_0 = np.zeros(raw_rgb.shape[:2], dtype=np.uint8)
        else:
            workspace_mask_0 = workspace_mask_0.astype(np.uint8)

        h, w = raw_rgb.shape[:2]
        crop_size = min(h, w)  
        x1 = w // 2 - crop_size // 2
        y1 = h // 2 - crop_size // 2
        self.x1, self.y1, self.crop_size = x1, y1, crop_size

        crop_rgb = raw_rgb[y1:y1+crop_size, x1:x1+crop_size]
        crop_depth = raw_depth[y1:y1+crop_size, x1:x1+crop_size]
        crop_mask_0 = workspace_mask_0[y1:y1+crop_size, x1:x1+crop_size]

        crop_cloth_mask = get_mask_v2(self.mask_generator, crop_rgb, debug=self.debug)
        self.cloth_mask = crop_cloth_mask
        self.coverage = np.sum(self.cloth_mask)
        if self.init_coverage is None: self.init_coverage = self.coverage

        norm_depth = process_depth(crop_depth)

        resized_rgb = cv2.resize(crop_rgb, self.resolution)
        resized_depth = cv2.resize(norm_depth, self.resolution)
        resized_mask = cv2.resize(crop_cloth_mask.astype(np.uint8), self.resolution)
        resized_r0_mask = cv2.resize(crop_mask_0, self.resolution, interpolation=cv2.INTER_NEAREST)

        if self.maskout_background:
            is_background = resized_mask == 0
            resized_rgb[is_background] = 0
            resized_depth[is_background] = 0

        info.update({
            'observation': {
                "rgb": resized_rgb,
                "depth": resized_depth,
                "mask": resized_mask.astype(np.bool_),
                "raw_rgb": raw_rgb,
                "action_step": self.action_step,
                "robot0_mask": resized_r0_mask.astype(np.bool_),
            },
            "eid": self.eid,
            "arena": self,
            "arena_id": self.id,
        })

        if flattened_obs:
            info['flattened_obs'] = self.get_flattened_obs()
            for k, v in info['flattened_obs'].items():
                info['observation'][f'flattened-{k}'] = v
        
        info['done'] = self.action_step >= self.action_horizon

        if task_related:
            info['evaluation'] = self.evaluate()
            if info['evaluation'].get('normalised_coverage', 0) > 0.9:
                self.last_flattened_step = self.action_step
           
            info['observation']['last_flattened_step'] = self.last_flattened_step
            info['success'] =  self.success()
            
            if info['evaluation'] != {}:
                print('evaluation', info['evaluation'])
                info['reward'] = self.task.reward(self.last_info, None, info)
            
            goals = self.task.get_goals()
            if len(goals) > 0:
                goal = goals[0]
                info['goal'] = {}
                for k, v in goal[-1]['observation'].items():
                    if k == 'rgb' and self.maskout_background and ('mask' in  goal[-1]['observation']):
                        # apply resized_mask on resized_rgb and resized_depeth.
                        is_background = goal[-1]['observation']['mask'] == 0
                        v[is_background] = 0

                    info['goal'][k] = v
                    info['observation'][f'goal_{k}'] = v

        self.last_info = info
        self.single_arm.home(MOVE_SPEED, MOVE_ACC)
        return info
    
    def clear_frames(self): self.frames = []
    def evaluate(self): return self.task.evaluate(self) if hasattr(self, 'task') else {}
    def success(self): return self.task.success(self) if hasattr(self, 'task') else False

    def get_action_space(self):
        return self._action_space

    def sample_random_action(self):
        return self._action_space.sample()

    def get_no_op(self) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)