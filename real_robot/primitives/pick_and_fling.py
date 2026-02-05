# skills/pick_and_fling_skill.py
import math
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from real_robot.utils.transform_utils import (
    point_on_table_base, transform_point, points_to_gripper_pose,
    points_to_action_frame, get_base_fling_poses, transform_pose, 
    GRIPPER_OFFSET_UR5e, GRIPPER_OFFSET_UR16e, SURFACE_HEIGHT, FLING_LIFT_DIST,
    MOVE_ACC, MOVE_SPEED
)

MIN_Z = 0.015
APPROACH_DIST = 0.08        
LIFT_DIST = 0.12            
FLING_SPEED = 3.0
FLING_ACC = 1.5
HANG_HEIGHT = 0.3
HOME_AFTER = True
MIN_STRETCH_DIST = 0.3

# --- HELPER: Apply Rotation ---
def apply_local_z_rotation(axis_angle, angle_rad):
    if abs(angle_rad) < 1e-4:
        return axis_angle
    r_current = R.from_rotvec(axis_angle)
    r_diff = R.from_euler('z', angle_rad, degrees=False)
    r_new = r_current * r_diff
    return r_new.as_rotvec()

# --- HELPER: Fling Path ---
def points_to_fling_path(
        right_point, left_point,
        width=None,   
        swing_stroke=0.6, 
        swing_angle=np.pi/4,
        lift_height=HANG_HEIGHT,
        place_height=0.05):
    tx_world_action = points_to_action_frame(right_point, left_point)
    tx_world_fling_base = tx_world_action.copy()
    tx_world_fling_base[2,3] = 0
    base_fling = get_base_fling_poses(
        stroke=swing_stroke,
        swing_angle=swing_angle,
        lift_height=lift_height,
        place_height=place_height)
    if width is None:
        width = np.linalg.norm((right_point - left_point)[:2])
    right_path = base_fling.copy()
    right_path[:,0] = -width/2
    left_path = base_fling.copy()
    left_path[:,0] = width/2
    right_path_w = transform_pose(tx_world_fling_base, right_path)
    left_path_w = transform_pose(tx_world_fling_base, left_path)
    return right_path_w, left_path_w


class PickAndFlingSkill:
    def __init__(self, scene):
        self.scene = scene
        self.vertical_rotvec = [math.pi, 0.0, 0.0]
        self.min_z = 0.015
        self.approach_dist = 0.08
        self.lift_dist = 0.1
        self.move_speed = 0.2
        self.move_acc = 0.2

    def reset(self):
        print("[PickAndFlingSkill] Resetting...")
        self.scene.both_home()
        self.scene.both_open_gripper()
        time.sleep(0.5)

    def _append_scene_trajectory(self, accumulator):
        """Helper to append the last recorded trajectory from the scene to our list."""
        if hasattr(self.scene, 'last_trajectory') and self.scene.last_trajectory:
            traj = self.scene.last_trajectory
            accumulator['ur5e'].extend(traj['ur5e'])
            accumulator['ur16e'].extend(traj['ur16e'])

    def step(self, action, record_debug=False):
        print('recording ?', record_debug)
        # Initialize accumulator 
        full_trajectory = {'ur5e': [], 'ur16e': []}

        # 1. Unpack Action
        if len(action) >= 6:
            pick_0_xy = action[0:2]
            pick_1_xy = action[2:4]
            angle_0 = action[4]
            angle_1 = action[5]
        else:
            pick_0_xy = action[0:2]
            pick_1_xy = action[2:4]
            angle_0, angle_1 = 0.0, 0.0

        # 2. Sort Logic
        pair_0 = {'pick': pick_0_xy, 'angle': angle_0}
        pair_1 = {'pick': pick_1_xy, 'angle': angle_1}

        if pair_0['pick'][0] < pair_1['pick'][0]:
            pair_0, pair_1 = pair_1, pair_0
            
        pick_0 = pair_0['pick']
        pick_1 = pair_1['pick']
        rot_angle_0 = pair_0['angle']
        rot_angle_1 = pair_1['angle']
        
        # 3. Calculate Table Coordinates
        p_base_pick_0 = point_on_table_base(pick_0[0], pick_0[1], self.scene.intr, self.scene.T_ur5e_cam, SURFACE_HEIGHT)
        p_base_pick_1 = point_on_table_base(pick_1[0], pick_1[1], self.scene.intr, self.scene.T_ur16e_cam, SURFACE_HEIGHT)

        def clamp_z(arr):
            a = np.array(arr, dtype=float).copy()
            if a.shape[0] < 3:
                a = np.pad(a, (0, 3 - a.shape[0]), 'constant', constant_values=0.0)
            a[2] = max(MIN_Z, float(a[2]))
            return a

        p_base_pick_0 = clamp_z(p_base_pick_0)
        p_base_pick_1 = clamp_z(p_base_pick_1)

        p_base_pick_0 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR5e])
        p_base_pick_1 += np.array([0.0, 0.0, GRIPPER_OFFSET_UR16e])

        # 4. Calculate Orientation
        pose_0_home = self.scene.ur5e.get_tcp_pose()
        pose_1_home = self.scene.ur16e.get_tcp_pose()

        rot_0_base = pose_0_home[3:6]
        rot_1_base = pose_1_home[3:6]

        rot_0 = apply_local_z_rotation(rot_0_base, rot_angle_0)
        rot_1 = apply_local_z_rotation(rot_1_base, rot_angle_1)

        # Compose poses
        approach_pick_0 = p_base_pick_0 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_0 = p_base_pick_0
        lift_after_0 = grasp_pick_0.copy()
        lift_after_0[2] += FLING_LIFT_DIST
        
        approach_pick_1 = p_base_pick_1 + np.array([0.0, 0.0, APPROACH_DIST])
        grasp_pick_1 = p_base_pick_1
        lift_after_1 = grasp_pick_1.copy()
        lift_after_1[2] += FLING_LIFT_DIST
        
        # Motion sequence
        self.scene.both_open_gripper()
        self.scene.both_home(speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True)

        # Approach (Not recorded)
        self.scene.both_movel(
            np.concatenate([approach_pick_0, rot_0]),
            np.concatenate([approach_pick_1, rot_1]),
            speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True
        )

        # Grasp (Not recorded)
        self.scene.both_movel(
            np.concatenate([grasp_pick_0, rot_0]),
            np.concatenate([grasp_pick_1, rot_1]),
            speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True
        )

        # --- GRASP ---
        self.scene.both_close_gripper()
        time.sleep(1.0)

        # --- RECORDING START ---
        
        # 1. Lift
        self.scene.both_movel(
            np.concatenate([lift_after_0, rot_0]),
            np.concatenate([lift_after_1, rot_1]),
            speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True,
            record=record_debug 
        )
        if record_debug: self._append_scene_trajectory(full_trajectory)

        # 2. Centering & Aligning
        print("Centering and Aligning grippers between robots...")
        p0_local = lift_after_0[:3]
        p1_local = transform_point(np.linalg.inv(self.scene.T_ur5e_ur16e), lift_after_1[:3])
        curr_width = np.linalg.norm(p1_local - p0_local)
        curr_width = max(curr_width, MIN_STRETCH_DIST)

        base_to_base_vec = self.scene.T_ur5e_ur16e[:3, 3]
        center_point = base_to_base_vec / 2.0
        axis_vec = base_to_base_vec / np.linalg.norm(base_to_base_vec)

        target_p0_local = center_point - (axis_vec * curr_width / 2.0)
        target_p1_local = center_point + (axis_vec * curr_width / 2.0)

        avg_z = (p0_local[2] + p1_local[2]) / 2.0
        target_p0_local[2] = HANG_HEIGHT
        target_p1_local[2] = HANG_HEIGHT

        target_pose_0 = np.concatenate([target_p0_local, rot_0])
        target_p1_ur16e = transform_point(np.linalg.inv(self.scene.T_ur5e_ur16e), target_p1_local)
        target_pose_1 = np.concatenate([target_p1_ur16e, rot_1])

        self.scene.both_movel(
            target_pose_0, 
            target_pose_1, 
            speed=MOVE_SPEED, acc=MOVE_ACC, blocking=True,
            record=record_debug
        )
        if record_debug: self._append_scene_trajectory(full_trajectory)

        # 3. Stretch, Fling, Drag
        self.dual_arm_stretch_and_fling(
            target_p0_local, 
            transform_point(self.scene.T_ur5e_ur16e, target_p1_ur16e), # Pass correct func arg
           # target_p1_ur16e, # Pass correct target
            record_debug=record_debug,
            full_trajectory_ref=full_trajectory # Pass accumulator
        )

        self.scene.both_home()
        return full_trajectory

    def dual_arm_stretch_and_fling(self, 
            ur5e_pick_point_world, # Actually in UR5e Base from step() call above
            ur16e_pick_point_world, # Actually in UR16e Base from step() call above
            stretch_force=15,
            stretch_max_speed=0.15,
            stretch_max_width=0.7, 
            stretch_max_time=5,
            swing_stroke=0.4,
            swing_height=0.45,
            swing_angle=np.pi/4,
            lift_height=HANG_HEIGHT,
            place_height=0.15,
            fling_speed=FLING_SPEED,  
            fling_acc=FLING_ACC,
            drag_speed=0.2,   
            drag_acc=0.2,
            record_debug=False,
            full_trajectory_ref=None
            ):
        
        # 1. Prepare Grasp/Stretch Poses
        # Note: Input points are already target centers from 'step', 
        # so we convert them to full pose with rotation if needed or use directly.
        # Since 'step' passed calculated points, let's treat them as points to gripper pose.
        
        # Re-using the points_to_gripper_pose logic might be redundant if points are already aligned,
        # but ensures rotation format is consistent.
        ur5e_pose_world, ur16e_pose_world = points_to_gripper_pose(
            ur5e_pick_point_world, ur16e_pick_point_world, max_width=stretch_max_width)
        
        # 2. Stretch
        r = self.dual_arm_stretch(ur5e_pose_world, ur16e_pose_world, 
            force=stretch_force, 
            max_speed=stretch_max_speed, 
            max_width=stretch_max_width,
            max_time=stretch_max_time,
            record_debug=record_debug,
            full_trajectory_ref=full_trajectory_ref)
            
        if not r: return False
        
        # --- FIX: Update Start Points from Actual Robots ---
        current_p5_base = self.scene.ur5e.get_tcp_pose()
        current_p16_base = self.scene.ur16e.get_tcp_pose() 

        new_ur5e_point = current_p5_base[:3]
        # Transform UR16e Base -> UR5e Base
        current_p16_in_ur5ebase = transform_pose(self.scene.T_ur5e_ur16e, current_p16_base)
        new_ur16e_point = current_p16_in_ur5ebase[:3]

        # 3. Generate Path
        ur5e_path_full, ur16e_path_full = points_to_fling_path(
            right_point=new_ur5e_point,
            left_point=new_ur16e_point,
            width=None, 
            swing_stroke=swing_stroke,
            swing_angle=swing_angle,
            lift_height=lift_height,
            place_height=place_height
        )
        
        # 4. Overwrite Start Points
        ur5e_path_full[0][:3] = new_ur5e_point
        ur16e_path_full[0][:3] = new_ur16e_point
        
        # 5. Slice paths
        ur5e_fling_path = ur5e_path_full[:-1]
        ur16e_fling_path = ur16e_path_full[:-1]
        ur5e_drag_target = ur5e_path_full[-1]
        ur16e_drag_target = ur16e_path_full[-1]

        # 6. Fast Fling
        ur16e_fling_path_base = transform_pose(np.linalg.inv(self.scene.T_ur5e_ur16e), ur16e_fling_path)
        
        self.scene.both_fling(
            ur5e_fling_path, 
            ur16e_fling_path_base, 
            fling_speed, 
            fling_acc,
            record=record_debug 
        )
        if record_debug: self._append_scene_trajectory(full_trajectory_ref)
        
        # 7. Slow Drag
        ur16e_drag_target_base = transform_pose(np.linalg.inv(self.scene.T_ur5e_ur16e), ur16e_drag_target)

        self.scene.both_movel(
            ur5e_drag_target.flatten(), 
            ur16e_drag_target_base.flatten(), 
            speed=drag_speed, 
            acc=drag_acc, 
            blocking=True,
            record=record_debug
        )
        if record_debug: self._append_scene_trajectory(full_trajectory_ref)
        
        self.scene.both_open_gripper()
        return True

    def dual_arm_stretch(self, 
        ur5e_pose_world, ur16e_pose_world,
        force=8,        # Increased default
        init_force=10,   # Increased default ("The Kick")
        max_speed=0.15, 
        max_width=0.7, 
        max_time=5,
        speed_threshold=0.005, # Lowered threshold (more sensitive)
        record_debug=False,
        full_trajectory_ref=None): 
        
        ur16e_pose_base = transform_pose(np.linalg.inv(self.scene.T_ur5e_ur16e), ur16e_pose_world)

        # Move to initial grasp pose
        self.scene.both_movel(ur5e_pose_world, \
            ur16e_pose_base, \
            speed=max_speed,
            acc=1.2,
            record=record_debug) 
        if record_debug and full_trajectory_ref:
            self._append_scene_trajectory(full_trajectory_ref)
    
        # Task frame = Base Frame
        task_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        selection_vector = [0, 1, 0, 0, 0, 0] 
        force_type = 2
        limits = [max_speed, 2, 2, 1, 1, 1]
        dt = 0.1 

        start_width = self.scene.get_tcp_distance()
        safe_limit_width = min(max_width, start_width * 1.5) 

        print(f"[Stretch] Start Width: {start_width:.3f}, Limit: {safe_limit_width:.3f}")
        print(f"[Stretch] Applying Init Force: {init_force}N, Steady Force: {force}N")

        with self.scene.ur5e.start_force_mode() as right_force_guard:
            with self.scene.ur16e.start_force_mode() as left_force_guard:
                start_time = time.time()
                prev_time = start_time
                
                while True:
                    if record_debug and full_trajectory_ref:
                        full_trajectory_ref['ur5e'].append(self.scene.ur5e.get_tcp_pose()[:3])
                        full_trajectory_ref['ur16e'].append(self.scene.ur16e.get_tcp_pose()[:3])
                    
                    elapsed = time.time() - start_time
                    
                    if elapsed >= max_time:
                        print(f'[Stretch] eplased time {elapsed} bigger than max time f{max_time}, break')
                        break
                    
                    if elapsed < 0.2:
                        f = init_force
                    else:
                        f = force
                    
                    right_wrench = [0, f, 0, 0, 0, 0]
                    left_wrench = [0, f, 0, 0, 0, 0]

                    r = right_force_guard.apply_force(task_frame, selection_vector, 
                        right_wrench, force_type, limits)
                    if not r: 
                        print("[Stretch] not r is true for right, return.")
                        return False
                    r = left_force_guard.apply_force(task_frame, selection_vector, 
                        left_wrench, force_type, limits)
                    if not r: 
                        print("[Stretch] not r is true for left, return.")
                        return False
                    
                    tcp_distance = self.scene.get_tcp_distance()
                    if tcp_distance >= safe_limit_width:
                        print(f"[Stretch] Hit width limit: {tcp_distance:.3f}")
                        break

                    l_speed = np.linalg.norm(self.scene.ur5e.get_tcp_speed()[:3])
                    r_speed = np.linalg.norm(self.scene.ur16e.get_tcp_speed()[:3])
                    actual_speed = max(l_speed, r_speed)
                    
                    if elapsed > 0.5:
                        if actual_speed < speed_threshold:
                            print(f"[Stretch] Stopped moving (Speed: {actual_speed:.4f}). Cloth taut.")
                            break

                    curr_time = time.time()
                    duration = curr_time - prev_time
                    if duration < dt:
                        time.sleep(dt - duration)
                    prev_time = curr_time
        return True