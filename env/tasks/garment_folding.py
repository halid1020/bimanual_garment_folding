import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from statistics import median
from statistics import mean

from scipy.spatial.distance import cdist
from agent_arena import Task
from agent_arena import save_video
from agent_arena.utilities.visual_utils import save_numpy_as_gif as sg

from .utils import get_max_IoU
from .folding_rewards import *
from .garment_task import GarmentTask
from ..utils.garment_utils import simple_rigid_align

SUCCESS_TRESHOLD = 0.05

def save_point_cloud_ply(path, points):
    N = points.shape[0]
    header = f"""ply
            format ascii 1.0
            element vertex {N}
            property float x
            property float y
            property float z
            end_header
            """
    
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def load_point_cloud_ply(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # find end of header robustly
    end_header_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "end_header":
            end_header_idx = idx + 1
            break
    if end_header_idx is None:
        raise ValueError(f"No 'end_header' found in PLY file: {path}")

    points = []
    for line in lines[end_header_idx:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        x, y, z = map(float, parts[:3])
        points.append([x, y, z])
    return np.array(points)


class GarmentFoldingTask(GarmentTask):
    def __init__(self, config):
        super().__init__(config)
        self.num_goals = config.num_goals
        self.name = config.task_name

        self.config = config
        self.demonstrator = config.demonstrator ## TODO: This needs to be initialised before the class.
        self.goals = [] # This needs to be loaded  or generated

        
        

    def reset(self, arena):
        """Reset environment and generate goals if necessary."""
        self.goal_dir = os.path.join(self.asset_dir, 'goals', arena.get_name(), \
                                     self.name, arena.get_mode(), f"eid_{arena.get_episode_id()}")
        
        os.makedirs(self.goal_dir, exist_ok=True)

        # Load or create semantic keypoints
        self.semkey2pid = self._load_or_create_keypoints(arena)

        # Generate goals (10 small variations)
        self.goals = []
        self.goals = self._load_or_generate_goals(arena, self.num_goals)

        self.aligned_pairs = []

        return {"goals": self.goals, "keypoints": self.semkey2pid}

    def _generate_a_goal(self, arena):
        goal = []
        particle_pos = arena.get_particle_positions()
        info = arena.set_to_flatten()
        self.demonstrator.reset([arena.id])
        goal.append(info)
        while not self.demonstrator.terminate()[arena.id]: ## The demonstrator does not need update and init function
            #print('here!')
            action = self.demonstrator.single_act(info) # Fold action
            #print('action', action)
            info = arena.step(action)
            goal.append(info)
            if self.config.debug:
                rgb = info['observation']['rgb']
                cv2.imwrite("tmp/step_rgb.png", rgb)
        
        if self.config.debug:
            frames = arena.get_frames()
            if len(frames) > 0:
                save_video(np.stack(arena.get_frames()), 'tmp', 'demo_videos')
                sg(
                    np.stack(arena.get_frames()), 
                    path='tmp',
                    filename="demo_videos"
                )
            
        arena.set_particle_positions(particle_pos)

        return goal


    def _load_or_generate_goals(self, arena, num_goals):
        goals = []
        for i in range(num_goals):
            goal_path = os.path.join(self.goal_dir, f"goal_{i}")
            if not os.path.exists(goal_path):
                print(f'Generating goal {i} for episode id {arena.eid}')
                goal = self._generate_a_goal(arena)
                os.makedirs(goal_path, exist_ok=True)

                for i, subgoal in enumerate(goal):
                    # Save RGB
                    plt.imsave(os.path.join(goal_path, f"rgb_step_{i}.png"), subgoal['observation']['rgb']/255.0)

                    # Save particles as PLY
                    save_point_cloud_ply(os.path.join(goal_path, f"particles_step_{i}.ply"),
                                        subgoal['observation']["particle_positions"])

                goals.append(goal)
            else:
                goal = []
                for i in range(self.config.goal_steps):
                    # Load existing goal
                    rgb = (plt.imread(os.path.join(goal_path, f"rgb_step_{i}.png"))*255).astype(np.uint8)

                    particles = load_point_cloud_ply(os.path.join(goal_path, f"particles_step_{i}.ply"))
                    subgoal = {
                        'observation': {
                            'rgb': rgb[:, :, :3],
                            'particle_positions': particles
                        }
                    }
                    goal.append(subgoal.copy())
                goals.append(goal.copy())
        return goals

    def evaluate(self, arena):
        """Evaluate folding quality using particle alignment and semantic keypoints."""
        if len(self.goals) == 0:
            return {}
        cur_particles = arena.get_mesh_particles_positions()
        #print('len cur', len(cur_particles))

        # Evaluate particle alignment against each goal
        particle_distances = []
        key_distances = []
        for goal in self.goals:
            goal_particles = goal[-1]['observation']["particle_positions"]
            #print('goal len', len(goal_particles))
            mdp, kdp = self._compute_particle_distance(cur_particles, goal_particles, arena)
            particle_distances.append(mdp)
            key_distances.append(kdp)
       
        mean_particle_distance = median(particle_distances)
        key_particle_distance = median(key_distances)
        #print('MPD', mean_particle_distance)

        #semantic_dist = self._compute_keypoint_distance(arena, cur_particles, goal_particles)

        return {
            "mean_particle_distance": mean_particle_distance,
            "semantic_keypoint_distance": key_particle_distance,
            'max_IoU': self._get_max_IoU(arena),
            'max_IoU_to_flattened':  self._get_max_IoU_to_flattened(arena),
            'normalised_coverage': self._get_normalised_coverage(arena)
        }

    def _align_points(self, arena, cur, goal):
        """
        Align cur points to goal points using Procrustes rigid alignment.
        Returns aligned points and mean distance.
        """
        # if len(self.aligned_pairs) == arena.action_step + 1:
        #     return self.aligned_pairs[-1]

        if self.config.alignment == 'simple_rigid':
            # Center both sets
            aligned_curr, aligned_goal = simple_rigid_align(cur, goal)
            #return aligned, goal_centered
        else:
            raise NotImplementedError
        
        # Safety check for NaNs
        assert not (np.isnan(aligned_curr).any() or np.isnan(aligned_goal).any()), \
            "NaN values detected after point alignment!"
        
        # self.aligned_pairs.append((aligned_curr, aligned_goal))
        
        return aligned_curr, aligned_goal

    def _compute_particle_distance(self, cur, goal, arena):
        """Align particles and compute mean distance."""
        #print('len cur', len(cur))
        aligned_curr, aligned_goal = self._align_points(arena, cur.copy(), goal.copy())
        mdp = np.mean(np.linalg.norm(aligned_curr - aligned_goal, axis=1))

        cur_pts = []
        goal_pts = []
        for name, pid in self.semkey2pid.items():
            
            cur_pts.append(aligned_curr[pid])
            goal_pts.append(aligned_goal[pid])
        cur_pts = np.stack(cur_pts)
        goal_pts = np.stack(goal_pts)
        kdp = np.mean(np.linalg.norm(cur_pts - goal_pts, axis=1))
        

        if self.config.debug:
            save_point_cloud_ply(os.path.join('tmp', f"cur_particles_step_{arena.action_step}.ply"), cur)
            save_point_cloud_ply(os.path.join('tmp', "goal_particles.ply"), goal)
            for align_type in ['simple_rigid']:
                if align_type == 'simple_rigid':
                    aligned_curr, aligned_goal = simple_rigid_align(cur, goal)
                # elif align_type == 'complex_rigid':
                #     #print('Cloth Area', arena.get_cloth_area())
                #     aligned_curr, aligned_goal = rigid_align(cur, goal, arena.get_cloth_area())
                # elif align_type == 'deform':
                #     aligned_curr, aligned_goal = deformable_align(cur, goal, arena.get_cloth_area())
                # elif align_type == 'chamfer_rotation':
                #     aligned_curr, aligned_goal = chamfer_alignment_with_rotation(cur, goal)
                mdp_ = np.mean(np.linalg.norm(aligned_curr - aligned_goal, axis=1))
                project_aligned, _ = arena.get_visibility(aligned_curr)
                project_goal, _ = arena.get_visibility(aligned_goal)
                canvas = np.zeros((480, 480, 3), dtype=np.uint8)

                for p in project_aligned:
                    x, y = map(int, p)
                    if x > 480 or y> 480:
                        continue
                    canvas[x, y] = (255, 255, 255)

                for p in project_goal:
                    x, y = map(int, p)
                    if x > 480 or y> 480:
                        continue
                    canvas[x, y] = (0, 255, 0)

                # Save both images
                combined = np.hstack((canvas, arena._render('rgb')))

                # 🔹 Overlay mdp value on combined image
                cv2.putText(
                    combined,
                    f"MDP: {mdp_:.4f}",
                    (10, 30),  # position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,       # font scale
                    (0, 0, 255),  # color (red)
                    2,         # thickness
                    cv2.LINE_AA
                )
                cv2.imwrite(f"tmp/{align_type}_combined_{arena.action_step}.png", combined)

        return mdp, kdp



    # def _compute_keypoint_distance(self, arena, cur, goal):
    #     """Align semantic keypoints and compute mean distance."""

    #     aligned_cur, aligned_goal = self._align_points(arena, cur, goal)
        
    #     cur_pts = []
    #     goal_pts = []
    #     for name, pid in self.semkey2pid.items():
            
    #         cur_pts.append(aligned_cur[pid])
    #         goal_pts.append(aligned_goal[pid])
    #     cur_pts = np.stack(cur_pts)
    #     goal_pts = np.stack(goal_pts)


    #     return np.mean(np.linalg.norm(cur_pts - goal_pts, axis=1))
    
    def reward(self, last_info, action, info): 
        mpd = info['evaluation']['mean_particle_distance']
        mkd = info['evaluation']["semantic_keypoint_distance"]
        
        pdr = particle_distance_reward(mpd)
        pdr_ = pdr

        
        #Multi stage reward
        if last_info == None:
            last_info = info
        last_particles = last_info['observation']['particle_positions']
        cur_particles = info['observation']['particle_positions']
        
        
        multi_stage_reward = coverage_alignment_reward(last_info, action, info)
        if info['evaluation']['normalised_coverage'] > 0.9 and info['evaluation']['max_IoU_to_flattened'] > 0.85:
            multi_stage_reward = 1
        arena = info['arena']
        for i in range(self.config.goal_steps)[1:]:
            cur_mdps = []
            last_mdps = []
            
            for goal in self.goals:
                cur_goal_particles = goal[i]['observation']["particle_positions"]
                last_goal_particles = goal[i-1]['observation']["particle_positions"]
                #print('goal len', len(goal_particles))
                cur_mdp, kdp = self._compute_particle_distance(cur_particles, cur_goal_particles, arena)
                last_mdp, kdp = self._compute_particle_distance(last_particles, last_goal_particles, arena)
                cur_mdps.append(cur_mdp)
                last_mdps.append(last_mdp)

            last_mdp = median(last_mdps)
            cur_mdp = median(cur_mdps)
            last_action_step = last_info['observation']['action_step']
            # print(f'\nlast_mdp for goal step {i-1} at action step {last_action_step}:', last_mdp)
            # print(f'\ncur_mdp for goal step {i} at action step {last_action_step+1}:', cur_mdp)
            
            if last_mdp < SUCCESS_TRESHOLD:
                # print(f'!!match at last goal step {i-1}, current step mdp', cur_mdp)
                multi_stage_reward = i + particle_distance_reward(cur_mdp)
            
        if info['success']:
            multi_stage_reward += self.config.goal_steps*(info['arena'].horizon - info['observation']['action_step'])
            pdr_ += (info['arena'].horizon - info['observation']['action_step'])

        threshold =  self.config.get('overstretch_penality_threshold', 0)
        if info['overstretch'] > threshold:
            pdr_ -= self.config.get("overstretch_penality_scale", 0) * (info['overstretch'] - threshold)
            #multi_stage_reward -= self.config.get("overstretch_penality_scale", 0) * (info['overstretch'] - threshold)

        return {
            'particle_distance': pdr,
            'particle_distance_with_stretch_penality': pdr_,
            'keypoint_distance': particle_distance_reward(mkd),
            'multi_stage_reward': multi_stage_reward,
        }
    

    def get_goals(self):
        return self.goals

    def get_goal(self):
        return self.goals[0]
    
    def success(self, arena):
        cur_eval = self.evaluate(arena)
        if cur_eval == {}:
            return False
        return cur_eval['mean_particle_distance'] < SUCCESS_TRESHOLD
    
    def _get_max_IoU(self, arena):
        cur_mask = arena.cloth_mask
        max_IoU = 0
        for goal in self.goals[:1]:
            goal_mask = goal[-1]['observation']["rgb"].sum(axis=2) > 0 ## only useful for background is black
            
            IoU, matched_IoU = get_max_IoU(cur_mask, goal_mask, debug=self.config.debug)
            if IoU > max_IoU:
                max_IoU = IoU
        
        return IoU
    
    def _get_max_IoU_to_flattened(self, arena):
        cur_mask = arena.cloth_mask
        IoU, matched_IoU = get_max_IoU(cur_mask, arena.get_flattened_obs()['observation']['mask'], debug=self.config.debug)
        
        return IoU
    
    def _get_normalised_coverage(self, arena):
        res = arena._get_coverage() / arena.flatten_coverage
        
        # clip between 0 and 1
        return np.clip(res, 0, 1)

    def compare(self, results_1, results_2):
        threshold=0.95

        # --- Compute averages for results_1 ---
        score_1 = mean([ep["mean_particle_distance"][-1] for ep in results_1])
        # --- Compute averages for results_2 ---
        score_2 = mean([ep["mean_particle_distance"][-1] for ep in results_2])

        # --- Otherwise prefer higher score ---
        if score_1 < score_2:
            return 1
        elif score_1 > score_2:
            return -1
        else:
            return 0