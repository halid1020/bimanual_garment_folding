import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from scipy.spatial.distance import cdist
from agent_arena import Task
from agent_arena import save_video
from ..utils.garment_utils import KEYPOINT_SEMANTICS, rigid_align, deformable_align, \
    simple_rigid_align, chamfer_alignment_with_rotation
from ..utils.keypoint_gui import KeypointGUI
from .utils import get_max_IoU
from .folding_rewards import *

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


class GarmentFoldingTask(Task):
    def __init__(self, config):
        self.num_goals = config.num_goals
        self.name = config.task_name
        self.asset_dir = config.asset_dir
        self.config = config
        self.demonstrator = config.demonstrator ## TODO: This needs to be initialised before the class.
        
        self.keypoint_semantics = KEYPOINT_SEMANTICS[config.object]

        self.semkey2pid = None # This needs to be loaded or annotated
        self.goals = [] # This needs to be loaded  or generated

        self.keypoint_assignment_gui = KeypointGUI(self.keypoint_semantics )
        self.keypoint_dir = os.path.join(self.asset_dir, 'keypoints')
        os.makedirs(self.keypoint_dir, exist_ok=True)
        self.name
        

    def reset(self, arena):
        """Reset environment and generate goals if necessary."""
        self.goal_dir = os.path.join(self.asset_dir, 'goals', arena.get_name(), \
                                     self.name, arena.get_mode(), f"eid_{arena.get_episode_id()}")
        
        os.makedirs(self.goal_dir, exist_ok=True)

        # Load or create semantic keypoints
        self.semkey2pid = self._load_or_create_keypoints(arena)

        # Generate goals (10 small variations)
        self.goals = self._load_or_generate_goals(arena, self.num_goals)

        self.aligned_pairs = []

        return {"goals": self.goals, "keypoints": self.semkey2pid}

    def _generate_a_goal(self, arena):
        
        particle_pos = arena.get_particle_positions()
        info = arena.set_to_flatten()
        self.demonstrator.reset([arena.id])
        while not self.demonstrator.terminate()[arena.id]: ## The demonstrator does not need update and init function
            #print('here!')
            action = self.demonstrator.single_act(info) # Fold action
            #print('action', action)
            info = arena.step(action)

            if self.config.debug:
                rgb = info['observation']['rgb']
                cv2.imwrite("tmp/step_rgb.png", rgb)
        
        if self.config.debug:
            frames = arena.get_frames()
            if len(frames) > 0:
                save_video(np.stack(arena.get_frames()), 'tmp', 'demo_videos')
        
        arena.set_particle_positions(particle_pos)

        return info


    def _load_or_generate_goals(self, arena, num_goals):
        goals = []
        for i in range(num_goals):
            goal_path = os.path.join(self.goal_dir, f"goal_{i}")
            if not os.path.exists(goal_path):
                print(f'Generating goal {i} for episode id {arena.eid}')
                goal = self._generate_a_goal(arena)
                os.makedirs(goal_path, exist_ok=True)

                # Save RGB
                plt.imsave(os.path.join(goal_path, "rgb.png"), goal['observation']['rgb']/255.0)

                # Save particles as PLY
                save_point_cloud_ply(os.path.join(goal_path, "particles.ply"),
                                    goal['observation']["particle_positions"])

                goals.append(goal)
            else:
                # Load existing goal
                rgb = (plt.imread(os.path.join(goal_path, "rgb.png"))*255).astype(np.uint8)
                #print('max rgb', np.max(rgb))
                #print('rgb', rgb.shape)
                particles = load_point_cloud_ply(os.path.join(goal_path, "particles.ply"))
                goal = {
                    'observation': {
                        'rgb': rgb[:, :, :3],
                        'particle_positions': particles
                    }
                }
                goals.append(goal)
        return goals


    def _load_or_create_keypoints(self, arena):
        """Load semantic keypoints if they exist, otherwise ask user to assign them."""

        mesh_id = arena.init_state_params['pkl_path'].split('/')[-1].split('.')[0]  # e.g. 03346_Tshirt
        keypoint_file = os.path.join(self.keypoint_dir, f"{mesh_id}.json")

        if os.path.exists(keypoint_file):
            with open(keypoint_file, "r") as f:
                keypoints = json.load(f)
            if self.config.debug:
                print("annotated keypoint ids", keypoints)
            return keypoints

        # Get flattened garment observation
        flatten_obs = arena.get_flattened_obs()
        flatten_rgb = flatten_obs['observation']["rgb"]
        particle_positions = flatten_obs['observation']["particle_positions"]  # (N, 3)

        # Ask user to click semantic keypoints
        keypoints_pixel = self.keypoint_assignment_gui.run(flatten_rgb)  # dict: {name: (u, v)}
        
        # Project all garment particles
        pixels, visible = arena.get_visibility(particle_positions)
        
        if self.config.debug:
            H, W = (480, 480)


            print('annotated keypoints', keypoints_pixel)

            # Make sure tmp folder exists
            os.makedirs("tmp", exist_ok=True)

            # Start with black canvases
            non_visible_img = np.zeros((H, W, 3), dtype=np.uint8)
            visible_img = np.zeros((H, W, 3), dtype=np.uint8)

            for pix, vis in zip(pixels, visible):
                x, y = pix  # assuming pix = (x, y)
                x = int(x)
                y = int(y)
                if not vis:
                    # non-visible -> gray pixel
                    non_visible_img[x, y] = (128, 128, 128)
                else:
                    # visible -> white pixel
                    visible_img[x, y] = (255, 255, 255)

            # Save both images
            cv2.imwrite("tmp/non-visible.png", non_visible_img)
            cv2.imwrite("tmp/visible.png", visible_img)


        keypoints = {}
        for name, pix in keypoints_pixel.items():
            y, x = pix
            dists = np.linalg.norm(pixels - np.array((x, y)), axis=1)
            particle_id = np.argmin(dists)
            keypoints[name] = int(particle_id)
        
        if self.config.debug:
            annotated = np.zeros((H, W, 3), dtype=np.uint8)
            for pid in keypoints.values():
                x, y = pixels[pid]
                x = int(x)
                y = int(y)
                annotated[x, y] = (255, 255, 255)
            cv2.imwrite("tmp/annotated.png", annotated)


        with open(keypoint_file, "w") as f:
            json.dump(keypoints, f, indent=2)
        return keypoints


    def evaluate(self, arena):
        """Evaluate folding quality using particle alignment and semantic keypoints."""
        if len(self.goals) == 0:
            return {}
        cur_particles = arena.get_mesh_particles_positions()

        # Evaluate particle alignment against each goal
        particle_distances = []
        key_distances = []
        for goal in self.goals:
            goal_particles = goal['observation']["particle_positions"][:arena.num_mesh_particles]
            mdp, kdp = self._compute_particle_distance(cur_particles, goal_particles, arena)
            particle_distances.append(mdp)
            key_distances.append(kdp)
       
        mean_particle_distance = min(particle_distances)
        key_particle_distance = min(key_distances)
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
        if len(self.aligned_pairs) == arena.action_step + 1:
            return self.aligned_pairs[-1]

        if self.config.alignment == 'simple_rigid':
            # Center both sets
            aligned_curr, aligned_goal = simple_rigid_align(cur, goal)
            #return aligned, goal_centered
        elif self.config.alignment == 'complex_rigid':
            aligned_curr, aligned_goal = rigid_align(cur, goal, arena.get_cloth_area())
        elif self.config.alignment == 'deform':
            aligned_curr, aligned_goal = deformable_align(cur, goal, arena.get_cloth_area())
        else:
            raise NotImplementedError
        
        # Safety check for NaNs
        assert not (np.isnan(aligned_curr).any() or np.isnan(aligned_goal).any()), \
            "NaN values detected after point alignment!"
        
        self.aligned_pairs.append((aligned_curr, aligned_goal))
        
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
            for align_type in ['simple_rigid', 'complex_rigid', 'deform', 'chamfer_rotation']:
                if align_type == 'simple_rigid':
                    aligned_curr, aligned_goal = simple_rigid_align(cur, goal)
                elif align_type == 'complex_rigid':
                    #print('Cloth Area', arena.get_cloth_area())
                    aligned_curr, aligned_goal = rigid_align(cur, goal, arena.get_cloth_area())
                elif align_type == 'deform':
                    aligned_curr, aligned_goal = deformable_align(cur, goal, arena.get_cloth_area())
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
        
        multi_stage_reward = coverage_alignment_reward(last_info, action, info) - 1 
        if info['observation']['action_step'] - info['observation']['last_flattened_step'] <= 3:
            multi_stage_reward = particle_distance_reward(mpd) # 0 to 1
        
        if info['success']:
            multi_stage_reward = info['arena'].horizon - info['observation']['action_step']
            
        return {
            'particle_distance': particle_distance_reward(mpd),
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
        return cur_eval['mean_particle_distance'] < 0.07
    
    def _get_max_IoU(self, arena):
        cur_mask = arena.cloth_mask
        max_IoU = 0
        for goal in self.goals[:1]:
            goal_mask = goal['observation']["rgb"].sum(axis=2) > 0 ## only useful for background is black
            
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