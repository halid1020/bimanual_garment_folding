import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from scipy.spatial.distance import cdist
from agent_arena import Task
from agent_arena import save_video
from ..utils.garment_utils import KEYPOINT_SEMANTICS
from ..utils.keypoint_gui import KeypointGUI

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
        self.task_name = config.task_name
        self.asset_dir = config.asset_dir
        self.config = config
        self.demonstrator = config.demonstrator ## TODO: This needs to be initialised before the class.
        
        self.keypoint_semantics = KEYPOINT_SEMANTICS[config.object]

        self.semkey2pid = None # This needs to be loaded or annotated
        self.goals = [] # This needs to be loaded  or generated

        self.keypoint_assignment_gui = KeypointGUI(self.keypoint_semantics )
        self.keypoint_dir = os.path.join(self.asset_dir, 'keypoints')
        os.makedirs(self.keypoint_dir, exist_ok=True)
        

    def reset(self, arena):
        """Reset environment and generate goals if necessary."""
        self.goal_dir = os.path.join(self.asset_dir, 'goals', arena.get_name(), \
                                     self.task_name, arena.get_mode(), f"eid_{arena.get_episode_id()}")
        
        os.makedirs(self.goal_dir, exist_ok=True)

        # Load or create semantic keypoints
        self.semkey2pid = self._load_or_create_keypoints(arena)

        # Generate goals (10 small variations)
        self.goals = self._load_or_generate_goals(arena, self.num_goals)

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
            save_video(np.stack(arena.get_frames()), 'tmp', 'demo_videos')
        
        arena.set_particle_positions(particle_pos)

        return info


    def _load_or_generate_goals(self, arena, num_goals):
        goals = []
        for i in tqdm(range(num_goals), desc="Generating goals"):
            goal_path = os.path.join(self.goal_dir, f"goal_{i}")
            if not os.path.exists(goal_path):
                goal = self._generate_a_goal(arena)
                os.makedirs(goal_path, exist_ok=True)

                # Save RGB
                plt.imsave(os.path.join(goal_path, "rgb.png"), goal['observation']['rgb'])

                # Save particles as PLY
                save_point_cloud_ply(os.path.join(goal_path, "particles.ply"),
                                    goal['observation']["particle_positions"])

                goals.append(goal)
            else:
                # Load existing goal
                rgb = plt.imread(os.path.join(goal_path, "rgb.png"))
                particles = load_point_cloud_ply(os.path.join(goal_path, "particles.ply"))
                goal = {
                    'observation': {
                        'rgb': rgb,
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
            print("annotated keypoint ids", keypoints)
            return keypoints

        # Get flattened garment observation
        flatten_obs = arena.get_flattened_obs()
        flatten_rgb = flatten_obs['observation']["rgb"]
        particle_positions = flatten_obs['observation']["particle_positions"]  # (N, 3)

        # Ask user to click semantic keypoints
        keypoints_pixel = self.keypoint_assignment_gui.run(flatten_rgb)  # dict: {name: (u, v)}
        print('annotated keypoints', keypoints_pixel)
        # Project all garment particles
        pixels, visible = arena.get_visibility(particle_positions)
        
        if self.config.debug:
            H, W = (480, 480)

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
        cur_particles = arena.get_particle_positions()

        # Evaluate particle alignment against each goal
        particle_distances = []
        for goal in self.goals:
            goal_particles = goal["particles"]
            aligned_dist = self._compute_particle_distance(cur_particles, goal_particles)
            particle_distances.append(aligned_dist)
        mean_particle_distance = min(particle_distances)

        # Evaluate semantic keypoints
        cur_keypoints = arena.get_keypoints()  # Arena should provide detected keypoints
        semantic_dist = self._compute_keypoint_distance(cur_keypoints, self.semkey2pid)

        return {
            "mean_particle_distance": mean_particle_distance,
            "semantic_keypoint_distance": semantic_dist
        }

    def _align_points(self, cur, goal):
        """
        Align cur points to goal points using Procrustes rigid alignment.
        Returns aligned points and mean distance.
        """
        # Center both sets
        cur_centered = cur - np.mean(cur, axis=0)
        goal_centered = goal - np.mean(goal, axis=0)

        # Compute optimal rotation via SVD
        H = cur_centered.T @ goal_centered
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        aligned = cur_centered @ R

        return aligned, goal_centered

    def _compute_particle_distance(self, cur, goal):
        """Align particles and compute mean distance."""
        aligned, goal_centered = self._align_points(cur, goal)
        return np.mean(np.linalg.norm(aligned - goal_centered, axis=1))

    def _compute_keypoint_distance(self, cur, goal):
        """Align semantic keypoints and compute mean distance."""

        aligned_cur, aligned_goal = self._align_points(cur, goal)
        
        cur_pts = []
        goal_pts = []
        for name, pid in self.semantic_keypoints.items():
            
            cur_pts.append(aligned_cur[pid])
            goal_pts.append(aligned_goal[pid])


        return np.mean(np.linalg.norm(cur_pts - goal_pts, axis=1))
    
    def reward(self, last_info, action, info): # TODO: implement this with keypoint distance and particle distance.
        return {
            'dummy': 0
        }
    
    def get_goal(self):
        return self.goals
    
    def success(self, arena):
        cur_eval = self.evaluate(arena)
        if cur_eval == {}:
            return False
        return cur_eval['mean_particle_distance'] < 0.01
