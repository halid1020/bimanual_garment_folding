import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from agent_arena import Task
from ..utils.garment_utils import KEYPOINT_SEMANTICS
from ..utils.keypoint_gui import KeypointGUI

class GarmentFoldingTask(Task):
    def __init__(self, config):
        self.num_goals = config.num_goals
        self.task_name = config.task_name
        self.asset_dir = config.asset_dir
        self.demonstrator = config.demonstrator ## TODO: This needs to be initialised before the class.
        
        self.keypoint_semantics = KEYPOINT_SEMANTICS[config.object]

        self.semkey2pid = None # This needs to be loaded or annotated
        self.goals = [] # This needs to be loaded  or generated

        self.keypoint_assignment_gui = KeypointGUI(self.keypoint_semantics )
        
        

    def reset(self, arena):
        """Reset environment and generate goals if necessary."""
        self.goal_dir = os.path.join(self.asset_dir, 'goals', arena.get_name(), self.task_name, arena.get_mode(), f"eid_{arena.get_episode_id()}")
        os.makedirs(self.goal_dir, exist_ok=True)

        # Load or create semantic keypoints
        self.semkey2pid = self._load_or_create_keypoints(arena)

        # Generate goals (10 small variations)
        self.goals = self._generate_goals(arena, self.num_goals)

        return {"goals": self.goals, "keypoints": self.semkey2pid}

    def _generate_a_goal(self, arena):
        
        particle_pos = arena.get_particle_positions()
        info = arena.set_to_flatten()
        self.demonstrator.reset([arena.id])
        while not self.demonstrator.terminate()[arena.id]: ## The demonstrator does not need update and init function
            print('here!')
            action = self.demonstrator.single_act(info) # Fold action
            print('action', action)
            info = arena.step(action)
        
        arena.set_particle_positions(particle_pos)

        return info


    def _generate_goals(self, arena, num_goals):
        """Generate multiple folding goals with variations and save them."""
        goals = []
        for i in range(num_goals):
            goal = self._generate_a_goal(arena)  # You need to implement this in arena
            goal_path = os.path.join(self.goal_dir, f"goal_{i}")
            os.makedirs(goal_path, exist_ok=True)
            plt.imsave(os.path.join(goal_path, "rgb.png"), goal["rgb"])
            np.save(os.path.join(goal_path, "depth.npy"), goal["depth"])
            np.save(os.path.join(goal_path, "particles.npy"), goal["particles"])
            goals.append(goal)
        return goals

    def _load_or_create_keypoints(self, arena):
        """Load semantic keypoints if they exist, otherwise ask user to assign them."""
        keypoint_file = os.path.join(self.goal_dir, "keypoints.npy")

        if os.path.exists(keypoint_file):
            keypoints = np.load(keypoint_file, allow_pickle=True).item()
            print('annotated keypoint ids', keypoints)
            return np.load(keypoint_file, allow_pickle=True).item()

        # Get flattened garment observation
        flatten_obs = arena.get_flattened_obs()
        flatten_rgb = flatten_obs['observation']["rgb"]
        particle_positions = flatten_obs['observation']["particle_position"]  # (N, 3)

        # Ask user to click semantic keypoints
        keypoints_pixel = self.keypoint_assignment_gui.run(flatten_rgb)  # dict: {name: (u, v)}
        print('annotated keypoints', keypoints_pixel)
        # Project all garment particles
        pixels, visible = arena.get_visibility(particle_positions)

        keypoints = {}
        for name, pix in keypoints_pixel.items():
            # Only consider visible particles
            visible_pixels = pixels[visible]
            visible_ids = np.where(visible)[0]

            if len(visible_pixels) == 0:
                raise RuntimeError("No visible particles to assign keypoints.")

            # Find nearest visible particle in pixel space
            dists = np.linalg.norm(visible_pixels - np.array(pix), axis=1)
            nearest_idx = np.argmin(dists)
            particle_id = visible_ids[nearest_idx]

            keypoints[name] = particle_id

        print('annotated keypoint ids', keypoints)
        # Save particle IDs instead of pixel coords
        np.save(keypoint_file, keypoints)
        return keypoints


    def evaluate(self, arena):
        """Evaluate folding quality using particle alignment and semantic keypoints."""
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
    
    def reward(self): # TODO: implement this with keypoint distance and particle distance.
        return {
            'dummy': 0
        }
    
    def get_goal(self):
        return self.goals
    
    def success(self):
        cur_eval = self.evaluate()
        return cur_eval['mean_particle_distance'] < 0.01
