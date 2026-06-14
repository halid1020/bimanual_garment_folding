from actoris_harena import Agent
import numpy as np
import cv2
from .utils import (
    overlay_workspaces, overlay_active_goal_contour, append_goal_grid, 
    get_user_clicks_with_undo, normalize_clicks, draw_evaluation_metrics
)

class HumanPickAndFling(Agent):
    """Interactive policy for executing a Pick-and-Fling sequence."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "human-pixel-pick-and-fling"
        self.overlay_goal_contour = config.get('overlay_goal_contour', False)

    def act(self, info_list, update=False):
        return [self.single_act(info) for info in info_list]

    def single_act(self, state, update=False):
        rgb = cv2.resize(cv2.cvtColor(state['observation']['rgb'], cv2.COLOR_BGR2RGB), (512, 512))
        rgb = overlay_workspaces(rgb, state)

        if self.overlay_goal_contour:
            rgb = overlay_active_goal_contour(rgb, state)

        # Build full UI layout
        img = append_goal_grid(rgb, state)
        
        # Draw metrics on bottom right
        draw_evaluation_metrics(img, state)

        # Collect user input (2 clicks for Fling)
        clicks = get_user_clicks_with_undo(img, num_clicks=2, window_name='Pick & Fling (2 clicks)')
        
        # Normalize and package action
        h, w = rgb.shape[:2]
        norm = normalize_clicks(clicks, w, h)
        
        return np.asarray(norm)

    def init(self, state): pass
    def update(self, state, action): pass