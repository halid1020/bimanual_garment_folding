import cv2
import numpy as np
from actoris_harena import Agent

from .human_pick_and_fling import HumanPickAndFling
from .human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
from .human_single_picker_pick_and_place import HumanSinglePickerPickAndPlace
from .no_operation import NoOperation
from .utils import (
    overlay_workspaces, overlay_active_goal_contour, 
    append_goal_grid,
    get_user_primitive_selection
)

class HumanMultiPrimitive(Agent):
    """Router Agent allowing a human user to dynamically switch between action primitives."""
    
    def __init__(self, config):
        super().__init__(config)
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-dual-pick-and-place",
            "norm-pixel-single-pick-and-place",
            "no-operation"
        ]
        self.overlay_goal_contour = config.get('overlay_goal_contour', False)

        self.primitive_instances = [
            HumanPickAndFling(config),
            HumanDualPickersPickAndPlace(config),
            HumanSinglePickerPickAndPlace(config),
            NoOperation(config)
        ]
    
    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        self.last_primitive = None
    
    def single_act(self, state, update=False):
        """Displays observation context, queries user for primitive selection, and executes it."""
        rgb = cv2.resize(cv2.cvtColor(state['observation']['rgb'], cv2.COLOR_BGR2RGB), (512, 512))
        
        rgb = overlay_workspaces(rgb, state)

        if self.overlay_goal_contour:
            rgb = overlay_active_goal_contour(rgb, state)

        img = append_goal_grid(rgb, state)
        
        obs_dir = "tmp/human_rgb.png"
        cv2.imwrite(obs_dir, img)
        
        choice_idx = get_user_primitive_selection(img, self.primitive_names, 'Select Action Primitive')
        
        chosen_primitive = self.primitive_names[choice_idx]
        self.current_primitive = self.primitive_instances[choice_idx]
       
        action = self.current_primitive.single_act(state)
        self.last_primitive = chosen_primitive

        return {chosen_primitive: action}
    
    def terminate(self):
        return {arena_id: (self.last_primitive == 'no-operation') for arena_id in self.internal_states}

    def init(self, infos): pass
    def update(self, infos, actions): pass