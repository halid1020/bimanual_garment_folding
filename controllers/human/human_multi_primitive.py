import cv2
import numpy as np
from actoris_harena import Agent

from .human_pick_and_fling import HumanPickAndFling
from .human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
from .human_single_picker_pick_and_place import HumanSinglePickerPickAndPlace
from .no_operation import NoOperation
from .utils import overlay_workspaces, overlay_active_goal_contour, append_goal_grid, draw_evaluation_metrics

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

        # 1. First, build the combined layout (Current State | 2x2 Grid)
        img = append_goal_grid(rgb, state)
        
        # 2. Then, draw the metrics onto the bottom-right corner of the entire layout
        draw_evaluation_metrics(img, state)
        
        obs_dir = "tmp/human_rgb.png"
        cv2.imwrite(obs_dir, img)
        print(f'[human-multi-primitive] Current Observation logged to {obs_dir}')

        chosen_primitive = None
        while True:
            print("\nSelect a Primitive:")
            for i, prim in enumerate(self.primitive_names):
                print(f"{i + 1}. {prim}")
        
            try:
                choice = int(input("Selection [1-4]: ")) - 1
                if 0 <= choice < len(self.primitive_names):
                    chosen_primitive = self.primitive_names[choice]
                    self.current_primitive = self.primitive_instances[choice]
                    break
                print("Invalid index. Try again.")
            except ValueError:
                print("Invalid input. Enter an integer.")

        action = self.current_primitive.single_act(state)
        self.last_primitive = chosen_primitive

        return {chosen_primitive: action}
    
    def terminate(self):
        return {arena_id: (self.last_primitive == 'no-operation') for arena_id in self.internal_states}

    def init(self, infos): pass
    def update(self, infos, actions): pass