from agent_arena import Agent
import numpy as np
import cv2

from ..human.pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker
from ..human.pick_and_fling.pixel_human import PixelHumanFling

class RandomMultiPrimitive(Agent):
    
    def __init__(self, config):
        
        super().__init__(config)
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place"
        ]
        self.primitive_instances = [
            PixelRandomFling(config),
            PixelRandomTwoPicker(config)]
    
    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def act(self, info_list, update=False):
        """
        Pop up a window shows the RGB image, and user can click on the image to
        produce normalised pick-and-place action ranges from [-1, 1]
        """
        actions = []
        for info in info_list:
            actions.append(self.single_act(info))
        
        return actions
    

    def single_act(self, state):
        """
        Allow user to choose a primitive, then delegate to the chosen primitive's act method.
        Shows rgb and goal_rgb images while prompting for input.
        """

        # Extract the RGB and goal images
        rgb = state['observation']['rgb']
        goal_rgb = state['goal']['rgb']

        # Ensure both are in correct format (uint8 BGR for OpenCV)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        if goal_rgb.dtype != np.uint8:
            goal_rgb = (goal_rgb * 255).astype(np.uint8)

        chosen_primitive = # Random
        # Delegate action
        action = self.current_primitive.single_act(state)

        return {
            chosen_primitive: action
        }
