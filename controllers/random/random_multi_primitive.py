from agent_arena import Agent
import numpy as np
import cv2
import random

from .random_pick_and_place import RandomPickAndPlace
from .random_pick_and_fling import RandomPickAndFling
from .random_pick_and_drag import RandomPickAndDrag
from .random_fold import RandomFold

class RandomMultiPrimitive(Agent):
    
    def __init__(self, config):
        
        super().__init__(config)
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place",
            "norm-pixel-pick-and-drag",
            "norm-pixel-fold",
        ]
        self.primitive_instances = [
            RandomPickAndFling(config),
            RandomPickAndPlace(config),
            RandomPickAndDrag(config),
            RandomFold(config)]
    
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
        
        print('random action', actions[0])
        
        return actions
    

    def single_act(self, state):
        """
        Allow user to choose a primitive, then delegate to the chosen primitive's act method.
        Shows rgb and goal_rgb images while prompting for input.
        """

        pid = random.choice(range(len(self.primitive_instances)))
        # Delegate action
        action = self.primitive_instances[pid].single_act(state)

        return {
            self.primitive_names[pid]: action
        }
