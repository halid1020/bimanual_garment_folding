from actoris_harena import Agent
import numpy as np
import cv2

class NoOperation(Agent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "human-pixel-pick-and-place-two"

    def act(self, info_list, update=False):
        """
        Pop up a window shows the RGB image, and user can click on the image to
        produce normalised pick-and-place action ranges from [-1, 1]
        """
        actions = []
        for info in info_list:
            actions.append(self.single_act(info))
        
        return actions
    
    def single_act(self, state, update=False):
        """
        Pop up a window shows the RGB image, and user can click on the image to
        produce normalised pick-and-place actions for two objects, ranges from [-1, 1]
        """
        return np.zeros(0)
        
    def init(self, state):
        pass
    
    def update(self, state, action):
        pass