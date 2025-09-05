from agent_arena import Agent
import numpy as np
import cv2

class RandomPickAndPlace(Agent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "random-pixel-pick-and-place-two"

    def act(self, info_list, update=False):
        actions = []
        for info in info_list:
            actions.append(self.single_act(info))
        
        return actions
    
    def single_act(self, state):
        
        normalized_action1 =  np.random.uniform(-1, 1, size=4)
        normalized_action2 =  np.random.uniform(-1, 1, size=4)
        
        return {
            'pick_0': normalized_action1[:2],
            'place_0': normalized_action1[2:],
            'pick_1': normalized_action2[:2],
            'place_1': normalized_action2[2:],
        }
        
    def init(self, state):
        pass
    
    def update(self, state, action):
        pass