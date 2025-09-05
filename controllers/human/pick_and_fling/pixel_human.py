from agent_arena import Agent
import numpy as np
import cv2

class PixelHumanFling(Agent):
    
        def __init__(self, config):
            #self.config = config
            self.name = "human-pixel-pick-and-fling"

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
            
            normalized_action = np.random.uniform(-1, 1, size=(4))
            
            return {
                'pick_0': normalized_action[:2],
                'pick_1': normalized_action[2:]
            }
            
            
        def get_phase(self):
            return "default"
        
        def terminate(self):
            return self.step >= 1
    