from actoris_harena import Agent
import numpy as np
import cv2

class RandomFold(Agent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "random-pixel-pick-and-place-two"

    def act(self, info_list, update=False):
        actions = []
        for info in info_list:
            actions.append(self.single_act(info))
        
        return actions
    
    def single_act(self, state):

        mask = state['observation']['mask']
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_coords = np.argwhere(mask)
        if len(mask_coords) == 0:
            return {
                'pick_0': np.random.uniform(-1, 1, 2),
                'place_0': np.random.uniform(-1, 1, 2),
                'pick_1': np.random.uniform(-1, 1, 2),
                'place_1': np.random.uniform(-1, 1, 2)
            }
        
        pick0_pixel = mask_coords[np.random.randint(len(mask_coords))]
        pick0_pixel = pick0_pixel.astype(np.float32)
        pick0_pixel[0] = pick0_pixel[0] / mask.shape[0] * 2 - 1
        pick0_pixel[1] = pick0_pixel[1] / mask.shape[1] * 2 - 1

        pick1_pixel = mask_coords[np.random.randint(len(mask_coords))]
        pick1_pixel = pick1_pixel.astype(np.float32)
        pick1_pixel[0] = pick1_pixel[0] / mask.shape[0] * 2 - 1
        pick1_pixel[1] = pick1_pixel[1] / mask.shape[1] * 2 - 1
        
        
        action = {
            
            'pick_0': pick0_pixel,
            'place_0': np.random.uniform(-1, 1, 2),
            'pick_1': pick1_pixel,
            'place_1': np.random.uniform(-1, 1, 2)
        }
      
        
        return action
        
    def init(self, state):
        pass
    
    def update(self, state, action):
        pass