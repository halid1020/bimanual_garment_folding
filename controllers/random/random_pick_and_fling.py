from agent_arena import Agent
import numpy as np
import cv2

class RandomPickAndFling(Agent):
    
        def __init__(self, config):
            #self.config = config
            self.name = "random-pixel-pick-and-fling"

        def act(self, info_list, update=False):
            """
            Pop up a window shows the RGB image, and user can click on the image to
            produce normalised pick-and-place action ranges from [-1, 1]
            """
            actions = []
            for info in info_list:
                actions.append(self.single_act(info))
            
            return actions
        
        def reset(self, arena_ids):
            self.internal_states = {arena_id: {} for arena_id in arena_ids}

        def single_act(self, state, update=False):

            mask = state['observation']['mask']
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            mask_coords = np.argwhere(mask)
            if len(mask_coords) == 0:
                return {
                    'pick_0': normalized_action[:2],
                    'pick_1': normalized_action[2:]
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
                'pick_1': pick1_pixel,
            }
            action["norm-pixel-pick-and-fling"] = {
                'pick_0': pick0_pixel,
                'pick_1': pick1_pixel,
            }
            return action
            
    