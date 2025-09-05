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
            
            rgb = state['observation']['rgb']
            goal_rgb = state['goal']['rgb']

            ## make it bgr to rgb using cv2
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            goal_rgb = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2RGB)

            ## resize
            rgb = cv2.resize(rgb, (512, 512))
            goal_rgb = cv2.resize(goal_rgb, (512, 512))
            
            # Create a copy of the image to draw on
            img = rgb.copy()

            # put img and goal_img side by side
            img = np.concatenate([img, goal_rgb], axis=1)
            
            # Store click coordinates
            clicks = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicks.append((x, y))
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                    cv2.imshow('Click Two Pick Points for Fling', img)
            
            cv2.imshow('Click Two Pick Points for Fling', img)
            cv2.setMouseCallback('Click Two Pick Points for Fling', mouse_callback)
            
            while len(clicks) < 2:
                cv2.waitKey(1)
            
            cv2.destroyAllWindows()
            
            # Normalize the coordinates to [-1, 1]
            height, width = rgb.shape[:2]
            pick_y, pick_x = clicks[0]
            place_y, place_x = clicks[1]
            
            normalized_action = [
                (pick_x / width) * 2 - 1,
                (pick_y / height) * 2 - 1,
                (place_x / width) * 2 - 1,
                (place_y / height) * 2 - 1
            ]
            
            return {
                'pick_0': normalized_action[:2],
                'pick_1': normalized_action[2:]
            }
            
            
        def get_phase(self):
            return "default"
        
        def terminate(self):
            return self.step >= 1
    