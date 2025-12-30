from agent_arena import Agent
import numpy as np
import cv2
import os

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
        

            ## make it bgr to rgb using cv2
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            

            ## resize
            rgb = cv2.resize(rgb, (512, 512))
            
            
            # Create a copy of the image to draw on
            img = rgb.copy()

            # put img and goal_img side by side
            # if 'goal' in state.keys():
            #     goal_rgb = state['goal']['rgb']
            #     goal_rgb = cv2.resize(goal_rgb, (512, 512))
            #     goal_rgb = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2RGB)
            #     img = np.concatenate([img, goal_rgb], axis=1)

            if 'goals' in state.keys():
                goals = state['goals']  # list of goal infos

                # Extract goal RGBs
                rgbs = []
                for goal in goals[:4]:  # max 4 for 2x2 grid
                    g = goal['observation']['rgb']

                    # Ensure RGB
                    if g.shape[-1] == 3:
                        g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)

                    # Resize to half-size (for 2x2 grid)
                    g = cv2.resize(g, (256, 256))
                    rgbs.append(g)

                # Pad with black images if fewer than 4
                while len(rgbs) < 4:
                    rgbs.append(np.zeros((256, 256, 3), dtype=np.uint8))

                # Arrange into 2x2 grid
                top_row = np.concatenate([rgbs[0], rgbs[1]], axis=1)
                bottom_row = np.concatenate([rgbs[2], rgbs[3]], axis=1)
                goal_rgb = np.concatenate([top_row, bottom_row], axis=0)
                img = np.concatenate([img, goal_rgb], axis=1)

            #os.environ["DISPLAY"] = "localhost:10.0"
            # Draw vertical white line between the two images
            line_x = rgb.shape[1]   # x-position = width of left image (512)
            cv2.line(img, (line_x, 0), (line_x, img.shape[0]), (255, 255, 255), 2)
            # Store click coordinates
            clicks = []
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicks.append((x, y))
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow('Click Two Pick Points for Fling', img)
            
            cv2.imshow('Click Two Pick Points for Fling', img)
            cv2.setMouseCallback('Click Two Pick Points for Fling', mouse_callback)
            
            while len(clicks) < 2:
                cv2.waitKey(1)
            
            cv2.destroyAllWindows()
            #os.environ["DISPLAY"] = ""
            
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
            
            return np.asarray(normalized_action)
            
            
        def get_phase(self):
            return "default"
        
        def terminate(self):
            return self.step >= 1
    