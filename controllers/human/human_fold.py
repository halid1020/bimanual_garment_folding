from agent_arena import Agent
import numpy as np
import cv2

class HumanFold(Agent):
    
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
                if len(clicks) % 2 == 1:  # Pick action (odd clicks)
                    color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
                    cv2.circle(img, (x, y), 5, color, -1)
                else:  # Place action (even clicks)
                    color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
                    cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                cv2.imshow('Click Pick and Place Points (4 clicks needed)', img)
        
        cv2.imshow('Click Pick and Place Points (4 clicks needed)', img)
        cv2.setMouseCallback('Click Pick and Place Points (4 clicks needed)', mouse_callback)
        
        while len(clicks) < 4:
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        
        # Normalize the coordinates to [-1, 1]
        height, width = rgb.shape[:2]
        pick1_y, pick1_x = clicks[0]
        place1_y, place1_x = clicks[1]
        pick2_y, pick2_x = clicks[2]
        place2_y, place2_x = clicks[3]
        
        normalized_action1 = [
            (pick1_x / width) * 2 - 1,
            (pick1_y / height) * 2 - 1,
            (place1_x / width) * 2 - 1,
            (place1_y / height) * 2 - 1
        ]
        
        normalized_action2 = [
            (pick2_x / height) * 2 - 1,
            (pick2_y / width) * 2 - 1,
            (place2_x / height) * 2 - 1,
            (place2_y / width) * 2 - 1
        ]
        
        return np.concatenate([
            normalized_action1[:2], normalized_action2[:2], 
            normalized_action1[2:], normalized_action2[2:] ])
        
    def init(self, state):
        pass
    
    def update(self, state, action):
        pass