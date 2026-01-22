from agent_arena import Agent
import numpy as np
import cv2
from .utils import draw_text_top_right
from .utils import draw_text_top_right, apply_workspace_shade, CV2_DISPLAY, SIM_DISPLAY
import os

class HumanSinglePickerPickAndPlace(Agent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "human-single-picker-pixel-pick-and-place"

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
        
        ## make it bgr to rgb using cv2
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        

        ## resize
        rgb = cv2.resize(rgb, (512, 512))

        # Overlay success + IoU info BEFORE concatenation
        if 'evaluation' in state.keys() and state['evaluation'] != {}:
            success = state['success']
            max_iou_flat = state['evaluation']['max_IoU_to_flattened']
            
            text_lines = [
                (f"Success: {success}", (0, 255, 0) if success else (0, 0, 255)),
                (f"IoU(flat): {max_iou_flat:.3f}", (255, 255, 255))
            ]

            if 'max_IoU' in state['evaluation'].keys():
                max_iou_goal = state['evaluation']['max_IoU']
                text_lines.append( (f"IoU(fold): {max_iou_goal:.3f}", (255, 255, 255)))

            draw_text_top_right(rgb, text_lines)
        
        
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

        # Draw vertical white line between the two images
        line_x = rgb.shape[1]   # x-position = width of left image (512)
        cv2.line(img, (line_x, 0), (line_x, img.shape[0]), (255, 255, 255), 2)
        
        # Store click coordinates
        clicks = []
        os.environ["DISPLAY"] = CV2_DISPLAY
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))
                if len(clicks) % 2 == 1:  # Pick action (odd clicks)
                    color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
                    cv2.circle(img, (x, y), 5, color, -1)
                else:  # Place action (even clicks)
                    color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
                    cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                cv2.imshow('Click Pick and Place Points (2 clicks needed)', img)
        
        cv2.imshow('Click Pick and Place Points (2 clicks needed)', img)
        cv2.setMouseCallback('Click Pick and Place Points (2 clicks needed)', mouse_callback)
        
        while len(clicks) < 2:
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        os.environ["DISPLAY"] = SIM_DISPLAY

        # Normalize the coordinates to [-1, 1]
        height, width = rgb.shape[:2]
        pick1_y, pick1_x = clicks[0]
        place1_y, place1_x = clicks[1]
        
        normalized_action1 = [
            (pick1_x / width) * 2 - 1,
            (pick1_y / height) * 2 - 1,
            (place1_x / width) * 2 - 1,
            (place1_y / height) * 2 - 1
        ]
        
      
        
        return np.asarray(normalized_action1)
        
    def init(self, state):
        pass
    
    def update(self, state, action):
        pass