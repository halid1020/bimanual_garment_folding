from agent_arena import Agent
import numpy as np
import cv2
from .utils import draw_text_top_right, apply_workspace_shade
import os

class HumanDualPickersPickAndPlace(Agent):
    
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
        
        ## make it bgr to rgb using cv2
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        

        ## resize
        rgb = cv2.resize(rgb, (512, 512))
        H, W = rgb.shape[:2]
        if 'robot0_mask' in state['observation']:
            mask0 = state['observation']['robot0_mask'].astype(bool)

            if mask0.shape[:2] != (H, W):
                mask0 = cv2.resize(
                    mask0.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            rgb = apply_workspace_shade(
                rgb,
                mask0,
                color=(255, 0, 0),  # Blue in BGR
                alpha=0.2
            )

        if 'robot1_mask' in state['observation']:
            mask1 = state['observation']['robot1_mask'].astype(bool)

            if mask1.shape[:2] != (H, W):
                mask1 = cv2.resize(
                    mask1.astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            rgb = apply_workspace_shade(
                rgb,
                mask1,
                color=(0, 0, 255),  # Red in BGR
                alpha=0.2
            )


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
        
        clicks = []
        os.environ["DISPLAY"] = "localhost:10.0"

        # Create a separate panel for buttons (e.g., 80 pixels wide)
        button_panel_width = 80
        panel = np.ones((img.shape[0], button_panel_width, 3), dtype=np.uint8) * 50  # dark gray panel
        img_with_panel = np.concatenate([img, panel], axis=1)

        # Define Undo button rectangle inside panel
        button_top_left = (10, 10)
        button_bottom_right = (button_panel_width - 10, 50)

        def redraw_image():
            temp_img = img.copy()
            # Draw pick-and-place clicks
            for i, (x, y) in enumerate(clicks):
                if i % 2 == 0:  # pick
                    color = (0, 255, 0) if i < 2 else (0, 0, 255)
                    cv2.circle(temp_img, (x, y), 5, color, -1)
                else:  # place
                    color = (0, 255, 0) if i < 2 else (0, 0, 255)
                    cv2.drawMarker(temp_img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            # Draw button panel
            panel_img = np.ones((img.shape[0], button_panel_width, 3), dtype=np.uint8) * 50
            cv2.rectangle(panel_img, button_top_left, button_bottom_right, (0, 0, 0), -1)
            cv2.putText(panel_img, 'UNDO', (button_top_left[0] + 5, button_top_left[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            display_img = np.concatenate([temp_img, panel_img], axis=1)
            cv2.imshow('Click Pick and Place Points (4 clicks needed)', display_img)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is in panel
                if x >= img.shape[1]:  # clicked in button panel
                    panel_x = x - img.shape[1]
                    if button_top_left[0] <= panel_x <= button_bottom_right[0] and \
                    button_top_left[1] <= y <= button_bottom_right[1]:
                        if clicks:
                            clicks.pop()
                            redraw_image()
                else:
                    clicks.append((x, y))
                    redraw_image()

        redraw_image()
        cv2.setMouseCallback('Click Pick and Place Points (4 clicks needed)', mouse_callback)

        while len(clicks) < 4:
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        os.environ["DISPLAY"] = ""

        # # Store click coordinates
        # clicks = []
        # os.environ["DISPLAY"] = "localhost:10.0"
        # def mouse_callback(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         clicks.append((x, y))
        #         if len(clicks) % 2 == 1:  # Pick action (odd clicks)
        #             color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
        #             cv2.circle(img, (x, y), 5, color, -1)
        #         else:  # Place action (even clicks)
        #             color = (0, 255, 0) if len(clicks) <= 2 else (0, 0, 255)  # Green for first, Red for second
        #             cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        #         cv2.imshow('Click Pick and Place Points (4 clicks needed)', img)
        
        # cv2.imshow('Click Pick and Place Points (4 clicks needed)', img)
        # cv2.setMouseCallback('Click Pick and Place Points (4 clicks needed)', mouse_callback)
        
        # while len(clicks) < 4:
        #     cv2.waitKey(1)
        
        # cv2.destroyAllWindows()
        # os.environ["DISPLAY"] = ""
        
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
        
    
    # def single_act(self, state, update=False):
    #     rgb = state['observation']['rgb']
    #     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #     rgb = cv2.resize(rgb, (512, 512))

    #     # Overlay success + IoU info
    #     if 'evaluation' in state.keys() and state['evaluation'] != {}:
    #         success = state['success']
    #         max_iou_flat = state['evaluation']['max_IoU_to_flattened']
    #         text_lines = [
    #             (f"Success: {success}", (0, 255, 0) if success else (0, 0, 255)),
    #             (f"IoU(flat): {max_iou_flat:.3f}", (255, 255, 255))
    #         ]
    #         if 'max_IoU' in state['evaluation'].keys():
    #             max_iou_goal = state['evaluation']['max_IoU']
    #             text_lines.append((f"IoU(fold): {max_iou_goal:.3f}", (255, 255, 255)))
    #         draw_text_top_right(rgb, text_lines)

    #     img = rgb.copy()

    #     # Display goal images if present
    #     if 'goals' in state.keys():
    #         goals = state['goals'][:4]  # max 4 for 2x2
    #         rgbs = []
    #         for goal in goals:
    #             g = goal['observation']['rgb']
    #             if g.shape[-1] == 3:
    #                 g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    #             g = cv2.resize(g, (256, 256))
    #             rgbs.append(g)
    #         while len(rgbs) < 4:
    #             rgbs.append(np.zeros((256, 256, 3), dtype=np.uint8))
    #         top_row = np.concatenate([rgbs[0], rgbs[1]], axis=1)
    #         bottom_row = np.concatenate([rgbs[2], rgbs[3]], axis=1)
    #         goal_rgb = np.concatenate([top_row, bottom_row], axis=0)
    #         img = np.concatenate([img, goal_rgb], axis=1)

    #     # Draw vertical line between images
    #     line_x = rgb.shape[1]
    #     cv2.line(img, (line_x, 0), (line_x, img.shape[0]), (255, 255, 255), 2)

    #     # Create a separate panel for buttons (e.g., 80 pixels wide)
    #     button_panel_width = 80
    #     panel = np.ones((img.shape[0], button_panel_width, 3), dtype=np.uint8) * 50  # dark gray panel
    #     img_with_panel = np.concatenate([img, panel], axis=1)

    #     # Define Undo button rectangle inside panel
    #     button_top_left = (10, 10)
    #     button_bottom_right = (button_panel_width - 10, 50)

    #     clicks = []
    #     os.environ["DISPLAY"] = "localhost:10.0"

    #     def redraw_image():
    #         temp_img = img.copy()
    #         # Draw pick-and-place clicks
    #         for i, (x, y) in enumerate(clicks):
    #             if i % 2 == 0:  # pick
    #                 color = (0, 255, 0) if i < 2 else (0, 0, 255)
    #                 cv2.circle(temp_img, (x, y), 5, color, -1)
    #             else:  # place
    #                 color = (0, 255, 0) if i < 2 else (0, 0, 255)
    #                 cv2.drawMarker(temp_img, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    #         # Draw button panel
    #         panel_img = np.ones((img.shape[0], button_panel_width, 3), dtype=np.uint8) * 50
    #         cv2.rectangle(panel_img, button_top_left, button_bottom_right, (0, 0, 0), -1)
    #         cv2.putText(panel_img, 'UNDO', (button_top_left[0] + 5, button_top_left[1] + 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #         display_img = np.concatenate([temp_img, panel_img], axis=1)
    #         cv2.imshow('Click Pick and Place Points (4 clicks needed)', display_img)

    #     def mouse_callback(event, x, y, flags, param):
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             # Check if click is in panel
    #             if x >= img.shape[1]:  # clicked in button panel
    #                 panel_x = x - img.shape[1]
    #                 if button_top_left[0] <= panel_x <= button_bottom_right[0] and \
    #                 button_top_left[1] <= y <= button_bottom_right[1]:
    #                     if clicks:
    #                         clicks.pop()
    #                         redraw_image()
    #             else:
    #                 clicks.append((x, y))
    #                 redraw_image()

    #     redraw_image()
    #     cv2.setMouseCallback('Click Pick and Place Points (4 clicks needed)', mouse_callback)

    #     while len(clicks) < 4:
    #         cv2.waitKey(1)

    #     cv2.destroyAllWindows()
    #     os.environ["DISPLAY"] = ""

    #     height, width = rgb.shape[:2]
    #     pick1_x, pick1_y = clicks[0]
    #     place1_x, place1_y = clicks[1]
    #     pick2_x, pick2_y = clicks[2]
    #     place2_x, place2_y = clicks[3]

    #     normalized_action1 = [
    #         (pick1_x / width) * 2 - 1,
    #         (pick1_y / height) * 2 - 1,
    #         (place1_x / width) * 2 - 1,
    #         (place1_y / height) * 2 - 1
    #     ]

    #     normalized_action2 = [
    #         (pick2_x / width) * 2 - 1,
    #         (pick2_y / height) * 2 - 1,
    #         (place2_x / width) * 2 - 1,
    #         (place2_y / height) * 2 - 1
    #     ]

    #     return np.concatenate([
    #         normalized_action1[:2], normalized_action2[:2],
    #         normalized_action1[2:], normalized_action2[2:]
    #     ])

    def init(self, state):
        pass
    
    def update(self, state, action):
        pass