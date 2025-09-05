from agent_arena import Agent
import numpy as np
import cv2

from .pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker
from .pick_and_fling.pixel_human import PixelHumanFling

class PixelMultiPrimitive(Agent):
    
    def __init__(self, config):
        
        super().__init__(config)
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place"
        ]
        self.primitive_instances = [
            PixelHumanFling(config),
            PixelHumanTwoPicker(config)]
    
    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

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
        """
        Allow user to choose a primitive, then delegate to the chosen primitive's act method.
        Shows rgb and goal_rgb images while prompting for input.
        """

        # Extract the RGB and goal images
        rgb = state['observation']['rgb']
        goal_rgb = state['goal']['rgb']

        # Ensure both are in correct format (uint8 BGR for OpenCV)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        if goal_rgb.dtype != np.uint8:
            goal_rgb = (goal_rgb * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV display
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        goal_bgr = cv2.cvtColor(goal_rgb, cv2.COLOR_RGB2BGR)

        # Combine images side by side
        combined = np.hstack((rgb_bgr, goal_bgr))

        # Show window
        cv2.imshow("Current RGB (left) | Goal RGB (right)", combined)
        cv2.waitKey(1)  # Keeps the window responsive

        chosen_primitive = None
        while True:
            print("\nChoose a primitive:")
            for i, primitive in enumerate(self.primitive_names):
                print(f"{i + 1}. {primitive}")
        
            try:
                choice = int(input("Enter the number of your choice: ")) - 1
                if 0 <= choice < len(self.primitive_names):
                    chosen_primitive = self.primitive_names[choice]
                    self.current_primitive = self.primitive_instances[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Close OpenCV window once a valid choice is made
        cv2.destroyAllWindows()

        # Delegate action
        action = self.current_primitive.single_act(state)

        return {
            chosen_primitive: action
        }
