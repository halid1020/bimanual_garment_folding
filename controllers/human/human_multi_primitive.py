from agent_arena import Agent
import numpy as np
import cv2

from .pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker
from .pick_and_fling.pixel_human import PixelHumanFling
from .human_pick_and_drag import HumanPickAndDrag
from .human_fold import HumanFold

class HumanMultiPrimitive(Agent):
    
    def __init__(self, config):
        
        super().__init__(config)
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place",
            "norm-pixel-pick-and-drag",
            "norm-pixel-fold",
        ]
        self.primitive_instances = [
            PixelHumanFling(config),
            PixelHumanTwoPicker(config),
            HumanPickAndDrag(config),
            HumanFold(config)
        ]
    
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
    

    def single_act(self, state, update=False):
        """
        Allow user to choose a primitive, then delegate to the chosen primitive's act method.
        Shows rgb and goal_rgb images while prompting for input.
        """

        # Extract the RGB and goal images
                    ## make it bgr to rgb using cv2
        rgb = state['observation']['rgb']
        goal_rgb = state['goal']['rgb']

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        goal_rgb = cv2.cvtColor(goal_rgb, cv2.COLOR_BGR2RGB)

        ## resize
        rgb = cv2.resize(rgb, (512, 512))
        goal_rgb = cv2.resize(goal_rgb, (512, 512))
        
        # Create a copy of the image to draw on
        img = rgb.copy()

        # put img and goal_img side by side
        img = np.concatenate([img, goal_rgb], axis=1)

        # Show window
        cv2.imwrite("tmp/human_rgb.png", img)

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
        # cv2.destroyAllWindows()

        # Delegate action
        action = self.current_primitive.single_act(state)

        return {
            chosen_primitive: action
        }
