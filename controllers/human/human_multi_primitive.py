from agent_arena import Agent
import numpy as np
import cv2

from .pick_and_place.pixel_human_two_picker import PixelHumanTwoPicker
from .pick_and_fling.pixel_human import PixelHumanFling
from .human_pick_and_drag import HumanPickAndDrag
from .human_dual_pickers_pick_and_place import HumanDualPickersPickAndPlace
from .no_operation import NoOperation

from .utils import draw_text_top_right, apply_workspace_shade


class HumanMultiPrimitive(Agent):
    
    def __init__(self, config):
        
        super().__init__(config)
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            # "norm-pixel-pick-and-place",
            # "norm-pixel-pick-and-drag",
            "norm-pixel-pick-and-place",
            "no-operation"
        ]
        print(f'XXXXXXXXXXXXXXXXXX [human-multi-primitive] Available config: {self.config}')
        self.primitive_instances = [
            PixelHumanFling(config),
            # PixelHumanTwoPicker(config),
            # HumanPickAndDrag(config),
            HumanDualPickersPickAndPlace(config),
            NoOperation(config)
        ]
    
    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        self.last_primitive = None
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
        

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
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

        # Show window
        obs_dir = "tmp/human_rgb.png"
        cv2.imwrite(obs_dir, img)
        print(f'[human-multi-primitive] Check {obs_dir} for current and goal observation.')

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
        self.last_primitive = chosen_primitive

        return {
            chosen_primitive: action
        }
    
    def terminate(self):
        return {arena_id: (self.last_primitive == 'no-operation')
                for arena_id in self.internal_states.keys()}
