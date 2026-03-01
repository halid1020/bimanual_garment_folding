import numpy as np
import cv2
from statistics import mean

from .utils import *
from real_robot.utils.mask_utils import calculate_iou
from .garment_flattening_task import RealWorldGarmentFlatteningTask

class RealWorldGarmentCanonicalisationAlignmentTask(RealWorldGarmentFlatteningTask):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'canonicalisation-alignment'
        self.randomise_goal = config.get('randomise_goal', False)
    
    # def reset(self, arena):
    #     self.info = super().reset(arena)

    #     flattened_goal = self.info['goals'][0][0]

    #     # TODO: create a UI interface, 
    #     # where the user can drag the goal cloth mask on the canvas.
    #     # The canvase is composed of the workspace mask of the two arms.
    #     # after user fix the goal position, self.goals are updated.

       
    #     return self.info

    def reset(self, arena):
        self.info = super().reset(arena)

        # self.goals[0][0] holds a reference to arena.flattened_obs
        flattened_goal = arena.flattened_obs
        obs = flattened_goal['observation']
        
        # --- 1. Extract masks to build the UI canvas ---
        cloth_mask = obs['mask'].astype(np.uint8) * 255
        h, w = cloth_mask.shape
        
        # FIX: Convert boolean masks to 0-255 scale so they are visible on the image
        r0_mask_vis = obs['robot0_mask'].astype(np.uint8) * 255
        r1_mask_vis = obs['robot1_mask'].astype(np.uint8) * 255
        
        # Create background canvas: Robot 0 in Red, Robot 1 in Blue
        canvas_base = np.zeros((h, w, 3), dtype=np.uint8)
        canvas_base[:, :, 2] = r0_mask_vis  # Red channel
        canvas_base[:, :, 0] = r1_mask_vis  # Blue channel

        # --- 2. Set up interactive variables ---
        dragging = False
        ix, iy = -1, -1
        offset_x, offset_y = 0, 0
        current_offset_x, current_offset_y = 0, 0
        current_angle = 0  # ADDED: Track rotation in degrees

        def drag_mouse_callback(event, x, y, flags, param):
            nonlocal dragging, ix, iy, offset_x, offset_y, current_offset_x, current_offset_y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging:
                    current_offset_x = offset_x + (x - ix)
                    current_offset_y = offset_y + (y - iy)
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                offset_x = current_offset_x
                offset_y = current_offset_y

        window_name = 'Drag Goal Mask (A/D to Rotate, ENTER to confirm)'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, drag_mouse_callback)

        print("\n[Task] Please drag the GREEN cloth mask to the desired goal location.")
        print("[Task] Use 'A' and 'D' keys to rotate.")
        print("[Task] Press ENTER to confirm.\n")

        # Define center of rotation (center of the image)
        center = (w // 2, h // 2)

        # --- 3. Render Loop ---
        while True:
            # Create transformation matrix: Rotate first, then Translate
            M = cv2.getRotationMatrix2D(center, current_angle, 1.0)
            M[0, 2] += current_offset_x
            M[1, 2] += current_offset_y
            
            shifted_mask = cv2.warpAffine(cloth_mask, M, (w, h), flags=cv2.INTER_NEAREST)
            
            # Overlay cloth mask in Green channel
            display = canvas_base.copy()
            display[:, :, 1] = np.maximum(display[:, :, 1], shifted_mask)
            
            # Add on-screen instructions
            cv2.putText(display, "Drag mouse to move.", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "'A' / 'D' to rotate.", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "ENTER to confirm.", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, display)
            
            # Break loop on Enter (13) or ESC (27)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 27: 
                break
            elif key == ord('a') or key == ord('A'):
                current_angle += 5  # Rotate counter-clockwise 5 degrees
            elif key == ord('d') or key == ord('D'):
                current_angle -= 5  # Rotate clockwise 5 degrees
                
        cv2.destroyAllWindows()

        # --- 4. Apply the final transformation ---
        if offset_x != 0 or offset_y != 0 or current_angle != 0:
            M_final = cv2.getRotationMatrix2D(center, current_angle, 1.0)
            M_final[0, 2] += offset_x
            M_final[1, 2] += offset_y
            
            # Warp the boolean mask using INTER_NEAREST to keep sharp 0/1 edges
            new_mask = cv2.warpAffine(obs['mask'].astype(np.uint8), M_final, (w, h), flags=cv2.INTER_NEAREST)
            obs['mask'] = new_mask > 0
            
            # Warp RGB and Depth
            if 'rgb' in obs:
                # INTER_LINEAR is fine for RGB
                obs['rgb'] = cv2.warpAffine(obs['rgb'], M_final, (w, h), flags=cv2.INTER_LINEAR)
            if 'depth' in obs:
                # INTER_NEAREST is strictly needed for depth to prevent floating point interpolation artifacts
                obs['depth'] = cv2.warpAffine(obs['depth'], M_final, (w, h), flags=cv2.INTER_NEAREST)
                
            print(f"[Task] Goal updated. Offset X:{offset_x}, Y:{offset_y}, Angle: {current_angle} deg")

        return self.info
    
    def success(self, arena):
        cur_eval = self.evaluate(arena)
        IoU = cur_eval['canon_IoU_to_flattened']
        coverage = cur_eval['normalised_coverage']
        return IoU > IOU_FLATTENING_TRESHOLD and coverage > NC_FLATTENING_TRESHOLD
    
    def compare(self, results_1, results_2):
        threshold=0.95

        # --- Compute averages for results_1 ---
        avg_nc_1 = mean([ep["normalised_coverage"][-1] for ep in results_1])
        avg_iou_1 = mean([ep["canon_IoU_to_flattened"][-1] for ep in results_1])
        avg_len_1 = mean([len(ep["canon_IoU_to_flattened"]) for ep in results_1])
        score_1 = avg_nc_1 + avg_iou_1

        # --- Compute averages for results_2 ---
        avg_nc_2 = mean([ep["normalised_coverage"][-1] for ep in results_2])
        avg_iou_2 = mean([ep["canon_IoU_to_flattened"][-1] for ep in results_2])
        avg_len_2 = mean([len(ep["canon_IoU_to_flattened"]) for ep in results_2])
        score_2 = avg_nc_2 + avg_iou_2

        # --- Both are very good → prefer shorter trajectory ---
        if score_1 > 2 * threshold and score_2 > 2 * threshold:
            if avg_len_1 < avg_len_2:
                return 1
            elif avg_len_1 > avg_len_2:
                return -1
            else:
                return 0

        # --- Otherwise prefer higher score ---
        if score_1 > score_2:
            return 1
        elif score_1 < score_2:
            return -1
        else:
            return 0