import os
import json
import numpy as np
import cv2

from actoris_harena import Task

from ..utils.garment_utils import KEYPOINT_SEMANTICS, rigid_align, deformable_align, \
    simple_rigid_align, chamfer_alignment_with_rotation
from ..utils.keypoint_gui import KeypointGUI

class GarmentTask(Task):
    def __init__(self, config):
        if config.garment_type == 'all':
            ## we assum the keypoints are generateld
            pass
        else:
           
            self.keypoint_semantics = KEYPOINT_SEMANTICS[config.garment_type]
            self.keypoint_assignment_gui = KeypointGUI(self.keypoint_semantics)

        self.semkey2pid = None # This needs to be loaded or annotated
        self.asset_dir = config.asset_dir

        
        self.keypoint_dir = os.path.join(self.asset_dir, 'keypoints')
        os.makedirs(self.keypoint_dir, exist_ok=True)

    def _load_or_create_keypoints(self, arena):
        """Load semantic keypoints if they exist, otherwise ask user to assign them."""
        mesh_id = arena.init_state_params['pkl_path'].split('/')[-1].split('.')[0]  
        keypoint_file = os.path.join(self.keypoint_dir, f"{mesh_id}.json")
        print('[GarmentTask] mesh id', mesh_id)

        if os.path.exists(keypoint_file):
            with open(keypoint_file, "r") as f:
                keypoints = json.load(f)
            if self.config.debug:
                print("annotated keypoint ids", keypoints)
            return keypoints

        # Get flattened garment observation
        flatten_obs = arena.get_flattened_obs()
        
        # --- CRITICAL FIX 1: Use raw_rgb ---
        # raw_rgb is the uncropped camera view. This perfectly aligns with 
        # the coordinates returned by arena.get_visibility().
        flatten_rgb = flatten_obs['observation']["raw_rgb"]
        particle_positions = flatten_obs['observation']["particle_positions"]  # (N, 3)

        # Ask user to click semantic keypoints
        keypoints_pixel = self.keypoint_assignment_gui.run(flatten_rgb)  
        
        # Project all garment particles (pixels array format is [y, x])
        pixels, visible = arena.get_visibility(particle_positions)
        
        if self.config.debug:
            print('annotated keypoints', keypoints_pixel)
            os.makedirs("tmp", exist_ok=True)
            H, W = flatten_rgb.shape[:2]
            
            non_visible_img = np.zeros((H, W, 3), dtype=np.uint8)
            visible_img = np.zeros((H, W, 3), dtype=np.uint8)

            for pix, vis in zip(pixels, visible):
                # pixels is [v, u] which means [y, x]
                py, px = int(pix[0]), int(pix[1]) 
                
                # Ensure bounds
                if 0 <= py < H and 0 <= px < W:
                    if not vis:
                        non_visible_img[py, px] = (128, 128, 128)
                    else:
                        visible_img[py, px] = (255, 255, 255)

            cv2.imwrite("tmp/non-visible.png", non_visible_img)
            cv2.imwrite("tmp/visible.png", visible_img)

        keypoints = {}
        for name, pix in keypoints_pixel.items():
            # GUI returns [x, y] (column, row)
            click_x, click_y = pix
            
            # arena.get_visibility returns pixels as [v, u] (which is [y, x])
            target_pixel = np.array([click_y, click_x])
            
            # Calculate distance between click and all particles
            dists = np.linalg.norm(pixels - target_pixel, axis=1)
            
            
            particle_id = np.argmin(dists)
            keypoints[name] = int(particle_id)
        
        if self.config.debug:
            annotated = np.zeros((H, W, 3), dtype=np.uint8)
            for pid in keypoints.values():
                py, px = int(pixels[pid][0]), int(pixels[pid][1])
                if 0 <= py < H and 0 <= px < W:
                    annotated[py, px] = (255, 255, 255)
            cv2.imwrite("tmp/annotated.png", annotated)

        with open(keypoint_file, "w") as f:
            json.dump(keypoints, f, indent=2)
            
        return keypoints