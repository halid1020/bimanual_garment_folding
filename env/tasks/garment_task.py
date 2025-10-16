import os
import json
import numpy as np
import cv2

from agent_arena import Task

from ..utils.garment_utils import KEYPOINT_SEMANTICS, rigid_align, deformable_align, \
    simple_rigid_align, chamfer_alignment_with_rotation
from ..utils.keypoint_gui import KeypointGUI

class GarmentTask(Task):
    def __init__(self, config):
        self.keypoint_semantics = KEYPOINT_SEMANTICS[config.garment_type]

        self.semkey2pid = None # This needs to be loaded or annotated
        self.asset_dir = config.asset_dir

        self.keypoint_assignment_gui = KeypointGUI(self.keypoint_semantics )
        self.keypoint_dir = os.path.join(self.asset_dir, 'keypoints')
        os.makedirs(self.keypoint_dir, exist_ok=True)

    def _load_or_create_keypoints(self, arena):
        """Load semantic keypoints if they exist, otherwise ask user to assign them."""
        #print('state keys', arena.init_state_params.keys())
        mesh_id = arena.init_state_params['pkl_path'].split('/')[-1].split('.')[0]  # e.g. 03346_Tshirt
        keypoint_file = os.path.join(self.keypoint_dir, f"{mesh_id}.json")
        print('mesh id', mesh_id)

        if os.path.exists(keypoint_file):
            with open(keypoint_file, "r") as f:
                keypoints = json.load(f)
            if self.config.debug:
                print("annotated keypoint ids", keypoints)
            return keypoints

        # Get flattened garment observation
        flatten_obs = arena.get_flattened_obs()
        flatten_rgb = flatten_obs['observation']["rgb"]
        particle_positions = flatten_obs['observation']["particle_positions"]  # (N, 3)

        # Ask user to click semantic keypoints
        keypoints_pixel = self.keypoint_assignment_gui.run(flatten_rgb)  # dict: {name: (u, v)}
        
        # Project all garment particles
        pixels, visible = arena.get_visibility(particle_positions)
        
        if self.config.debug:
            H, W = (480, 480)


            print('annotated keypoints', keypoints_pixel)

            # Make sure tmp folder exists
            os.makedirs("tmp", exist_ok=True)

            # Start with black canvases
            non_visible_img = np.zeros((H, W, 3), dtype=np.uint8)
            visible_img = np.zeros((H, W, 3), dtype=np.uint8)

            for pix, vis in zip(pixels, visible):
                x, y = pix  # assuming pix = (x, y)
                x = int(x)
                y = int(y)
                if not vis:
                    # non-visible -> gray pixel
                    non_visible_img[x, y] = (128, 128, 128)
                else:
                    # visible -> white pixel
                    visible_img[x, y] = (255, 255, 255)

            # Save both images
            cv2.imwrite("tmp/non-visible.png", non_visible_img)
            cv2.imwrite("tmp/visible.png", visible_img)


        keypoints = {}
        for name, pix in keypoints_pixel.items():
            y, x = pix
            dists = np.linalg.norm(pixels - np.array((x, y)), axis=1)
            particle_id = np.argmin(dists)
            keypoints[name] = int(particle_id)
        
        if self.config.debug:
            annotated = np.zeros((H, W, 3), dtype=np.uint8)
            for pid in keypoints.values():
                x, y = pixels[pid]
                x = int(x)
                y = int(y)
                annotated[x, y] = (255, 255, 255)
            cv2.imwrite("tmp/annotated.png", annotated)


        with open(keypoint_file, "w") as f:
            json.dump(keypoints, f, indent=2)
        return keypoints