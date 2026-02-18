import time
import cv2
import numpy as np
from actoris_harena import Agent

class RealWorldSingleArmHumanPickAndPlacePolicy(Agent):
    """
    Interactive human-in-the-loop policy for Single-Arm Pick & Place.
    Returns: np.array([pick_x, pick_y, place_x, place_y]) in normalized [-1, 1] range.
    """

    def __init__(self, config):
        super().__init__(config)
        self.measure_time = config.get('measure_time', False)
        self.debug = config.get('debug', False)
        self.window_name = "Single Arm Human Policy"

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}
        for arena_id in arena_ids:
            self.internal_states[arena_id]['inference_time'] = []

    def single_act(self, info, update=False):
        """Get a single-arm action from user clicks."""
        
        # --- Start Timer ---
        if self.measure_time:
            start_time = time.time()
            arena_id = info['arena_id']


        # 2. Process Observation for Visualization
        obs = info['observation']
        rgb = obs["rgb"]
        
        # Handle mask: simpler logic for single arm
        # We only care about robot0_mask (the UR5e)
        workspace_mask = obs.get("robot0_mask", np.ones(rgb.shape[:2], dtype=bool))
        
        # Visualization: Gray out invalid areas, tint blue valid areas
        display_rgb = self.apply_workspace_mask(rgb, workspace_mask)
        
        h, w = rgb.shape[:2]

        # 3. Handle Primitives
        action_array = np.zeros(4, dtype=np.float32)

        # Get 2 clicks: Pick (Green), Place (Red)
        print(f"Please click: 1. Pick Point, 2. Place Point")
        clicks = self.get_user_clicks(display_rgb, num_points=2)
        
        if len(clicks) < 2: 
            print("Action cancelled.")
            # Return zeros (No-Op) if cancelled
            return np.zeros(4, dtype=np.float32)

        pick_pt, place_pt = clicks

        # Normalize to [-1, 1]
        def norm_xy(pt):
            x, y = pt
            return ((x / w) * 2 - 1, (y / h) * 2 - 1)

        pick_norm = norm_xy(pick_pt)
        place_norm = norm_xy(place_pt)

        # Return flattened array directly: [px, py, lx, ly]
        action_array = np.concatenate([pick_norm, place_norm]).astype(np.float32)

        # --- End Timer ---
        if self.measure_time:
            self.internal_states[arena_id]['inference_time'].append(time.time() - start_time)
        
        return action_array

    def apply_workspace_mask(self, rgb, mask):
        """
        Visualizes the valid workspace for the single arm.
        Valid area = Blue tint.
        Invalid area = Grayed out.
        """
        # Ensure boolean
        mask = mask.astype(bool)
        if mask.ndim == 3: mask = mask[:, :, 0]

        rgb_f = rgb.astype(np.float32) / 255.0
        
        # Colors
        valid_tint = np.array([0.2, 0.6, 1.0]) # Nice Blue
        gray_tint  = np.array([0.3, 0.3, 0.3]) # Dark Gray

        blend_valid = 0.2
        blend_invalid = 0.6

        shaded = rgb_f.copy()

        # Apply Gray to invalid
        shaded[~mask] = (
            rgb_f[~mask] * (1 - blend_invalid) + gray_tint * blend_invalid
        )

        # Apply Blue to valid
        shaded[mask] = (
            rgb_f[mask] * (1 - blend_valid) + valid_tint * blend_valid
        )

        return (np.clip(shaded, 0, 1) * 255).astype(np.uint8)

    def get_user_clicks(self, img, num_points=2):
        """
        Self-contained OpenCV click listener.
        Returns a list of (x, y) tuples.
        """
        points = []
        img_display = img.copy()
        
        # Colors for dots: Green (Pick), Red (Place)
        colors = [(0, 255, 0), (0, 0, 255)] 

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < num_points:
                    points.append((x, y))
                    # Draw circle
                    idx = len(points) - 1
                    color = colors[idx] if idx < len(colors) else (255, 255, 0)
                    cv2.circle(img_display, (x, y), 5, color, -1)
                    cv2.imshow(self.window_name, img_display)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        cv2.imshow(self.window_name, img_display)

        print(f"Waiting for {num_points} clicks...")
        
        while len(points) < num_points:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27: # q or ESC
                break
        
        # Brief pause to show the final dots
        cv2.waitKey(200)
        cv2.destroyWindow(self.window_name)
        
        return points