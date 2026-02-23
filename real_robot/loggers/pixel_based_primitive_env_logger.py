import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from real_robot.utils.draw_utils import *
from real_robot.robot.video_logger import VideoLogger

class PixelBasedPrimitiveEnvLogger(VideoLogger):

    def __call__(self, episode_config, result, filename=None):
        super().__call__(episode_config, result, filename=filename)
        eid = episode_config['eid']
        frames = [info["observation"]["rgb"] for info in result["information"]]
        robot0_masks = None
        robot1_masks = None
        if 'robot0_mask' in result["information"][0]["observation"]:
            robot0_masks = [info["observation"]["robot0_mask"] for info in result["information"]]
        if 'robot1_mask' in result["information"][0]["observation"]:
            robot1_masks = [info["observation"]["robot1_mask"] for info in result["information"]]
        
        actions = result["actions"]
        H, W = 512, 512

        if filename is None:
            filename = 'manupilation'
        out_dir = os.path.join(self.log_dir, filename, 'performance_visualisation')
        os.makedirs(out_dir, exist_ok=True)

        # --- Helper: Project 3D points to 2D image coordinates ---
        def project_trajectory(points_3d, T_base_cam, intr, arena_meta):
            """
            Projects 3D points (base frame) -> 2D pixels (observation frame).
            Accounts for Camera Extrinsics -> Intrinsics -> Crop -> Resize.
            """
            if not points_3d: return []
            
            # --- FIX: Handle RealSense Intrinsics Object vs Numpy Matrix ---
            if hasattr(intr, 'fx'): # It is a pyrealsense2.intrinsics object
                fx, fy = intr.fx, intr.fy
                cx, cy = intr.ppx, intr.ppy
            else: # It is a numpy matrix
                fx, fy = intr[0,0], intr[1,1]
                cx, cy = intr[0,2], intr[1,2]
            # -------------------------------------------------------------

            # 1. Base -> Camera Frame
            T_cam_base = np.linalg.inv(T_base_cam)
            points_2d = []
            
            # Arena Crop/Resize parameters
            # Note: We must access these from the arena instance stored in info
            crop_x1 = arena_meta.x1
            crop_y1 = arena_meta.y1
            crop_size = arena_meta.crop_size
            target_res = 512.0 # The resolution of the image we are drawing on
            
            scale = target_res / crop_size

            for p in points_3d:
                p_h = np.append(p, 1.0) # Homogeneous
                p_cam = T_cam_base @ p_h
                
                z = p_cam[2]
                if z <= 0: continue # Skip points behind camera

                # 2. Camera -> Raw Pixel
                u_raw = (p_cam[0] * fx / z) + cx
                v_raw = (p_cam[1] * fy / z) + cy

                # 3. Raw Pixel -> Cropped & Resized Pixel
                u_final = (u_raw - crop_x1) * scale
                v_final = (v_raw - crop_y1) * scale
                
                points_2d.append((int(u_final), int(v_final)))
            
            return points_2d

        images = []
        for i, act in enumerate(actions):
            key = list(act.keys())[0]
            val = act[key]

            img = frames[i].copy()
            # --- Resize to 512×512 BEFORE drawing text ---
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            if robot0_masks is not None:
                mask0 = cv2.resize(
                    robot0_masks[i].astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                img = apply_workspace_shade(
                    img,
                    mask0,
                    color=(0, 0, 255),  # Blue in BGR
                    alpha=0.2
                )
                
            if robot1_masks is not None:
                mask1 = cv2.resize(
                    robot1_masks[i].astype(np.uint8),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                img = apply_workspace_shade(
                    img,
                    mask1,
                    color=(255, 0, 0),  # Red in BGR
                    alpha=0.2
                )
            

            primitive_color = PRIMITIVE_COLORS.get(key, PRIMITIVE_COLORS["default"])

            step_text = f"Step {i+1}: "
            if key == "norm-pixel-pick-and-place":
                step_text += "Pick and Place"
            elif key == "norm-pixel-pick-and-fling":
                step_text += "Pick and Fling"
            elif key == "no-operation":
                step_text += "No Operation"
            else:
                step_text += key

            draw_text_with_bg(
                img,
                step_text,
                (10, TEXT_Y_STEP),
                primitive_color
            )

            # ================================
            #          FOLD / PICK-PLACE
            # ================================
            if key == "norm-pixel-pick-and-place":
                # Ensure we have info for the next step to get 'applied_action'
                if i + 1 < len(result["information"]):
                    info_next = result["information"][i+1]
                    if 'applied_action' in info_next:
                        applied_action = info_next['applied_action']
                        applied_action = applied_action.get('norm-pixel-pick-and-place')
                        if applied_action is not None:
                            img = draw_pick_and_place(img, applied_action)

            # ================================
            #        PICK & FLING
            # ================================
            elif key == "norm-pixel-pick-and-fling":
                # 1. Prepare Trajectories
                traj_px_0, traj_px_1 = None, None
                
                # Check next step for trajectory data
                if i + 1 < len(result["information"]):
                    info_next = result["information"][i+1]
                    traj_data = info_next.get("debug_trajectory")
                    arena_instance = info_next.get("arena")

                    if traj_data and arena_instance:
                        scene = arena_instance.dual_arm
                        # Project points
                        traj_px_0 = project_trajectory(traj_data['ur5e'], scene.T_ur5e_cam, scene.intr, arena_instance)
                        traj_px_1 = project_trajectory(traj_data['ur16e'], scene.T_ur16e_cam, scene.intr, arena_instance)

                # 2. Call the clean draw function
                # Note: val contains the action parameters for this step
                img = draw_pick_and_fling(img, val, traj_0=traj_px_0, traj_1=traj_px_1)

            images.append(img)

        # Add final frame
        if len(frames) > len(images):
             images.append(frames[-1])

        # ===========================================
        #  CONCAT images in a matplotlib grid
        # ===========================================

        MAX_COLS = 6
        num_images = len(images)

        # Compute number of rows
        num_rows = (num_images + MAX_COLS - 1) // MAX_COLS
        if num_rows == 0: num_rows = 1

        # Create the figure
        fig, axes = plt.subplots(
            num_rows, MAX_COLS,
            figsize=(3 * MAX_COLS, 3 * num_rows),  # adjust tile size here
        )

        # If only one row/col, axes may not be 2D — fix it:
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if MAX_COLS == 1:
            axes = np.expand_dims(axes, axis=1)

        # Fill the grid
        idx = 0
        for r in range(num_rows):
            for c in range(MAX_COLS):
                ax = axes[r][c]

                if idx < num_images:
                    # RGB images in OpenCV are BGR — convert before plotting
                    img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.axis("off")
                    idx += 1
                else:
                    # Hide empty cells
                    ax.axis("off")

        # Tight layout avoids big gaps
        plt.tight_layout()

        # Save final image
        save_path = os.path.join(out_dir, f"episode_{eid}_trajectory.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)