# This file is adapted from the running script of clothmate repository: https://github.com/chongchongjjj/clothmate
import os
import numpy as np
import torch
from actoris_harena import TrainableAgent

from .utils.utils import *
from .network import MaximumValuePolicy  
from .keypoint.model import UNet

class ClothMateAdapter(TrainableAgent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'cloth-mate'
        self.debug = config.get('debug', False)
        self.action_primitives = config.get("action_primitives", ['fling', 'place', 'pick-and-stretch'])
        
        # Set up compute device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # --- CRITICAL COMMENT: Spatial Action Maps ---
        # The Fling Policy Module evaluates actions by transforming the observation stack 
        # across K discrete rotations and scales.
        # The number of transformations defines the channel depth of the network's output.
        self.num_rotations = config.get("num_rotations", 24)
        assert self.num_rotations % 4 == 0
        self.rotations = np.linspace(-180, 180, self.num_rotations + 1)

        self.scale_factors = np.array(config.get("scale_factors", [1.0]))
        self.use_adaptive_scaling = config.get("use_adaptive_scaling", False)
        self.adaptive_scale_factors = self.scale_factors.copy()

        # Primitive-specific valid rotation ranges:
        # e.g., 'fling' only requires frontal grasps (-90 to 90 degrees), whereas 
        # 'pick-and-stretch' might require full 360-degree reachability.
        self.rotation_indices = {
            'fling': np.where(np.logical_and(self.rotations >= -90, self.rotations <= 90))[0],
            'place': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 167.5))[0],
            'pick-and-stretch': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 180))[0]
        }

        self.primitive_vmap_indices = {}
        for primitive, indices in self.rotation_indices.items():
            self.primitive_vmap_indices[primitive] = [None, None]
            if len(indices) > 0:
                self.primitive_vmap_indices[primitive][0] = indices[0] * len(self.scale_factors)
                self.primitive_vmap_indices[primitive][1] = (indices[-1] + 1) * len(self.scale_factors)

        # --- Model Initializations ---
        
        # 1. Keypoint Detector (U-Net)
        self.keypoint_detector = UNet(in_ch=3, out_ch=4).to(self.device)

        # 2. Policy Network (MaximumValuePolicy)
        input_channel_types = config.get('input_channel_types', 'rgb_pos')
        
        # Initialize the network with the factorized configuration required by ClothMate
        self.policy = MaximumValuePolicy(
            action_primitives=self.action_primitives,
            num_rotations=self.num_rotations,
            scale_factors=self.scale_factors.tolist(),
            obs_dim=config.get('obs_dim', 128),
            pix_grasp_dist=config.get('pix_grasp_dist', 16),
            pix_drag_dist=config.get('pix_drag_dist', 16),
            pix_place_dist=config.get('pix_place_dist', 10),
            deformable_weight=config.get('deformable_weight', 0.65),
            action_expl_prob=config.get('action_expl_prob', 0.0),
            action_expl_decay=config.get('action_expl_decay', 0.0),
            value_expl_prob=config.get('value_expl_prob', 0.0),
            value_expl_decay=config.get('value_expl_decay', 0.0),
            input_channel_types=input_channel_types,
            device=self.device,
            gpu=0,
            deformable_pos=config.get('deformable_pos', True),
            unfactorized_networks=config.get('unfactorized_networks', False),
            coverage_reward=config.get('coverage_reward', False),
            # bilinear=config.get('bilinear', True)
        ).to(self.device)

    def load_best(self, path = None) -> int:
        # Construct the path to load from
        load_path = path if path is not None else self.save_dir
        load_path = os.path.join(load_path, 'checkpoints', 'model_best.pth')
        
        # --- FIX: Use 'load_path' here instead of 'path' ---
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        
        if self.policy is not None:
            # We don't need to strip the 'value_net.' prefix anymore because MaximumValuePolicy expects it!
            state_dict = checkpoint.get('net', checkpoint)
            self.policy.load_state_dict(state_dict, strict=False)
            self.policy.eval()
            
        if self.keypoint_detector is not None:
            #kps_model_path = self.config.get('kps_model', 'models/keypoint_model.pth')
            # Assuming you also want to load kps from the constructed directory:
            kps_model_path = os.path.join(path if path is not None else self.save_dir, 'checkpoints', 'keypoint_model.pth')
            
            if os.path.exists(kps_model_path):
                kps_ckpt = torch.load(kps_model_path, map_location="cpu", weights_only=True)
                self.keypoint_detector.load_state_dict(kps_ckpt.get('model_state_dict', kps_ckpt))
                self.keypoint_detector.eval()

        return True
    
    def rotate_point_back(self, x_rot, y_rot, orig_W, orig_H):
        x_orig = orig_W - 1 - y_rot
        y_orig = x_rot
        return x_orig, y_orig

    def normalize_pixel(self, x, y, W, H):
        norm_x = (x / W) * 2.0 - 1.0
        norm_y = (y / H) * 2.0 - 1.0
        return norm_x, norm_y

    def single_act(self, info, update=False):
        obs = info['observation']
        rgb = obs['rgb']
        orig_H, orig_W = rgb.shape[:2]
        
        # 1. Rotate 90 deg CCW
        prerot_rgb = rgb.copy()
        in_obs = {}
        for key in ['rgb', 'depth', 'mask', 'robot0_mask', 'robot1_mask']:
            assert key in obs, f"Key {key} not found in observation"
            in_obs[key] = np.rot90(obs[key].copy(), 1).copy()

        self.transformed_obs = self.generate_transformed_obs(in_obs) 
        self.transformed_obs['prerot_rgb'] = prerot_rgb

        # 2. Forward Pass
        action_primitive = None
        action_params = {}
        
        with torch.no_grad():
            # Pass the 128x128 feature map directly to the inner value_net to get raw maps
            obs_tensor = self.transformed_obs['transformed_obs'].to(self.device)
            value_maps = self.policy.value_net(obs_tensor, use_random_value=False)
            
            best_primitive, vmap_params = self.get_max_value_valid_action(value_maps)
            
            # --- CRITICAL COMMENT: The State Classifier ---
            # ClothMate utilizes a binary state classifier to determine if the cloth is 
            # sufficiently flattened to trigger the Pick-and-Stretch (P&S) module.
            # The threshold for this transition is empirically set to an RCA score < 0.15.
            if vmap_params.get('max_value', 1.0) < 0.15 and self.keypoint_detector is not None:
                from .keypoint.utils import get_keypoints 
                
                input_tensor = self.transformed_obs['transformed_obs'][40:41][:, :3].to(self.device)
                pred_heatmap, pred_cls = self.keypoint_detector(input_tensor)
                
                # Secondary validation: The U-Net classifier branch must also confidently 
                # recognize the state before executing P&S.
                # Secondary validation: The U-Net classifier branch must also confidently 
                # recognize the state before executing P&S.
                if pred_cls[0].item() >= 0.45:
                    
                    pred_coords = get_keypoints(pred_heatmap.cpu().numpy(), threshold=0.2)
                    
                    if np.all(pred_coords != -1):
                        action_primitive = 'pick-and-stretch'
                        
                        num_scales = len(self.adaptive_scale_factors)
                        rotation_idx = 40 // num_scales
                        scale_idx = 40 % num_scales
                        scale = self.adaptive_scale_factors[scale_idx]
                        rotation = self.rotations[rotation_idx]
                        
                        orig_dim = self.transformed_obs['original_dim']
                        mat = get_transform_matrix(
                            original_dim=orig_dim, 
                            resized_dim=128, 
                            rotation=-rotation, 
                            scale=scale
                        )
                        
                        mat_xy = mat.copy()
                        mat_xy[[0, 1], :] = mat_xy[[1, 0], :] 
                        mat_xy[:, [0, 1]] = mat_xy[:, [1, 0]] 
                        
                        coords_homo = np.concatenate((pred_coords, np.ones((4, 1))), axis=1)
                        orig_kps = np.matmul(coords_homo, mat_xy)[:, :2].astype(int)
                        
                        tr, tl, br, bl = orig_kps[0], orig_kps[1], orig_kps[2], orig_kps[3]
                        
                        top_width = np.linalg.norm(tl - tr)
                        bottom_width = np.linalg.norm(bl - br)
                        
                        mask_np = (input_tensor[0].sum(dim=0) > 0).cpu().numpy()
                        
                        # --- 1. Pants Fallback Classification ---
                        try:
                            from .keypoint.utils import is_line_all_ones
                            is_pants = not is_line_all_ones(mask_np, pred_coords[2], pred_coords[3])
                        except ImportError:
                            is_pants = False

                        if not is_pants:
                            bot_px_width = np.linalg.norm(pred_coords[3] - pred_coords[2])
                            top_px_width = np.linalg.norm(pred_coords[1] - pred_coords[0])
                            # Geometrically sound check for standard 2D space
                            if bot_px_width > top_px_width * 1.2: 
                                is_pants = True

                        # --- 2. Calculate Misalignment (Deviation) ---
                        # In 2D space, we check the deviation along the vertical Y-axis (index 1)
                        # to see which side is more "crooked" and needs flattening.
                        top_y_avg = (tl[1] + tr[1]) / 2.0
                        bottom_y_avg = (bl[1] + br[1]) / 2.0

                        top_deviation = abs(tl[1] - top_y_avg) + abs(tr[1] - top_y_avg)
                        bottom_deviation = abs(bl[1] - bottom_y_avg) + abs(br[1] - bottom_y_avg)

                        # --- 3. Pick and Stretch Logic ---
                        # Grasp the pair with the highest deviation from the bounding box
                        if top_deviation > bottom_deviation:
                            # Top is more wrinkled/deviated -> Grasp top
                            pick_L, pick_R = tl, tr
                            anchor_L, anchor_R = bl, br
                            target_width = top_width * 1.15
                        else:
                            # Bottom is more wrinkled/deviated -> Grasp bottom
                            pick_L, pick_R = bl, br
                            anchor_L, anchor_R = tl, tr
                            
                            if is_pants:
                                # Replicating original code's pants constraint: max_grasp_dist based on top keypoints
                                target_width = top_width * 1.15 
                            else:
                                target_width = bottom_width * 1.15
                                                    
                        midpoint = (pick_L + pick_R) / 2
                        anchor_mid = (anchor_L + anchor_R) / 2
                        
                        stretch_dir = (pick_L - pick_R) / (np.linalg.norm(pick_L - pick_R) + 1e-5)
                        tension_dir = (midpoint - anchor_mid) / (np.linalg.norm(midpoint - anchor_mid) + 1e-5)
                        tension_magnitude = orig_dim * 0.1  
                        
                        place_L = midpoint + stretch_dir * (target_width / 2) + tension_dir * tension_magnitude
                        place_R = midpoint - stretch_dir * (target_width / 2) + tension_dir * tension_magnitude
                        
                        action_params = {
                            'p1': pick_L.astype(int),
                            'p2': pick_R.astype(int),
                            'p3': place_L.astype(int),
                            'p4': place_R.astype(int)
                        }
                        
     
                        if self.debug:
                            import cv2
                            import os
                            os.makedirs('tmp/clothmate_debug/', exist_ok=True)
                            
                            # --- 1. Transformed Space (128x128 U-Net Input) ---
                            kp_img_ts = self.transformed_obs['transformed_obs'][40][:3].detach().cpu().numpy()
                            kp_img_ts = (kp_img_ts.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
                            kp_img_ts = cv2.cvtColor(kp_img_ts, cv2.COLOR_RGB2BGR)
                            
                            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # R, G, B, Y
                            for i, coord in enumerate(pred_coords):
                                cv2.circle(kp_img_ts, (int(coord[0]), int(coord[1])), 2, colors[i], -1)
                            cv2.imwrite('tmp/clothmate_debug/4_kp_transformed_space.png', kp_img_ts)
                            
                            # --- 2. Pretransform Space (480x480) ---
                            pre_img = self.transformed_obs['pretransform_rgb'].copy()
                            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)
                            
                            # Draw Original Keypoints (tl, tr, bl, br)
                            cv2.circle(pre_img, (int(tr[0]), int(tr[1])), 4, (0, 0, 255), -1)   # tr: Red
                            cv2.circle(pre_img, (int(tl[0]), int(tl[1])), 4, (0, 255, 0), -1)   # tl: Green
                            cv2.circle(pre_img, (int(br[0]), int(br[1])), 4, (255, 0, 0), -1)   # br: Blue
                            cv2.circle(pre_img, (int(bl[0]), int(bl[1])), 4, (0, 255, 255), -1) # bl: Yellow
                            
                            # Draw Action Points
                            pick_color = (255, 0, 255)  # Magenta (Picks)
                            place_color = (255, 255, 0) # Cyan (Anchors/Places)
                            
                            cv2.circle(pre_img, (int(pick_L[0]), int(pick_L[1])), 6, pick_color, -1)
                            cv2.circle(pre_img, (int(pick_R[0]), int(pick_R[1])), 6, pick_color, -1)
                            cv2.circle(pre_img, (int(place_L[0]), int(place_L[1])), 6, place_color, -1)
                            cv2.circle(pre_img, (int(place_R[0]), int(place_R[1])), 6, place_color, -1)
                            
                            # Draw Arrows for Stretch Direction
                            cv2.arrowedLine(pre_img, (int(pick_L[0]), int(pick_L[1])), (int(place_L[0]), int(place_L[1])), (255, 255, 255), 2)
                            cv2.arrowedLine(pre_img, (int(pick_R[0]), int(pick_R[1])), (int(place_R[0]), int(place_R[1])), (255, 255, 255), 2)
                            cv2.imwrite('tmp/clothmate_debug/5_kp_pretransform_space.png', pre_img)
                            
                            # --- 3. Absolute Original Space (Raw Camera) ---
                            orig_img = self.transformed_obs['prerot_rgb'].copy()
                            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
                            
                            # Helper to map points backwards
                            def map_back(pt):
                                x, y = self.rotate_point_back(pt[0], pt[1], orig_W, orig_H)
                                return (int(x), int(y))
                                
                            # Draw Mapped Keypoints
                            cv2.circle(orig_img, map_back(tr), 4, (0, 0, 255), -1)
                            cv2.circle(orig_img, map_back(tl), 4, (0, 255, 0), -1)
                            cv2.circle(orig_img, map_back(br), 4, (255, 0, 0), -1)
                            cv2.circle(orig_img, map_back(bl), 4, (0, 255, 255), -1)
                            
                            # Draw Mapped Actions & Arrows
                            cv2.circle(orig_img, map_back(pick_L), 6, pick_color, -1)
                            cv2.circle(orig_img, map_back(pick_R), 6, pick_color, -1)
                            cv2.circle(orig_img, map_back(place_L), 6, place_color, -1)
                            cv2.circle(orig_img, map_back(place_R), 6, place_color, -1)
                            
                            cv2.arrowedLine(orig_img, map_back(pick_L), map_back(place_L), (255, 255, 255), 2)
                            cv2.arrowedLine(orig_img, map_back(pick_R), map_back(place_R), (255, 255, 255), 2)
                            cv2.imwrite('tmp/clothmate_debug/6_kp_original_space.png', orig_img)
                        
                    else:
                        action_primitive = best_primitive
                        action_params = vmap_params
                else:
                    action_primitive = best_primitive
                    action_params = vmap_params
            else:
                action_primitive = best_primitive
                action_params = vmap_params
            
        # 3. Rotate back and Normalize pixel actions
        pts_rotated = []
        if 'p1' in action_params and 'p2' in action_params:
            pts_rotated.extend([action_params['p1'], action_params['p2']])
        if action_primitive in ['pick-and-stretch', 'stretch'] and 'p3' in action_params:
            pts_rotated.extend([action_params['p3'], action_params['p4']])
            
        norm_pts = []
        for p in pts_rotated:
            x_orig, y_orig = self.rotate_point_back(p[0], p[1], orig_W, orig_H)
            norm_x, norm_y = self.normalize_pixel(x_orig, y_orig, orig_W, orig_H)
            norm_pts.extend([norm_y, norm_x])
            
        norm_pts_array = np.array(norm_pts, dtype=np.float32)
        
        # 4. Map the ClothMate primitives to our environment's actions
        final_action = {}
        if action_primitive == 'fling':
            final_action['norm-pixel-pick-and-fling'] = norm_pts_array
        elif action_primitive == 'place':
            final_action['norm-pixel-single-pick-and-place'] = norm_pts_array
        elif action_primitive in ['pick-and-stretch', 'stretch']:
            final_action['norm-pixel-dual-pick-and-place'] = norm_pts_array
        else:
            final_action['no-operation'] = []
            
        return final_action

    def get_max_value_valid_action(self, value_maps) -> tuple:
        best_val = -float('inf')
        best_primitive = None
        best_max_indices = None
        
        deformable_weight = self.config.get('deformable_weight', 0.65)
        
        # --- 1. Find Highest Value Action ---
        for primitive in self.action_primitives:
            if primitive not in value_maps.get('rigid', {}): 
                continue
                
            vmap = (1 - deformable_weight) * value_maps['rigid'][primitive].squeeze(1) + \
                   deformable_weight * value_maps['deformable'][primitive].squeeze(1)
            
            mask = self.transformed_obs[f'{primitive}_mask'].to(self.device)
            
            if mask.any():
                vmap[~mask] = -float('inf')
                
                max_flat_index = torch.argmax(vmap).item()
                max_index = np.unravel_index(max_flat_index, vmap.shape) 
                max_value = vmap[max_index].item()
                
                if max_value > best_val:
                    best_val = max_value
                    best_primitive = primitive
                    best_max_indices = max_index

        if best_primitive is None:
            print("[ClothMateAdapter] No valid action found in masks. Skipping.")
            return 'no-operation', {}
            
        x, y, z = best_max_indices
        
        pix_grasp_dist = self.config.get('pix_grasp_dist', 16)
        pix_place_dist = self.config.get('pix_place_dist', 10)
        
        # --- 2. Apply Grasp Offsets ---
        # Represent points as [col(X), row(Y)]
        p1 = np.array([z, y], dtype=float) 
        p2 = np.array([z, y], dtype=float)
        
        # relative to the rotated bounding box.
        if best_primitive in ['fling', 'stretchdrag', 'pick-and-stretch']:
            p1[1] += pix_grasp_dist
            p2[1] -= pix_grasp_dist
        elif best_primitive == 'place':
            p2[1] += pix_place_dist
        elif best_primitive == 'drag':
            p2[1] += self.config.get('pix_drag_dist', 16)
            
        # --- 3. Inverse Affine Mapping to Pretransform Space ---
        num_scales = len(self.adaptive_scale_factors)
        rotation_idx = x // num_scales
        scale_idx = x % num_scales
        
        scale = self.adaptive_scale_factors[scale_idx]
        rotation = self.rotations[rotation_idx]
        
        orig_dim = self.transformed_obs['original_dim'] 
        resized_dim = 128 
        
        # Get standard [Y, X] transformation matrix from your utils
        mat = get_transform_matrix(
            original_dim=orig_dim, 
            resized_dim=resized_dim, 
            rotation=-rotation, 
            scale=scale
        )
        
        # Convert matrix to [X, Y] format to map our pixel coordinates correctly
        mat_xy = mat.copy()
        mat_xy[[0, 1], :] = mat_xy[[1, 0], :] # Swap rows
        mat_xy[:, [0, 1]] = mat_xy[:, [1, 0]] # Swap cols
        
        pixels = np.array([p1, p2])
        pixels_homo = np.concatenate((pixels, np.ones((2, 1))), axis=1)
        orig_pixels = np.matmul(pixels_homo, mat_xy)[:, :2].astype(int) # Points in 480x480 Pretransform space
        

        if self.debug:  # Set to True or pass via config
            import cv2
            import os
            os.makedirs('tmp/clothmate_debug/', exist_ok=True)
            
            # --- Base Images ---
            chosen_img_ts = self.transformed_obs['transformed_obs'][x][:3].detach().cpu().numpy()
            chosen_img_ts = (chosen_img_ts.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
            chosen_img_ts = cv2.cvtColor(chosen_img_ts, cv2.COLOR_RGB2BGR) 
            
            pre_img = self.transformed_obs['pretransform_rgb'].copy()
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)
            pre_H, pre_W = pre_img.shape[:2]
            
            orig_img = self.transformed_obs['prerot_rgb'].copy()
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            orig_H, orig_W = orig_img.shape[:2] 
            
            # --- Masks ---
            ws_mask = self.transformed_obs[f'{best_primitive}_workspace_mask'][x].detach().cpu().numpy().astype(bool)
            left_arm_ts = self.transformed_obs['left_arm_mask'][x].detach().cpu().numpy().astype(bool)
            right_arm_ts = self.transformed_obs['right_arm_mask'][x].detach().cpu().numpy().astype(bool)
            
            left_arm_pre = self.transformed_obs['pretransform_left_arm_mask'].detach().cpu().numpy().astype(bool)
            right_arm_pre = self.transformed_obs['pretransform_right_arm_mask'].detach().cpu().numpy().astype(bool)
            
            M_pre = mat_xy.T[:2, :].astype(np.float32) 
            
            # --- Image 1: Transformed Space (128x128) ---
            overlay_ts = chosen_img_ts.copy()
            overlay_ts[left_arm_ts] = [0, 0, 255]     # Red: Left Arm
            overlay_ts[right_arm_ts] = [0, 255, 0]    # Green: Right Arm
            overlay_ts[ws_mask] = [255, 0, 255]       # Magenta: Workspace
            
            chosen_img_ts = cv2.addWeighted(overlay_ts, 0.35, chosen_img_ts, 0.65, 0)
            cv2.circle(chosen_img_ts, (int(p1[0]), int(p1[1])), 3, (255, 0, 0), -1) 
            cv2.circle(chosen_img_ts, (int(p2[0]), int(p2[1])), 3, (0, 255, 255), -1)
            cv2.imwrite('tmp/clothmate_debug/1_transformed_space.png', chosen_img_ts)

            # --- Image 2: Pretransform Space (480x480) ---
            ws_mask_pre = cv2.warpAffine(ws_mask.astype(np.uint8), M_pre, (pre_W, pre_H), flags=cv2.INTER_NEAREST).astype(bool)
            
            overlay_pre = pre_img.copy()
            overlay_pre[left_arm_pre] = [0, 0, 255]
            overlay_pre[right_arm_pre] = [0, 255, 0]
            overlay_pre[ws_mask_pre] = [255, 0, 255]
            
            pre_img = cv2.addWeighted(overlay_pre, 0.35, pre_img, 0.65, 0)
            cv2.circle(pre_img, (int(orig_pixels[0][0]), int(orig_pixels[0][1])), 5, (255, 0, 0), -1)
            cv2.circle(pre_img, (int(orig_pixels[1][0]), int(orig_pixels[1][1])), 5, (0, 255, 255), -1)
            cv2.imwrite('tmp/clothmate_debug/2_pretransform_space.png', pre_img)

            # --- Image 3: Absolute Original Space (Raw Vision) ---
            ws_mask_orig = np.rot90(ws_mask_pre, -1)
            left_arm_orig = np.rot90(left_arm_pre, -1)
            right_arm_orig = np.rot90(right_arm_pre, -1)
            
            overlay_orig = orig_img.copy()
            overlay_orig[left_arm_orig] = [0, 0, 255]
            overlay_orig[right_arm_orig] = [0, 255, 0]
            overlay_orig[ws_mask_orig] = [255, 0, 255]
            
            orig_img = cv2.addWeighted(overlay_orig, 0.35, orig_img, 0.65, 0)
            
            # Use your function to map points to the final image for visual confirmation
            p1_final_x, p1_final_y = self.rotate_point_back(orig_pixels[0][0], orig_pixels[0][1], orig_W, orig_H)
            p2_final_x, p2_final_y = self.rotate_point_back(orig_pixels[1][0], orig_pixels[1][1], orig_W, orig_H)
            
            cv2.circle(orig_img, (int(p1_final_x), int(p1_final_y)), 6, (255, 0, 0), -1)
            cv2.circle(orig_img, (int(p2_final_x), int(p2_final_y)), 6, (0, 255, 255), -1)
            cv2.imwrite('tmp/clothmate_debug/3_original_space.png', orig_img)
       
        # Return the action params mapped to Pretransform Space. 
        # Your single_act function will handle translating these to Original Space 
        # and normalizing them to [-1, 1].
        action_params = {
            'p1': orig_pixels[0],
            'p2': orig_pixels[1],
            'max_value': best_val 
        }
        
        return best_primitive, action_params


    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir

    def generate_transformed_obs(self, obs_dict, input_dim=128, scale_factors=None, rotations=None):
        from .utils.env_utils import prepare_image, generate_workspace_mask, generate_primitive_cloth_mask
        from itertools import product
        
        if scale_factors is None: scale_factors = self.adaptive_scale_factors
        if rotations is None: rotations = self.rotations
        
        rgb = obs_dict['rgb']
        depth = obs_dict.get('depth', np.zeros(rgb.shape[:2]))
        
        obs_tensor = torch.cat((
            torch.tensor(rgb).float() / 255.0,
            torch.tensor(depth).float()
        ), dim=2).permute(2, 0, 1)
        
        transformations = list(product(rotations, scale_factors))
        
        retval = {}
        retval['original_dim'] = rgb.shape[0] 
        retval['pretransform_rgb'] = rgb.copy()

        device = self.device

        retval['transformed_obs'] = prepare_image(
            obs_tensor, 
            transformations, 
            input_dim,
            parallelize=False,
            nocs_mode='collapsed',
            inter_dim=256,
            constant_positional_enc=True
        ).to(device)
        
        cloth_mask_np = obs_dict.get('mask', rgb.sum(axis=-1) > 0)
        pretransform_cloth_mask = torch.tensor(cloth_mask_np).float()
        pretransform_left_arm_mask =  torch.tensor(obs_dict['robot1_mask'].copy()).float()
        pretransform_right_arm_mask =  torch.tensor(obs_dict['robot0_mask'].copy()).float()
            

        pretransform_mask = torch.stack([
            pretransform_cloth_mask, 
            pretransform_left_arm_mask, 
            pretransform_right_arm_mask
        ], dim=0)
        
        transformed_mask = prepare_image(
            pretransform_mask, 
            transformations, 
            input_dim,
            parallelize=False,
            nocs_mode='collapsed',
            inter_dim=256,
            constant_positional_enc=True
        ).to(device)
        
        cloth_mask = transformed_mask[:, 0]
        left_arm_mask = transformed_mask[:, 1]
        right_arm_mask = transformed_mask[:, 2]
        
        retval['left_arm_mask'] = left_arm_mask
        retval['right_arm_mask'] = right_arm_mask
        retval['pretransform_left_arm_mask'] = pretransform_left_arm_mask
        retval['pretransform_right_arm_mask'] = pretransform_right_arm_mask

        pix_place_dist = self.config.get("pix_place_dist", 10)
        pix_grasp_dist = self.config.get("pix_grasp_dist", 16)
        
        workspace_mask = generate_workspace_mask(
            left_arm_mask, right_arm_mask, self.action_primitives, pix_place_dist, pix_grasp_dist
        )
        
        cloth_mask_dict = generate_primitive_cloth_mask(
            cloth_mask, self.action_primitives, pix_place_dist, pix_grasp_dist
        )
        
        for primitive in self.action_primitives:
            offset = pix_grasp_dist if primitive in ['fling', 'pick-and-stretch', 'stretchdrag'] else pix_place_dist + 6
            
            valid_transforms_mask = torch.zeros_like(cloth_mask_dict[primitive]).bool()
            prim_vmap_indices = self.primitive_vmap_indices.get(primitive, [0, len(transformations)])
            
            valid_transforms_mask[prim_vmap_indices[0]:prim_vmap_indices[1], offset:-offset, offset:-offset] = True
            
            table_mask = retval['transformed_obs'][:, 3] > 0
            offset_table_mask_up = torch.zeros_like(table_mask).bool()
            offset_table_mask_down = torch.zeros_like(table_mask).bool()
            offset_table_mask_up[:, :-offset, :] = table_mask[:, offset:]
            offset_table_mask_down[:, offset:, :] = table_mask[:, :-offset]
            table_mask = offset_table_mask_up & offset_table_mask_down & table_mask
            
            primitive_workspace_mask = torch.logical_and(workspace_mask.get(primitive, left_arm_mask), table_mask)
            primitive_workspace_mask = torch.logical_and(primitive_workspace_mask, valid_transforms_mask)
            
            retval[f"{primitive}_cloth_mask"] = cloth_mask_dict[primitive]
            retval[f"{primitive}_workspace_mask"] = primitive_workspace_mask
            retval[f"{primitive}_mask"] = torch.logical_and(cloth_mask_dict[primitive], primitive_workspace_mask)
            
        return retval

    def save(self):
        pass

    def set_eval(self):
        self.policy.eval()
        self.keypoint_detector.eval()
    
    def set_train(self):
        self.policy.train()
        self.keypoint_detector.train()