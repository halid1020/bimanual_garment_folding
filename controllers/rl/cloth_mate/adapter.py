import os
import numpy as np
import torch
import cv2

# Relative imports to access core code
from .utils import *
from .network import MaximumValuePolicy  
from .keypoint.model import UNet
from actoris_harena import TrainableAgent

class ClothMateAdapter(TrainableAgent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'cloth-mate'
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
            bilinear=config.get('bilinear', True)
        ).to(self.device)

    def load_best(self, path=None):
        """
        Loads the provided checkpoint for the ClothMate policy and sets it to evaluation mode.
        """
        if path is None:
            path = self.config.get('checkpoint_path', 'models/latest_ckpt.pth')
            
        print(f"[ClothMateAdapter] Loading checkpoint from {path}")
        
        # Load weights
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        
        if self.policy is not None:
            # We don't need to strip the 'value_net.' prefix anymore because MaximumValuePolicy expects it!
            state_dict = checkpoint.get('net', checkpoint)
            self.policy.load_state_dict(state_dict, strict=False)
            self.policy.eval()
            
        if self.keypoint_detector is not None:
            kps_model_path = self.config.get('kps_model', 'models/keypoint_model.pth')
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
        rotated_rgb = np.rot90(rgb, k=1, axes=(0, 1))
        transformed_obs = obs.copy()
        transformed_obs['rgb'] = rotated_rgb
        if 'depth' in obs: transformed_obs['depth'] = np.rot90(obs['depth'], k=1, axes=(0, 1))
        if 'mask' in obs: transformed_obs['mask'] = np.rot90(obs['mask'], k=1, axes=(0, 1))

        self.transformed_obs = self.generate_transformed_obs(transformed_obs) 
        
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
                if pred_cls[0].item() >= 0.45:
                    pred_coords = get_keypoints(pred_heatmap.cpu().numpy(), threshold=0.2)[0]
                    
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
                        
                        coords_homo = np.concatenate((pred_coords, np.ones((4, 1))), axis=1)
                        orig_kps = np.matmul(coords_homo, mat)[:, :2].astype(int)
                        
                        tr, tl, br, bl = orig_kps[0], orig_kps[1], orig_kps[2], orig_kps[3]
                        
                        # --- CRITICAL COMMENT: Heuristic P&S Strategy ---
                        # The heuristic relies on comparing the horizontal deviation 
                        # (width) of the top and bottom keypoints.
                        top_width = np.linalg.norm(tl - tr)
                        bottom_width = np.linalg.norm(bl - br)
                        
                        # The algorithm selects the pair of keypoints with the largest 
                        # horizontal deviation (the more wrinkled side).
                        if top_width < bottom_width:
                            pick_L, pick_R = tl, tr
                            anchor_L, anchor_R = bl, br
                            # Safety constraint: Maximum stretching distance is limited 
                            # (e.g., 1.15x or 1.2x) to prevent over-stretching garments 
                            # lacking reliable bottom constraints, like trousers.
                            target_width = bottom_width * 1.15
                        else:
                            pick_L, pick_R = bl, br
                            anchor_L, anchor_R = tl, tr
                            target_width = top_width * 1.15
                            
                        midpoint = (pick_L + pick_R) / 2
                        anchor_mid = (anchor_L + anchor_R) / 2
                        
                        # Stretching is executed by repositioning keypoints to symmetric 
                        # target locations and applying vertical tension.
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
            norm_pts.extend([norm_x, norm_y])
            
        norm_pts_array = np.array(norm_pts, dtype=np.float32)
        
        # 4. Map the ClothMate primitives to our environment's actions
        final_action = {}
        if action_primitive == 'fling':
            final_action['norm-pixel-pick-and-fling'] = norm_pts_array
        elif action_primitive == 'place':
            final_action['norm-pixel-single-arm-pick-and-place'] = norm_pts_array
        elif action_primitive in ['pick-and-stretch', 'stretch']:
            final_action['norm-pixel-pick-and-place'] = norm_pts_array
        else:
            final_action['no-operation'] = np.zeros(0) 
            
        return final_action

    def get_max_value_valid_action(self, value_maps) -> tuple:
        best_val = -float('inf')
        best_primitive = None
        best_max_indices = None
        
        deformable_weight = self.config.get('deformable_weight', 0.65)
        
        # --- CRITICAL COMMENT: Rigid & Deformable Value Combination ---
        # The framework linearly combines rigid and deformable predictions to generate
        # the final value map.
        for primitive in self.action_primitives:
            if primitive not in value_maps.get('rigid', {}): 
                continue
                
            vmap = (1 - deformable_weight) * value_maps['rigid'][primitive].squeeze(1) + \
                   deformable_weight * value_maps['deformable'][primitive].squeeze(1)
            
            mask = self.transformed_obs[f'{primitive}_mask'].to(self.device)
            
            # Workspace and reachability constraints are applied as masks, zeroing out 
            # unreachable vertices (set to -inf).
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
        
        p1 = np.array([z, y]) 
        p2 = np.array([z, y])
        
        if best_primitive in ['fling', 'stretchdrag', 'pick-and-stretch']:
            p1[0] += pix_grasp_dist
            p2[0] -= pix_grasp_dist
        elif best_primitive == 'place':
            p2[0] += pix_place_dist
        elif best_primitive == 'drag':
            p2[0] += self.config.get('pix_drag_dist', 16)
            
        num_scales = len(self.adaptive_scale_factors)
        rotation_idx = x // num_scales
        scale_idx = x % num_scales
        
        scale = self.adaptive_scale_factors[scale_idx]
        rotation = self.rotations[rotation_idx]
        
        orig_dim = self.transformed_obs['original_dim'] 
        resized_dim = 128 
        
        # --- CRITICAL COMMENT: Inverse Affine Mapping ---
        # Because the network evaluated the action on a transformed (rotated/scaled) 
        # 128x128 observation stack, the extracted spatial coordinates (z, y) must be 
        # inversely mapped back to the original input dimension.
        mat = get_transform_matrix(
            original_dim=orig_dim, 
            resized_dim=resized_dim, 
            rotation=-rotation, 
            scale=scale
        )
        
        pixels = np.array([p1, p2])
        pixels_homo = np.concatenate((pixels, np.ones((2, 1))), axis=1)
        orig_pixels = np.matmul(pixels_homo, mat)[:, :2].astype(int)
        
        action_params = {
            'p1': orig_pixels[0],
            'p2': orig_pixels[1],
            'max_value': best_val 
        }
        
        return best_primitive, action_params
        
    def generate_transformed_obs(self, obs_dict, input_dim=128, scale_factors=None, rotations=None):
        from clothmate.utils.env_utils import prepare_image, generate_workspace_mask, generate_primitive_cloth_mask
        from itertools import product
        
        if scale_factors is None: scale_factors = self.adaptive_scale_factors
        if rotations is None: rotations = self.rotations
        
        rgb = obs_dict['rgb']
        depth = obs_dict.get('depth', np.zeros(rgb.shape[:2]))
        
        obs_tensor = torch.cat((
            torch.tensor(rgb).float() / 255.0,
            torch.tensor(depth).unsqueeze(2).float()
        ), dim=2).permute(2, 0, 1)
        
        transformations = list(product(rotations, scale_factors))
        
        retval = {}
        retval['original_dim'] = rgb.shape[0] 
        
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
        pretransform_left_arm_mask = torch.ones_like(pretransform_cloth_mask)
        pretransform_right_arm_mask = torch.ones_like(pretransform_cloth_mask)
        
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