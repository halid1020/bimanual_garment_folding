import os
import numpy as np
import cv2
import time

from actoris_harena import TrainableAgent
import torch
from itertools import product
from actoris_harena.utilities.torch_utils import np_to_ts

from .utils import prepare_image, generate_primitive_cloth_mask,\
    generate_workspace_mask, transform, get_transform_matrix
from .nets import MaximumValuePolicy
from .keypoint_inference import KeypointDetector


LOSS_NORMALIZATIONS = {
    'rigid':{
        'fling':{'manipulation':5, 'nocs':3}, 
        'place':{'manipulation':1.2, 'nocs':3}},
    'deformable':{
        'fling':{'manipulation':0.8, 'nocs':3}, 
        'place':{'manipulation':0.1, 'nocs':3}}
}

def shirt_folding_heuristic_2d(keypoint_positions):
    """Calculates 2D pick and place waypoints for a shirt."""
    top_midpoint = (keypoint_positions['top_right'] + keypoint_positions['top_left']) / 2
    bottom_midpoint = (keypoint_positions['bottom_right'] + keypoint_positions['bottom_left']) / 2

    alpha = 0.9
    bottom_right_quarter = alpha * bottom_midpoint + (1 - alpha) * keypoint_positions['bottom_right']
    bottom_left_quarter = alpha * bottom_midpoint + (1 - alpha) * keypoint_positions['bottom_left']

    # 2D norms work perfectly here
    right_arm_length = np.linalg.norm(keypoint_positions['right_shoulder'] - keypoint_positions['top_right'])
    left_arm_length = np.linalg.norm(keypoint_positions['left_shoulder'] - keypoint_positions['top_left'])
    arm_length = max([right_arm_length, left_arm_length])

    right_vec = (bottom_right_quarter - keypoint_positions['right_shoulder'])
    right_place = keypoint_positions['right_shoulder'] + (right_vec / (np.linalg.norm(right_vec) + 1e-6)) * arm_length

    left_vec = (bottom_left_quarter - keypoint_positions['left_shoulder'])
    left_place = keypoint_positions['left_shoulder'] + (left_vec / (np.linalg.norm(left_vec) + 1e-6)) * arm_length

    right_double_pick = keypoint_positions['bottom_right'] + (keypoint_positions['right_shoulder'] - keypoint_positions['bottom_right']) * 0.95
    left_double_pick = keypoint_positions['bottom_left'] + (keypoint_positions['left_shoulder'] - keypoint_positions['bottom_left']) * 0.95

    right_axis_gap = keypoint_positions['right_shoulder'] - keypoint_positions['bottom_right']
    left_axis_gap = keypoint_positions['left_shoulder'] - keypoint_positions['bottom_left']

    right_double_place = keypoint_positions['bottom_right'] + left_axis_gap * 0.2
    left_double_place = keypoint_positions['bottom_left'] + right_axis_gap * 0.2

    # Group actions
    return [
        [{"pick": keypoint_positions['top_right'], "place": right_place}],
        [{"pick": keypoint_positions['top_left'], "place": left_place}],
        [{"pick": left_double_pick, "place": left_double_place},
         {"pick": right_double_pick, "place": right_double_place}]
    ]

class ClothFunnelsAdapter(TrainableAgent):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'cloth-funnel'
        self.internal_states = {}

        self.device = config.device if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.update_step = -1
        self.debug = self.config.get('debug', False)
        self._init_network()
        
    
    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir

    def _init_network(self):

        self.network = MaximumValuePolicy(
            action_expl_prob=self.config.action_expl_prob,
            action_expl_decay=self.config.action_expl_decay,
            value_expl_decay=self.config.value_expl_decay,
            value_expl_prob=self.config.value_expl_prob,
            action_primitives=list(self.config.action_primitives),
            num_rotations=self.config.num_rotations,
            scale_factors=list(self.config.scale_factors),
            obs_dim=self.config.obs_dim,
            pix_grasp_dist=self.config.pix_grasp_dist,
            pix_place_dist=self.config.pix_place_dist,
            pix_drag_dist=self.config.pix_drag_dist,
            deformable_weight=self.config.deformable_weight,
            network_gpu=self.config.network_gpu,
            input_channel_types=self.config.input_channel_types,
            gpu=0)

        self.optimiser = torch.optim.Adam(
            self.network.value_net.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        
        # --- ADD THESE LINES HERE ---
        
        # 1. Total parameters for the ENTIRE network
        total_params = sum(p.numel() for p in self.network.parameters())
        
        # 2. Trainable parameters for the ENTIRE network
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        # 3. (Optional) Parameters just for the value_net that your Adam optimizer is updating
        value_net_params = sum(p.numel() for p in self.network.value_net.parameters())

        print(f"--- Network Parameter Info ---")
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Value Net Parameters:  {value_net_params:,}")
        print(f"------------------------------")
        
    def _init_action_primitives(self):

        self.action_primitives = list(self.config.action_primitives)
        self.adaptive_scale_factors = list(self.config.scale_factors)
        self.rotations = np.linspace(-180, 180, self.config.num_rotations + 1)
        # print("ALL ROTATIONS", self.rotations)
        self.rotation_indices = {
            'fling': np.where(np.logical_and(self.rotations >= -90, self.rotations <= 90)),
            'place': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 167.5)),
        }
        self.scale_factors = list(self.config.scale_factors)
        self.primitive_vmap_indices = {}
        for primitive, indices in self.rotation_indices.items():
            self.primitive_vmap_indices[primitive] = [None, None]
            self.primitive_vmap_indices[primitive][0] = indices[0] * len(self.scale_factors)
            self.primitive_vmap_indices[primitive][1] = (indices[-1]+1) * len(self.scale_factors)
        
        if self.config.include_fold:
            path = os.path.join(self.save_dir, self.config.keypoint_model_path)
            self.keypoint_detector = KeypointDetector(path)

    
    def load(self, path = None) -> int:
        load_path = path if path is not None else self.save_dir
        load_path = os.path.join(load_path, 'last_train.pth')
        checkpoint = torch.load(load_path)
        #print(checkpoint.keys())
        self.network.load_state_dict(checkpoint['net'])
        self.optimiser.load_state_dict(checkpoint['optimizer'])
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = checkpoint['replay_buffer']
            self.update_step = checkpoint['step']
        self.update_step = -1
        if 'step' in checkpoint:
            self.update_step = checkpoint['step']
        print(f"Loaded checkpoint {self.update_step}")
        return self.update_step
    
    def load_best(self, path = None) -> int:
        load_path = path if path is not None else self.save_dir
        load_path = os.path.join(load_path, 'checkpoints', 'model_best.pth')
        checkpoint = torch.load(load_path)
        #print(checkpoint.keys())
        self.network.load_state_dict(checkpoint['net'])
        self.optimiser.load_state_dict(checkpoint['optimizer'])
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = checkpoint['replay_buffer']
            self.update_step = checkpoint['step']
        if 'step' in checkpoint:
            self.update_step = checkpoint['step']
        print(f"[ClothFunnels] Loaded Best checkpoint")
        return self.update_step
    
    def load_checkpoint(self, checkpoint: int) -> bool:
        if checkpoint == -1:
            self.load()
        load_path = os.path.join(self.save_dir, f'checkpoint_{checkpoint}.pth')
        self.network.load_state_dict(torch.load(load_path)['net'])
        self.optimiser.load_state_dict(torch.load(load_path)['optimizer'])

        return True
    
    def reset(self, arena_ids):
        self._init_action_primitives()
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {
                'step': 0,
                'fold_action_queue': [],
                'has_folded': False  # Add this flag
            }

    def get_state(self):
        return self.internal_states
    
    def init(self, information):
        pass

    def set_eval(self):
        return self.network.eval()
    
    def set_train(self):
        return self.network.train()
    
    def update(self, info, action):
        pass

    def _preporcess_info(self, info):
        retval = {}

        ##GENERATE OBSERVATION

        H, W = info['observation']['rgb'].shape[:2]
        retval['prerot_rgb'] = info['observation']['rgb'].copy()


        for key in ['rgb', 'depth', 'mask', 'robot0_mask', 'robot1_mask']:
            assert key in info['observation'], f"Key {key} not found in observation"
            ## rotate the image clothwise 90 degrees
            info['observation'][key] = np.rot90(info['observation'][key], 1)


        retval['pretransform_depth'] = info['observation']['depth'].copy()
        retval['pretransform_rgb'] = info['observation']['rgb'].copy()
        
        masked_rgb = info['observation']['rgb'].copy()
        masked_rgb[~info['observation']['mask']] = 0
        rgbd = torch.cat([
            np_to_ts(masked_rgb, self.device).float()/255,
            np_to_ts(info['observation']['depth'].copy(), self.device).reshape(H, W, -1).float(),
        ], dim=2).permute(2, 0, 1)

        #print('RGBD shape', rgbd.shape)
        
        retval['transformed_obs'] = prepare_image(
            rgbd, 
            self._get_transformations(self.rotations), 
            self.config.obs_dim,
            orientation_net = None,
            parallelize=self.config.parallelize_prepare_image,
            nocs_mode=self.config.nocs_mode,
            inter_dim=256,
            constant_positional_enc=self.config.constant_positional_enc,)
    
        #print('Transformed obs shape', retval['transformed_obs'].shape)
        
        mask = np_to_ts(info['observation']['mask'].copy(), self.device)
        right_arm_mask =  torch.tensor(info['observation']['robot0_mask'].copy()).bool().to(self.device)
        left_arm_mask =  torch.tensor(info['observation']['robot1_mask'].copy()).bool().to(self.device)
                
        pretransform_mask = torch.stack([mask, left_arm_mask, right_arm_mask], dim=0)
        #print('Pretransform mask shape', pretransform_mask.shape)
        transformed_mask = prepare_image(
            pretransform_mask, 
            self._get_transformations(self.rotations), 
            self.config.obs_dim,
            parallelize=self.config.parallelize_prepare_image,
            nocs_mode=self.config.nocs_mode,
            inter_dim=128,
            constant_positional_enc=self.config.constant_positional_enc,)
        retval['pretransform_mask'] = pretransform_mask
        
        #print('Transformed mask shape', transformed_mask.shape)
        cloth_mask = transformed_mask[:, 0]
        #print('Cloth mask shape', cloth_mask.shape)
        left_arm_mask = transformed_mask[:, 1]
        right_arm_mask = transformed_mask[:, 2]

        workspace_mask = generate_workspace_mask(
            left_arm_mask,
            right_arm_mask,
            self.action_primitives,
            self.config.pix_place_dist,
            self.config.pix_grasp_dist,
        )

        cloth_mask = generate_primitive_cloth_mask(
            cloth_mask,
            self.action_primitives,
            self.config.pix_place_dist,
            self.config.pix_grasp_dist,
        )

        
        for primitive in self.action_primitives:
            GUARANTEE_OFFSET=6
            offset = self.config.pix_grasp_dist if primitive == 'fling' else self.config.pix_place_dist + GUARANTEE_OFFSET
            offset = int(offset)
            primitive_vmap_indices = self.primitive_vmap_indices[primitive] ## TODO: define this
            #print('Primitive vmap indices', primitive_vmap_indices)

            valid_transforms_mask = torch.zeros_like(cloth_mask[primitive]).bool()
            #print('valid_transforms_mask shape', valid_transforms_mask.shape)
            #print('vmp_idx dtype', type(primitive_vmap_indices[0]), type(primitive_vmap_indices[1]))
            id0s = torch.tensor(primitive_vmap_indices[0], device=self.device, dtype=torch.long)
            id1s = torch.tensor(primitive_vmap_indices[1], device=self.device, dtype=torch.long)

            first_dim_mask = torch.zeros(valid_transforms_mask.shape[0], dtype=torch.bool, device=self.device)
            for start, end in zip(id0s, id1s):
                first_dim_mask[start:end] = True

            #print('id0s', id0s, 'id1s', id1s)
            valid_transforms_mask[first_dim_mask, 
                        offset:-offset,
                        offset:-offset] = True
        
            table_mask = retval['transformed_obs'][:, 3] > 0
            offset_table_mask_up = torch.zeros_like(table_mask).bool()
            offset_table_mask_down = torch.zeros_like(table_mask).bool()
            offset_table_mask_up[:, :-offset, :] = table_mask[:, offset:]
            offset_table_mask_down[:, offset:, :] = table_mask[:, :-offset]
            table_mask = offset_table_mask_up & offset_table_mask_down & table_mask

            primitive_workspace_mask = torch.logical_and(workspace_mask[primitive], table_mask)
            primitive_workspace_mask = torch.logical_and(primitive_workspace_mask, valid_transforms_mask)

            retval[f"{primitive}_cloth_mask"] = cloth_mask[primitive]
            retval[f"{primitive}_workspace_mask"] = primitive_workspace_mask
            retval[f"{primitive}_mask"] = torch.logical_and(cloth_mask[primitive], primitive_workspace_mask)
        
        for key in ['rgb', 'depth', 'mask', 'robot0_mask', 'robot1_mask']:
            assert key in info['observation'], f"Key {key} not found in observation"
            ## rotate the image clothwise 90 degrees back
            info['observation'][key] = np.rot90(info['observation'][key], -1)

        #print('preproces keys', retval.keys())
        return retval



    def _get_transformations(self, rotations):
        #print('Rotations', rotations)
        #print('Adaptive scale factors', self.adaptive_scale_factors)
        return list(product(
            rotations, self.adaptive_scale_factors))

    def single_act(self, info, update=False):
        aid = info['arena_id']

        # =========================================================
        # DEBUG BLOCK 0: Sanity Check Raw Input (Before Preprocessing)
        # =========================================================
        if getattr(self, 'debug', False):
            os.makedirs('tmp/clothfunnels/', exist_ok=True)
            raw_rgb = info['observation']['rgb'].copy()
            r0_mask = info['observation']['robot0_mask'].copy()
            r1_mask = info['observation']['robot1_mask'].copy()

            # Overlay masks on the raw environment camera frame
            overlay = raw_rgb.copy()
            overlay[r0_mask.astype(bool)] = [255, 0, 0]  # Red: Robot 0
            overlay[r1_mask.astype(bool)] = [0, 255, 0]  # Green: Robot 1
            
            sanity_img = cv2.addWeighted(overlay, 0.35, raw_rgb, 0.65, 0)
            cv2.imwrite('tmp/clothfunnels/0_sanity_check_raw.png', cv2.cvtColor(sanity_img, cv2.COLOR_RGB2BGR))
        
        # 1. INTERCEPT: Are we currently executing a fold sequence?
        if len(self.internal_states[aid].get('fold_action_queue', [])) > 0:
            action = self.internal_states[aid]['fold_action_queue'].pop(0)
            self.internal_states[aid]['step'] += 1
            return action

        start_time = time.time()
        transformed_info = self._preporcess_info(info)

        # 2. TRIGGER FOLD: Have we reached the step limit to start folding?
        if (self.internal_states[aid].get('step', 0) >= self.config.flat_length - 1) \
            and self.config.include_fold \
            and not self.internal_states[aid]['has_folded']: # Check the flag
            
            fold_actions = self._generate_fold_actions(transformed_info)
            self.internal_states[aid]['fold_action_queue'].extend(fold_actions)
            self.internal_states[aid]['has_folded'] = True # Set the flag
            
            action = self.internal_states[aid]['fold_action_queue'].pop(0)
            self.internal_states[aid]['step'] += 1
            print('[ClothFunnels] action', action)
            return action

        # 3. NORMAL NETWORK POLICY
        action_tuple = self.network.act([transformed_info])[0]

        action = self._postprocess_action_tuple(action_tuple, transformed_info)
        self.internal_states[aid]['step'] += 1
        
        duration = time.time() - start_time
        print(f"Arena {info.get('arena_id', 'Unknown')}: Action planned in {duration:.4f} seconds.")
        print('[ClothFunnels] action', action)
        return action

    def _generate_fold_actions(self, transformed_info):
        keypoint_names = ['left_shoulder', 'right_shoulder', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        
        # Get the RGB image directly from your preprocessed dictionary
        img = transformed_info['pretransform_rgb']
        
        # 1. Detect Keypoints
        keypoint_coords = self.keypoint_detector.get_keypoints(img.astype(np.float32)/255)
        pixel_keypoints = {k: v for k, v in zip(keypoint_names, keypoint_coords)}

        # =========================================================
        # DEBUG BLOCK 4: Annotate Keypoints on Pretransform Image
        # =========================================================
        if getattr(self, 'debug', False):
            import cv2
            import os
            os.makedirs('tmp/clothfunnels/', exist_ok=True)
            
            # Ensure image is 0-255 uint8 and BGR for OpenCV
            debug_img = img.copy()
            if debug_img.max() <= 1.0:
                debug_img = (debug_img * 255).astype(np.uint8)
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            
            # Distinct colors for each keypoint
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
            
            for (name, coord), color in zip(pixel_keypoints.items(), colors):
                # Assuming coord is [y, x] based on your H, W normalization
                x, y = int(coord[0]), int(coord[1])
                cv2.circle(debug_img, (x, y), 5, color, -1)
                cv2.putText(debug_img, name.replace('_', ' '), (x + 8, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
            cv2.imwrite('tmp/clothfunnels/4_keypoint_debug.png', debug_img)
        # =========================================================

        # 2. Generate Heuristic 2D actions
        raw_actions = shirt_folding_heuristic_2d(pixel_keypoints)
        
        # 3. Setup Normalization based on image shape
        H, W = img.shape[:2]
        hw_array = np.array([H, W]) # Note: Make sure this aligns with (y, x) vs (x, y)
        
        def norm_coord(coord):
            return (coord / hw_array) * 2 - 1

        # 4. Format Action 1: Combine both single-arm folds into one dual-action
        arm_fold_action = {
            'norm-pixel-dual-pick-and-place': {
                'pick_0': norm_coord(raw_actions[0][0]['pick']),
                'place_0': norm_coord(raw_actions[0][0]['place']),
                'pick_1': norm_coord(raw_actions[1][0]['pick']),
                'place_1': norm_coord(raw_actions[1][0]['place'])
            }
        }
        
        # 5. Format Action 2: The bottom-up body fold
        body_fold_action = {
            'norm-pixel-dual-pick-and-place': {
                'pick_0': norm_coord(raw_actions[2][0]['pick']),
                'place_0': norm_coord(raw_actions[2][0]['place']),
                'pick_1': norm_coord(raw_actions[2][1]['pick']),
                'place_1': norm_coord(raw_actions[2][1]['place'])
            }
        }

        # Return the sequence (queue) of actions
        return [arm_fold_action, body_fold_action]
    
    
    def _postprocess_action_tuple(self, action_tuple, info):

        primitive = action_tuple['chosen_primitive']
        if self.config.fling_only:
            primitive = 'fling'
        elif self.config.place_only:
            primitive = 'place'
   
        chosen_indices = action_tuple[primitive]['chosen_index']

        chosen_deformable_value = action_tuple[primitive]['chosen_deformable_value']
        chosen_rigid_value = action_tuple[primitive]['chosen_rigid_value']
        chosen_value = action_tuple[primitive]['chosen_value']

        x, y, z = chosen_indices
        all_value_maps = action_tuple[primitive]['all_value_maps']
        value_map = all_value_maps[x]
        action_mask = torch.zeros(value_map.size())

        try:
            action_mask[y, z] = 1 
        except:
            print("Indices", chosen_indices)
            exit(1)

        num_scales = len(self.adaptive_scale_factors)
        rotation_idx = torch.div(x, num_scales, rounding_mode='floor')
        scale_idx = x - rotation_idx * num_scales
        
        scale = self.adaptive_scale_factors[scale_idx]
        rotation = self.rotations[rotation_idx]

        # 1. Get Reach Points in the 128x128 Transformed Space
        reach_points = np.array(self._get_action_params(
            action_primitive=primitive,
            max_indices=(x, y, z),
        ))
        p1, p2 = reach_points[:2]

        if (p1 is None) or (p2 is None):
            print("\n [SimEnv] Invalid pickpoints \n", primitive, p1, p2)
            raise ValueError("Invalid pickpoints")

        # --- Extract Masks for Debugging ---
        # 1. Primitive Workspace (Transformed 128x128)
        ws_mask = info[f'{primitive}_workspace_mask'][x].detach().cpu().numpy().astype(bool)
        
        # 2. Left & Right Robot Workspaces (Pretransform 480x480)
        left_arm_pre = info['pretransform_mask'][1].detach().cpu().numpy().astype(bool)
        right_arm_pre = info['pretransform_mask'][2].detach().cpu().numpy().astype(bool)

        # Calculate Transformation Matrices
        pre_H, pre_W = info['pretransform_rgb'].shape[:2]
        T2d = get_transform_matrix(
            original_dim=pre_H, 
            resized_dim=self.config.obs_dim, 
            rotation=-rotation, 
            scale=scale
        )
        M_pre = T2d.T[:2, :].astype(np.float32)            # Matrix: 128x128 -> 480x480
        M_fwd = cv2.invertAffineTransform(M_pre)           # Matrix: 480x480 -> 128x128


        # =========================================================
        # DEBUG BLOCK 1: Plot on the Transformed Observation Space
        # =========================================================
        if getattr(self, 'debug', False):
            os.makedirs('tmp/clothfunnels/', exist_ok=True)
            chosen_image = info['transformed_obs'][x][:3].detach().cpu().numpy()
            chosen_image = (chosen_image.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
            
            # Warp Robot Workspaces forward to 128x128 space
            left_arm_trans = cv2.warpAffine(left_arm_pre.astype(np.uint8), M_fwd, 
                                            (self.config.obs_dim, self.config.obs_dim), 
                                            flags=cv2.INTER_NEAREST).astype(bool)
            right_arm_trans = cv2.warpAffine(right_arm_pre.astype(np.uint8), M_fwd, 
                                             (self.config.obs_dim, self.config.obs_dim), 
                                             flags=cv2.INTER_NEAREST).astype(bool)
            
            # Workspace Overlays
            overlay = chosen_image.copy()
            overlay[left_arm_trans] = [255, 0, 0]   # Red: Left Arm
            overlay[right_arm_trans] = [0, 255, 0]  # Green: Right Arm
            overlay[ws_mask] = [255, 0, 255]        # Magenta: Valid Primitive Action
            
            chosen_image = cv2.addWeighted(overlay, 0.35, chosen_image, 0.65, 0)
            
            cv2.circle(chosen_image, (int(p1[1]), int(p1[0])), 3, (255, 0, 0), -1)
            cv2.circle(chosen_image, (int(p2[1]), int(p2[0])), 3, (0, 255, 0), -1)
            cv2.imwrite('tmp/clothfunnels/1_transformed_space.png', cv2.cvtColor(chosen_image, cv2.COLOR_RGB2BGR))


        # =========================================================
        # DEBUG BLOCK 2: Plot on the Pretransform Space
        # =========================================================
        transform_pixels = np.concatenate((np.array([p1, p2]), np.ones((2, 1))), axis=1)
        pixels_pre = np.matmul(transform_pixels, T2d)[:, :2].astype(int)

        if getattr(self, 'debug', False):
            pretransform_img = info['pretransform_rgb'].copy()
            
            # Un-warp Primitive WS to Pretransform 480x480 space
            ws_mask_pre = cv2.warpAffine(ws_mask.astype(np.uint8), M_pre, 
                                         (pre_W, pre_H), flags=cv2.INTER_NEAREST).astype(bool)
            
            # Workspace Overlays
            overlay_pre = pretransform_img.copy()
            overlay_pre[left_arm_pre] = [255, 0, 0]    # Red: Left Arm
            overlay_pre[right_arm_pre] = [0, 255, 0]   # Green: Right Arm
            overlay_pre[ws_mask_pre] = [255, 0, 255]   # Magenta: Valid Primitive Action
            
            pretransform_img = cv2.addWeighted(overlay_pre, 0.35, pretransform_img, 0.65, 0)

            cv2.circle(pretransform_img, (int(pixels_pre[0][1]), int(pixels_pre[0][0])), 5, (255, 0, 0), -1)
            cv2.circle(pretransform_img, (int(pixels_pre[1][1]), int(pixels_pre[1][0])), 5, (0, 255, 0), -1)
            cv2.imwrite('tmp/clothfunnels/2_pretransform_space.png', cv2.cvtColor(pretransform_img, cv2.COLOR_RGB2BGR))


        # =========================================================
        # DEBUG BLOCK 3: Plot on the Absolute Original Image Space
        # =========================================================
        H, W = info['prerot_rgb'].shape[:2]
        p_orig = np.zeros_like(pixels_pre)
        
        p_orig[:, 0] = pixels_pre[:, 1]                 
        p_orig[:, 1] = (W - 1) - pixels_pre[:, 0]       
        p_orig[:, 0] = np.clip(p_orig[:, 0], 0, H - 1)
        p_orig[:, 1] = np.clip(p_orig[:, 1], 0, W - 1)
        p1_final, p2_final = p_orig[0], p_orig[1]

        if getattr(self, 'debug', False):
            prerot_img = info['prerot_rgb'].copy()
            
            # Un-rotate Workspaces
            left_arm_orig = np.rot90(left_arm_pre, -1)
            right_arm_orig = np.rot90(right_arm_pre, -1)
            ws_mask_orig = np.rot90(ws_mask_pre, -1)
            
            # Workspace Overlays
            overlay_orig = prerot_img.copy()
            overlay_orig[left_arm_orig] = [255, 0, 0]   # Red: Left Arm
            overlay_orig[right_arm_orig] = [0, 255, 0]  # Green: Right Arm
            overlay_orig[ws_mask_orig] = [255, 0, 255]  # Magenta: Valid Primitive Action
            
            prerot_img = cv2.addWeighted(overlay_orig, 0.35, prerot_img, 0.65, 0)

            cv2.circle(prerot_img, (int(p1_final[1]), int(p1_final[0])), 6, (255, 0, 0), -1)
            cv2.circle(prerot_img, (int(p2_final[1]), int(p2_final[0])), 6, (0, 255, 0), -1)
            cv2.imwrite('tmp/clothfunnels/3_original_space.png', cv2.cvtColor(prerot_img, cv2.COLOR_RGB2BGR))


        # 4. Normalize to [-1, 1] and Pack Action
        chosen_primitive_name = 'norm-pixel-single-pick-and-place' if primitive == 'place' else 'norm-pixel-pick-and-fling'
        
        chosen_primitive_params = {
            'pick_0': p1_final / np.array([H, W]) * 2 - 1,
            'pick_1': p2_final / np.array([H, W]) * 2 - 1,
        }

        if primitive == 'place':
            chosen_primitive_params['place_0'] = chosen_primitive_params['pick_1']
            del chosen_primitive_params['pick_1']
        
        # if self.config.place_only:
        #     return np.stack([chosen_primitive_params['pick_0'], chosen_primitive_params['place_0']]).flatten()

        action = {
            chosen_primitive_name: chosen_primitive_params
        }

        return action
    
    def _get_action_params(self, action_primitive, max_indices, cloth_mask=None):
        x, y, z = max_indices
        if action_primitive == 'fling' or\
                action_primitive == 'stretchdrag':
            center = np.array([x, y, z])
            p1 = center[1:].copy()
            p2 = center[1:].copy()

            p1[0] = p1[0] + self.config.pix_grasp_dist
            p2[0] = p2[0] - self.config.pix_grasp_dist
        elif action_primitive == 'place':
            p1 = np.array([y, z])
            p2 = p1.copy()
            p2[0] += self.config.pix_place_dist
        else:
            raise Exception(
                f'Action Primitive not supported: {action_primitive}')
        if (p1 is None) or (p2 is None):
            raise Exception(
                f'None reach points: {action_primitive}')
        return p1, p2


    def save(self, path=None) -> bool:
        save_path = path if path is not None else self.config.save_path
        checkpoint_path = os.path.join(save_path, f'checkpoint_{self.update_step}.pth')
        
        # save network to the checkpoint path,
        # and save replay buffer, optimiser and network with train_{} prefix
        torch.save({
            'network': self.network.state_dict(),
            'optimiser': self.optimiser.state_dict(),
        }, checkpoint_path)

        torch.save({
            'network': self.network.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'replay_buffer': self.replay_buffer,
            'step': self.update_step
            }, os.path.join(save_path, f'last_train.pth'))
            
    
        return True

    def train(self, update_steps, arena=None) -> bool:
        
        if self.config.train_mode == 'offline':
            return self._train_offline(update_steps)
        elif self.config.train_mode == 'online':
            return self._train_online(update_steps, arena)
    
    def _train_online(self, update_steps, arena):
        target_update_step = self.update_step + update_steps
        
        if self.replay_buffer.size() < self.config.pretrain_collect_steps:
            ## collect data using network output
            ## in the original code, it runs paralelle environments to collect data
            ## but here we provide sequential version.
            collect_steps = self.config.pretrain_collect_steps - self.replay_buffer.size()
            self._collect_data(collect_steps, arena)

        for u in range(self.update_step, target_update_step):
            
            self._update_network()

            if u%self.config.updates_per_collect == 0:
                # updates_per_collect is 8 and steps_per_collect is 128 
                collect_steps = (u // self.config.updates_per_collect) \
                    * self.config.steps_per_collect - self.replay_buffer.size()
                self._collect_data(collect_steps, arena)
    
    def _train_offline(self, update_steps):
        target_update_step = self.update_step + update_steps
        for u in range(self.update_step, target_update_step):
            self._update_network()

    def _update_network(self):
        self.network.train()
        value_net = self.network.value_net
        
        sample = self.replay_buffer.sample(self.config.batch_size)

        #COMPUTE LOSSES
        # 2-picker pick-and-place and pick-and-fling in the original paper.
        stats = dict()
        unfactorized_losses = dict()
        distance_types = ['rigid', 'deformable'] # rigid and deformable distances in the original paper

        losses = {
            dt: {primitive: {'manipulation': 0} for primitive in self.config.action_primitives}
            for dt in distance_types
        }
        l2_error = {
            dt: {primitive: {'manipulation': 0} for primitive in self.config.action_primitives}
            for dt in distance_types
        }
        unfactorized_losses = {primitive: 0 for primitive in self.config.action_primitives}

        for primitive_id in range(len(self.config.action_primitives)): 

            action_primitive = self.config.action_primitives[primitive_id]

            in_dict = sample[primitive_id]
            obs = in_dict['obs']
            action_mask = in_dict['action']
            weighted_reward = in_dict['weighted_reward']
            deformable_reward = in_dict['deformable_reward']
            rigid_reward = in_dict['rigid_reward']
            l2_reward = in_dict['l2_reward']
            cov_reward = in_dict['coverage_reward']
            is_terminal = in_dict['is_terminal'].to(self.device)

            action_mask = action_mask.unsqueeze(1)
            
            rewards = {'rigid': rigid_reward, 'deformable': deformable_reward}

            #preprocess here so that we can log the obs
            obs = value_net.preprocess_obs(obs.to(self.device, non_blocking=True))
            out = value_net.forward_for_optimize(obs, action_primitive, preprocess=False)

            action_mask = torch.cat([action_mask], dim=1)

            # deformable_weight is 0.65 in the original code
            unfactorized_value_pred_dense = (1-self.config.deformable_weight) \
                  * out['rigid'][action_primitive] + \
                 self.config.deformable_weight * out['deformable'][action_primitive]
            unfactorized_value_pred = torch.masked_select(
                unfactorized_value_pred_dense, action_mask.\
                    to(self.device, non_blocking=True))

            if self.config.unfactorized_rewards: # False in the original code
                if self.config.coverage_reward: # False in the original code
                    print("[Network] Using coverage reward")
                    unfactorized_reward = cov_reward.to(self.device)
                else:
                    print("[Network] Using unfactorized reward")
                    unfactorized_reward = l2_reward.to(self.device)
            else:
                print("[Network] Using factorized reward")
                unfactorized_reward = weighted_reward.to(self.device)

          
            unfactorized_losses[action_primitive] = \
                torch.nn.functional.smooth_l1_loss(
                    unfactorized_value_pred, unfactorized_reward.to(self.device, non_blocking=True))

            for distance in distance_types:

                value_pred_dense = out[distance][action_primitive]
                value_pred = torch.masked_select(
                    value_pred_dense,
                    action_mask.to(self.device, non_blocking=True))
    
                reward = rewards[distance].to(self.device)

                manipulation_loss = torch.nn.functional.smooth_l1_loss(value_pred, reward)

                losses[distance][action_primitive]['manipulation'] = \
                    manipulation_loss / LOSS_NORMALIZATIONS[distance][action_primitive]['manipulation']
             
                l2_error[distance][action_primitive]['manipulation'] = manipulation_loss

                log_idx = 0
                # visualizations[distance][action_primitive]['manipulation'] = value_pred_dense[log_idx].detach().cpu().numpy()
                # visualizations[distance][action_primitive]['obs'] = obs[log_idx].detach().cpu().numpy()

        #OPTIMIZE
        loss = 0

        for distance in distance_types:
            for primitive in self.config.action_primitives:
                stats[f'loss/{primitive}/unfactorized']= unfactorized_losses[primitive] / len(self.config.action_primitives)
                stats[f'loss/{primitive}/{distance}/factorized'] = losses[distance][primitive]['manipulation'] / len(self.config.action_primitives)
                stats[f'l2_error/{primitive}/{distance}/factorized'] = l2_error[distance][primitive]['manipulation'] / len(self.config.action_primitives)

        self.optimiser.zero_grad()
        if self.config.unfactorized_networks: # False in the original code
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/unfactorized' in k)
        else:
            loss = sum(v for k,v in stats.items() if 'loss/' in k and '/factorized' in k)
        loss.backward()
        self.optimiser.step()