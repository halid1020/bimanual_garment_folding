
from numpy import np
import torch
from .utils import *
from actoris_harena import TrainableAgent

class ClothMateAdapter(TrainbaleAgent):
    
    def __init__(self, config):
        super().__init__(config)
        self.name = 'cloth-mate'
        self.num_rotations = config.num_rotations
        assert self.num_rotations % 4 == 0
        self.rotations = np.linspace(-180, 180, self.num_rotations + 1)

        self.scale_factors = np.array(config.scale_factors)
        self.use_adaptive_scaling = config.use_adaptive_scaling
        self.adaptive_scale_factors = self.scale_factors.copy()


        self.rotation_indices = {
            'fling': np.where(np.logical_and(self.rotations >= -90, self.rotations <= 90)),
            'place': np.where(np.logical_and(self.rotations >= -180, self.rotations <= 167.5)),
        }

        self.primitive_vmap_indices = {}
        for primitive, indices in self.rotation_indices.items():
            self.primitive_vmap_indices[primitive] = [None, None]
            self.primitive_vmap_indices[primitive][0] = indices[0] * len(self.scale_factors)
            self.primitive_vmap_indices[primitive][1] = (indices[-1]+1) * len(self.scale_factors)

        
        

        
        self.keypoint_detector = UNet(in_ch=3, out_ch=4).to('cuda')
        checkpoint = torch.load(self.keypoint_detector_path, map_location="cpu", weights_only=True)
        self.keypoint_detector.load_state_dict(checkpoint['model_state_dict'])
        self.keypoint_detector.eval()

        self.policy_nentwork = 

    def get_max_value_valid_action(self, action_tuple) -> dict:

        primitive = action_tuple['best_primitive']

        if self.recreate_primitive:
            primitive = self.recreate_primitive

        max_indices = action_tuple[primitive]['max_index']

        max_deformable_value = action_tuple[primitive]['max_deformable_value']
        max_rigid_value = action_tuple[primitive]['max_rigid_value']
        max_value = action_tuple[primitive]['max_value']

        x, y, z = max_indices
        
        all_value_maps = action_tuple[primitive]['all_value_maps']
        value_map = all_value_maps[x]
        action_mask = torch.zeros(value_map.size())

        try:
            action_mask[y, z] = 1 
        except:
            print("Indices", max_indices)
            exit(1)

        num_scales = len(self.adaptive_scale_factors)
        rotation_idx = torch.div(x, num_scales, rounding_mode='floor')
        scale_idx = x - rotation_idx * num_scales
        scale = self.adaptive_scale_factors[scale_idx]

        rotation = self.rotations[rotation_idx]

        reach_points = np.array(self.get_action_params(
            action_primitive=primitive,
            max_indices=(x, y, z),
            # cloth_mask = transform_cloth_mask
            ))

        p1, p2 = reach_points[:2]

        if (p1 is None) or (p2 is None):
            print("\n [SimEnv] Invalid pickpoints \n", primitive, p1, p2)
            raise ValueError("Invalid pickpoints")

        action_kwargs = {
            'observation': self.transformed_obs['transformed_obs'][x],
            'mask': self.transformed_obs[f'{primitive}_mask'][x],
            'workspace_mask': self.transformed_obs[f'{primitive}_workspace_mask'][x],
            'action_primitive': str(primitive),
            'primitive': primitive,
            'p1': p1,
            'p2': p2,
            'scale': scale,
            'nonadaptive_scale': self.scale_factors[scale_idx],
            'rotation': rotation,
            'predicted_deformable_value': float(max_deformable_value),
            'predicted_rigid_value': float(max_rigid_value),
            'predicted_weighted_value': float(max_value),
            'max_indices': np.array(max_indices),
            'action_mask': action_mask,
            'value_map': value_map,
            'info': None,
            'all_value_maps': all_value_maps,
        }

        if action_tuple[primitive].get('raw_value_map') is not None:
            action_kwargs['raw_value_maps'] = action_tuple[primitive]['raw_value_maps']


        assert ((action_kwargs['p1'] is not None) and (action_kwargs['p2'] is not None))

        action_kwargs.update({
            'transformed_depth':
            action_kwargs['observation'][3, :, :],
            'transformed_rgb':
            action_kwargs['observation'][:3, :, :],
        })

        if self.recreate_verts is not None:
            for primitive in self.action_primitives:
                action_kwargs[f'{primitive}_value_maps'] = action_tuple[primitive]['all_value_maps']
                action_kwargs[f'{primitive}_raw_value_maps'] = action_tuple[primitive]['raw_value_maps']


        action_params = self.check_action(
            pixels=np.array([p1, p2]),
            **action_kwargs)

        try:
            reachable, left_or_right = self.check_action_reachability(
                action=primitive,
                p1=action_params['p1'],
                p2=action_params['p2'])
        except ValueError as e:
            raise ValueError("Reach pos none")

        if primitive == 'place':
            action_kwargs['left_or_right'] = left_or_right
        
        for k in ['valid_action',
                    'pretransform_pixels',]:
            del action_params[k]

        return action_kwargs['action_primitive'], action_params
    
    def generate_transformed_obs(self, obs,
                                 input_dim = None, 
                                 scale_factors = None, 
                                 rotations = None):
        """
        Generates transformed observations and masks
        """
        if all(x is not None for x in [input_dim, scale_factors, rotations]):
            input_dim = input_dim
            scale_factors = scale_factors
            rotations = rotations
            input_flag = True
        else:
            input_dim = self.obs_dim
            scale_factors = self.adaptive_scale_factors
            rotations = self.rotations
            input_flag = False
        retval = {}

        ##GENERATE OBSERVATION

        retval['transformed_obs'] = prepare_image(
                        obs, 
                        self.get_transformations(rotations, scale_factors), 
                        input_dim,
                        orientation_net = self.orn_net_handle,
                        parallelize=self.parallelize_prepare_image,
                        nocs_mode=self.nocs_mode,
                        inter_dim=256,
                        constant_positional_enc=self.constant_positional_enc,)   

        ##GENERATE MASKS
        pretransform_cloth_mask = self.get_cloth_mask(obs[:3]) #TODO: info['observation']['mask']
        if input_flag:
            pretransform_left_arm_mask = torch.ones_like(self.get_cloth_mask(obs[:3]))
            pretransform_right_arm_mask = torch.ones_like(self.get_cloth_mask(obs[:3])) 
        else:
            pretransform_left_arm_mask = self.left_arm_mask
            pretransform_right_arm_mask = self.right_arm_mask 

        pretransform_mask = torch.stack([pretransform_cloth_mask, 
                                        pretransform_left_arm_mask, 
                                        pretransform_right_arm_mask], 
                                        dim=0)
        
        transformed_mask = prepare_image(
                        pretransform_mask, 
                        self.get_transformations(rotations, scale_factors), 
                        input_dim,
                        parallelize=self.parallelize_prepare_image,
                        nocs_mode=self.nocs_mode,
                        inter_dim=256,
                        constant_positional_enc=self.constant_positional_enc,)   

        cloth_mask = transformed_mask[:, 0]
        left_arm_mask = transformed_mask[:, 1]
        right_arm_mask = transformed_mask[:, 2]

        workspace_mask = generate_workspace_mask(left_arm_mask, 
                                                right_arm_mask, 
                                                self.action_primitives, 
                                                self.pix_place_dist, 
                                                self.pix_grasp_dist)
        
        cloth_mask = generate_primitive_cloth_mask(
                                cloth_mask,
                                self.action_primitives,
                                self.pix_place_dist,
                                self.pix_grasp_dist)

        for primitive in self.action_primitives:

            GUARANTEE_OFFSET=6
            offset = self.pix_grasp_dist if primitive == 'fling' else self.pix_place_dist + GUARANTEE_OFFSET
            primitive_vmap_indices = self.primitive_vmap_indices[primitive]

            if input_flag: primitive_vmap_indices = [0, cloth_mask[primitive].shape[0]]

            valid_transforms_mask = torch.zeros_like(cloth_mask[primitive]).bool()
            valid_transforms_mask[primitive_vmap_indices[0]:primitive_vmap_indices[1], 
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
        return retval
    
    def single_act(self, info, update=False):
        
        # TODO: rotate the rgb of info clockwise 90 degree
        

        self.transformed_obs = self.generate_transformed_obs(info)

        value_maps = self.network(info)
        state, keypoints = self.keypoint_detect()
        
        action_primitive, action = self.get_max_value_valid_action(value_maps)

        # TODO: return primtive action as dict in normalised pixel space.

        # TODO: rotate the pixel based action anticlock wise 90 degree.