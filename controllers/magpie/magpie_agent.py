import os
import time
import torch
import numpy as np
from collections import deque
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
cv2.setNumThreads(0)

from actoris_harena import TrainableAgent
from actoris_harena.utilities.networks.utils import ts_to_np

# --- NEW EXTRACTED MODULES ---
from .magpie_transform import DiffusionTransform
from .magpie_network_builder import build_networks_and_optimizers, build_primitive_action_masks
from .magpie_trainer import MagpieTrainer
from .action_sampler import ActionSampler
from .constrain_action_functions import name2func

class MagpieAgent(TrainableAgent):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'diffusion'
        self.config = config
        self.internal_states = {}
        self.buffer_actions = {}
        self.last_actions = {}
        self.obs_deque = {}
        
        # Configuration flags
        self.collect_on_success = self.config.get('collect_on_success', True)
        self.measure_time = config.get('measure_time', False)
        self.debug = config.get('debug', False)
        self.constrain_action = name2func[config.get('constrain_action', 'identity')]
        self.val_interval = self.config.get('val_interval', 100)
        self.validate_training = self.config.get('validate_training', 100)

        # Primitive Integration Setup
        self.primitive_integration = self.config.get('primitive_integration', 'none')
        if self.primitive_integration != 'none':
            self.primitives = config.primitives
            self.K = len(self.primitives)
            self.action_dims = [prim['dim'] if isinstance(prim, dict) else prim.dim for prim in self.primitives]
            self.prim_name2id = {item['name']: i for i, item in enumerate(self.primitives)}
            
            self.network_action_dim = max(self.action_dims)
            if self.primitive_integration == 'bin_as_output':
                self.network_action_dim += 1
                
            self.data_save_action_dim = self.network_action_dim
            if self.primitive_integration == 'one-hot-encoding':
                self.data_save_action_dim += 1
                
            self.mask_out_irrelavent_action_dim = self.config.get('mask_out_irrelavent_action_dim', False)
        else:
            self.network_action_dim = config.action_dim
            self.data_save_action_dim = config.action_dim

        # Extracted Network Initialization
        self.nets, self.optimizer, self.lr_scheduler, self.ema, self.clip_norm = build_networks_and_optimizers(self)
        
        if self.primitive_integration != 'none':
             self.primitive_action_masks = build_primitive_action_masks(self)

        self.loaded = False
        self.eval_action_sampler = ActionSampler[self.config.eval_action_sampler]()
        self.update_step = 0
        self.total_update_steps = self.config.total_update_steps
        self.dataset_inited = False
        
        from data_augmentation.register_augmeters import build_data_augmenter
        self.data_augmenter = build_data_augmenter(config.data_augmenter)
        self.device = self.config.get('device', 'cpu')
        
        # Extracted Trainer Logic
        self.trainer = MagpieTrainer(self)

    def train(self, update_steps, arenas):
        """Delegates the heavy training loop to the dedicated Trainer class."""
        self.trainer.train(update_steps, arenas)

    def validate(self):
        """Delegates validation logic."""
        self.trainer.validate()

    def set_log_dir(self, logdir, project_name, exp_name, disable_wandb=False):
        super().set_log_dir(logdir, project_name, exp_name, disable_wandb=disable_wandb)
        self.save_dir = logdir

    def save(self):
        ckpt_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        for filename in os.listdir(ckpt_dir):
            if filename.startswith('net_') and filename.endswith('.pt') and 'best' not in filename:
                try:
                    os.remove(os.path.join(ckpt_dir, filename))
                except OSError as e:
                    print(f"Error deleting old checkpoint: {e}")

        ckpt_path = os.path.join(ckpt_dir, f'net_{self.update_step}.pt')
        torch.save(self.nets.state_dict(), ckpt_path)

    def save_best(self):
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(self.nets.state_dict(), os.path.join(ckpt_path, 'net_best.pt'))

    def load_checkpoint(self, checkpoint):
        ckpt_path = os.path.join(self.save_dir, 'checkpoints', f'net_{checkpoint}.pt')
        self.nets.load_state_dict(torch.load(ckpt_path))
        print(f'Loaded checkpoint: {checkpoint}')
        self.loaded = True

    def load(self):
        ckpt_path = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_files = [c for c in os.listdir(ckpt_path) if c.endswith('.pt') and 'best' not in c]
        ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if not ckpt_files:
            print('[MultiPrimitiveDiffusion, load] No checkpoint found')
            return 0
            
        ckpt_file = ckpt_files[-1]
        self.nets.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_file)))
        print(f'Loaded checkpoint: {ckpt_file}')
        self.loaded = True
        self.update_step = int(ckpt_file.split('_')[1].split('.')[0])
        return self.update_step

    def load_best(self):
        ckpt_path = os.path.join(self.save_dir, 'checkpoints', 'net_best.pt')
        if not os.path.exists(ckpt_path):
            print(f"[MultiPrimitiveDiffusion, load_best] Checkpoint not found at: {ckpt_path}")
            return 0 
        
        self.nets.load_state_dict(torch.load(ckpt_path))
        self.loaded = True
        print(f"[MultiPrimitiveDiffusion, load_best] Best checkpoint is loaded")
        return -2

    @torch.no_grad()
    def single_act(self, info, update=False):
        start_time = time.time()
        arena_id = info['arena_id']

        if update:
            last_action = self.last_actions.get(arena_id)
            if last_action is not None:
                self.update([info], [last_action])
            else:
                self.init([info])

        if len(self.buffer_actions[arena_id]) == 0:
            sample_state = self._prepare_eval_state(info)
            obs_features = self._extract_vision_features(sample_state['image'], info)
            
            if self.config.include_state:
                vector_state = torch.stack([x['vector_state'] for x in self.obs_deque[arena_id]])
                obs_features = torch.cat([obs_features, vector_state], dim=-1)

            if getattr(self, 'use_projector', False):
                obs_features = self.nets['obs_projector'](obs_features)

            obs_cond, cur_prim_id, prim_probs_log = self._process_primitives(obs_features, info)
            
            naction = self.eval_action_sampler.sample(
                state=sample_state, 
                horizon=self.config.pred_horizon, 
                action_dim=getattr(self, 'diffusion_dim', self.network_action_dim)
            ).to(self.device)
            
            naction, noise_actions = self._run_diffusion_loop(naction, obs_cond, cur_prim_id, info)
            self._store_predicted_actions(naction, arena_id)
            self._log_internal_states(naction, noise_actions, prim_probs_log, obs_features, info)

        action = self.buffer_actions[arena_id].popleft()
        action = action[:self.network_action_dim].flatten()
        out_action = self._format_output_action(action, cur_prim_id if 'cur_prim_id' in locals() else None)
        
        self.last_actions[arena_id] = action

        if self.measure_time:
            self.internal_states[arena_id].setdefault('inference_time', []).append(time.time() - start_time)
            print(f"Arena {arena_id}: Action planned in {time.time() - start_time:.4f} seconds.")
            
        return out_action

    def act(self, infos, updates):
        return [self.single_act(info, upd) for info, upd in zip(infos, updates)]

    def reset(self, arena_ids):
        if not self.loaded:
            self.load()
            
        for arena_id in arena_ids:
            self.internal_states[arena_id] = {}
            self.buffer_actions[arena_id] = deque(maxlen=self.config.action_horizon)
            self.last_actions[arena_id] = None

    def get_state(self):
        return self.internal_states

    def init(self, infos):
        for info in infos:
            obs = self._process_info(info)
            self.obs_deque[info['arena_id']] = deque([obs]*self.config.obs_horizon, maxlen=self.config.obs_horizon)

    def update(self, infos, actions):
        for info, action in zip(infos, actions):
            obs = self._process_info(info)
            self.obs_deque[info['arena_id']].append(obs)

    # --------------------------------------------------------------------------------
    # --- PRIVATE HELPERS FOR SINGLE_ACT (Keeps the main logic clean) ----------------
    # --------------------------------------------------------------------------------

    def _prepare_eval_state(self, info):
        arena_id = info['arena_id']
        image = torch.stack([x[self.config.input_obs] for x in self.obs_deque[arena_id]])
        state = {'image': image}
        if self.config.use_mask:
            state['mask'] = torch.stack([x['mask'] for x in self.obs_deque[arena_id]])
        return state

    def _extract_vision_features(self, image, info):
        # The logic here mirrors the existing vision branching
        if self.vision_encoder_type == 'original':
            return self.nets['vision_encoder'](image)
        elif self.vision_encoder_type == 'vit':
            import torchvision.transforms.functional as TF
            B = 1 # single_act operates unbatched
            T = image.shape[0] if image.ndim == 4 else image.shape[1]

            rgb_resized = TF.resize(image[:, :3, :, :], [224, 224], antialias=True)
            goal_resized = TF.resize(image[:, 3:6, :, :], [224, 224], antialias=True)
            
            obs_feature = self.nets['vision_encoder'](rgb_resized.to(self.device))
            goal_feature = self.nets['vision_encoder'](goal_resized.to(self.device))
            
            # Restored reshape logic to maintain (Batch, Time, Dim)
            image_features = torch.cat([obs_feature, goal_feature], dim=-1)
            return image_features.reshape(B, T, -1)
            
        elif self.vision_encoder_type == 'gc_rssm_encoder':
            return torch.cat([self.nets['vision_encoder'](image[:, :3, :, :]), 
                              self.nets['vision_encoder'](image[:, 3:6, :, :])], dim=-1)
        elif self.vision_encoder_type == 'gc_rssm_dynamic':
            image_b = image.unsqueeze(0)
            B, T = image_b.shape[:2]
            obs_emb = self.nets['vision_encoder'](image_b[:, :, :3].flatten(end_dim=1)).view(B, T, -1).transpose(0, 1)
            goal_emb = self.nets['vision_encoder'](image_b[:, :, 3:6].flatten(end_dim=1)).view(B, T, -1).transpose(0, 1)
            
            hidden = self.nets['transition_model'](
                prev_state=torch.zeros(B, self.config.stochastic_latent_dim, device=self.device),
                actions=torch.ones(T, B, self.network_action_dim, device=self.device),
                prev_belief=torch.zeros(B, self.config.deterministic_latent_dim, device=self.device),
                goal_observations=goal_emb, observations=obs_emb,
                nonterminals=torch.ones(T, B, 1, device=self.device)
            )
            return torch.cat([hidden[0], hidden[4]], dim=-1).transpose(0, 1).squeeze(0)
        
    def _process_primitives(self, obs_features, info):
        cur_prim_id = 0
        prim_probs_log = None
        
        if self.primitive_integration in ['one-hot-encoding', 'separate_networks']:
            prim_logits = self.nets['prim_class_head'](obs_features)
            prim_probs_log = torch.softmax(prim_logits, dim=-1).cpu().detach().numpy()
            prim_id = torch.argmax(prim_logits, dim=-1)
            cur_prim_id = prim_id[-1].cpu().detach().item()
            
            p_obj = self.primitives[cur_prim_id]
            info['prim_name'] = p_obj['name'] if isinstance(p_obj, dict) else p_obj.name

            if self.primitive_integration == 'one-hot-encoding':
                prim_enc = torch.nn.functional.one_hot(prim_id, num_classes=self.K).float()
                obs_cond = torch.cat([obs_features, prim_enc], dim=-1)
                # Ensure the one-hot appended tensor is flattened identically
                return obs_cond.unsqueeze(0).flatten(start_dim=1), cur_prim_id, prim_probs_log
                
        return obs_features.unsqueeze(0).flatten(start_dim=1), cur_prim_id, prim_probs_log
    
    def _run_diffusion_loop(self, naction, obs_cond, cur_prim_id, info):
        start = self.config.obs_horizon - 1
        end = start + self.config.action_horizon
        dim_k = self.diffusion_dims[cur_prim_id] if self.primitive_integration == 'separate_networks' else self.diffusion_dim
        
        # RESTORED: Capture the very first fully-noised frame for debug visualizations
        noise_actions = [ts_to_np(naction[:, start:end, :self.network_action_dim])]

        if dim_k == 0:
            naction = torch.zeros_like(naction)
            return naction, [ts_to_np(naction[:, start:end, :self.network_action_dim])] * (self.config.num_diffusion_iters + 1)

        active_net = self.nets[f'noise_pred_net_{cur_prim_id}'] if self.primitive_integration == 'separate_networks' else self.nets['noise_pred_net']
        
        if self.config.get('loss_type', 'diffusion') == 'ot_flow_match':
            num_steps = self.config.num_diffusion_iters
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_val = i / num_steps
                t_tensor = torch.tensor([t_val * num_steps], device=self.device, dtype=torch.float32)
                v_pred = active_net(sample=naction[..., :dim_k], timestep=t_tensor, global_cond=obs_cond)
                naction[..., :dim_k] += v_pred * dt
                if self.primitive_integration == 'one-hot-encoding':
                    naction = self._apply_action_constraints(naction, info, i)
                noise_actions.append(ts_to_np(naction[:, start:end, :self.network_action_dim]))
        else:
            self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
            for k in self.noise_scheduler.timesteps:
                noise_pred = active_net(sample=naction[..., :dim_k], timestep=k, global_cond=obs_cond)
                naction[..., :dim_k] = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction[..., :dim_k]).prev_sample
                if self.primitive_integration == 'one-hot-encoding':
                    naction = self._apply_action_constraints(naction, info, k)
                noise_actions.append(ts_to_np(naction[:, start:end, :self.network_action_dim]))
                
        return naction, noise_actions
    
    def _apply_action_constraints(self, naction, info, timestep):
        act_part = naction[..., :self.network_action_dim]
        state_part = naction[..., self.network_action_dim:]
        act_part = self.constrain_action(act_part, info, t=timestep, debug=self.debug)
        return torch.cat([act_part, state_part], dim=-1)

    def _store_predicted_actions(self, naction, arena_id):
        start = self.config.obs_horizon - 1
        end = start + self.config.action_horizon
        final_naction = naction[..., :self.network_action_dim]
        action_pred = self.data_augmenter.postprocess({'action': ts_to_np(final_naction)})['action'][0]
        self.buffer_actions[arena_id] = deque(action_pred[start:end, :], maxlen=self.config.action_horizon)

    def _format_output_action(self, action, cur_prim_id):
        if self.config.primitive_integration == 'none':
            return action
        elif self.config.primitive_integration == 'bin_as_output':
            prim_idx = int(np.clip(((action[0] + 1)/2)*self.K - 1e-6, 0, self.K - 1))
            p_obj = self.primitives[prim_idx]
            prim_name = p_obj['name'] if isinstance(p_obj, dict) else p_obj.name
            return {prim_name: action[1:]}
        elif self.primitive_integration in ['one-hot-encoding', 'separate_networks']:
            p_obj = self.primitives[cur_prim_id]
            prim_name = p_obj['name'] if isinstance(p_obj, dict) else p_obj.name
            return {prim_name: action[:self.action_dims[cur_prim_id]]}
        raise NotImplementedError

    def _log_internal_states(self, naction, noise_actions, prim_probs_log, obs_features, info):
        pred_keypoints = None
        if naction.shape[-1] > self.network_action_dim:
            pred_keypoints = ts_to_np(naction[..., self.network_action_dim:])
        elif getattr(self, 'rep_learn', None) == 'predict-state' and 'state_predictor' in self.nets:
            with torch.no_grad():
                pred_keypoints = ts_to_np(self.nets['state_predictor'](obs_features))

        gt_keypoints = info['observation'].get('semkey_norm_pixel', info['observation'].get('vector_state'))
        self.internal_states[info['arena_id']].update({
            'noise_actions_history': noise_actions,
            'primitive_probabilities': prim_probs_log,
            'predicted_keypoints': pred_keypoints,
            'gt_keypoints': gt_keypoints
        })

    def _process_info(self, info):
        if 'depth' in info['observation'].keys():
            depth = info['observation']['depth'] #get the view from first camera.

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
                info['observation']['depth'] = depth

        if self.config.input_obs == 'rgbd':
            info['observation']['rgbd'] = np.concatenate(
                [info['observation']['rgb'][0].astype(np.float32), depth], axis=-1)
        
        if self.config.input_obs == 'rgb+goal_rgb':
            info['observation']['rgb+goal_rgb'] = np.concatenate(
                [info['observation']['rgb'].astype(np.float32), info['observation']['goal_rgb'].astype(np.float32)], axis=-1)

        def resize_mask_to_rgb(mask):
                H, W = rgb.shape[:2]

                # Ensure numpy array (already true, but safe)
                mask = np.asarray(mask)

                # Remove channel if present
                if mask.ndim == 3:
                    mask = mask[..., 0]

                # CRITICAL: cast dtype
                if mask.dtype != np.uint8 and mask.dtype != np.float32:
                    mask = mask.astype(np.float32)

                mask = cv2.resize(
                    mask,
                    (W, H),                      # (width, height)
                    interpolation=cv2.INTER_NEAREST
                )

                return mask[..., None]           # (H, W, 1)
        
        if self.config.input_obs == 'rgb-workspace-mask':
            rgb = info['observation']['rgb'].astype(np.float32)

            m0 = resize_mask_to_rgb(info['observation']['robot0_mask'])
            m1 = resize_mask_to_rgb(info['observation']['robot1_mask'])

            info['observation']['rgb-workspace-mask'] = np.concatenate(
                [rgb, m0, m1], axis=-1
            )
            
        if self.config.input_obs == 'rgb-workspace-mask-goal':
            rgb = info['observation']['rgb'].astype(np.float32)
            goal = info['observation']['goal_rgb'].astype(np.float32)

            m0 = resize_mask_to_rgb(info['observation']['robot0_mask'])
            m1 = resize_mask_to_rgb(info['observation']['robot1_mask'])

            info['observation']['rgb-workspace-mask-goal'] = np.concatenate(
                [rgb, m0, m1, goal], axis=-1
            )
        
        if self.config.input_obs == 'rgb+goal_mask':
            rgb = info['observation']['rgb'].astype(np.float32)
            mask = resize_mask_to_rgb(info['observation']['mask'])

            info['observation']['rgb+goal_mask'] = np.concatenate(
                [rgb, mask], axis=-1
            )
            if self.debug: print('input shape', info['observation']['rgb+goal_mask'].shape)

        input_data = {
            self.config.input_obs: info['observation'][self.config.input_obs]\
                .reshape(1, 1, *info['observation'][self.config.input_obs].shape),
        }
        
        if 'use_mask' in self.config and self.config.use_mask:
            input_data['mask'] = info['observation']['mask']\
                .reshape(1, 1, *info['observation']['mask'].shape, 1)
            
        if self.config.include_state:
            input_data['vector_state'] = info['observation']['vector_state']\
                .reshape(1, 1, *info['observation']['vector_state'].shape)

        input_data = self.data_augmenter(input_data, train=False, device=self.device) 
        
        vis = input_data[self.config.input_obs].squeeze(0).squeeze(0)

        obs = {
            self.config.input_obs: vis.cpu(),  
        }
        if 'use_mask' in self.config and self.config.use_mask:
            mask = input_data['mask'].squeeze(0).squeeze(0)
            obs['mask'] = mask.cpu()

        if self.config.include_state:
            vector_state = input_data['vector_state'].squeeze(0).squeeze(0)
            obs['vector_state'] = vector_state.cpu()

        input_obs = self.data_augmenter.postprocess(obs)[self.config.input_obs]

        self.internal_states[info['arena_id']].update(
            {'input_obs': input_obs.transpose(1,2,0),
             'input_type': self.config.input_obs}
        )
        
        return obs

    def set_eval(self): pass
    def set_train(self): pass