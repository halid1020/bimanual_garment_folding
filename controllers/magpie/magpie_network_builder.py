import os
import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .utils import get_resnet, replace_bn_with_gn
from .networks import ConditionalUnet1D, MLPClassifier, ResNetDecoder, ConditionalMLP1D

def build_networks_and_optimizers(agent):
    """
    Extracts the massive _init_networks and _init_optimizer logic.
    Returns nets, optimizer, lr_scheduler, ema, and clip_norm.
    """
    config = agent.config
    device = config.get('device', 'cpu')
    
    # 1. Determine Input Channels
    agent.rep_learn = config.get('rep_learn', 'none')
    input_channel_map = {
        'depth': 1, 'rgbd': 4, 'rgb+goal_mask': 4,
        'rgb-workspace-mask': 5, 'rgb+goal_rgb': 6,
        'rgb-workspace-mask-goal': 8
    }
    agent.input_channel = input_channel_map.get(config.input_obs, 3)

    # 2. Vision Encoder Initialization
    agent.vision_encoder_type = config.get('vision_encoder', 'original')
    if agent.vision_encoder_type == 'original':
        vision_encoder = get_resnet('resnet18', input_channel=agent.input_channel)
        vision_encoder = replace_bn_with_gn(vision_encoder)
    elif agent.vision_encoder_type == 'vit':
        vision_encoder = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vision_encoder.heads = nn.Identity() 
    elif agent.vision_encoder_type in ['gc_rssm_encoder', 'gc_rssm_dynamic']:
        from ..rl.lagarnet.networks import ImageEncoder
        vision_encoder = ImageEncoder(
            image_dim=config.input_obs_dim, embedding_size=config.embedding_dim,
            activation_function=config.activation, batchnorm=config.encoder_batchnorm,
            residual=config.encoder_residual
        )
        # Note: Dynamic transition logic/weight loading is handled in the main trainer if needed
    else:
        vision_encoder = nn.Identity()

    if config.get('freeze_encoder', False):
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()

    # 3. Projectors and Classifiers
    agent.use_projector = config.get('use_projector', False)
    agent.effective_obs_dim = config.obs_dim
    
    obs_projector = nn.Identity()
    if agent.use_projector:
        proj_hidden = config.get('projector_hidden_dims', [])
        proj_out = config.get('projector_out_dim', 128)
        layers = []
        in_dim = config.obs_dim
        for h in proj_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, proj_out))
        obs_projector = nn.Sequential(*layers)
        agent.effective_obs_dim = proj_out

    prim_class_head = None
    global_cond_dim = agent.effective_obs_dim * config.obs_horizon
    if agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
        cls_cfg = config.get("primitive_classifier", {}) 
        prim_class_head = MLPClassifier(
            input_dim=agent.effective_obs_dim, output_dim=agent.K,
            hidden_dims=cls_cfg.get("hidden_dims", []), activation=cls_cfg.get("activation", "relu"),
            dropout=cls_cfg.get("dropout", 0.0), use_layernorm=cls_cfg.get("use_layernorm", False),
        )
        if agent.primitive_integration == 'one-hot-encoding':
            global_cond_dim = (agent.effective_obs_dim + agent.K) * config.obs_horizon

    # 4. Diffusion Network Setup
    agent.diffusion_dim = agent.network_action_dim
    agent.diffusion_dims = {}
    extra_state_dim = config.state_dim + config.get('goal_state_dim', 30) if agent.rep_learn == 'predict-state-with-action' else 0
    agent.diffusion_dim += extra_state_dim

    for k in range(getattr(agent, 'K', 1)):
        agent.diffusion_dims[k] = agent.action_dims[k] + extra_state_dim if hasattr(agent, 'action_dims') else agent.diffusion_dim

    net_dict = {'vision_encoder': vision_encoder, 'obs_projector': obs_projector}
    if prim_class_head: 
        net_dict['prim_class_head'] = prim_class_head

    backbone_type = config.get('noise_pred_net', 'unet')
    mlp_cfg = config.get('mlp_backbone_config', {})

    if agent.primitive_integration == 'separate_networks':
        for k in range(agent.K):
            dim_k = agent.diffusion_dims[k]
            if dim_k == 0: continue
            
            if backbone_type == 'mlp':
                net = ConditionalMLP1D(input_dim=dim_k, global_cond_dim=global_cond_dim, pred_horizon=config.pred_horizon, **mlp_cfg)
            else:
                net = ConditionalUnet1D(input_dim=dim_k, global_cond_dim=global_cond_dim, diable_updown=config.get('disable_updown', False))
            
            setattr(agent, f'noise_pred_net_{k}', net)
            net_dict[f'noise_pred_net_{k}'] = net
    else:
        if backbone_type == 'mlp':
            noise_pred_net = ConditionalMLP1D(input_dim=agent.diffusion_dim, global_cond_dim=global_cond_dim, pred_horizon=config.pred_horizon, **mlp_cfg)
        else:
            noise_pred_net = ConditionalUnet1D(input_dim=agent.diffusion_dim, global_cond_dim=global_cond_dim, diable_updown=config.get('disable_updown', False))
            
        setattr(agent, 'noise_pred_net', noise_pred_net)
        net_dict['noise_pred_net'] = noise_pred_net

    agent.noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters, 
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True, 
        prediction_type='epsilon'
    )

    nets = nn.ModuleDict(net_dict).to(device)

    # 5. Optimizers
    trainable_params = [p for p in nets.parameters() if p.requires_grad]
    ema = EMAModel(parameters=trainable_params, power=config.get('ema_power', 0.75))
    
    opt_params = config.get('optimiser_params', {})
    if hasattr(opt_params, 'toDict'): 
        opt_params = opt_params.toDict()
    
    optimizer = torch.optim.AdamW(params=trainable_params, **opt_params)
    lr_scheduler = get_scheduler(
        name=config.get('lr_scheduler', 'cosine'), 
        optimizer=optimizer,
        num_warmup_steps=config.get('num_warmup_steps', 500), 
        num_training_steps=config.total_update_steps
    )
    clip_norm = config.get('grad_clip_norm', -1)

    return nets, optimizer, lr_scheduler, ema, clip_norm

def build_primitive_action_masks(agent):
    """Builds action masks for different primitive outputs."""
    masks = {}
    
    start = None
    if agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
        start = 0
    elif agent.primitive_integration == 'bin_as_output':
        start = 1

    for pid, prim in enumerate(agent.primitives):
        mask = torch.zeros(agent.network_action_dim, dtype=torch.float32)
        if start == 1: 
            mask[0] = 1.0
            
        dim = prim['dim'] if isinstance(prim, dict) else prim.dim
        mask[start:start + dim] = 1.0
        masks[pid] = mask
        
    return masks