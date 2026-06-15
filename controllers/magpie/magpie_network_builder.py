"""
Magpie Network Builder Module.

This module acts as a factory for constructing the complex neural network 
architectures required by the Magpie diffusion agent. It handles the instantiation 
of vision encoders, observation projectors, primitive classification heads, 
noise prediction backbones (U-Net or MLP), and representation learning modules. 
It also configures the optimizers and learning rate schedulers.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .utils import get_resnet, replace_bn_with_gn
from .networks import ConditionalUnet1D, MLPNetwork, ResNetDecoder, ConditionalMLP1D

def build_networks_and_optimizers(agent):
    """
    Constructs all neural network components, optimizers, and schedulers for the agent.

    This function extracts the massive initialization logic from the main agent class 
    to maintain modularity. It dynamically scales network input/output dimensions 
    based on the configured observation space and primitive action integration.

    Args:
        agent: The MagpieAgent instance containing the `.config` and `.primitives` definitions.

    Returns:
        tuple: A 5-tuple containing:
            - nets (nn.ModuleDict): A dictionary of all trainable and non-trainable networks.
            - optimizer (torch.optim.Optimizer): The AdamW optimizer.
            - lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
            - ema (EMAModel): The Exponential Moving Average model for stable diffusion sampling.
            - clip_norm (float): The maximum gradient norm for clipping.
    """
    config = agent.config
    device = config.get('device', 'cpu')
    
    # =========================================================================
    # 1. Determine Input Channels
    # Dynamically map the sensory modality to the required image channel depth.
    # =========================================================================
    agent.rep_learn = config.get('rep_learn', 'none')
    input_channel_map = {
        'depth': 1, 
        'rgbd': 4, 
        'rgb+goal_mask': 4,
        'rgb-workspace-mask': 5, 
        'rgb+goal_rgb': 6,
        'rgb-workspace-mask-goal': 8
    }
    agent.input_channel = input_channel_map.get(config.input_obs, 3)

    # =========================================================================
    # 2. Vision Encoder Initialization
    # Supports standard ResNets, Vision Transformers, and recurrent state-space models.
    # =========================================================================
    agent.vision_encoder_type = config.get('vision_encoder', 'original')
    if agent.vision_encoder_type == 'original':
        vision_encoder = get_resnet('resnet18', input_channel=agent.input_channel)
        # CRITICAL: Replace BatchNorm with GroupNorm. RL/Robotics datasets often require
        # small batch sizes that cause BatchNorm statistics to become highly unstable.
        vision_encoder = replace_bn_with_gn(vision_encoder)
        
    elif agent.vision_encoder_type == 'vit':
        vision_encoder = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vision_encoder.heads = nn.Identity() # Strip the classification head to extract pure embeddings
        
    elif agent.vision_encoder_type in ['gc_rssm_encoder', 'gc_rssm_dynamic']:
        # Import state-space models specifically designed for transition dynamics
        from ..rl.lagarnet.networks import ImageEncoder
        vision_encoder = ImageEncoder(
            image_dim=config.input_obs_dim, embedding_size=config.embedding_dim,
            activation_function=config.activation, batchnorm=config.encoder_batchnorm,
            residual=config.encoder_residual
        )
    else:
        vision_encoder = nn.Identity()

    # Freezing the encoder prevents catastrophic forgetting if fine-tuning from a pre-trained base
    if config.get('freeze_encoder', False):
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_encoder.eval()

    # =========================================================================
    # 3. Projectors and Classifiers
    # Compresses vision features and sets up routing for primitive skills.
    # =========================================================================
    agent.use_projector = config.get('use_projector', False)
    agent.effective_obs_dim = config.obs_dim
    
    obs_projector = nn.Identity()
    if agent.use_projector:
        proj_hidden = config.get('projector_hidden_dims', [])
        proj_out = config.get('projector_out_dim', 128)
        
        # Pull new flexibility parameters with safe defaults
        act_name = config.get('projector_activation', 'relu')
        dropout_p = config.get('projector_dropout', 0.0)
        use_ln = config.get('projector_use_layernorm', False)
        
        # Reuse your existing MLP class for a fully featured projector
        obs_projector = MLPNetwork(
            input_dim=config.obs_dim, 
            output_dim=proj_out,
            hidden_dims=proj_hidden, 
            activation=act_name,
            dropout=dropout_p, 
            use_layernorm=use_ln
        )
        agent.effective_obs_dim = proj_out

    # Initialize the primitive classification head if using hybrid action spaces
    prim_class_head = None
    global_cond_dim = agent.effective_obs_dim * config.obs_horizon
    
    if agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
        cls_cfg = config.get("primitive_classifier", {}) 
        prim_class_head = MLPNetwork(
            input_dim=agent.effective_obs_dim, output_dim=agent.K,
            hidden_dims=cls_cfg.get("hidden_dims", []), activation=cls_cfg.get("activation", "relu"),
            dropout=cls_cfg.get("dropout", 0.0), use_layernorm=cls_cfg.get("use_layernorm", False),
        )
        if agent.primitive_integration == 'one-hot-encoding':
            # Expand the global conditioning dimension to accommodate the concatenated one-hot vector
            global_cond_dim = (agent.effective_obs_dim + agent.K) * config.obs_horizon

    # =========================================================================
    # 4. Diffusion Network Setup
    # Builds the core noise prediction networks (U-Net or MLP).
    # =========================================================================
    agent.diffusion_dim = agent.network_action_dim
    agent.diffusion_dims = {}
    
    # If jointly diffusing states and actions, extend the diffusion dimension
    extra_state_dim = config.state_dim + config.get('goal_state_dim', 30) if agent.rep_learn == 'predict-state-with-action' else 0
    agent.diffusion_dim += extra_state_dim

    for k in range(getattr(agent, 'K', 1)):
        agent.diffusion_dims[k] = agent.action_dims[k] + extra_state_dim if hasattr(agent, 'action_dims') else agent.diffusion_dim

    # Accumulate networks into a dictionary for easy `nn.ModuleDict` instantiation
    net_dict = {'vision_encoder': vision_encoder, 'obs_projector': obs_projector}
    if prim_class_head: 
        net_dict['prim_class_head'] = prim_class_head

    backbone_type = config.get('noise_pred_net', 'unet')
    mlp_cfg = config.get('mlp_backbone_config', {})

    # 'separate_networks' instantiates an independent denoiser for *each* primitive skill
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
        # Standard monolithic denoiser handling all actions
        if backbone_type == 'mlp':
            noise_pred_net = ConditionalMLP1D(input_dim=agent.diffusion_dim, global_cond_dim=global_cond_dim, pred_horizon=config.pred_horizon, **mlp_cfg)
        else:
            noise_pred_net = ConditionalUnet1D(input_dim=agent.diffusion_dim, global_cond_dim=global_cond_dim, diable_updown=config.get('disable_updown', False))
            
        setattr(agent, 'noise_pred_net', noise_pred_net)
        net_dict['noise_pred_net'] = noise_pred_net

    # Standard DDPM Scheduler from Hugging Face Diffusers
    agent.noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters, 
        beta_schedule='squaredcos_cap_v2', # Cosine schedule prevents noise explosion at boundaries
        clip_sample=True, 
        prediction_type='epsilon'
    )

    # =========================================================================
    # 5. Representation Learning Networks
    # Auxiliary tasks (like state/keypoint prediction) to force the vision 
    # encoder to learn geometrically meaningful embeddings.
    # =========================================================================
    if agent.rep_learn == 'auto-encoder':
        net_dict['vision_decoder'] = ResNetDecoder(
            input_dim=512, 
            output_channel=agent.input_channel
        )
    elif agent.rep_learn == 'predict-state':
        agent.predict_goal_state = config.get('predict_goal_state', False)
        agent.goal_state_dim = config.get('goal_state_dim', 30) 
        
        out_dim = config.state_dim
        if agent.predict_goal_state:
            out_dim += agent.goal_state_dim
            print(f"[MultiPrimitiveDiffusion] state_predictor out_dim extended to {out_dim} (State + Goal)")

        pred_cfg = config.get('state_predictor_config', {})
        hidden_dims = pred_cfg.get('hidden_dims', [256])
        use_layernorm = pred_cfg.get('use_layernorm', False)
        dropout_p = pred_cfg.get('dropout', 0)
        activation_name = pred_cfg.get('activation', 'relu').lower()

        layers = []
        in_dim = agent.effective_obs_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            
            if activation_name == 'relu':
                layers.append(nn.ReLU())
            elif activation_name == 'gelu':
                layers.append(nn.GELU())
            elif activation_name == 'silu':
                layers.append(nn.SiLU())
            else:
                layers.append(nn.ReLU())
            
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
                
            in_dim = h_dim 

        layers.append(nn.Linear(in_dim, out_dim))
        net_dict['state_predictor'] = nn.Sequential(*layers)

    # =========================================================================
    # 6. Initialization and Optimizers
    # =========================================================================
    nets = nn.ModuleDict(net_dict).to(device)
    
    trainable_params = [p for p in nets.parameters() if p.requires_grad]
    
    # Exponential Moving Average (EMA) maintains a smoothed version of the weights.
    # This is practically mandatory for stable inference in diffusion models.
    ema = EMAModel(parameters=trainable_params, power=config.get('ema_power', 0.75))
    
    opt_params = config.get('optimiser_params', {})
    if hasattr(opt_params, 'toDict'): 
        opt_params = opt_params.toDict()
    
    # AdamW is preferred to decouple weight decay from the gradient updates
    optimizer = torch.optim.AdamW(params=trainable_params, **opt_params)
    
    # Cosine learning rate scheduler with warmup to prevent early training collapse
    lr_scheduler = get_scheduler(
        name=config.get('lr_scheduler', 'cosine'), 
        optimizer=optimizer,
        num_warmup_steps=config.get('num_warmup_steps', 500), 
        num_training_steps=config.total_update_steps
    )
    clip_norm = config.get('grad_clip_norm', -1)

    return nets, optimizer, lr_scheduler, ema, clip_norm


def build_primitive_action_masks(agent):
    """
    Builds binary masks to zero out irrelevant action dimensions for different primitives.

    When using a shared action space for multiple distinct robotic skills (e.g., 
    a 4-DoF pick vs an 8-DoF bimanual operation), this ensures the network only 
    focuses its loss and predictions on the valid dimensions for the active skill.

    Args:
        agent: The MagpieAgent instance.

    Returns:
        dict: A mapping from primitive IDs (int) to a binary `torch.Tensor` mask.
    """
    masks = {}
    
    start = None
    # Determine the offset index based on whether we reserve dimension 0 for the primitive bin
    if agent.primitive_integration in ['one-hot-encoding', 'separate_networks']:
        start = 0
    elif agent.primitive_integration == 'bin_as_output':
        start = 1

    for pid, prim in enumerate(agent.primitives):
        mask = torch.zeros(agent.network_action_dim, dtype=torch.float32)
        
        if start == 1: 
            mask[0] = 1.0 # Always unmask the bin selector if using it
            
        dim = prim['dim'] if isinstance(prim, dict) else prim.dim
        mask[start:start + dim] = 1.0
        masks[pid] = mask
        
    return masks