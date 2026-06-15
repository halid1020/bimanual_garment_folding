"""
Magpie Core Networks Module.

This module contains the foundational neural network architectures for the diffusion 
policy. It includes the 1D U-Net and MLP backbones for the reverse diffusion process, 
positional embeddings for timestep conditioning, FiLM-modulated residual blocks for 
state conditioning, and auxiliary networks for representation learning and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union

class ResNetDecoder(nn.Module):
    """
    Decodes a 1D latent representation back into a 2D spatial image mask or observation.

    Typically used as an auxiliary task (Auto-Encoder) during representation learning 
    to force the vision encoder to capture geometrically meaningful features.
    """
    def __init__(self, input_dim=512, output_channel=3):
        super().__init__()
        # Map the flat 512-dim vector into a low-resolution spatial feature map (6x6)
        self.fc = nn.Linear(input_dim, 256 * 6 * 6)
        
        # Sequentially upsample: 6x6 -> 12x12 -> 24x24 -> 48x48 -> 96x96
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, output_channel, kernel_size=4, stride=2, padding=1),
            # Squash output to [0, 1] range to match normalized image inputs
            nn.Sigmoid() 
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Latent vector of shape (Batch, 512).
        Returns:
            torch.Tensor: Reconstructed spatial tensor of shape (Batch, C, 96, 96).
        """
        x = self.fc(x)
        x = x.view(-1, 256, 6, 6)
        x = self.decoder(x)
        return x
    
class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding.
    
    Used to encode the scalar diffusion timestep `k` into a high-dimensional 
    vector so the neural network can condition its denoising logic on the current 
    noise level.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 1D tensor of timesteps (Batch,).
        Returns:
            torch.Tensor: Embedded timesteps of shape (Batch, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """Halves the temporal resolution of the action sequence."""
    def __init__(self, dim, disable=False):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
        self.disable = disable

    def forward(self, x):
        if self.disable:
            return x
        return self.conv(x)

class Upsample1d(nn.Module):
    """Doubles the temporal resolution of the action sequence."""
    def __init__(self, dim, disable=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        self.disable = disable

    def forward(self, x):
        if self.disable:
            return x
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Standard 1D Convolutional Block using GroupNorm and Mish activation.

    GroupNorm is preferred over BatchNorm in sequence modeling and RL because 
    batch sizes are often small or highly correlated, which destabilizes BatchNorm.
    Mish provides a smooth, non-monotonic gradient flow.
    """
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    A 1D Residual Block modulated by global conditioning features via FiLM.

    Feature-wise Linear Modulation (FiLM) is the standard mechanism to inject 
    the environmental observation (image features) and the diffusion timestep 
    into the action denoiser. It predicts a per-channel scale and bias to shift 
    the convolutional activations.
    """
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM Modulator: maps the global condition to scale and bias vectors
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)) # Prepare for broadcasting over the time dimension
        )

        # 1x1 conv to match channel dimensions for the residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x (torch.Tensor): Action sequence of shape (Batch, in_channels, Horizon).
            cond (torch.Tensor): Global condition (vision + timestep) of shape (Batch, cond_dim).

        Returns:
            torch.Tensor: Modulated action sequence of shape (Batch, out_channels, Horizon).
        """
        out = self.blocks[0](x)
        
        # Generate FiLM scale and bias
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        
        # Apply affine modulation
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    The core Diffusion Backbone: A 1D U-Net.

    Unlike standard 2D U-Nets used for images, this operates over the temporal 
    horizon of the action trajectory. The spatial dimension is `T` (prediction horizon), 
    and the "channels" are the action dimensions.
    """
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8,
        diable_updown=False): # Note: retained original typo 'diable_updown' for compatibility
        """
        Args:
            input_dim: Action dimension (e.g., 7 DoF).
            global_cond_dim: Dimension of the flattened vision/state features.
            diffusion_step_embed_dim: Dimension of the sinusoidal time embedding.
            down_dims: List of channel dimensions for the UNet hierarchy.
            kernel_size: Convolutional kernel size along the time axis.
            n_groups: Group count for GroupNorm.
            diable_updown: If True, skips spatial downsampling/upsampling.
        """
        print('disable_updown', diable_updown)
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Process diffusion timestep `k` into an embedding
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # The total conditioning dimension injected into every FiLM block
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        
        # Bottleneck blocks
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        # Contracting Path
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out, diable_updown) if not is_last else nn.Identity()
            ]))

        # Expanding Path
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                # dim_out * 2 because of skip connections concatenated from down_modules
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in, diable_updown) if not is_last else nn.Identity()
            ]))

        # Final projection back to action dimensions
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(sum(p.numel() for p in self.parameters())))

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        Args:
            sample (torch.Tensor): Noisy action sequence (Batch, Horizon, ActionDim).
            timestep (Tensor/int): Current diffusion step.
            global_cond (torch.Tensor): Visual/State conditioning (Batch, CondDim).

        Returns:
            torch.Tensor: Predicted noise vector (Batch, Horizon, ActionDim).
        """
        # PyTorch 1D Convs expect (Batch, Channels, Length). We swap ActionDim and Horizon.
        # (B, T, C) -> (B, C, T)
        sample = sample.moveaxis(-1,-2)

        # 1. Process Timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
            
        # Broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # 2. Fuse Timestep and Vision features
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        
        # 3. Contracting Path
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x) # Save for skip connection
            x = downsample(x)

        # 4. Bottleneck
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # 5. Expanding Path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # Concatenate skip connection
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # Revert back to (Batch, Horizon, ActionDim)
        x = x.moveaxis(-1,-2)
        return x


def get_activation(name: str):
    """Utility to map string names to PyTorch activation modules."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish":
        return nn.SiLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    else:
        raise ValueError(f"Unsupported activation: {name}")


class MLPNetwork(nn.Module):
    """
    Standard Multi-Layer Perceptron.

    Used primarily as the classification head for primitive skill selection 
    (e.g., determining whether to execute a pick, place, or dual-arm operation).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims,
        activation="relu",
        dropout=0.0,
        use_layernorm=False,
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + list(hidden_dims)
        act = get_activation(activation)

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))

            layers.append(act)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConditionalMLP1D(nn.Module):
    """
    An alternative Diffusion Backbone: A Dense MLP.

    While the U-Net leverages temporal convolutions to maintain structural awareness 
    across the trajectory horizon, the MLP flattens the entire horizon into a single 
    vector. This is faster and uses fewer parameters, but generally performs worse 
    on highly complex continuous control tasks compared to the U-Net.
    """
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        pred_horizon,
        diffusion_step_embed_dim=256,
        hidden_dims=[512, 512, 512],
        activation="relu",
        dropout=0.1
    ):
        """
        Args:
            input_dim: Dim of a single action step.
            global_cond_dim: Dim of global conditioning.
            pred_horizon: The length of the action sequence (needed to flatten/unflatten).
            diffusion_step_embed_dim: Size of positional encoding for diffusion iteration.
            hidden_dims: List of hidden layer dimensions.
        """
        super().__init__()
        self.input_dim = input_dim
        self.pred_horizon = pred_horizon

        # Time step embedding (identical to ConditionalUnet1D for consistent scaling)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # The MLP takes the flattened action + time embedding + global conditioning
        cond_dim = dsed + global_cond_dim
        mlp_input_dim = (pred_horizon * input_dim) + cond_dim
        output_dim = pred_horizon * input_dim

        # Build the dense layers
        layers = []
        dims = [mlp_input_dim] + list(hidden_dims)
        act = get_activation(activation)

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Final projection to the flattened action shape
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

        print("number of parameters (MLP backbone): {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None
    ):
        """
        Args:
            sample: (Batch, Horizon, ActionDim)
            timestep: (Batch,) or int
            global_cond: (Batch, CondDim)
            
        Returns:
            torch.Tensor: Predicted noise vector (Batch, Horizon, ActionDim)
        """
        B, T, C = sample.shape

        # 1. Process time steps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B)

        global_feature = self.diffusion_step_encoder(timesteps)

        # 2. Concatenate conditioning features
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # 3. Flatten the action sample (B, T*C) and concat with conditioning
        sample_flat = sample.reshape(B, -1) 
        mlp_in = torch.cat([sample_flat, global_feature], dim=-1) 

        # 4. Forward pass
        out_flat = self.net(mlp_in) 

        # 5. Unflatten back to original temporal shape
        out = out_flat.reshape(B, T, C)
        return out