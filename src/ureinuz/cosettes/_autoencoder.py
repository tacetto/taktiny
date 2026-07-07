# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import jax
from ureinuz import nn
from ureinuz import layers

class SpatialAttention(nn.Module):
    def __init__(self, dim: int, seed: nn.Rngs):
        # Standard VAE attention uses 1 head with head_dim = dim
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim)
        self.attn = layers.Attention(
            hidden_size=dim,
            num_heads=1,
            head_dim=dim,
            bias=True,
            seed=seed
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        
        # Pre-Norm
        h = self.norm(x)
        
        # Flatten spatial dimensions
        h = h.reshape(B, H * W, C)
        
        # Attention
        h, _ = self.attn(h)
        
        # Unflatten spatial dimensions
        h = h.reshape(B, H, W, C)
        
        # Residual connection
        return x + h

class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        dims: list[int], 
        latent_dim: int, 
        depths: list[int] = None, 
        seed: nn.Rngs = None
    ):
        self.conv_in = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding="SAME", seed=seed)
        
        self.down_blocks = nn.List()
        layers_per_block = depths[0] if depths is not None else 2
        current_channels = dims[0]
        
        for i, dim in enumerate(dims):
            # ResNet blocks
            for _ in range(layers_per_block):
                self.down_blocks.layers.append(
                    layers.ResnetBlock2D(in_channels=current_channels, out_channels=dim, seed=seed)
                )
                current_channels = dim
            
            # Downsample (except for the last block)
            if i != len(dims) - 1:
                self.down_blocks.layers.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding="SAME", seed=seed)
                )
                
        # Mid Block
        self.mid_block1 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)
        self.mid_attn = SpatialAttention(dim=current_channels, seed=seed)
        self.mid_block2 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)
        
        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=current_channels)
        self.conv_out = nn.Conv2d(current_channels, latent_dim, kernel_size=3, padding="SAME", seed=seed)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_in(x)
        
        for layer in self.down_blocks:
            x = layer(x)
            
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        x = self.conv_out(x)
        
        return x

class Decoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        dims: list[int], 
        latent_dim: int, 
        depths: list[int] = None, 
        seed: nn.Rngs = None
    ):
        self.conv_in = nn.Conv2d(latent_dim, dims[-1], kernel_size=3, padding="SAME", seed=seed)
        
        # Mid Block
        current_channels = dims[-1]
        self.mid_block1 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)
        self.mid_attn = SpatialAttention(dim=current_channels, seed=seed)
        self.mid_block2 = layers.ResnetBlock2D(in_channels=current_channels, out_channels=current_channels, seed=seed)
        
        self.up_blocks = nn.List()
        layers_per_block = depths[0] if depths is not None else 2
        
        reversed_dims = list(reversed(dims))
        
        for i, dim in enumerate(reversed_dims):
            # ResNet blocks (Diffusers uses layers_per_block + 1 for upsampling blocks)
            for _ in range(layers_per_block + 1):
                self.up_blocks.layers.append(
                    layers.ResnetBlock2D(in_channels=current_channels, out_channels=dim, seed=seed)
                )
                current_channels = dim
                
            # Upsample (except for the last block)
            if i != len(reversed_dims) - 1:
                self.up_blocks.layers.append(
                    nn.Sequential(
                        nn.Upsample2d(scale_factor=2),
                        nn.Conv2d(current_channels, current_channels, kernel_size=3, padding="SAME", seed=seed)
                    )
                )
                
        # Output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=current_channels)
        self.conv_out = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding="SAME", seed=seed)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_in(x)
        
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        for layer in self.up_blocks:
            x = layer(x)
            
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        x = self.conv_out(x)
        
        return x

class Autoencoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        dims: list[int], 
        latent_dim: int, 
        depths: list[int] = None, 
        seed: nn.Rngs = None
    ):
        self.encoder = Encoder(
            in_channels=in_channels, 
            dims=dims, 
            latent_dim=latent_dim, 
            depths=depths, 
            seed=seed
        )
        self.decoder = Decoder(
            in_channels=in_channels, 
            dims=dims, 
            latent_dim=latent_dim, 
            depths=depths, 
            seed=seed
        )
        
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
