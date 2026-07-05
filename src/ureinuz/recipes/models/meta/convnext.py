import jax
import jax.numpy as jnp
from .... import nn, Rngs
from ..configs.cnn import CNNModelConfig

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, seed: Rngs):
        # 7x7 depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding="SAME", groups=dim, seed=seed)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Point-wise convolutions are implemented with Linear layers since NHWC perfectly matches Linear
        self.pwconv1 = nn.Linear(dim, 4 * dim, seed=seed)
        self.pwconv2 = nn.Linear(4 * dim, dim, seed=seed)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = jax.nn.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        return x + res


class ConvNeXt(nn.Module):
    def __init__(self, config: CNNModelConfig, seed: Rngs):
        self.downsample_layers = nn.List()
        # Stem (4x4 conv stride 4)
        stem = nn.Sequential(
            nn.Conv2d(3, config.dims[0], kernel_size=4, stride=4, padding="VALID", seed=seed),
            nn.LayerNorm(config.dims[0], eps=1e-6)
        )
        self.downsample_layers.layers.append(stem)
        
        # 3 intermediate downsampling layers
        for i in range(3):
            downsample = nn.Sequential(
                nn.LayerNorm(config.dims[i], eps=1e-6),
                nn.Conv2d(config.dims[i], config.dims[i+1], kernel_size=2, stride=2, padding="VALID", seed=seed)
            )
            self.downsample_layers.layers.append(downsample)
            
        self.stages = nn.List()
        for i in range(4):
            # Efficient SequentialStack for multiple identical blocks per stage
            stage = nn.SequentialStack(ConvNeXtBlock, config.dims[i], num_stack=config.depths[i], seed=seed)
            self.stages.layers.append(stage)
            
        self.norm = nn.LayerNorm(config.dims[-1], eps=1e-6) # Final norm
        self.head = nn.Linear(config.dims[-1], config.num_classes, seed=seed)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is [B, H, W, 3]
        
        def forward_block(layer, carry):
            return layer(carry)
            
        for i in range(4):
            x = self.downsample_layers[i](x)
            # Apply the sequential stack of blocks via scan
            x = self.stages[i](forward_block, x)
            
        # Global average pooling over spatial dimensions H, W (indices 1, 2 for NHWC)
        x = jnp.mean(x, axis=(1, 2))
        x = self.norm(x)
        x = self.head(x)
        return x