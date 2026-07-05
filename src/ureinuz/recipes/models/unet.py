import jax
import jax.numpy as jnp
from ... import nn, Rngs

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, seed: Rngs):
        self.sin_pos_emb = nn.SinusoidalPositionalEmbedding(time_dim)
        self.linear_1 = nn.Linear(time_dim, time_dim * 4, seed=seed)
        self.linear_2 = nn.Linear(time_dim * 4, time_dim * 4, seed=seed)

    def __call__(self, timesteps: jax.Array) -> jax.Array:
        t = self.sin_pos_emb(timesteps)
        t = jax.nn.silu(self.linear_1(t))
        t = self.linear_2(t)
        return t

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, seed: Rngs):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="SAME", seed=seed)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels, seed=seed)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="SAME", seed=seed)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, seed=seed)
        else:
            self.res_conv = None
            
    def __call__(self, x: jax.Array, t_emb: jax.Array) -> jax.Array:
        res = x if self.res_conv is None else self.res_conv(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = jax.nn.silu(h)
        
        # Inject time embedding
        # t_emb is (B, time_emb_dim) or (time_emb_dim,). we project to out_channels
        time_bias = self.time_proj(jax.nn.silu(t_emb))
        
        if time_bias.ndim == 1:
            # Unbatched: time_bias is (C,), h is (H, W, C)
            h = h + time_bias[None, None, :]
        else:
            # Batched: time_bias is (B, C), h is (B, H, W, C)
            h = h + time_bias[:, None, None, :]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = jax.nn.silu(h)
        
        return h + res

class UNet(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        dims: list[int], 
        seed: Rngs = None
    ):
        time_dim = dims[0]
        self.time_embedding = TimeEmbedding(time_dim, seed=seed)
        
        time_emb_dim = time_dim * 4
        
        self.conv_in = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding="SAME", seed=seed)
        
        self.down_blocks = nn.List()
        current_channels = dims[0]
        for dim in dims:
            self.down_blocks.layers.append(
                UNetBlock(current_channels, dim, time_emb_dim, seed=seed)
            )
            # Add downsample layer except for last
            if dim != dims[-1]:
                self.down_blocks.layers.append(
                    nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding="SAME", seed=seed)
                )
            current_channels = dim
            
        self.mid_block1 = UNetBlock(dims[-1], dims[-1], time_emb_dim, seed=seed)
        self.mid_block2 = UNetBlock(dims[-1], dims[-1], time_emb_dim, seed=seed)
        
        self.up_blocks = nn.List()
        reversed_dims = list(reversed(dims))
        
        for i, dim in enumerate(reversed_dims):
            # Upsample logic
            if i != 0:
                self.up_blocks.layers.append(nn.Upsample2d(scale_factor=2))
                
            # Input to upblock is the previous output + the skip connection
            # Which means the in_channels is dim * 2 (except the first one which is from mid_block)
            block_in_channels = current_channels + dim
            self.up_blocks.layers.append(
                UNetBlock(block_in_channels, dim, time_emb_dim, seed=seed)
            )
            current_channels = dim
            
        self.conv_out = nn.Conv2d(dims[0], in_channels, kernel_size=3, padding="SAME", seed=seed)

    def __call__(self, x: jax.Array, timesteps: jax.Array) -> jax.Array:
        t_emb = self.time_embedding(timesteps)
        
        x = self.conv_in(x)
        skips = [x]
        
        for layer in self.down_blocks:
            if isinstance(layer, UNetBlock):
                x = layer(x, t_emb)
                skips.append(x)
            else:
                x = layer(x)
                
        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)
        
        # Up blocks
        for layer in self.up_blocks:
            if isinstance(layer, nn.Upsample2d):
                x = layer(x)
            else:
                # UNetBlock
                skip = skips.pop()
                x = jnp.concatenate([x, skip], axis=-1)
                x = layer(x, t_emb)
                
        return self.conv_out(x)
