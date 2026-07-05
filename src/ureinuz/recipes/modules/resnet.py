import jax
import jax.numpy as jnp
from ... import nn, Rngs

class ResnetBlock2D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = None, 
        time_emb_dim: int = None, 
        groups: int = 32,
        eps: float = 1e-5,
        seed: Rngs = None
    ):
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="SAME", seed=seed)
        
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, out_channels, seed=seed)
        else:
            self.time_proj = None
            
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="SAME", seed=seed)
        
        if self.in_channels != self.out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, seed=seed)
        else:
            self.res_conv = None

    def __call__(self, x: jax.Array, t_emb: jax.Array = None) -> jax.Array:
        # Pre-Norm architecture used by CompVis / Diffusers
        res = x if self.res_conv is None else self.res_conv(x)
        
        h = self.norm1(x)
        h = jax.nn.silu(h)
        h = self.conv1(h)
        
        if self.time_proj is not None and t_emb is not None:
            time_bias = self.time_proj(jax.nn.silu(t_emb))
            if time_bias.ndim == 1:
                # Unbatched: time_bias is (C,), h is (H, W, C)
                h = h + time_bias[None, None, :]
            else:
                # Batched: time_bias is (B, C), h is (B, H, W, C)
                h = h + time_bias[:, None, None, :]
                
        h = self.norm2(h)
        h = jax.nn.silu(h)
        h = self.conv2(h)
        
        return h + res
