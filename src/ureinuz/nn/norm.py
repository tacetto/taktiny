import jax
import jax.numpy as jnp
from .module import Module, Parameter

class LayerNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(jnp.ones((hidden_size,), dtype=jnp.float32))
        self.bias = Parameter(jnp.zeros((hidden_size,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return x_norm * self.weight.value + self.bias.value

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"

class RMSNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(jnp.ones((hidden_size,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(var + self.eps)
        return x_norm * self.weight.value

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"

class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = Parameter(jnp.ones((num_channels,), dtype=jnp.float32))
        self.bias = Parameter(jnp.zeros((num_channels,), dtype=jnp.float32))

        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (B, H, W, C) or (H, W, C)
        is_unbatched = x.ndim == 3
        if is_unbatched:
            x = jnp.expand_dims(x, 0)
            
        B, H, W, C = x.shape
        G = self.num_groups
        D = C // G
        
        # Reshape to (B, H, W, G, D)
        x_reshaped = x.reshape((B, H, W, G, D))
        
        # Calculate mean and variance over H, W, and D
        # We want to normalize over the spatial dimensions and the channel group
        mean = jnp.mean(x_reshaped, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x_reshaped, axis=(1, 2, 4), keepdims=True)
        
        x_norm = (x_reshaped - mean) * jax.lax.rsqrt(var + self.eps)
        x_norm = x_norm.reshape((B, H, W, C))
        
        out = x_norm * self.weight.value + self.bias.value
        
        if is_unbatched:
            out = jnp.squeeze(out, 0)
            
        return out
        
    def extra_repr(self):
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"
