import jax
import jax.numpy as jnp
from ... import nn, Rngs

class AdaLayerNorm(nn.Module):
    """
    Generic Adaptive Layer Normalization.
    Computes a parameter-free normalization on `x` and projects a condition vector `vec` 
    into a desired dimension. The caller is responsible for splitting the modulation 
    output into scale/shift/gate chunks as needed for their specific architecture.
    """
    def __init__(
        self, 
        embedding_dim: int, 
        out_dim: int, 
        norm_type: str = "layer_norm",
        eps: float = 1e-6, 
        seed: Rngs = None
    ):
        self.eps = eps
        self.norm_type = norm_type
        
        # Linear projection from the condition vector (e.g. time/text)
        self.linear = nn.Linear(embedding_dim, out_dim, seed=seed)
        
    def __call__(self, x: jax.Array, vec: jax.Array) -> tuple[jax.Array, jax.Array]:
        # 1. Parameter-free normalization
        if self.norm_type == "layer_norm":
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            normed_x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        elif self.norm_type == "rms_norm":
            var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
            normed_x = x * jax.lax.rsqrt(var + self.eps)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
            
        # 2. Compute modulation (typically SiLU is applied before projection in DiTs)
        modulation = self.linear(jax.nn.silu(vec))
        
        return normed_x, modulation

class AdaLayerNormChunks(AdaLayerNorm):
    """
    Adaptive Layer Normalization that generates multiple chunks (e.g. 6 chunks for shift/scale/gate of Q/K/V).
    """
    def __init__(
        self, 
        embedding_dim: int, 
        out_dim: int,
        num_chunks: int,
        norm_type: str = "layer_norm",
        eps: float = 1e-6, 
        seed: Rngs = None
    ):
        self.num_chunks = num_chunks
        super().__init__(
            embedding_dim=embedding_dim,
            out_dim=out_dim * num_chunks,
            norm_type=norm_type,
            eps=eps,
            seed=seed
        )
        
    def __call__(self, x: jax.Array, vec: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        normed_x, modulation = super().__call__(x, vec)
        
        # Split modulation into chunks along the last dimension
        chunks = tuple(jnp.split(modulation, self.num_chunks, axis=-1))
        
        # Return the parameter-free normalized x and the chunks tuple
        return normed_x, chunks
