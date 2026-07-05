from ...nn.module import Module

import jax, jax.numpy as jnp
from typing import Callable
from ... import nn, Rngs

class GateMLP(Module):
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        activation: Callable | str = jax.nn.silu,
        bias: bool = False, 
        dtype: str = None,
        seed: Rngs = None
    ):
        self.activation = activation if isinstance(activation, Callable) else getattr(jax.nn, activation)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype, seed=seed)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, dtype=dtype, seed=seed)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, seed=seed)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class FusedGateMLP(Module):
    """
    GateMLP where the gate and up projections are fused into a single linear layer.
    """
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        activation: Callable | str = jax.nn.silu,
        bias: bool = False, 
        dtype: str = None,
        seed: Rngs = None
    ):
        self.activation = activation if isinstance(activation, Callable) else getattr(jax.nn, activation)
        
        self.linear_in = nn.Linear(hidden_size, intermediate_size * 2, bias=bias, dtype=dtype, seed=seed)
        self.linear_out = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, seed=seed)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear_in(x)
        x, gate = jnp.split(x, 2, axis=-1)
        return self.linear_out(x * self.activation(gate))