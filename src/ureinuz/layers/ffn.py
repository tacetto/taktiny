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
"""Feed Forward Network modules"""

from __future__ import annotations

import jax, jax.numpy as jnp
from typing import Callable

from ureinuz.utils.typing import ShardMode
from ureinuz.nn.module import Module
from ureinuz import nn

class GateMLP(Module):
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        activation: Callable | str = jax.nn.silu,
        bias: bool = False, 
        dtype: str = None,
        rngs: nn.Rngs = None,
        gate_axis_names: tuple[str | None, ...] | None = None,
        up_axis_names: tuple[str | None, ...] | None = None,
        down_axis_names: tuple[str | None, ...] | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant=None,
        dot_general=None,
    ):
        self.activation = activation if isinstance(activation, Callable) else getattr(jax.nn, activation)
        
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, 
            bias=bias, dtype=dtype, rngs=rngs, 
            axis_names=gate_axis_names, 
            shard_mode=shard_mode, 
            quant=quant, dot_general=dot_general
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, 
            bias=bias, dtype=dtype, rngs=rngs, 
            axis_names=up_axis_names, 
            shard_mode=shard_mode, 
            quant=quant, dot_general=dot_general
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, 
            bias=bias, dtype=dtype, rngs=rngs, 
            axis_names=down_axis_names, 
            shard_mode=shard_mode, 
            quant=quant, dot_general=dot_general
        )
        
    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up, out_sharding=out_sharding)

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
        seed: nn.Rngs = None,
        linear_in_axis_names: tuple[str | None, ...] | None = None,
        linear_out_axis_names: tuple[str | None, ...] | None = None,
        shard_mode: ShardMode = ShardMode.AUTO,
        quant=None,
        dot_general=None,
    ):
        self.activation = activation if isinstance(activation, Callable) else getattr(jax.nn, activation)
        
        self.linear_in = nn.Linear(hidden_size, intermediate_size * 2, bias=bias, dtype=dtype, seed=seed, axis_names=linear_in_axis_names, shard_mode=shard_mode, quant=quant, dot_general=dot_general)
        self.linear_out = nn.Linear(intermediate_size, hidden_size, bias=bias, dtype=dtype, seed=seed, axis_names=linear_out_axis_names, shard_mode=shard_mode, quant=quant, dot_general=dot_general)
        
    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        x = self.linear_in(x)
        x, gate = jnp.split(x, 2, axis=-1)
        return self.linear_out(x * self.activation(gate), out_sharding=out_sharding)
    
class MLP(Module):
    ...