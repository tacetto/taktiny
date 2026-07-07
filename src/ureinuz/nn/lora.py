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
"""LoRA modules"""

import jax
import jax.numpy as jnp
import re

from ureinuz.nn.module import Module, Parameter, iter_children
from ureinuz.nn.rng import Rngs

class LoRALinear(Module):
    def __init__(self, base_layer: Module, rank: int, alpha: float, seed: Rngs):
        self.base_layer = base_layer
        self.in_features = getattr(base_layer, 'in_features', None)
        self.out_features = getattr(base_layer, 'out_features', None)
        self.rank = rank
        self.scaling = alpha / rank
        
        if self.in_features is None or self.out_features is None:
            raise ValueError("Base layer must have in_features and out_features attributes.")
            
        # Detect if base_layer is stacked (used inside SequentialStack)
        sample_param = getattr(base_layer, 'weight', getattr(base_layer, 'weights_q', None))
        if sample_param is not None:
            expected_dims = len(self.in_features) + len(self.out_features)
            # If it's dynamically quantized, weights_q might have fewer dims due to packing, but it still has an extra leading dim if stacked.
            # A safer check is if it has more dims than expected, or we can just check if it's 1 dim longer than expected.
            is_stacked = len(sample_param.shape) > expected_dims or (len(sample_param.shape) > 2 and 'weights_q' in vars(base_layer))
        else:
            is_stacked = False
            
        num_layers = sample_param.shape[0] if is_stacked else None
            
        import math
        self.in_features_flat = math.prod(self.in_features)
        self.out_features_flat = math.prod(self.out_features)
            
        w_key = seed()
        if is_stacked:
            self.lora_A = Parameter(jax.random.normal(w_key, (num_layers, self.in_features_flat, self.rank), dtype=jnp.float32) * (1 / self.in_features_flat))
            self.lora_B = Parameter(jnp.zeros((num_layers, self.rank, self.out_features_flat), dtype=jnp.float32))
        else:
            self.lora_A = Parameter(jax.random.normal(w_key, (self.in_features_flat, self.rank), dtype=jnp.float32) * (1 / self.in_features_flat))
            self.lora_B = Parameter(jnp.zeros((self.rank, self.out_features_flat), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        base_out = jax.lax.stop_gradient(self.base_layer(x))
        # Flatten the input's feature dimensions to match in_features_flat
        in_dims = len(self.in_features)
        x_flat = x.reshape(x.shape[:-in_dims] + (self.in_features_flat,))
        
        lora_out = jnp.dot(jnp.dot(x_flat, self.lora_A.value.astype(x.dtype)), self.lora_B.value.astype(x.dtype)) * self.scaling
        
        # Reshape the output to match base_out shape
        out_dims = self.out_features
        lora_out = lora_out.reshape(lora_out.shape[:-1] + out_dims)
        
        return base_out + lora_out.astype(x.dtype)

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.scaling * self.rank}"


def inject_lora(model: Module, target_modules: list[str], rank: int, alpha: float, seed: Rngs, prefix: str = "") -> Module:
    """
    Recursively wraps target modules (e.g. ['q_proj', 'v_proj']) with LoRALinear.
    """
    for name, child in iter_children(model):
        full_name = f"{prefix}.{name}" if prefix else name
        
        # Check if the current child is a target module
        is_target = any(re.search(target, full_name) for target in target_modules)
        
        # We only wrap it if it has an __call__ and in_features/out_features
        if is_target and hasattr(child, 'in_features') and hasattr(child, 'out_features'):
            # Swap it!
            lora_layer = LoRALinear(base_layer=child, rank=rank, alpha=alpha, seed=seed)
            setattr(model, name, lora_layer)
        elif isinstance(child, Module):
            # Recurse
            inject_lora(child, target_modules, rank, alpha, seed, full_name)
            
    return model

apply_lora = inject_lora