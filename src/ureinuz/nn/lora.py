import jax
import jax.numpy as jnp
import re
from .module import Module, Parameter, iter_children
from .linear import Linear
from ..rng import Rngs

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
        is_stacked = sample_param is not None and len(sample_param.shape) == 3
        num_layers = sample_param.shape[0] if is_stacked else None
            
        w_key = seed()
        if is_stacked:
            self.lora_A = Parameter(jax.random.normal(w_key, (num_layers, self.in_features, self.rank), dtype=jnp.float32) * (1 / self.in_features))
            self.lora_B = Parameter(jnp.zeros((num_layers, self.rank, self.out_features), dtype=jnp.float32))
        else:
            self.lora_A = Parameter(jax.random.normal(w_key, (self.in_features, self.rank), dtype=jnp.float32) * (1 / self.in_features))
            self.lora_B = Parameter(jnp.zeros((self.rank, self.out_features), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        # Stop gradient on the base layer so JAX doesn't try to compute gradients for the frozen quantized weights
        # Actually JAX will only compute gradients for what we ask, but stop_gradient is safer memory-wise
        base_out = jax.lax.stop_gradient(self.base_layer(x))
        lora_out = jnp.dot(jnp.dot(x, self.lora_A.value.astype(x.dtype)), self.lora_B.value.astype(x.dtype)) * self.scaling
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
