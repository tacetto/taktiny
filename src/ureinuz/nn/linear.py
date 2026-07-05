import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_uniform, zeros
from .module import Module, Parameter
from ..rng import Rngs

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, dtype: str = None, seed: Rngs = None, initializer = lecun_uniform()):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if seed is None:
            raise ValueError("A seed must be provided to initialize Linear layer")
            
        if dtype is None:
            dtype = jnp.float32
        elif isinstance(dtype, str):
            dtype = getattr(jnp, dtype.lower(), jnp.float32)

        w_key = seed()
        self.weight = Parameter(initializer(w_key, (in_features, out_features), dtype))

        if bias:
            b_key = seed()
            self.bias = Parameter(jnp.zeros((out_features,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        out = jnp.dot(x, self.weight.value)
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}"

class LinearFP8(Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 128, *, bias: bool = False, seed: Rngs = None):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_bias = bias
        
        assert in_features % block_size == 0
        num_blocks = in_features // block_size
        
        self.weights_q = Parameter(jnp.zeros((in_features, out_features), dtype=jnp.float8_e4m3fn))
        self.scales_q = Parameter(jnp.zeros((num_blocks, out_features), dtype=jnp.float8_e4m3fn))
        self.scale_of_scales = Parameter(jnp.zeros((1, out_features), dtype=jnp.float32))
        
        if bias:
            self.bias = Parameter(jnp.zeros((out_features,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        # 1. Unpack scales
        scales = self.scales_q.value.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        # Repeat scales along the block dimension
        scales = jnp.repeat(scales, self.block_size, axis=0)
        
        # 2. Unpack weights
        weights = self.weights_q.value.astype(x.dtype) * scales
        
        # 3. Dense Projection
        out = jnp.dot(x, weights)
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, FP8 DoubleQuant(b={self.block_size})"


class LinearINT8(Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 128, *, bias: bool = False, seed: Rngs = None):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_bias = bias
        
        assert in_features % block_size == 0
        num_blocks = in_features // block_size
        
        self.weights_q = Parameter(jnp.zeros((in_features, out_features), dtype=jnp.int8))
        self.scales_q = Parameter(jnp.zeros((num_blocks, out_features), dtype=jnp.int8))
        self.scale_of_scales = Parameter(jnp.zeros((1, out_features), dtype=jnp.float32))
        
        if bias:
            self.bias = Parameter(jnp.zeros((out_features,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        scales = self.scales_q.value.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        weights = self.weights_q.value.astype(x.dtype) * scales
        
        out = jnp.dot(x, weights)
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, INT8 DoubleQuant(b={self.block_size})"


class LinearINT4(Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 128, *, bias: bool = False, seed: Rngs = None):
        self.in_features = in_features
        self.out_features = out_features
        if in_features % block_size != 0:
            block_size = 64
        assert in_features % block_size == 0
        self.block_size = block_size
        self.use_bias = bias
        
        assert out_features % 2 == 0
        num_blocks = in_features // block_size
        
        # Pack 2 weights along the out_features dimension into 1 uint8
        self.weights_q = Parameter(jnp.zeros((in_features, out_features // 2), dtype=jnp.uint8))
        self.scales_q = Parameter(jnp.zeros((num_blocks, out_features // 2), dtype=jnp.uint8))
        self.scale_of_scales = Parameter(jnp.zeros((1, out_features), dtype=jnp.float32))
        
        if bias:
            self.bias = Parameter(jnp.zeros((out_features,), dtype=jnp.float32))

    def _unpack_int4(self, packed: jax.Array) -> jax.Array:
        # Expand packed [..., N] into [..., N, 2] then reshape to [..., N*2]
        # Low nibble: val & 0x0F, High nibble: val >> 4
        # Since standard quantization usually maps -8 to 7, we subtract 8.
        low = (packed & 0x0F).astype(jnp.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(jnp.int8) - 8
        unpacked = jnp.stack([low, high], axis=-1)
        return unpacked.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))

    def __call__(self, x: jax.Array) -> jax.Array:
        scales_unpacked = self._unpack_int4(self.scales_q.value)
        scales = scales_unpacked.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        
        weights_unpacked = self._unpack_int4(self.weights_q.value)
        weights = weights_unpacked.astype(x.dtype) * scales
        
        out = jnp.dot(x, weights)
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, INT4 DoubleQuant(b={self.block_size})"


class LinearFP4(Module):
    def __init__(self, in_features: int, out_features: int, block_size: int = 128, *, bias: bool = False, seed: Rngs = None):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_bias = bias
        
        assert in_features % block_size == 0
        assert out_features % 2 == 0
        num_blocks = in_features // block_size
        
        self.weights_q = Parameter(jnp.zeros((in_features, out_features // 2), dtype=jnp.uint8))
        self.scales_q = Parameter(jnp.zeros((num_blocks, out_features // 2), dtype=jnp.uint8))
        self.scale_of_scales = Parameter(jnp.zeros((1, out_features), dtype=jnp.float32))
        
        if bias:
            self.bias = Parameter(jnp.zeros((out_features,), dtype=jnp.float32))

    def _unpack_fp4(self, packed: jax.Array) -> jax.Array:
        # To unpack custom FP4 (E2M1), we would use a Look-Up Table (LUT) mapped from the 16 possible values.
        # For pure simplicity here, we assume the LUT values are standard NF4/FP4 mappings.
        # This is a placeholder standard uniform mapping for now to maintain the structural architecture.
        # (A true FP4 LUT will be supplied by the Quantizer!)
        low = (packed & 0x0F).astype(jnp.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(jnp.int8) - 8
        unpacked = jnp.stack([low, high], axis=-1)
        return unpacked.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))

    def __call__(self, x: jax.Array) -> jax.Array:
        scales_unpacked = self._unpack_fp4(self.scales_q.value)
        scales = scales_unpacked.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        
        weights_unpacked = self._unpack_fp4(self.weights_q.value)
        weights = weights_unpacked.astype(x.dtype) * scales
        
        out = jnp.dot(x, weights)
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, FP4 DoubleQuant(b={self.block_size})"


def build_linear(in_features: int, out_features: int, dtype: str = None, **kwargs) -> Module:
    if dtype is None:
        return Linear(in_features, out_features, dtype=jnp.float32, **kwargs)
        
    dtype_str = str(dtype).lower()
    if dtype_str == "fp8":
        return LinearFP8(in_features, out_features, **kwargs)
    elif dtype_str == "int8":
        return LinearINT8(in_features, out_features, **kwargs)
    elif dtype_str == "fp4":
        return LinearFP4(in_features, out_features, **kwargs)
    elif dtype_str == "int4":
        return LinearINT4(in_features, out_features, **kwargs)
        
    return Linear(in_features, out_features, dtype=dtype, **kwargs)
