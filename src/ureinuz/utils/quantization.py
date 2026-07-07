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
"""Utilities for Quantization"""

import numpy as np

def quantize_to_fp8_double(weight_tensor: np.ndarray, block_size: int = 128):
    """
    Quantizes a [in_features, out_features] float16/bfloat16 weight tensor 
    into a double-quantized FP8 format along the in_features dimension.
    
    Returns:
        weights_q: uint8 array representing float8_e4m3fn
        scales_q: uint8 array representing float8_e4m3fn
        scale_of_scales: float32 array
    """
    in_features, out_features = weight_tensor.shape
    assert in_features % block_size == 0
    num_blocks = in_features // block_size
    
    # 1. Reshape to block format
    weight_blocks = weight_tensor.reshape(num_blocks, block_size, out_features)
    
    # 2. First Pass: Block-wise Scales
    max_abs = np.max(np.abs(weight_blocks), axis=1, keepdims=True)
    # E4M3 has max representable value of 448.0
    scales = max_abs / 448.0
    scales = np.where(scales == 0, 1.0, scales)
    
    # 3. Quantize Weights to FP8
    weights_q_np = np.round(weight_blocks / scales)
    weights_q_np = np.clip(weights_q_np, -448.0, 448.0)
    
    scales = scales.squeeze(1) # [num_blocks, out_features]
    
    # 4. Second Pass: Scale of Scales
    scale_of_scales = np.max(np.abs(scales), axis=0, keepdims=True) # [1, out_features]
    scale_of_scales = scale_of_scales / 448.0
    scale_of_scales = np.where(scale_of_scales == 0, 1.0, scale_of_scales)
    
    # 5. Quantize Scales to FP8
    scales_q_np = np.round(scales / scale_of_scales)
    scales_q_np = np.clip(scales_q_np, -448.0, 448.0)
    
    # JAX will cast these to jnp.float8_e4m3fn during device_put
    return (
        weights_q_np.reshape(in_features, out_features), 
        scales_q_np, 
        scale_of_scales.astype(np.float32)
    )

def quantize_to_int8_double(weight_tensor: np.ndarray, block_size: int = 128):
    """
    Quantizes to double-quantized INT8 format.
    """
    in_features, out_features = weight_tensor.shape
    assert in_features % block_size == 0
    num_blocks = in_features // block_size
    
    weight_blocks = weight_tensor.reshape(num_blocks, block_size, out_features)
    
    # INT8 max representable value is 127
    max_abs = np.max(np.abs(weight_blocks), axis=1, keepdims=True)
    scales = max_abs / 127.0
    scales = np.where(scales == 0, 1.0, scales)
    
    weights_q_np = np.round(weight_blocks / scales)
    weights_q_np = np.clip(weights_q_np, -127, 127).astype(np.int8)
    
    scales = scales.squeeze(1)
    
    scale_of_scales = np.max(np.abs(scales), axis=0, keepdims=True)
    scale_of_scales = scale_of_scales / 127.0
    scale_of_scales = np.where(scale_of_scales == 0, 1.0, scale_of_scales)
    
    scales_q_np = np.round(scales / scale_of_scales)
    scales_q_np = np.clip(scales_q_np, -127, 127).astype(np.int8)
    
    return (
        weights_q_np.reshape(in_features, out_features), 
        scales_q_np, 
        scale_of_scales.astype(np.float32)
    )

def quantize_to_int4_double(weight_tensor: np.ndarray, block_size: int = 128):
    """
    Quantizes to double-quantized INT4 format (packed into uint8).
    """
    in_features, out_features = weight_tensor.shape
    assert in_features % block_size == 0
    assert out_features % 2 == 0
    num_blocks = in_features // block_size
    
    weight_blocks = weight_tensor.reshape(num_blocks, block_size, out_features)
    
    # INT4 (signed) range is [-8, 7]. We scale by max/7.
    max_abs = np.max(np.abs(weight_blocks), axis=1, keepdims=True)
    scales = max_abs / 7.0
    scales = np.where(scales == 0, 1.0, scales)
    
    weights_q_np = np.round(weight_blocks / scales)
    weights_q_np = np.clip(weights_q_np, -8, 7).astype(np.int8)
    
    scales = scales.squeeze(1)
    
    scale_of_scales = np.max(np.abs(scales), axis=0, keepdims=True)
    scale_of_scales = scale_of_scales / 7.0
    scale_of_scales = np.where(scale_of_scales == 0, 1.0, scale_of_scales)
    
    scales_q_np = np.round(scales / scale_of_scales)
    scales_q_np = np.clip(scales_q_np, -8, 7).astype(np.int8)
    
    # Pack weights: convert [-8, 7] to [0, 15] then pack low and high nibbles
    w_flat = weights_q_np.reshape(in_features, out_features) + 8
    w_low = w_flat[:, 0::2].astype(np.uint8)
    w_high = w_flat[:, 1::2].astype(np.uint8)
    packed_weights = (w_low & 0x0F) | ((w_high & 0x0F) << 4)
    
    # Pack scales
    s_flat = scales_q_np + 8
    s_low = s_flat[:, 0::2].astype(np.uint8)
    s_high = s_flat[:, 1::2].astype(np.uint8)
    packed_scales = (s_low & 0x0F) | ((s_high & 0x0F) << 4)
    
    return (
        packed_weights, 
        packed_scales, 
        scale_of_scales.astype(np.float32)
    )

def quantize_to_fp4_double(weight_tensor: np.ndarray, block_size: int = 128):
    """
    Fallback identical to INT4 packing structural layout until a LUT is integrated.
    """
    return quantize_to_int4_double(weight_tensor, block_size)
