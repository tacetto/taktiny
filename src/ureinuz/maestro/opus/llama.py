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
"""Llama architectures"""

from __future__ import annotations

import jax.numpy as jnp

from ureinuz.maestro._livret import repertoire
from ureinuz.cosettes.transformer._common import TransformerCausalLM
from ureinuz.cosettes.transformer.llama import LlamaTransformerBlock
from ureinuz.maestro._config import ModelConfig
from ureinuz.utils.typing import ShardMode
from ureinuz import nn
from ureinuz.utils.sharding import create_sharding


class LlamaCausalLM(TransformerCausalLM):
    # Default Megatron-LM Tensor Parallelism rules
    default_sharding_rules = [
        # (Logical Axis Name, Physical Mesh Axis Name)
        
        # --- Weight Axes ---
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        
        # --- Activation Axes ---
        ('batch', 'fsdp'),
        ('sequence', None),
    ]

    def __init__(
        self, 
        config, 
        rngs: nn.Rngs = None, 
        mesh=None, 
        sharding_rules=None
    ):
        if rngs is None:
            rngs = nn.Rngs(0)
            
        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)

        assert hasattr(config, 'vocab_size')
        assert hasattr(config, 'hidden_size')
        assert hasattr(config, 'rms_norm_eps')
        
        super().__init__(
            config,
            LlamaTransformerBlock,
            embedder=lambda c, s: nn.Embedding(c.vocab_size, c.hidden_size, rngs=s),
            lm_head=lambda c, s: nn.Linear(
                c.hidden_size, 
                c.vocab_size, 
                bias=False, 
                dtype=jnp.float32, 
                rngs=s, 
                axis_names=('embed', 'vocab'), 
                shard_mode=shard_mode, 
                quant=quant, 
                dot_general=dot_general
            ),
            rngs=rngs
        )
        
        if hasattr(self.embed_tokens, 'embedding'):
            self.embed_tokens.embedding.axis_names = ('vocab', 'embed')
            
        self.norm = nn.RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps, 
            dtype=jnp.float32, 
            shard_mode=shard_mode, 
            axis_names=('embed',)
        )

        if sharding_rules is None:
            sharding_rules = self.default_sharding_rules

        self.out_sharding = None
        if mesh is not None and shard_mode == ShardMode.EXPLICIT:
            self.out_sharding = create_sharding(
                mesh, 
                ('batch', 'sequence', 'embed'), 
                rules=sharding_rules
            )

    def __call__(self, x, attention_mask = None, aux=None):
        return super().__call__(x, attention_mask, aux)

    @classmethod
    def from_pretrained(cls, path_or_repo, mesh=None, sharding_rules=None, local=False, **kwargs):
        # Load config
        config = ModelConfig.load_config(path_or_repo, local=local)
        
        # We define how HuggingFace weights map to our components using our new Tuple format
        module_map = [
            ("model.", ""),
            ("input_layernorm", "norm1"),
            ("post_attention_layernorm", "norm2"),
            ("self_attn", "attn"),
            ("embed_tokens.weight", "embed_tokens.embedding"),
        ]
        
        # Call the base class safetensors loader
        # (Note: PretrainedModel.from_pretrained will need to be updated to pass mesh and sharding_rules down!)
        return super().from_pretrained(
            path_or_repo, 
            config=config, 
            module_map=module_map, 
            local=local, 
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )

repertoire.register('LlamaForCausalLM', LlamaCausalLM)

__all__ = ['LlamaCausalLM']