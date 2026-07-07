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
"""Common base modules for transformer architectures"""

from __future__ import annotations

from ureinuz import nn
from ureinuz.cosettes._base import PretrainedModel
from ureinuz.maestro._config import ModelConfig

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from dataclasses import replace


@dataclass(frozen=True)
class TransformerAuxilialary:
    key_cache: jax.Array
    value_cache: jax.Array
    position_idx: jax.Array
    is_causal: bool


class CausalLM(PretrainedModel):
    def _sample(
        self, 
        logits: jax.Array, 
        temperature: float, 
        top_k: int, 
        top_p: float, 
        key: jax.Array
    ) -> jax.Array:
        logits = logits / jnp.maximum(temperature, 1e-5)
        
        if top_k > 0:
            top_k_logits, _ = jax.lax.top_k(logits, top_k)
            min_top_k = top_k_logits[:, -1:]
            logits = jnp.where(logits >= min_top_k, logits, -jnp.inf)
            
        if top_p < 1.0:
            sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token that crosses the threshold
            sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
            sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
            
            # Map back to original order
            indices_to_remove = jnp.empty_like(sorted_indices_to_remove)
            indices_to_remove = indices_to_remove.at[
                jnp.arange(logits.shape[0])[:, None], sorted_indices
            ].set(sorted_indices_to_remove)
            
            logits = jnp.where(indices_to_remove, -jnp.inf, logits)
            
        return jax.random.categorical(key, logits)[:, None]

    from functools import partial

    @partial(jax.jit, static_argnames=['max_seq_len', 'top_k', 'top_p'])
    def _decode_step(
        self, carry, 
        max_seq_len: int = None, 
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 1.0
    ):
        token, k_cache, v_cache, pos, rng = carry
        
        decode_aux = TransformerAuxilialary(
            key_cache=k_cache,
            value_cache=v_cache,
            position_idx=pos,
            is_causal=False
        )
        
        # Mask to attend to all past tokens up to pos
        mask = jnp.arange(max_seq_len) <= pos
        mask = mask.reshape(1, 1, 1, max_seq_len)
        
        step_logits, decode_aux = self(token, attention_mask=mask, aux=decode_aux)
        
        rng, subkey = jax.random.split(rng)
        next_t = self._sample(step_logits[:, -1, :], temperature, top_k, top_p, subkey)
        
        return (next_t, decode_aux.key_cache, decode_aux.value_cache, pos + 1, rng), next_t

    def generate(
        self, 
        input_ids: jax.Array, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        key: jax.Array = None
    ) -> jax.Array:
        if key is None:
            key = jax.random.key(42)
            
        batch_size, seq_len = input_ids.shape
        max_seq_len = seq_len + max_new_tokens

        assert (num_layers := self.config.num_hidden_layers) is not None, \
            'Cannot specified `num_hidden_layers` for key value cache generation.'

        assert (num_attention_heads := self.config.num_attention_heads) is not None, \
            'Cannot specified `num_attention_heads` for key value cache generation.'
            
        assert (num_kv_heads := self.config.num_key_value_heads) is not None, \
            'Cannot specified `num_key_value_heads` for key value cache generation.'
            
        assert (hidden_size := self.config.hidden_size) is not None, \
            'Cannot specified `head_dim` for key value cache generation'
            
        head_dim = hidden_size // num_attention_heads
        
        # Initialize KV Cache with the model's actual dtype (e.g. bfloat16)
        leaves = jax.tree_util.tree_leaves(self)
        arrays = [leaf for leaf in leaves if getattr(leaf, 'dtype', None) is not None]
        model_dtype = arrays[0].dtype if arrays else jnp.float32
        
        k_cache = jnp.zeros((num_layers, batch_size, max_seq_len, num_kv_heads, head_dim), dtype=model_dtype)
        v_cache = jnp.zeros((num_layers, batch_size, max_seq_len, num_kv_heads, head_dim), dtype=model_dtype)
        
        # Prefill phase
        position_idx = jnp.array(0, dtype=jnp.int32)
        aux = TransformerAuxilialary(
            key_cache=k_cache,
            value_cache=v_cache,
            position_idx=position_idx,
            is_causal=True # JAX native dot_product_attention handles causal masking if True
        )
        
        logits, aux = self(input_ids, attention_mask=None, aux=aux)
        next_token_logits = logits[:, -1, :]
        
        key, subkey = jax.random.split(key)
        next_token = self._sample(next_token_logits, temperature, top_k, top_p, subkey)
        
        # 3. Decoding phase
        def scan_decode_step(carry, _):
            return self._decode_step(
                carry, 
                max_seq_len=max_seq_len, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            
        initial_pos = jnp.array(seq_len, dtype=jnp.int32)
        initial_carry = (next_token, aux.key_cache, aux.value_cache, initial_pos, key)
        _, new_tokens = jax.lax.scan(scan_decode_step, initial_carry, None, length=max_new_tokens - 1)
        
        # new_tokens is shape [max_new_tokens - 1, batch_size, 1] -> swap to [batch_size, max_new_tokens - 1]
        new_tokens = new_tokens.swapaxes(0, 1).reshape(batch_size, -1)
        
        return jnp.concatenate([input_ids, next_token, new_tokens], axis=1)


class MaskedLM(PretrainedModel):
    pass


class Seq2SeqLM(PretrainedModel):
    pass


class PrefixLM(PretrainedModel):
    pass


class SeqClassificationLM(PretrainedModel):
    pass


class TokenClassificationLM(PretrainedModel):
    pass

class QALM(PretrainedModel):
    pass


class ImgClassificationVM(PretrainedModel):
    pass


class ObjDetectionVM(PretrainedModel):
    pass


class SemanticSegmentationVM(PretrainedModel):
    pass


class AudioClassificationAM(PretrainedModel):
    pass


class CTCAM(PretrainedModel):
    pass


default_embedder = lambda c, s: nn.Embedding(c.vocab_size, c.hidden_size, rngs=s)
default_lm_head = lambda c, s: nn.Linear(
    c.hidden_size, 
    c.vocab_size, 
    bias=False, 
    dtype=jnp.float32, 
    rngs=s
)
class TransformerCausalLM(CausalLM):
    def __init__(
        self, config, 
        decoder, 
        embedder =  default_embedder,
        lm_head = default_lm_head, 
        rngs: nn.Rngs = None
    ):
        if rngs is None:
            rngs = nn.Rngs(0)
            
        self.config = config

        self.embed_tokens = embedder(config, rngs)
        self.layers = nn.SequentialStack(
            decoder, config, rngs, 
            num_stack=config.num_hidden_layers
        )
        self.lm_head = lm_head(config, rngs)


    # def __call__(self, *args, **kwds):
    #     return super().__call__(*args, **kwds)

    def __call__(
        self, x: jax.Array, 
        attention_mask: jax.Array = None, 
        aux = None
    ):
        x = self.embed_tokens(x)
        
        has_cache = aux is not None and getattr(aux, 'key_cache', None) is not None
        if has_cache:
            carry = (x, (aux.key_cache, aux.value_cache), 0)

        else:
            carry = (x, None, 0)
            
        position_idx = getattr(aux, 'position_idx', None)
        is_causal = getattr(aux, 'is_causal', False)
        
        def forward_stack(layer, carry):
            h, full_kv_cache, layer_idx = carry
            current_kv = (full_kv_cache[0][layer_idx], full_kv_cache[1][layer_idx]) if has_cache else None
            
            h, next_cache = layer(
                h, 
                attention_mask=attention_mask, 
                kv_cache=current_kv,
                position_idx=position_idx,
                is_causal=is_causal,
                out_sharding=self.out_sharding
            )
            
            if has_cache:
                new_k_cache = full_kv_cache[0].at[layer_idx].set(next_cache[0])
                new_v_cache = full_kv_cache[1].at[layer_idx].set(next_cache[1])
                full_kv_cache = (new_k_cache, new_v_cache)
                
            return h, full_kv_cache, layer_idx + 1
            
        x, final_kv_cache, _ = self.layers(forward_stack, carry)
        
        x = self.norm(x, out_sharding=self.out_sharding)
        logits = self.lm_head(x, out_sharding=self.out_sharding)
        
        if has_cache:
            aux = replace(
                aux, 
                key_cache=final_kv_cache[0], 
                value_cache=final_kv_cache[1]
            )
            
        return logits, aux

    @classmethod
    def from_pretrained(cls, path_or_repo, config, module_map=None, **kwargs):
        module_map = module_map or []
        if isinstance(module_map, dict):
            module_map = list(module_map.items())
            
        tied = getattr(config, 'tie_word_embeddings', False) or getattr(config, 'use_tie_lm_head', False)
        
        new_module_map = []
        for rule in module_map:
            if len(rule) == 2:
                source, target = rule
                if tied and target == "embed_tokens.embedding":
                    new_module_map.append((source, ["embed_tokens.embedding", "lm_head.weight"], lambda x: [x, x]))
                    continue
            new_module_map.append(rule)
            
        return super().from_pretrained(path_or_repo, config=config, module_map=new_module_map, **kwargs)