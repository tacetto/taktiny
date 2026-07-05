import jax
from dataclasses import replace
import jax.numpy as jnp

from ..configs.transformer import CausalLM, TransformerAuxilialary
from ...modules import RotaryEmbedding, GateMLP, Attention
from .... import nn, Rngs


class QwenTransformerBlock(nn.Module):
    def __init__(self, config, seed: Rngs):
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
            pos_emb=RotaryEmbedding(
                config.head_dim, **(getattr(config, 'posemb_kwargs', {}) or {})
            ),
            bias=config.use_attention_bias,
            use_qkv_norm=config.use_qkv_norm,
            dtype=config.dtype,
            seed=seed
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        activation_fn = getattr(jax.nn, config.activation_fn)
        self.mlp = GateMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=activation_fn,
            bias=config.use_mlp_bias,
            dtype=config.dtype,
            seed=seed
        )

    def __call__(
        self, 
        x: jax.Array, 
        attention_mask: jax.Array = None, 
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        is_causal: bool = False
    ):
        attn_out, new_cache = self.attn(
            self.norm1(x), 
            attention_mask=attention_mask, 
            is_causal=is_causal, 
            kv_cache=kv_cache, 
            position_idx=position_idx
        )
        
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class QwenModel(nn.Module):
    def __init__(self, config, seed: Rngs):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, seed=seed)
        self.layers = nn.SequentialStack(
            QwenTransformerBlock, config, num_stack=config.num_hidden_layers, seed=seed
        )
        self.norm = nn.RMSNorm(config.hidden_size)

    def __call__(self, x: jax.Array, attention_mask: jax.Array = None, aux: TransformerAuxilialary = None):
        x = self.embed_tokens(x)
        
        has_cache = aux is not None and aux.key_cache is not None
        if has_cache:
            carry = (x, (aux.key_cache, aux.value_cache), 0)
        else:
            carry = (x, None, 0)
            
        position_idx = getattr(aux, 'position_idx', None)
        is_causal = getattr(aux, 'is_causal', False)
        
        def forward_stack(layer, carry):
            h, full_kv_cache, layer_idx = carry
            
            current_kv = (full_kv_cache[0][layer_idx], full_kv_cache[1][layer_idx]) if has_cache else None
            
            # TODO: Handle sliding window attention per layer_types if config supports it.
            h, next_cache = layer(
                h, 
                attention_mask=attention_mask, 
                kv_cache=current_kv,
                position_idx=position_idx,
                is_causal=is_causal
            )
            
            if has_cache:
                full_kv_cache = (
                    full_kv_cache[0].at[layer_idx].set(next_cache[0]),
                    full_kv_cache[1].at[layer_idx].set(next_cache[1])
                )
                
            return h, full_kv_cache, layer_idx + 1
            
        x, final_kv_cache, _ = self.layers(forward_stack, carry)
        
        if has_cache:
            aux = replace(aux, 
                key_cache=final_kv_cache[0], 
                value_cache=final_kv_cache[1]
            )
            
        return self.norm(x), aux


class QwenCausalLM(CausalLM):
    def __init__(self, config, seed: Rngs):
        self.config = config
        self.model = QwenModel(config, seed=seed)
        self.lm_head = nn.build_linear(config.hidden_size, config.vocab_size, bias=config.use_bias, dtype=config.dtype, seed=seed)

    @classmethod
    def from_pretrained(cls, path_or_repo, local=False, **kwargs):
        from ..configs.transformer import TransformerModelConfig
        config = TransformerModelConfig.from_pretrained(path_or_repo, local=local)
        module_map = {
            "input_layernorm": "norm1",
            "post_attention_layernorm": "norm2",
            "self_attn": "attn",
            "embed_tokens.weight": "embed_tokens.embedding",
        }
        return super().from_pretrained(
            path_or_repo, 
            config=config, 
            module_map=module_map, 
            local=local, 
            **kwargs
        )

    def __call__(self, x: jax.Array, attention_mask: jax.Array = None, aux: TransformerAuxilialary = None):
        x, aux = self.model(x, attention_mask=attention_mask, aux=aux)
        return self.lm_head(x), aux