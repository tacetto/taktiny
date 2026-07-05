from dataclasses import dataclass
from .model import PretrainedModel
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    use_mlp_bias: bool
    use_attention_bias: bool
    use_qkv_norm: bool
    use_tie_lm_head: bool
    use_bias: bool
    num_hidden_layers: int
    activation_fn: str = 'silu'
    
    norm_type: str = 'rmsnorm'
    norm_eps: float = 1e-6

    posemb_type: str = None
    posemb_kwargs: dict = None
    
    dtype: str = None

    @classmethod
    def from_pretrained(cls, path_or_repo: str, local: bool = False) -> 'TransformerModelConfig':
        from .model import PretrainedModel
        hf_config = PretrainedModel.load_config(path_or_repo, local=local)
        
        if hf_config is None:
            raise ValueError(f"Failed to load config from {path_or_repo}")
            
        head_dim = hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"])
        
        return cls(
            vocab_size=hf_config["vocab_size"],
            hidden_size=hf_config["hidden_size"],
            intermediate_size=hf_config["intermediate_size"],
            num_heads=hf_config["num_attention_heads"],
            head_dim=head_dim,
            num_kv_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
            use_mlp_bias=hf_config.get("mlp_bias", False),
            use_attention_bias=hf_config.get("attention_bias", False),
            use_qkv_norm=hf_config.get("model_type") == "qwen3",
            use_tie_lm_head=hf_config.get("tie_word_embeddings", False),
            use_bias=False,
            num_hidden_layers=hf_config["num_hidden_layers"],
            activation_fn=hf_config.get("hidden_act", "silu"),
            norm_type="rmsnorm",
            norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            posemb_type="rotary",
            posemb_kwargs={
                "max_position_embeddings": hf_config.get("max_position_embeddings", 4096),
                "base": hf_config.get("rope_theta", 10000.0),
                "rope_scaling": hf_config.get("rope_scaling", None)
            }
        )


@dataclass(frozen=True)
class TransformerAuxilialary:
    key_cache: jax.Array
    value_cache: jax.Array
    position_idx: jax.Array
    is_causal: bool


class CausalLM(PretrainedModel):
    def _sample(self, logits: jax.Array, temperature: float, top_k: int, top_p: float, key: jax.Array) -> jax.Array:
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
    def _decode_step(self, carry, max_seq_len: int = None, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0):
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
        prng_key: jax.Array = None
    ) -> jax.Array:
        if prng_key is None:
            prng_key = jax.random.PRNGKey(42)
            
        batch_size, seq_len = input_ids.shape
        max_seq_len = seq_len + max_new_tokens
        
        # We rely on the subclass having self.config
        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_kv_heads
        head_dim = self.config.head_dim
        
        # 1. Initialize KV Cache with the model's actual dtype (e.g. bfloat16)
        leaves = jax.tree_util.tree_leaves(self)
        arrays = [leaf for leaf in leaves if getattr(leaf, 'dtype', None) is not None]
        model_dtype = arrays[0].dtype if arrays else jnp.float32
        
        k_cache = jnp.zeros((num_layers, batch_size, max_seq_len, num_kv_heads, head_dim), dtype=model_dtype)
        v_cache = jnp.zeros((num_layers, batch_size, max_seq_len, num_kv_heads, head_dim), dtype=model_dtype)
        
        # 2. Prefill phase
        position_idx = jnp.array(0, dtype=jnp.int32)
        aux = TransformerAuxilialary(
            key_cache=k_cache,
            value_cache=v_cache,
            position_idx=position_idx,
            is_causal=True # JAX native dot_product_attention handles causal masking if True
        )
        
        logits, aux = self(input_ids, attention_mask=None, aux=aux)
        next_token_logits = logits[:, -1, :]
        
        prng_key, subkey = jax.random.split(prng_key)
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
        initial_carry = (next_token, aux.key_cache, aux.value_cache, initial_pos, prng_key)
        _, new_tokens = jax.lax.scan(scan_decode_step, initial_carry, None, length=max_new_tokens - 1)
        
        # new_tokens is shape [max_new_tokens - 1, batch_size, 1] -> swap to [batch_size, max_new_tokens - 1]
        new_tokens = new_tokens.swapaxes(0, 1).reshape(batch_size, -1)
        
        return jnp.concatenate([input_ids, next_token, new_tokens], axis=1)
