import jax
import jax.numpy as jnp
from .... import nn, Rngs
from ...modules import Attention
from ..configs.mmdit import MMDiTConfig
from ..configs.model import PretrainedModel

from ...modules.norm import AdaLayerNorm

class SiLU(nn.Module):
    def __call__(self, x):
        return jax.nn.silu(x)

class FluxAdaLN(AdaLayerNorm):
    def __init__(self, in_channels: int, emb_dim: int, seed: Rngs = None):
        # FLUX AdaLN generates 6 chunks! 
        # (shift_q, scale_q, shift_k, scale_k, shift_v, scale_v)
        super().__init__(
            embedding_dim=emb_dim,
            out_dim=in_channels * 6,
            norm_type="layer_norm",
            eps=1e-6,
            seed=seed
        )
        
    def __call__(self, x: jax.Array, vec: jax.Array) -> tuple[jax.Array, tuple]:
        normed_x, modulation = super().__call__(x, vec)
        
        # Split modulation into 6 chunks along the last dimension
        chunks = jnp.split(modulation, 6, axis=-1)
        
        # Return the parameter-free normalized x and the chunks!
        return normed_x, chunks

from ...modules.attention import JointAttention
from ...modules.posemb import RotaryEmbedding

class JointAttentionBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, seed: Rngs = None):
        self.config = config
        
        # AdaLN for entire block
        self.img_ada_ln = FluxAdaLN(config.hidden_size, config.hidden_size, seed=seed)
        self.txt_ada_ln = FluxAdaLN(config.hidden_size, config.hidden_size, seed=seed)
        
        head_dim = config.hidden_size // config.num_heads
        
        # Joint Attention
        self.attn = JointAttention(
            hidden_size1=config.hidden_size,
            hidden_size2=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            use_qkv_norm=True, # FLUX strictly uses QK-Normalization!
            pos_emb=RotaryEmbedding(head_dim),
            seed=seed
        )
        
class FluxMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int, seed: Rngs = None):
        # ffn_dim in FLUX 2 is 3072 * 3 = 9216. We project to 9216 * 2 = 18432.
        self.linear_in = nn.Linear(hidden_size, ffn_dim * 2, seed=seed)
        self.linear_out = nn.Linear(ffn_dim, hidden_size, seed=seed)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear_in(x)
        x, gate = jnp.split(x, 2, axis=-1)
        # FLUX uses GELU approximate tanh
        gate = jax.nn.gelu(gate, approximate=True)
        return self.linear_out(x * gate)

class JointAttentionBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, seed: Rngs = None):
        self.config = config
        
        # AdaLN for entire block
        self.img_ada_ln = FluxAdaLN(config.hidden_size, config.hidden_size, seed=seed)
        self.txt_ada_ln = FluxAdaLN(config.hidden_size, config.hidden_size, seed=seed)
        
        head_dim = config.hidden_size // config.num_heads
        
        # Joint Attention
        self.attn = JointAttention(
            hidden_size1=config.hidden_size,
            hidden_size2=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            use_qkv_norm=True, # FLUX strictly uses QK-Normalization!
            pos_emb=RotaryEmbedding(head_dim),
            seed=seed
        )
        
        # MLPs
        ffn_dim = config.hidden_size * 3
        self.img_mlp = FluxMLP(config.hidden_size, ffn_dim, seed=seed)
        self.txt_mlp = FluxMLP(config.hidden_size, ffn_dim, seed=seed)
        
    def __call__(self, img: jnp.ndarray, txt: jnp.ndarray, vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # 1. Get 6 chunks for Attention and MLP
        img_norm, img_mod = self.img_ada_ln(img, vec)
        txt_norm, txt_mod = self.txt_ada_ln(txt, vec)
        
        img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = img_mod
        txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = txt_mod
        
        # 2. Modulate input for Attention
        img_norm = img_norm * (1 + img_scale_msa[:, None, :]) + img_shift_msa[:, None, :]
        txt_norm = txt_norm * (1 + txt_scale_msa[:, None, :]) + txt_shift_msa[:, None, :]
        
        # 3. Joint Attention
        img_attn, txt_attn = self.attn(
            x1=img_norm, 
            x2=txt_norm,
        )
        
        # 4. Residual connections for attention with gate
        img = img + img_attn * img_gate_msa[:, None, :]
        txt = txt + txt_attn * txt_gate_msa[:, None, :]
        
        # 5. Modulate input for MLP (parallel)
        img_mlp_out = self.img_mlp(img_norm * (1 + img_scale_mlp[:, None, :]) + img_shift_mlp[:, None, :])
        txt_mlp_out = self.txt_mlp(txt_norm * (1 + txt_scale_mlp[:, None, :]) + txt_shift_mlp[:, None, :])
        
        img = img + img_mlp_out * img_gate_mlp[:, None, :]
        txt = txt + txt_mlp_out * txt_gate_mlp[:, None, :]
        
        return img, txt

from ...modules.attention import Attention

class FluxSingleStreamAttention(Attention):
    def __call__(self, x: jax.Array, mod: tuple[jax.Array, ...] = None) -> jax.Array:
        B, L, _ = x.shape
        
        # 1. Project
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # 2. QK Norms
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
            
        # 3. Apply Flux QKV Modulation!
        if mod is not None:
            shift_q, scale_q, shift_k, scale_k, shift_v, scale_v = mod
            shift_q, scale_q = shift_q.reshape(B, 1, self.num_heads, self.head_dim), scale_q.reshape(B, 1, self.num_heads, self.head_dim)
            shift_k, scale_k = shift_k.reshape(B, 1, self.num_heads, self.head_dim), scale_k.reshape(B, 1, self.num_heads, self.head_dim)
            shift_v, scale_v = shift_v.reshape(B, 1, self.num_heads, self.head_dim), scale_v.reshape(B, 1, self.num_heads, self.head_dim)
            
            q = q * (1 + scale_q) + shift_q
            k = k * (1 + scale_k) + shift_k
            v = v * (1 + scale_v) + shift_v
            
        # Apply Positional Embeddings (e.g. RoPE)
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k)
            
        # 4. Native Attention
        out = jax.nn.dot_product_attention(q, k, v)
        
        # 5. Output Projection
        out = out.reshape(B, L, -1)
        return self.o_proj(out)

class SingleStreamBlock(nn.Module):
    def __init__(self, config: MMDiTConfig, seed: Rngs = None):
        self.config = config
        
        # AdaLN
        self.ada_ln = FluxAdaLN(config.hidden_size, config.hidden_size, seed=seed)
        
        head_dim = config.hidden_size // config.num_heads
        
        # Single Stream Attention
        self.attn = FluxSingleStreamAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=head_dim,
            use_qkv_norm=True, # FLUX uses QK-Normalization in single blocks too!
            pos_emb=RotaryEmbedding(head_dim),
            seed=seed
        )
        
        # MLP
        ffn_dim = config.hidden_size * 3
        self.mlp = FluxMLP(config.hidden_size, ffn_dim, seed=seed)
        
    def __call__(self, x: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
        # 1. Normalize and get modulation chunks
        x_norm, mod = self.ada_ln(x, vec)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod
        
        # 2. Attention
        x_attn_in = x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x_attn = self.attn(x_attn_in)
        
        # 3. MLP
        x_mlp_in = x_norm * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x_mlp = self.mlp(x_mlp_in)
        
        # 4. Residual (Parallel!)
        x = x + x_attn * gate_msa[:, None, :] + x_mlp * gate_mlp[:, None, :]
        return x

class Flux(PretrainedModel):
    def __init__(self, config: MMDiTConfig, seed: Rngs = None):
        self.config = config
        
        # Embeddings
        self.img_in = nn.Linear(config.in_channels, config.hidden_size, seed=seed)
        self.txt_in = nn.Linear(config.context_dim, config.hidden_size, seed=seed)
        
        # Time and Pooled Text (Vector) Embeddings
        self.time_in = nn.Sequential(
            nn.SinusoidalPositionalEmbedding(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size, seed=seed),
            SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, seed=seed)
        )
        self.vector_in = nn.Sequential(
            nn.Linear(config.pooled_projection_dim, config.hidden_size, seed=seed),
            SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, seed=seed)
        )
        if config.guidance_embeds:
            self.guidance_in = nn.Sequential(
                nn.SinusoidalPositionalEmbedding(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size, seed=seed),
                SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size, seed=seed)
            )
        else:
            self.guidance_in = None
            
        # Blocks
        self.joint_blocks = nn.SequentialStack(
            JointAttentionBlock, config, num_stack=config.num_layers, seed=seed
        )
        self.single_blocks = nn.SequentialStack(
            SingleStreamBlock, config, num_stack=config.num_single_layers, seed=seed
        )

        # Output Projections
        self.norm_out = AdaLayerNorm(
            embedding_dim=config.hidden_size,
            out_dim=config.hidden_size * 2, # Scale and Shift
            norm_type="layer_norm",
            eps=1e-6,
            seed=seed
        )
        self.proj_out = nn.Linear(config.hidden_size, config.out_channels, seed=seed)

    def __call__(self, img: jnp.ndarray, txt: jnp.ndarray, timestep: jnp.ndarray, pooled_txt: jnp.ndarray, guidance: jnp.ndarray = None):
        B, L, _ = img.shape
        txt_len = txt.shape[1]
        
        img = self.img_in(img)
        txt = self.txt_in(txt)
        
        vec = self.time_in(timestep) + self.vector_in(pooled_txt)
        if self.config.guidance_embeds and guidance is not None:
            vec = vec + self.guidance_in(guidance)
            
        def apply_joint(layer, carry, vec):
            img, txt = carry
            out_img, out_txt = layer(img, txt, vec)
            return (out_img, out_txt), None

        (img, txt), _ = self.joint_blocks(apply_joint, (img, txt), vec)
        
        x = jnp.concatenate([txt, img], axis=1)
        
        def apply_single(layer, carry, vec):
            return layer(carry, vec), None
            
        x, _ = self.single_blocks(apply_single, x, vec)
        
        x_img = x[:, txt_len:]
        
        x_norm, mod = self.norm_out(x_img, vec)
        shift, scale = jnp.split(mod, 2, axis=-1)
        x_norm = x_norm * (1 + scale[:, None, :]) + shift[:, None, :]
        
        out = self.proj_out(x_norm)
        return out

    @classmethod
    def load_from_diffusers(cls, path_or_repo, config, local=False, dtype=jnp.bfloat16, subfolder="transformer"):
        import os
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import jax
        import numpy as np
        from ureinuz import Rngs
        import re
        
        path_or_repo_str = str(path_or_repo)
        file_name = "diffusion_pytorch_model.safetensors"
        
        if local:
            shard_path = os.path.join(path_or_repo_str, subfolder if subfolder else "", file_name)
        else:
            shard_path = hf_hub_download(repo_id=path_or_repo_str, subfolder=subfolder, filename=file_name)
            
        print(f"Loading weights from {shard_path}...")
        
        # 1. Instantiate skeleton
        state = jax.eval_shape(lambda: cls(config, Rngs(0)))
        current_state_dict = state.flat_state_dict()
        
        new_state = {}
        stacked_states = {}
        
        # 2. Key mapping function
        def map_key(k):
            # Time and embedders
            k = k.replace("time_text_embed.timestep_embedder.linear_1", "time_in.1")
            k = k.replace("time_text_embed.timestep_embedder.linear_2", "time_in.3")
            k = k.replace("time_text_embed.text_embedder.linear_1", "vector_in.0")
            k = k.replace("time_text_embed.text_embedder.linear_2", "vector_in.2")
            k = k.replace("time_text_embed.guidance_embedder.linear_1", "guidance_in.1")
            k = k.replace("time_text_embed.guidance_embedder.linear_2", "guidance_in.3")
            k = k.replace("context_embedder", "txt_in")
            k = k.replace("x_embedder", "img_in")
            
            # Joint Blocks
            if k.startswith("transformer_blocks."):
                k = k.replace("transformer_blocks.", "joint_blocks.")
                k = k.replace(".attn.to_q.", ".attn.q_proj_1.")
                k = k.replace(".attn.to_k.", ".attn.k_proj_1.")
                k = k.replace(".attn.to_v.", ".attn.v_proj_1.")
                k = k.replace(".attn.add_q_proj.", ".attn.q_proj_2.")
                k = k.replace(".attn.add_k_proj.", ".attn.k_proj_2.")
                k = k.replace(".attn.add_v_proj.", ".attn.v_proj_2.")
                k = k.replace(".attn.to_out.0.", ".attn.out_proj_1.")
                k = k.replace(".attn.to_add_out.", ".attn.out_proj_2.")
                k = k.replace(".attn.norm_q.", ".attn.q_norm_1.")
                k = k.replace(".attn.norm_k.", ".attn.k_norm_1.")
                k = k.replace(".attn.norm_added_q.", ".attn.q_norm_2.")
                k = k.replace(".attn.norm_added_k.", ".attn.k_norm_2.")
                k = k.replace(".ff.linear_in.", ".img_mlp.0.")
                k = k.replace(".ff.linear_out.", ".img_mlp.2.")
                k = k.replace(".ff_context.linear_in.", ".txt_mlp.0.")
                k = k.replace(".ff_context.linear_out.", ".txt_mlp.2.")
                k = k.replace(".norm1.linear.", ".img_ada_ln.linear.")
                k = k.replace(".norm1_context.linear.", ".txt_ada_ln.linear.")
                
            # Single Blocks
            elif k.startswith("single_transformer_blocks."):
                k = k.replace("single_transformer_blocks.", "single_blocks.")
                k = k.replace(".attn.to_q.", ".attn.q_proj.")
                k = k.replace(".attn.to_k.", ".attn.k_proj.")
                k = k.replace(".attn.to_v.", ".attn.v_proj.")
                k = k.replace(".attn.to_out.0.", ".attn.out_proj.")
                k = k.replace(".attn.to_out.", ".attn.o_proj.")
                k = k.replace(".attn.norm_q.", ".attn.q_norm.")
                k = k.replace(".attn.norm_k.", ".attn.k_norm.")
                k = k.replace(".proj_mlp.", ".mlp.linear_in.")
                k = k.replace(".proj_out.", ".mlp.linear_out.")
                k = k.replace(".norm.linear.", ".ada_ln.linear.")
                
            # Final Layer
            k = k.replace("proj_out", "proj_out")
            k = k.replace("norm_out.linear", "norm_out.linear")
            
            return k
            
        with safe_open(shard_path, framework="np", device="cpu") as f:
            for k_str in f.keys():
                k_mapped = map_key(k_str)
                
                if k_mapped in current_state_dict:
                    value = f.get_tensor(k_str)
                    
                    if value.ndim == 2:
                        value = value.T
                        
                    sharding = getattr(current_state_dict[k_mapped], "sharding", None)
                    value = jax.device_put(value, sharding).astype(dtype)
                    new_state[k_mapped] = value
                    
                else:
                    match = re.search(r'\.(\d+)\.', k_mapped)
                    if match:
                        idx = int(match.group(1))
                        k_stacked = k_mapped[:match.start()] + '.stacked.' + k_mapped[match.end():]
                        
                        # Handle fused to_qkv_mlp_proj in SingleStreamBlock
                        if k_str.endswith(".attn.to_qkv_mlp_proj.weight") or k_str.endswith(".attn.to_qkv_mlp_proj.bias"):
                            value = f.get_tensor(k_str)
                            if value.ndim == 2:
                                value = value.T
                                
                            hidden = 3072
                            q = value[..., :hidden]
                            k = value[..., hidden:hidden*2]
                            v = value[..., hidden*2:hidden*3]
                            mlp_in = value[..., hidden*3:]
                            
                            is_weight = k_str.endswith(".weight")
                            suffix = ".weight" if is_weight else ".bias"
                            
                            q_key = f"single_blocks.stacked.attn.q_proj{suffix}"
                            k_key = f"single_blocks.stacked.attn.k_proj{suffix}"
                            v_key = f"single_blocks.stacked.attn.v_proj{suffix}"
                            mlp_key = f"single_blocks.stacked.mlp.linear_in{suffix}"
                            
                            for split_key, split_val in [(q_key, q), (k_key, k), (v_key, v), (mlp_key, mlp_in)]:
                                if split_key in current_state_dict:
                                    target_var = current_state_dict[split_key]
                                    if split_key not in stacked_states:
                                        stacked_states[split_key] = np.zeros(target_var.shape, dtype=np.float16)
                                    stacked_states[split_key][idx] = split_val.astype(np.float16)
                            continue
                            
                        # Handle fused to_out in SingleStreamBlock
                        elif k_str.endswith(".attn.to_out.weight") or k_str.endswith(".attn.to_out.bias"):
                            value = f.get_tensor(k_str)
                            if value.ndim == 2:
                                value = value.T
                                
                            hidden = 3072
                            attn_out = value[:hidden, ...] if value.ndim == 1 else value[:hidden, ...]
                            mlp_out = value[hidden:, ...] if value.ndim == 1 else value[hidden:, ...]
                            
                            is_weight = k_str.endswith(".weight")
                            suffix = ".weight" if is_weight else ".bias"
                            
                            attn_key = f"single_blocks.stacked.attn.o_proj{suffix}"
                            mlp_key = f"single_blocks.stacked.mlp.linear_out{suffix}"
                            
                            for split_key, split_val in [(attn_key, attn_out), (mlp_key, mlp_out)]:
                                if split_key in current_state_dict:
                                    target_var = current_state_dict[split_key]
                                    if split_key not in stacked_states:
                                        stacked_states[split_key] = np.zeros(target_var.shape, dtype=np.float16)
                                    stacked_states[split_key][idx] = split_val.astype(np.float16)
                            continue
                            
                        elif k_stacked in current_state_dict:
                            target_var = current_state_dict[k_stacked]
                            value = f.get_tensor(k_str)
                            layer_shape = target_var.shape[1:]
                            
                            if value.ndim == 2 and layer_shape == value.shape[::-1]:
                                value = value.T
                                
                            if target_var.shape[1:] != value.shape:
                                print(f"SHAPE MISMATCH! k_str: {k_str}, k_stacked: {k_stacked}, target: {target_var.shape}, value: {value.shape}")
                                
                            if k_stacked not in stacked_states:
                                stacked_states[k_stacked] = np.zeros(target_var.shape, dtype=np.float16)
                            stacked_states[k_stacked][idx] = value.astype(np.float16)
                            continue
                    
                    print(f"Warning: {k_str} (mapped to {k_mapped}) found in checkpoint but not in model.")
                    
        for k_stacked, stacked_array in stacked_states.items():
            sharding = getattr(current_state_dict[k_stacked], "sharding", None)
            new_state[k_stacked] = jax.device_put(stacked_array, sharding).astype(dtype)
            
        # Check missing keys
        missing_keys = set(current_state_dict.keys()) - set(new_state.keys())
        
        # Fill missing biases with zeros!
        for k in list(missing_keys):
            if k.endswith('.bias'):
                target_var = current_state_dict[k]
                sharding = getattr(target_var, "sharding", None)
                new_state[k] = jax.device_put(jnp.zeros(target_var.shape, dtype=jnp.float32), sharding).astype(dtype)
                missing_keys.remove(k)
        
        if missing_keys:
            print(f"Warning: {len(missing_keys)} keys missing from checkpoint!")
            for k in list(missing_keys)[:10]:
                print(f"  Missing: {k}")
                
        state.load_flat_state_dict(new_state)
        return state
