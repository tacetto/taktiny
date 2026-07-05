import os
import json
import jax
import jax.numpy as jnp
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.flax import save_file
from ....nn import Module, Rngs

class PretrainedModel(Module):

    def save_pretrained(self, path):
        """
        Saves the model's weights to a local directory in safetensors format.
        Always saves an index json for unified loading.
        """
        os.makedirs(path, exist_ok=True)
        weights_path = os.path.join(path, "model.safetensors")
        index_path = os.path.join(path, "model.safetensors.index.json")
        
        # Extract flat state dict (mapping string paths to JAX arrays)
        state_dict = self.flat_state_dict()
        save_file(state_dict, weights_path)
        
        # Always create an index file for consistency
        index_data = {
            "metadata": {"total_size": os.path.getsize(weights_path)},
            "weight_map": {k: "model.safetensors" for k in state_dict.keys()}
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

    @classmethod
    def load_config(cls, path_or_repo, subfolder=None, local=False):
        if local:
            config_path = Path(path_or_repo).resolve()
            if subfolder:
                config_path = config_path / subfolder
            config_path = config_path / 'config.json'
        else:
            try:
                if subfolder:
                    config_path = hf_hub_download(repo_id=str(path_or_repo), subfolder=subfolder, filename="config.json")
                else:
                    config_path = hf_hub_download(repo_id=str(path_or_repo), filename="config.json")
            except Exception as e:
                print(f'config.json not found in repo: {e}')
                return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f'Error loading config.json: {e}')
            return None

    @classmethod
    def from_pretrained(cls, path_or_repo, config, module_map=None, local=False, dtype=None, subfolder=None):
        """
        Loads safetensors weights into a newly instantiated model.
        Supports both single-file (model.safetensors) and sharded models.
        """
        if dtype is not None:
            import dataclasses
            config = dataclasses.replace(config, dtype=dtype)
            
        path_or_repo_str = str(path_or_repo)
        module_map = module_map or {}

        # 1. Determine if model is sharded or single file
        is_sharded = False
        if local:
            index_path = os.path.join(path_or_repo_str, subfolder if subfolder else "", "model.safetensors.index.json")
            if os.path.exists(index_path):
                is_sharded = True
        else:
            from huggingface_hub import repo_info
            try:
                info = repo_info(repo_id=path_or_repo_str)
                files = [f.rfilename for f in info.siblings]
                target_index = f"{subfolder}/model.safetensors.index.json" if subfolder else "model.safetensors.index.json"
                if target_index in files:
                    is_sharded = True
                    index_path = hf_hub_download(repo_id=path_or_repo_str, subfolder=subfolder, filename="model.safetensors.index.json")
            except Exception as e:
                print(f"Failed to fetch repo info: {e}")
                is_sharded = False

        # 2. Build files_to_load mapping: file_name -> list of keys (or None for all)
        files_to_load = {}
        if is_sharded:
            with open(index_path, "r") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            for k_str, file_name in weight_map.items():
                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(k_str)
        else:
            files_to_load["model.safetensors"] = None

        # 3. Instantiate model skeleton using eval_shape (no memory allocation)
        state = jax.eval_shape(lambda: cls(config, Rngs(0)))
        current_state_dict = state.flat_state_dict()
        new_state = {}
        not_found_some = False

        # 4. Load weights
        import re
        import numpy as np
        from safetensors import safe_open
        stacked_states = {}

        for file_name, keys_in_file in files_to_load.items():
            if local:
                shard_path = os.path.join(path_or_repo_str, subfolder if subfolder else "", file_name)
            else:
                shard_path = hf_hub_download(repo_id=path_or_repo_str, subfolder=subfolder, filename=file_name)
                
            with safe_open(shard_path, framework="np", device="cpu") as f:
                keys_to_process = keys_in_file if keys_in_file is not None else f.keys()
                
                for k_str in keys_to_process:
                    # Apply module_map via string replacements
                    k_mapped = k_str
                    for old, new in module_map.items():
                        k_mapped = k_mapped.replace(old, new)
                    
                    
                    if k_mapped in current_state_dict:
                        value = f.get_tensor(k_str)
                        target_var = current_state_dict[k_mapped]
                        
                        if value.ndim == 2 and target_var.shape == value.shape[::-1]:
                            value = value.T
    
                        sharding = getattr(target_var, "sharding", None)
                        value = jax.device_put(value, sharding)
                            
                        new_state[k_mapped] = value
                    elif k_mapped.replace('.weight', '.weights_q') in current_state_dict:
                        # Dynamic Quantization Interceptor for non-stacked modules
                        k_weights_q = k_mapped.replace('.weight', '.weights_q')
                        k_scales_q = k_mapped.replace('.weight', '.scales_q')
                        k_scale_of_scales = k_mapped.replace('.weight', '.scale_of_scales')
                        
                        target_var = current_state_dict[k_weights_q]
                        value = f.get_tensor(k_str)
                        
                        # Handle transpose for Linear
                        # For unquantized, it was [in, out]. Check target_var shape.
                        # Target shape for weights_q could be packed (e.g. out_features // 2). 
                        # We always transpose if value.shape[0] == out_features of original layer.
                        # Original layer out_features = value.shape[0] if it was (out, in).
                        if value.ndim == 2:
                            # We can infer in_features from value
                            # Let's just always transpose if config says so, but SAFETENSORS HF Linear weights are [out_features, in_features]
                            value = value.T
                            
                        from ....nn.quant.core import (
                            quantize_to_fp8_double, quantize_to_int8_double, 
                            quantize_to_int4_double, quantize_to_fp4_double
                        )
                        
                        # Infer block size from target_var and value
                        if target_var.dtype == jnp.uint8:
                            # INT4 / FP4 packed
                            # target_var is [in_features, out_features // 2]
                            pass # block_size is fixed to 128 for now
                        
                        if dtype == "fp8":
                            w_q, s_q, sos = quantize_to_fp8_double(value, block_size=128)
                        elif dtype == "int8":
                            w_q, s_q, sos = quantize_to_int8_double(value, block_size=128)
                        elif dtype == "int4":
                            w_q, s_q, sos = quantize_to_int4_double(value, block_size=128)
                        elif dtype == "fp4":
                            w_q, s_q, sos = quantize_to_fp4_double(value, block_size=128)
                            
                        sharding = getattr(target_var, "sharding", None)
                        
                        w_q = jax.device_put(w_q, sharding)
                        s_q = jax.device_put(s_q, sharding)
                        sos = jax.device_put(sos, sharding)
                        
                        # Cast to correct JAX types
                        if dtype == "fp8":
                            w_q = w_q.astype(jnp.float8_e4m3fn)
                            s_q = s_q.astype(jnp.float8_e4m3fn)
                        elif dtype == "int8":
                            w_q = w_q.astype(jnp.int8)
                            s_q = s_q.astype(jnp.int8)
                        
                        new_state[k_weights_q] = w_q
                        new_state[k_scales_q] = s_q
                        new_state[k_scale_of_scales] = sos

                    else:
                        # Check if it belongs to a SequentialStack
                        match = re.search(r'\.(\d+)\.', k_mapped)
                        if match:
                            idx = int(match.group(1))
                            k_stacked = k_mapped[:match.start()] + '.stacked.' + k_mapped[match.end():]
                            k_stacked_q = k_stacked.replace('.weight', '.weights_q')
                            
                            if k_stacked in current_state_dict:
                                target_var = current_state_dict[k_stacked]
                                value = f.get_tensor(k_str)
                                
                                layer_shape = target_var.shape[1:]
                                if value.ndim == 2 and layer_shape == value.shape[::-1]:
                                    value = value.T
                                    
                                if k_stacked not in stacked_states:
                                    stacked_states[k_stacked] = np.zeros(target_var.shape, dtype=value.dtype)
                                stacked_states[k_stacked][idx] = value
                                continue
                                
                            elif k_stacked_q in current_state_dict:
                                target_var = current_state_dict[k_stacked_q]
                                value = f.get_tensor(k_str)
                                
                                if value.ndim == 2:
                                    value = value.T
                                    
                                from ....nn.quant.core import (
                                    quantize_to_fp8_double, quantize_to_int8_double, 
                                    quantize_to_int4_double, quantize_to_fp4_double
                                )
                                
                                if dtype == "fp8":
                                    w_q, s_q, sos = quantize_to_fp8_double(value, block_size=128)
                                elif dtype == "int8":
                                    w_q, s_q, sos = quantize_to_int8_double(value, block_size=128)
                                elif dtype == "int4":
                                    block_size = 128
                                    if value.shape[0] % 128 != 0:
                                        block_size = 64
                                    w_q, s_q, sos = quantize_to_int4_double(value, block_size=block_size)
                                elif dtype == "fp4":
                                    w_q, s_q, sos = quantize_to_fp4_double(value, block_size=128)
                                    
                                k_scales_q = k_stacked.replace('.weight', '.scales_q')
                                k_scale_of_scales = k_stacked.replace('.weight', '.scale_of_scales')
                                
                                if k_stacked_q not in stacked_states:
                                    stacked_states[k_stacked_q] = np.zeros(target_var.shape, dtype=w_q.dtype)
                                    stacked_states[k_scales_q] = np.zeros(current_state_dict[k_scales_q].shape, dtype=s_q.dtype)
                                    stacked_states[k_scale_of_scales] = np.zeros(current_state_dict[k_scale_of_scales].shape, dtype=sos.dtype)
                                    
                                stacked_states[k_stacked_q][idx] = w_q
                                stacked_states[k_scales_q][idx] = s_q
                                stacked_states[k_scale_of_scales][idx] = sos
                                continue
                                
                        not_found_some = True
                        print(f"Warning: {k_str} (mapped to {k_mapped}) found in checkpoint but not in model.")

        # Move accumulated SequentialStack weights to JAX
        for k_stacked, stacked_array in stacked_states.items():
            target_var = current_state_dict[k_stacked]
            sharding = getattr(target_var, "sharding", None)
            
            stacked_array = jax.device_put(stacked_array, sharding)
            
            # If it's an FP8 or INT8 quantized parameter, we must explicitly cast it inside JAX
            # because NumPy arrays cannot natively represent FP8 E4M3, and we store them as uint8 temporarily.
            if target_var.dtype == jnp.float8_e4m3fn:
                stacked_array = stacked_array.astype(jnp.float8_e4m3fn)
            elif target_var.dtype == jnp.int8:
                stacked_array = stacked_array.astype(jnp.int8)
                
            new_state[k_stacked] = stacked_array

        if not_found_some:
            print("\nSome modules from the checkpoint were not found in this model.")
            print("You can try to map module names using module_map.")
            print("e.g. module_map = {'target_module': 'name_to_change'}")

        # Handle tied word embeddings
        if getattr(config, 'use_tie_lm_head', False) or getattr(config, 'tie_word_embeddings', False):
            embed_key = 'model.embed_tokens.embedding'
            lm_head_key = 'lm_head.weight'
            if embed_key in new_state:
                if lm_head_key in current_state_dict:
                    new_state[lm_head_key] = new_state[embed_key].T
                elif lm_head_key.replace('.weight', '.weights_q') in current_state_dict:
                    k_weights_q = lm_head_key.replace('.weight', '.weights_q')
                    k_scales_q = lm_head_key.replace('.weight', '.scales_q')
                    k_scale_of_scales = lm_head_key.replace('.weight', '.scale_of_scales')
                    
                    value = new_state[embed_key].T
                    
                    from ....nn.quant.core import (
                        quantize_to_fp8_double, quantize_to_int8_double, 
                        quantize_to_int4_double, quantize_to_fp4_double
                    )
                    
                    if dtype == "fp8":
                        w_q, s_q, sos = quantize_to_fp8_double(value, block_size=128)
                    elif dtype == "int8":
                        w_q, s_q, sos = quantize_to_int8_double(value, block_size=128)
                    elif dtype == "int4":
                        block_size = 128
                        if value.shape[0] % 128 != 0:
                            block_size = 64
                        w_q, s_q, sos = quantize_to_int4_double(value, block_size=block_size)
                    elif dtype == "fp4":
                        w_q, s_q, sos = quantize_to_fp4_double(value, block_size=128)
                        
                    target_var = current_state_dict[k_weights_q]
                    sharding = getattr(target_var, "sharding", None)
                    
                    w_q = jax.device_put(w_q, sharding)
                    s_q = jax.device_put(s_q, sharding)
                    sos = jax.device_put(sos, sharding)
                    
                    new_state[k_weights_q] = w_q
                    new_state[k_scales_q] = s_q
                    new_state[k_scale_of_scales] = sos

        # 5. Inject actual arrays into the PyTree skeleton
        state.load_flat_state_dict(new_state)
        return state