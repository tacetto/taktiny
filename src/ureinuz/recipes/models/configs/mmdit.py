from dataclasses import dataclass
from .model import PretrainedModel
from .transformer import TransformerModelConfig
from .cnn import CNNModelConfig
import jax

@dataclass
class MMDiTConfig:
    """
    Configuration for Multimodal Diffusion Transformers (like FLUX and SD3)
    that use Joint/Double-Stream Attention and Single-Stream blocks.
    """
    text_config: TransformerModelConfig
    vae_config: CNNModelConfig
    
    in_channels: int
    out_channels: int
    hidden_size: int
    num_heads: int
    num_layers: int          # Number of Double-Stream / Joint Attention blocks
    num_single_layers: int   # Number of Single-Stream blocks
    context_dim: int         # Dimension of the incoming text embeddings
    pooled_projection_dim: int
    guidance_embeds: bool = False
    axes_dim: list[int] = None
    
    @classmethod
    def from_pretrained(cls, path_or_repo: str, local: bool = False) -> 'MMDiTConfig':
        from .utils import load_flexible_config
        
        # (Assuming the user provides path to the transformer config directly, 
        # or we automatically append 'transformer' if it's a pipeline repo)
        hf_config = PretrainedModel.load_config(path_or_repo, local=local)
            
        if hf_config is None:
            # Try fetching from 'transformer' subfolder if it's a diffusers repo
            if not local:
                hf_config = PretrainedModel.load_config(path_or_repo, subfolder="transformer", local=local)
            else:
                hf_config = PretrainedModel.load_config(str(path_or_repo) + "/transformer", local=local)
                
        if hf_config is None:
            raise ValueError(f"Failed to load config from {path_or_repo}")
            
        # Map HuggingFace diffusers keys to our strict dataclass keys
        mapped_config = {
            "in_channels": hf_config.get("in_channels"),
            "out_channels": hf_config.get("out_channels") if hf_config.get("out_channels") is not None else hf_config.get("in_channels"),
            "num_heads": hf_config.get("num_attention_heads"),
            "hidden_size": hf_config.get("num_attention_heads") * hf_config.get("attention_head_dim"),
            "num_layers": hf_config.get("num_layers"),
            "num_single_layers": hf_config.get("num_single_layers"),
            "context_dim": hf_config.get("joint_attention_dim"),
            "pooled_projection_dim": hf_config.get("pooled_projection_dim", 768),
            "guidance_embeds": hf_config.get("guidance_embeds", False),
            "axes_dim": hf_config.get("axes_dims_rope"),
            
            # Sub-configs will be injected manually by the pipeline later
            "text_config": None, 
            "vae_config": None
        }
        
        # Merge any extra keys dynamically
        merged_config = {**hf_config, **mapped_config}
            
        return load_flexible_config(cls, merged_config)
