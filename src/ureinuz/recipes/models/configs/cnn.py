
from dataclasses import dataclass



@dataclass(frozen=True)
class CNNModelConfig:
    dims: list[int]
    depths: list[int] = None
    num_classes: int = None
    latent_dim: int = None
    in_channels: int = 3
    drop_path_rate: float = 0.0