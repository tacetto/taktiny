from dataclasses import fields, is_dataclass
from typing import TypeVar, Type, Any

T = TypeVar('T')

def load_flexible_config(config_cls: Type[T], config_dict: dict[str, Any]) -> T:
    """
    Loads a dictionary into a dataclass, safely absorbing any extra keys 
    so they are still accessible via dot notation without causing a TypeError.
    """
    if not is_dataclass(config_cls):
        raise ValueError("config_cls must be a dataclass")

    # Get the strict fields defined in your dataclass
    valid_keys = {f.name for f in fields(config_cls)}
    
    # Separate the dict into known and unknown keys
    known_kwargs = {k: v for k, v in config_dict.items() if k in valid_keys}
    unknown_kwargs = {k: v for k, v in config_dict.items() if k not in valid_keys}
    
    # Instantiate the strict dataclass
    config = config_cls(**known_kwargs)
    
    # Dynamically monkey-patch the unknown keys onto the instance
    for k, v in unknown_kwargs.items():
        setattr(config, k, v)
        
    return config
