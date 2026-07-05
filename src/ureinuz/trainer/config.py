import optax
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Iterable

@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 1e-3
    optimizer: Any = None # Optax optimizer, defaults to adamw if None
    weight_decay: float = 0.0
    log_interval: int = 10
    seed: int = 42
    jit_compile: bool = False

@dataclass(frozen=True)
class DatasetConfig:
    # A generic iterable that yields batches (e.g. tf.data, PyTorch DataLoader, or custom generator)
    dataloader: Iterable[Any]
    validation_dataloader: Optional[Iterable[Any]] = None
    shuffle: bool = True
