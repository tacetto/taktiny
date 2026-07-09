# Taktiny

**Taktiny** is a boutique, object-oriented neural network framework and universal training engine built natively on top of JAX. 

It provides an intuitive, stateful design where models encapsulate their own weights while leveraging the lightning-fast, purely functional compilation of JAX. Every `taktiny.nn.Module` and `Parameter` is a natively registered PyTree, allowing stateful objects to compile flawlessly with `jax.jit` and `optax`.

## ✨ Features
- **Boutique Architecture**: A meticulously handcrafted framework featuring a unique, cohesive orchestrator aesthetic (`Maestro`, `Repertoire`, and `cosettes/`).
- **Object-Oriented JAX**: Build models using stateful `nn.Module` and `Parameter` classes without manually separating parameters from architecture.
- **🎼 Maestro & Repertoire**: A powerful HuggingFace auto-model loader. Seamlessly load, quantize (int4/int8/fp8), and automatically shard masterpieces from the `Repertoire` (like LLaMA and Qwen) over any JAX Mesh configuration.
- **Dynamic Configs**: Fluid and adaptable configurations. Taktiny elegantly parses raw Hugging Face JSON architectures directly into dynamic Python objects.
- **Universal Trainer**: A polished, generic `Trainer` engine that can train models natively across multiple JAX-based frameworks.

## 🏗️ Building Modules

`taktiny` embraces an **Object-Oriented** philosophy. Parameters are stored directly inside the class using `nn.Parameter`. Because `nn.Module` is a registered JAX PyTree, JAX functions (like `jax.jit` or `jax.value_and_grad`) understand the objects natively.

```python
import jax
import jax.numpy as jnp
from taktiny import nn

class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rngs: nn.Rngs):
        self.fc1 = nn.Linear(in_features, hidden_features, rngs=rngs)
        self.fc2 = nn.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        return x

# Initialize with a random seed generator
rngs = nn.Rngs(42)
model = SimpleMLP(in_features=64, hidden_features=128, out_features=10, rngs=rngs)

# State management for saving and checkpointing
state_dict = model.state_dict() # or model.flat_state_dict()
model.load_state_dict(state_dict)
```

## 🎼 Maestro: HuggingFace & Quantization

Taktiny includes **Maestro**, an intelligent model loader that can pull HuggingFace repositories, match them against the registered **Repertoire**, and instantiate the equivalent Taktiny native architectures (the `opus`). Maestro also supports native dynamic quantization (INT4, INT8, FP8) on-the-fly during load time.

```python
import jax
from jax.sharding import Mesh
from taktiny import Maestro

# 1. Define your hardware mesh (e.g., for Tensor Parallelism)
mesh = Mesh(jax.devices(), ('dp', 'tp'))

# 2. Let Maestro download, quantize, and shard the weights dynamically
model = Maestro.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct", 
    dtype="int4", # Dynamically quantize weights to INT4 
    mesh=mesh     # Distribute weights according to the mesh
)

print("Model successfully loaded and sharded!")
```

## 🧠 The Universal Trainer

The `Trainer` class is designed to train neural networks robustly. When initialized, it automatically inspects the model and internally handles parameter extraction and state updates. This allows it to train models from native `taktiny` or external JAX frameworks (such as `flax.linen`, `flax.nnx`, or `equinox`).

You can customize training with `TrainingConfig` (which supports setting maximum steps, learning rates, epochs, and Optax optimizers) and pass any standard data iterable to the `DatasetConfig`.

```python
import optax
from taktiny.trainer import Trainer, TrainingConfig, DatasetConfig

trainer = Trainer(
    model=model,
    loss_fn=my_loss_function,
    training_config=TrainingConfig(
        epochs=5,
        max_steps=2000,
        learning_rate=1e-3,
        optimizer=optax.adamw(1e-3),
        log_interval=50
    ),
    dataset_config=DatasetConfig(dataloader=my_batch_generator)
)

trainer.train()
```
The trainer provides a gorgeous `rich` progress bar in the terminal, complete with time-per-step tracking!

## 🚀 Quick Start
<!-- 
```python
import jax.numpy as jnp
from taktiny import Rngs
from taktiny.recipes import CNNModelConfig, Autoencoder
from taktiny.trainer import Trainer, TrainingConfig, DatasetConfig

# 1. Initialize a Stateful Model from the Zoo
config = CNNModelConfig(in_channels=3, dims=[64, 128], latent_dim=32)
model = Autoencoder(config, rngs=Rngs(42))

# 2. Define a standard Loss Function
def mse_loss(params, batch):
    reconstructed, _ = params(batch)
    return jnp.mean((reconstructed - batch) ** 2)

# 3. Train!
trainer = Trainer(
    model=model,
    loss_fn=mse_loss,
    training_config=TrainingConfig(max_steps=1000),
    dataset_config=DatasetConfig(dataloader=my_batch_generator)
)
trainer.train()
``` -->
```python
from taktiny import Maestro
from transformers import AutoTokenizer
import jax.numpy as jnp

# currently supports
# - llama (under 4)
# - qwen 2
# - gemma (first gemma)
repo = 'google/gemma-2b-it'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(repo)

print("Loading model...")
model = Maestro.from_pretrained(repo)

prompt = "Once upon a time\n"
print(f"\nPrompt: {prompt}")

input_ids = tokenizer.encode(prompt, return_tensors='np')
input_ids = jnp.array(input_ids)

print("Generating...")
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nOutput:\n{output_text}")

```