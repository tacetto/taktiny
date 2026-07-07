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

from __future__ import annotations

import jax
from ureinuz.trainer.config import TrainingConfig, DatasetConfig

import jax.numpy as jnp
import optax

def _is_frozen(x):
    """Heuristic to detect if a parameter should be frozen."""
    if not hasattr(x, 'dtype') or not jnp.issubdtype(x.dtype, jnp.inexact):
        return True
    if x.dtype == getattr(jnp, 'float8_e4m3fn', None):
        return True
    # Freeze 1D arrays (LayerNorms, biases)
    if len(x.shape) == 1:
        return True
    # Freeze massive matrices (like 2048x151936 embeddings/lm_head). LoRA matrices have small rank (<= 256).
    if len(x.shape) == 2 and min(x.shape) > 256:
        return True
    return False

def _safe_add(p, u):
    """Safely apply updates only to trainable parameters."""
    if _is_frozen(p):
        return p
    return p + u

class Trainer:
    def __init__(self, model, loss_fn, training_config: TrainingConfig, dataset_config: DatasetConfig):
        self.model = model
        self.loss_fn = loss_fn
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.model_type = self._diagnose_model_type(model)
        
    def _diagnose_model_type(self, model) -> str:
        # Detect Ureinuz models
        if hasattr(model, "__module__") and "ureinuz" in model.__module__:
            return "ureinuz"
            
        # Detect Flax NNX models
        if hasattr(model, "__module__") and "flax.nnx" in model.__module__:
            return "nnx"
            
        # Detect classic Flax Linen models
        if hasattr(model, "__module__") and "flax.linen" in model.__module__:
            return "flax_linen"
            
        # Detect Equinox models
        if hasattr(model, "__module__") and "equinox" in model.__module__:
            return "equinox"
            
        return "unknown"
        
    def extract_params(self):
        """Extract params based on the diagnosed model type."""
        if self.model_type == "ureinuz":
            # Ureinuz models are fully registered PyTrees
            return self.model
        elif self.model_type == "nnx":
            from flax import nnx
            _, params = nnx.split(self.model)
            return params
        elif self.model_type == "flax_linen":
            # Assume self.model is a dict of params for Flax Linen in this simplified design
            # (In reality, Flax Trainer would need model.init or params passed in)
            return self.model
        elif self.model_type == "equinox":
            import equinox as eqx
            return eqx.filter(self.model, eqx.is_array)
        else:
            raise ValueError("Unsupported model type")
            
    def _setup_optimizer(self):
        """Configures the optimizer with auto-freezing for quantized/massive parameters."""
        if self.training_config.optimizer is not None:
            return self.training_config.optimizer
            
        base_opt = optax.adamw(self.training_config.learning_rate, weight_decay=self.training_config.weight_decay)
        
        def get_label(x):
            return 'frozen' if _is_frozen(x) else 'trainable'
            
        return optax.multi_transform(
            {'trainable': base_opt, 'frozen': optax.set_to_zero()},
            lambda p: jax.tree_util.tree_map(get_label, p)
        )
            
    def train(self):
        from rich.console import Console
        from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        
        console = Console()
        console.print(f"[bold green]Starting training for a [cyan]{self.model_type.upper()}[/cyan] model[/bold green]")
        console.print(f"Epochs: [bold]{self.training_config.epochs}[/bold] | Max Steps: [bold]{self.training_config.max_steps}[/bold]")
        
        # 1. Initialize Optimizer
        optimizer = self._setup_optimizer()
        params = self.extract_params()
        opt_state = optimizer.init(params)
        
        # 2. Define train_step
        def train_step(params, opt_state, batch):
            loss, grads = jax.value_and_grad(self.loss_fn, allow_int=True)(params, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = jax.tree_util.tree_map(_safe_add, params, updates)
            return new_params, new_opt_state, loss
            
        if getattr(self.training_config, "jit_compile", False):
            train_step = jax.jit(train_step)

        # 3. Training Loop
        import time
        step = 0
        should_stop = False
        start_time = time.time()
        
        # Try to guess total steps if dataloader has __len__
        total_steps = None
        if hasattr(self.dataset_config.dataloader, "__len__"):
            total_steps = len(self.dataset_config.dataloader) * self.training_config.epochs
        if self.training_config.max_steps is not None:
            total_steps = self.training_config.max_steps
            
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("• [bold magenta]Loss: {task.fields[loss]:.4f}[/bold magenta]"),
            console=console
        ) as progress:
            
            task_id = progress.add_task("[cyan]Training...", total=total_steps, loss=0.0)
            
            for epoch in range(self.training_config.epochs):
                if should_stop:
                    break
                    
                for batch in self.dataset_config.dataloader:
                    params, opt_state, loss = train_step(params, opt_state, batch)
                    step += 1
                    
                    if step % self.training_config.log_interval == 0:
                        progress.update(task_id, advance=self.training_config.log_interval, loss=float(loss))
                        
                        # Calculate timing
                        elapsed = time.time() - start_time
                        ms_per_step = (elapsed / max(1, self.training_config.log_interval)) * 1000
                        
                        # Persistent log above the progress bar
                        log_msg = (
                            f"[bold cyan]Epoch {epoch:<3}[/bold cyan] ┃ "
                            f"[bold yellow]Step {step:<6}[/bold yellow] ┃ "
                            f"Loss: [bold magenta]{loss:<7.4f}[/bold magenta] ┃ "
                            f"[dim]{ms_per_step:>6.1f} ms/it[/dim]"
                        )
                        progress.console.print(log_msg)
                        start_time = time.time()
                        
                    if self.training_config.max_steps is not None and step >= self.training_config.max_steps:
                        should_stop = True
                        break
                        
            # Force finish the progress bar
            progress.update(task_id, completed=total_steps if total_steps else step, loss=float(loss))
                
        # 4. Inject back into the object if needed
        self._inject_params(params)
        console.print("[bold green]✨ Training complete![/bold green]")
        
    def _inject_params(self, params):
        if self.model_type == "ureinuz":
            # The returned PyTree is a new ureinuz Module. We can update self.model in-place.
            self.model.load_state_dict(params.state_dict())
        elif self.model_type == "nnx":
            from flax import nnx
            # params is the state dict, we merge it back into the graph
            nnx.update(self.model, params)
        elif self.model_type == "flax_linen":
            self.params = params
        elif self.model_type == "equinox":
            self.model = params


__all__ = ['Trainer']