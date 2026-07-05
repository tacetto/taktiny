import jax
from .module import Module

class List(Module):
    def __init__(self, *modules):
        self.layers = list(modules)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def extra_repr(self):
        return f"{len(self.layers)}"

class Sequential(Module):
    def __init__(self, *modules):
        self.layers = tuple(modules)

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x

    def extra_repr(self):
        return f"{len(self.layers)}"

class SequentialStack(Module):
    def __init__(self, layer_cls, *args, num_stack: int, **kwargs):
        layers = [layer_cls(*args, **kwargs) for _ in range(num_stack)]
        self.stacked = jax.tree_util.tree_map(lambda *xs: jax.numpy.stack(xs), *layers)
        self.num_stack = num_stack

    def __call__(self, f, carry, *args, **kwargs):
        def apply_fn(carry, layer):
            out = jax.checkpoint(f)(layer, carry, *args, **kwargs)
            return out, None

        out, _ = jax.lax.scan(apply_fn, carry, self.stacked)
        return out

    def extra_repr(self):
        return f"{self.num_stack}"

class Stack(Module):
    def __init__(self, layer_cls, *args, num_stack: int, **kwargs):
        layers = [layer_cls(*args, **kwargs) for _ in range(num_stack)]
        self.stacked = jax.tree_util.tree_map(lambda *xs: jax.numpy.stack(xs), *layers)
        self.num_stack = num_stack

    def __call__(self, *args, in_axes=0, out_axes=0, **kwargs):
        if isinstance(in_axes, tuple):
            vmap_in_axes = (0,) + in_axes
        else:
            vmap_in_axes = (0,) + (in_axes,) * len(args)
            
        def apply_fn(layer, *positional_args):
            return layer(*positional_args, **kwargs)
            
        return jax.vmap(apply_fn, in_axes=vmap_in_axes, out_axes=out_axes)(self.stacked, *args)

    def extra_repr(self):
        return f"{self.num_stack}"