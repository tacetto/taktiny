import jax
import jax.numpy as jnp
from dataclasses import dataclass

from .... import nn, Rngs
from ..autoencoder import Encoder, Decoder

@dataclass
class FluxAutoencoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    latent_channels: int = 32
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True

class AutoencoderKLFlux2(nn.Module):
    """
    Pure JAX implementation of the FLUX 2 Autoencoder (VAE).
    """
    def __init__(self, config: FluxAutoencoderConfig, seed: Rngs = None):
        self.config = config
        
        # Diffusers' double_z=True means the encoder outputs 2 * latent_channels
        self.encoder = Encoder(
            in_channels=config.in_channels,
            dims=list(config.block_out_channels),
            latent_dim=2 * config.latent_channels,
            depths=[config.layers_per_block] * len(config.block_out_channels),
            seed=seed
        )
        
        self.decoder = Decoder(
            in_channels=config.out_channels,
            dims=list(config.block_out_channels),
            latent_dim=config.latent_channels,
            depths=[config.layers_per_block] * len(config.block_out_channels),
            seed=seed
        )
        
        if config.use_quant_conv:
            self.quant_conv = nn.Conv2d(
                2 * config.latent_channels, 
                2 * config.latent_channels, 
                kernel_size=1, 
                seed=seed
            )
        else:
            self.quant_conv = None
            
        if config.use_post_quant_conv:
            self.post_quant_conv = nn.Conv2d(
                config.latent_channels, 
                config.latent_channels, 
                kernel_size=1, 
                seed=seed
            )
        else:
            self.post_quant_conv = None

    def encode(self, x: jax.Array) -> jax.Array:
        """
        Encodes an image into the latent space (returns mean and logvar concatenated).
        Input shape: (B, H, W, C)
        Output shape: (B, H/16, W/16, 2 * latent_channels)
        """
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        return h

    def decode(self, z: jax.Array) -> jax.Array:
        """
        Decodes a latent vector back into an image.
        Input shape: (B, H/16, W/16, latent_channels)
        Output shape: (B, H, W, out_channels)
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
        
    def __call__(self, x: jax.Array, seed: Rngs = None) -> tuple[jax.Array, jax.Array]:
        """
        Full forward pass: encode, sample (if seed is provided), and decode.
        """
        h = self.encode(x)
        mean, logvar = jnp.split(h, 2, axis=-1)
        
        if seed is not None:
            std = jnp.exp(0.5 * logvar)
            noise = jax.random.normal(seed(), mean.shape, dtype=mean.dtype)
            z = mean + std * noise
        else:
            z = mean
            
        reconstructed = self.decode(z)
        return reconstructed, z