from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class AutoencoderConfig:
    sample_rate: int = 44100
    n_samples: int = 16384
    latent_dim: int = 64
    base_channels: int = 32
    num_down_blocks: int = 4


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DeconvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2, padding: Optional[int] = None, output_padding: int = 1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.deconv = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class KickConvAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_ch = 1
        out_ch = config.base_channels
        
        for _ in range(config.num_down_blocks):
            self.encoder_blocks.append(ConvBlock1d(in_ch, out_ch))
            in_ch = out_ch
            out_ch *= 2
        
        encoder_out_channels = in_ch
        
        # Dummy forward pass to compute encoded shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.config.n_samples)
            h = dummy
            for block in self.encoder_blocks:
                h = block(h)
            self._encoded_shape = h.shape  # (1, C, L)
            self._feature_dim = h.numel()
        
        # Latent bottleneck
        self.fc_mu = nn.Linear(self._feature_dim, self.config.latent_dim)
        
        # Decoder: Linear to feature_dim, then deconv blocks
        self.fc_decode = nn.Linear(self.config.latent_dim, self._feature_dim)
        
        # Build decoder blocks (mirror encoder)
        self.decoder_blocks = nn.ModuleList()
        # Start from encoder_out_channels and go back down
        dec_in_ch = encoder_out_channels
        dec_out_ch = encoder_out_channels // 2
        
        for _ in range(config.num_down_blocks):
            self.decoder_blocks.append(DeconvBlock1d(dec_in_ch, dec_out_ch))
            dec_in_ch = dec_out_ch
            dec_out_ch = max(dec_out_ch // 2, config.base_channels)
        
        # Final conv to output
        self.final_conv = nn.Conv1d(config.base_channels, 1, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, n_samples)
        returns: (batch, latent_dim)
        """
        h = x
        for block in self.encoder_blocks:
            h = block(h)
        h = h.view(h.size(0), -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim)
        returns: (batch, 1, n_samples)
        """
        h = self.fc_decode(z)
        h = h.view(z.size(0), self._encoded_shape[1], self._encoded_shape[2])
        for block in self.decoder_blocks:
            h = block(h)
        out = self.final_conv(h)
        out = torch.tanh(out)
        
        # Ensure output length matches config.n_samples
        if out.size(-1) > self.config.n_samples:
            out = out[..., : self.config.n_samples]
        elif out.size(-1) < self.config.n_samples:
            pad = self.config.n_samples - out.size(-1)
            out = torch.nn.functional.pad(out, (0, pad))
        
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          recon: reconstructed waveform
          z: latent code
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def create_model(config: Optional[AutoencoderConfig] = None) -> KickConvAutoencoder:
    if config is None:
        config = AutoencoderConfig()
    return KickConvAutoencoder(config)


if __name__ == "__main__":
    # Test instantiation and forward pass
    config = AutoencoderConfig()
    model = KickConvAutoencoder(config)
    
    # Create random input
    batch_size = 4
    x = torch.randn(batch_size, 1, config.n_samples)
    
    # Forward pass
    recon, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify shapes
    assert z.shape == (batch_size, config.latent_dim), f"Expected latent shape {(batch_size, config.latent_dim)}, got {z.shape}"
    assert recon.shape == (batch_size, 1, config.n_samples), f"Expected recon shape {(batch_size, 1, config.n_samples)}, got {recon.shape}"
    print("✓ All tests passed!")
