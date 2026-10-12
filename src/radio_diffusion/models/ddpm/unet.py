"""Building blocks for the noise-predicting U-Net."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Represent each diffusion timestep as a vector of sine and cosine values."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even number of at least 2.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a 1D tensor with one value per image.")

        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, device=timesteps.device)
            / half_dim
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class TimeConditionedConvBlock(nn.Module):
    """Process image features while conditioning them on a timestep embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if out_channels % num_groups != 0:
            raise ValueError("out_channels must be divisible by num_groups.")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.time_projection = nn.Linear(time_embedding_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self,
        features: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual(features)

        hidden = self.norm1(self.conv1(features))
        time_features = self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.activation(hidden + time_features)
        hidden = self.norm2(self.conv2(hidden))

        return self.activation(hidden + residual)
