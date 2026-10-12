"""Reusable U-Net architectures and building blocks."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encode diffusion timesteps using sinusoidal embeddings.

    Args:
        embedding_dim: Size of the embedding vector. Must be even and at least 2.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even number of at least 2.")
        self.embedding_dim = embedding_dim


    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Convert timesteps into sinusoidal embeddings.

        Args:
            timesteps: Tensor of shape [B] containing one timestep per image.

        Returns:
            Tensor of shape [B, embedding_dim] containing the time embeddings.
        """
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
    """Residual convolution block conditioned on diffusion time.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        time_embedding_dim: Size of the timestep embedding vector.
        num_groups: Number of groups used for GroupNorm.
    """

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
        """Process image features using timestep conditioning.

        Args:
            features: Input feature maps of shape [B, in_channels, H, W].
            time_embedding: Timestep embeddings of shape [B, time_embedding_dim].

        Returns:
            Processed feature maps of shape [B, out_channels, H, W].
        """
        residual = self.residual(features)

        hidden = self.norm1(self.conv1(features))
        time_features = self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.activation(hidden + time_features)
        hidden = self.norm2(self.conv2(hidden))

        return self.activation(hidden + residual)


class CompactUNet(nn.Module):
    """Compact time-conditioned U-Net for diffusion noise prediction.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output channels. Defaults to in_channels.
        channels: Feature channel sizes for the two encoder levels and bottleneck.
        time_embedding_dim: Size of the sinusoidal timestep embedding.
        num_groups: Number of groups used for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | None = None,
        channels: tuple[int, int, int] = (32, 64, 128),
        time_embedding_dim: int = 64,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if out_channels is not None and out_channels < 1:
            raise ValueError("out_channels must be positive.")
        if len(channels) != 3 or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain three positive channel sizes.")

        encoder_channels, middle_channels, bottleneck_channels = channels
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.channels = channels

        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)

        self.encoder_first = TimeConditionedConvBlock(
            in_channels,
            encoder_channels,
            time_embedding_dim,
            num_groups=num_groups,
        )
        self.down_first = nn.Conv2d(
            encoder_channels,
            middle_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder_second = TimeConditionedConvBlock(
            middle_channels,
            middle_channels,
            time_embedding_dim,
            num_groups=num_groups,
        )
        self.down_second = nn.Conv2d(
            middle_channels,
            bottleneck_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.bottleneck = TimeConditionedConvBlock(
            bottleneck_channels,
            bottleneck_channels,
            time_embedding_dim,
            num_groups=num_groups,
        )

        self.up_second = nn.ConvTranspose2d(
            bottleneck_channels,
            middle_channels,
            kernel_size=2,
            stride=2,
        )
        self.decoder_second = TimeConditionedConvBlock(
            middle_channels + middle_channels,
            middle_channels,
            time_embedding_dim,
            num_groups=num_groups,
        )
        self.up_first = nn.ConvTranspose2d(
            middle_channels,
            encoder_channels,
            kernel_size=2,
            stride=2,
        )
        self.decoder_first = TimeConditionedConvBlock(
            encoder_channels + encoder_channels,
            encoder_channels,
            time_embedding_dim,
            num_groups=num_groups,
        )
        self.output = nn.Conv2d(encoder_channels, self.out_channels, kernel_size=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy images and their diffusion timesteps.

        Args:
            noisy_images: Images of shape [B, in_channels, H, W].
                Height and width must be divisible by 4.
            timesteps: Tensor of shape [B] containing one timestep per image.

        Returns:
            Predicted noise of shape [B, out_channels, H, W].
        """
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.in_channels:
            raise ValueError(
                "noisy_images must be shaped [B, in_channels, H, W]."
            )
        if noisy_images.shape[2] % 4 != 0 or noisy_images.shape[3] % 4 != 0:
            raise ValueError("image height and width must be divisible by 4.")
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("timesteps must be shaped [B].")

        time_embedding = self.time_embedding(timesteps)

        first_skip = self.encoder_first(noisy_images, time_embedding)
        hidden = self.down_first(first_skip)
        second_skip = self.encoder_second(hidden, time_embedding)
        hidden = self.down_second(second_skip)

        hidden = self.bottleneck(hidden, time_embedding)

        hidden = self.up_second(hidden)
        hidden = torch.cat((hidden, second_skip), dim=1)
        hidden = self.decoder_second(hidden, time_embedding)

        hidden = self.up_first(hidden)
        hidden = torch.cat((hidden, first_skip), dim=1)
        hidden = self.decoder_first(hidden, time_embedding)

        return self.output(hidden)
