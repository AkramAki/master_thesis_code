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


class SpatialSelfAttention(nn.Module):
    """Apply multi-head self-attention across the spatial image features."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        if channels % num_groups != 0:
            raise ValueError("channels must be divisible by num_groups.")

        self.norm = nn.GroupNorm(num_groups, channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        tokens = self.norm(features).flatten(2).transpose(1, 2)
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return features + attended


class UNet(nn.Module):
    """Configurable time-conditioned U-Net for diffusion noise prediction.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output channels. Defaults to ``in_channels``.
        channels: Feature channels at each resolution, from highest to lowest.
            The image height and width must be divisible by
            ``2 ** (len(channels) - 1)``.
        blocks_per_level: Number of residual blocks on the encoder at each
            level, either one value shared across levels or one value per level.
            The deepest level may use zero blocks because it has a separate
            bottleneck block. Decoder levels always have at least one block to
            combine features with their skip connection.
        time_embedding_dim: Size of the sinusoidal timestep embedding.
        num_groups: Number of groups used for GroupNorm.
        attention_levels: Resolution-level indices that use spatial self-attention.
            Level zero is the highest-resolution level. Attention is most
            practical at lower-resolution levels.
        attention_heads: Number of attention heads at each selected level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | None = None,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        blocks_per_level: int | tuple[int, ...] = 2,
        time_embedding_dim: int = 128,
        num_groups: int = 8,
        attention_levels: tuple[int, ...] = (),
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if out_channels is not None and out_channels < 1:
            raise ValueError("out_channels must be positive.")
        if len(channels) < 2 or any(channel < 1 for channel in channels):
            raise ValueError("channels must contain at least two positive sizes.")
        if num_groups < 1 or attention_heads < 1:
            raise ValueError("num_groups and attention_heads must be positive.")

        levels = len(channels)
        if isinstance(blocks_per_level, int):
            block_counts = (blocks_per_level,) * levels
        else:
            block_counts = tuple(blocks_per_level)
            if len(block_counts) != levels:
                raise ValueError(
                    "blocks_per_level must have one value per resolution level."
                )
        if any(count < 0 for count in block_counts):
            raise ValueError("blocks_per_level values must be nonnegative.")
        if any(count < 1 for count in block_counts[:-1]):
            raise ValueError(
                "Every level except the deepest must have at least one block."
            )
        if len(set(attention_levels)) != len(attention_levels):
            raise ValueError("attention_levels must not contain duplicates.")
        if any(level < 0 or level >= levels for level in attention_levels):
            raise ValueError("attention_levels contain an invalid level index.")

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.channels = channels
        self.blocks_per_level = block_counts
        self.attention_levels = frozenset(attention_levels)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attention = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for level, level_channels in enumerate(channels):
            self.encoder_blocks.append(
                nn.ModuleList(
                    TimeConditionedConvBlock(
                        in_channels if level == 0 and block_index == 0 else level_channels,
                        level_channels,
                        time_embedding_dim,
                        num_groups=num_groups,
                    )
                    for block_index in range(block_counts[level])
                )
            )
            self.encoder_attention.append(
                SpatialSelfAttention(
                    level_channels,
                    num_heads=attention_heads,
                    num_groups=num_groups,
                )
                if level in self.attention_levels
                else nn.Identity()
            )
            if level < levels - 1:
                self.downsample.append(
                    nn.Conv2d(
                        level_channels,
                        channels[level + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )

        bottleneck_channels = channels[-1]
        self.bottleneck = TimeConditionedConvBlock(
            bottleneck_channels,
            bottleneck_channels,
            time_embedding_dim,
            num_groups=num_groups,
        )

        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        for level in reversed(range(levels - 1)):
            level_channels = channels[level]
            self.upsample.append(
                nn.ConvTranspose2d(
                    channels[level + 1],
                    level_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            blocks = [
                TimeConditionedConvBlock(
                    level_channels * 2,
                    level_channels,
                    time_embedding_dim,
                    num_groups=num_groups,
                )
            ]
            blocks.extend(
                TimeConditionedConvBlock(
                    level_channels,
                    level_channels,
                    time_embedding_dim,
                    num_groups=num_groups,
                )
                for _ in range(block_counts[level] - 1)
            )
            self.decoder_blocks.append(nn.ModuleList(blocks))
            self.decoder_attention.append(
                SpatialSelfAttention(
                    level_channels,
                    num_heads=attention_heads,
                    num_groups=num_groups,
                )
                if level in self.attention_levels
                else nn.Identity()
            )

        self.output_conv = nn.Conv2d(channels[0], self.out_channels, kernel_size=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from images shaped ``[B, C, H, W]`` and timesteps ``[B]``."""
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.in_channels:
            raise ValueError(
                "noisy_images must be shaped [B, in_channels, H, W]."
            )
        downsample_factor = 2 ** (len(self.channels) - 1)
        if (
            noisy_images.shape[2] % downsample_factor != 0
            or noisy_images.shape[3] % downsample_factor != 0
        ):
            raise ValueError(
                "image height and width must be divisible by "
                f"{downsample_factor}."
            )
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("timesteps must be shaped [B].")

        time_embedding = self.time_embedding(timesteps)
        hidden = noisy_images
        skip_features = []

        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                hidden = block(hidden, time_embedding)
            hidden = self.encoder_attention[level](hidden)
            if level < len(self.channels) - 1:
                skip_features.append(hidden)
                hidden = self.downsample[level](hidden)

        hidden = self.bottleneck(hidden, time_embedding)
        for upsample, blocks, attention, skip in zip(
            self.upsample,
            self.decoder_blocks,
            self.decoder_attention,
            reversed(skip_features),
        ):
            hidden = upsample(hidden)
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, time_embedding)
            hidden = attention(hidden)

        return self.output_conv(hidden)


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
