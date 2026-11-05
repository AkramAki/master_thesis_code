"""Configurable Diffusion Transformer for pixel-space noise prediction."""

import math

import torch
from torch import nn


def _sinusoidal_embedding(values: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Return a sinusoidal embedding for a one-dimensional batch of values."""
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half_dim, device=values.device, dtype=torch.float32)
        / half_dim
    )
    angles = values.float()[:, None] * frequencies[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def _position_embedding(
    grid_height: int,
    grid_width: int,
    embedding_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a fixed 2D sinusoidal position embedding for a patch grid."""
    axis_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(axis_dim // 2, device=device, dtype=torch.float32)
        / (axis_dim // 2)
    )
    y_positions, x_positions = torch.meshgrid(
        torch.arange(grid_height, device=device, dtype=torch.float32),
        torch.arange(grid_width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    y_angles = y_positions.reshape(-1, 1) * frequencies[None, :]
    x_angles = x_positions.reshape(-1, 1) * frequencies[None, :]
    y_embedding = torch.cat((y_angles.sin(), y_angles.cos()), dim=1)
    x_embedding = torch.cat((x_angles.sin(), x_angles.cos()), dim=1)
    return torch.cat((y_embedding, x_embedding), dim=1)


class DiTBlock(nn.Module):
    """Transformer block with timestep-conditioned AdaLN-Zero modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(condition).chunk(6, dim=-1)

        attention_input = self.norm1(tokens) * (1 + scale_attention[:, None])
        attention_input = attention_input + shift_attention[:, None]
        attended, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        tokens = tokens + gate_attention[:, None] * attended

        mlp_input = self.norm2(tokens) * (1 + scale_mlp[:, None])
        mlp_input = mlp_input + shift_mlp[:, None]
        return tokens + gate_mlp[:, None] * self.mlp(mlp_input)


class DiT(nn.Module):
    """Configurable pixel-space Diffusion Transformer.

    The model splits an image into non-overlapping patches, processes the patch
    tokens with timestep-conditioned transformer blocks, and maps the output
    tokens back to a noise prediction with the same shape as the input.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of predicted noise channels. Defaults to
            ``in_channels``.
        patch_size: Height and width of each square image patch.
        hidden_size: Transformer token and timestep-embedding size. Must be
            divisible by both four and ``num_heads``.
        depth: Number of transformer blocks.
        num_heads: Number of self-attention heads in each block.
        mlp_ratio: Hidden-layer width in each block's MLP, relative to
            ``hidden_size``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | None = None,
        patch_size: int = 8,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if out_channels is not None and out_channels < 1:
            raise ValueError("out_channels must be positive.")
        if patch_size < 1:
            raise ValueError("patch_size must be positive.")
        if hidden_size < 4 or hidden_size % 4 != 0:
            raise ValueError("hidden_size must be a positive multiple of 4.")
        if num_heads < 1 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if depth < 1:
            raise ValueError("depth must be positive.")
        if (
            isinstance(mlp_ratio, bool)
            or not isinstance(mlp_ratio, (int, float))
            or not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0
            or int(hidden_size * mlp_ratio) < 1
        ):
            raise ValueError("mlp_ratio must be a finite positive number.")

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embedding = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.output_projection = nn.Linear(
            hidden_size,
            patch_size * patch_size * self.out_channels,
        )

        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise for images shaped ``[B, C, H, W]`` and timesteps ``[B]``."""
        if (
            noisy_images.ndim != 4
            or not noisy_images.is_floating_point()
            or noisy_images.shape[1] != self.in_channels
        ):
            raise ValueError(
                "noisy_images must be floating-point and shaped "
                "[B, in_channels, H, W]."
            )
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("timesteps must be shaped [B].")
        if (
            noisy_images.shape[2] % self.patch_size != 0
            or noisy_images.shape[3] % self.patch_size != 0
        ):
            raise ValueError(
                "image height and width must be divisible by patch_size."
            )

        hidden = self.patch_embedding(noisy_images)
        batch_size, _, grid_height, grid_width = hidden.shape
        tokens = hidden.flatten(2).transpose(1, 2)
        position_embedding = _position_embedding(
            grid_height,
            grid_width,
            self.hidden_size,
            noisy_images.device,
        )
        tokens = tokens + position_embedding.to(dtype=tokens.dtype)[None, :, :]

        condition = self.time_embedding(
            _sinusoidal_embedding(timesteps, self.hidden_size).to(dtype=tokens.dtype)
        )
        for block in self.blocks:
            tokens = block(tokens, condition)

        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        patch_predictions = self.output_projection(tokens)
        patch_predictions = patch_predictions.reshape(
            batch_size,
            grid_height,
            grid_width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        # after reshape, the axes are:
        # [batch, patch_row, patch_col, pixel_row, pixel_col, channel] but need
        # [batch, channel, patch_row, pixel_row, patch_col, pixel_col]
        return patch_predictions.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            self.out_channels,
            grid_height * self.patch_size,
            grid_width * self.patch_size,
        )
