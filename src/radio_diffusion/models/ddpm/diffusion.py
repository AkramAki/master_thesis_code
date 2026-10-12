"""Forward diffusion, following DDPM equation (4): https://arxiv.org/abs/2006.11239."""

import torch


def linear_beta_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Return one noise variance per step, increasing linearly."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if not 0 < beta_start <= beta_end < 1:
        raise ValueError("The schedule requires 0 < beta_start <= beta_end < 1.")
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def add_noise(
    images: torch.Tensor,
    timesteps: torch.Tensor,
    betas: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return noisy images and the Gaussian noise used to construct them.

    Images have shape [B, C, H, W]. Timesteps are a torch.long tensor of
    shape [B], selecting one schedule index per image. Index 0 is the first
    noisy step; the original image is passed separately as ``images``.

    A supplied noise tensor must match the images and represent standard
    normal noise. Otherwise, fresh noise is sampled. Outputs are not clipped.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("images must be a floating-point tensor shaped [B, C, H, W].")
    if timesteps.shape != (images.shape[0],) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must have dtype torch.long and shape [B].")
    if betas.ndim != 1 or betas.numel() == 0 or not ((betas > 0) & (betas < 1)).all():
        raise ValueError("betas must be a nonempty 1D tensor with values between 0 and 1.")
    if ((timesteps < 0) | (timesteps >= len(betas))).any():
        raise ValueError("timesteps must be between 0 and len(betas) - 1.")
    if noise is None:
        noise = torch.randn_like(images)
    elif (
        noise.shape != images.shape
        or noise.dtype != images.dtype
        or noise.device != images.device
    ):
        raise ValueError("noise must match the images' shape, dtype, and device.")

    # alpha_bar is the fraction of signal variance retained after each step.
    # Calculate the schedule in at least float32, even for lower-precision images.
    schedule_dtype = torch.float64 if betas.dtype == torch.float64 else torch.float32
    betas = betas.to(device=images.device, dtype=schedule_dtype)
    alpha_bar = torch.cumprod(1 - betas, dim=0)
    alpha_bar_t = alpha_bar[timesteps.to(images.device)].view(-1, 1, 1, 1)

    signal_scale = alpha_bar_t.sqrt().to(images.dtype)
    noise_scale = (1 - alpha_bar_t).sqrt().to(images.dtype)
    noisy_images = signal_scale * images + noise_scale * noise
    return noisy_images, noise
