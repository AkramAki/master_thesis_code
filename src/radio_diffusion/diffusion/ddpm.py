"""Gaussian forward diffusion for denoising diffusion models."""

from typing import Callable

import torch
from torch import nn

from tqdm.auto import tqdm


class GaussianDiffusion(nn.Module):
    """Manage a Gaussian noise schedule and its forward diffusion process.

    By default the beta schedule is linear. Passing ``betas`` allows a custom
    schedule while retaining the same precomputed diffusion coefficients.

    Args:
        num_steps: Number of steps in the default linear schedule.
        beta_start: First variance in the default linear schedule.
        beta_end: Last variance in the default linear schedule.
        betas: Optional custom one-dimensional variance schedule.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        betas: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if betas is None:
            if num_steps < 1:
                raise ValueError("num_steps must be positive.")
            if not 0 < beta_start <= beta_end < 1:
                raise ValueError(
                    "The schedule requires 0 < beta_start <= beta_end < 1."
                )
            betas = torch.linspace(
                beta_start,
                beta_end,
                num_steps,
                dtype=torch.float32,
            )
        elif (
            betas.ndim != 1
            or betas.numel() == 0
            or not ((betas > 0) & (betas < 1)).all()
        ):
            raise ValueError(
                "betas must be a nonempty 1D tensor with values between 0 and 1."
            )

        # Keep schedule calculations in at least float32 for lower-precision images.
        schedule_dtype = (
            torch.float64 if betas.dtype == torch.float64 else torch.float32
        )
        betas = betas.detach().clone().to(dtype=schedule_dtype)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        alphas_cumprod_previous = torch.cat(
            (
                torch.ones(1, dtype=schedule_dtype, device=betas.device),
                alphas_cumprod[:-1],
            )
        )
        posterior_variance = (
            betas
            * (1 - alphas_cumprod_previous)
            / (1 - alphas_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            (1 - alphas_cumprod).sqrt(),
        )
        self.register_buffer("sqrt_reciprocal_alphas", alphas.rsqrt())
        self.register_buffer("posterior_variance", posterior_variance)

    @property
    def num_steps(self) -> int:
        """Return the number of steps in the noise schedule."""
        return self.betas.numel()

    def add_noise(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return noisy images and the Gaussian noise used to construct them.

        Images have shape ``[B, C, H, W]``. Timesteps are a ``torch.long``
        tensor of shape ``[B]`` selecting one schedule index per image. A
        supplied noise tensor must match the images; otherwise fresh standard
        normal noise is sampled. Outputs are not clipped.
        """
        if images.ndim != 4 or not images.is_floating_point():
            raise ValueError(
                "images must be a floating-point tensor shaped [B, C, H, W]."
            )
        if timesteps.shape != (images.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError(
                "timesteps must have dtype torch.long and shape [B].")
        if ((timesteps < 0) | (timesteps >= self.num_steps)).any():
            raise ValueError("timesteps must be between 0 and num_steps - 1.")
        if noise is None:
            noise = torch.randn_like(images)
        elif (
            noise.shape != images.shape
            or noise.dtype != images.dtype
            or noise.device != images.device
        ):
            raise ValueError(
                "noise must match the images' shape, dtype, and device.")

        coefficient_indices = timesteps.to(self.betas.device)
        signal_scale = self.sqrt_alphas_cumprod[coefficient_indices]
        noise_scale = self.sqrt_one_minus_alphas_cumprod[coefficient_indices]
        signal_scale = signal_scale.to(
            device=images.device, dtype=images.dtype)
        noise_scale = noise_scale.to(device=images.device, dtype=images.dtype)
        signal_scale = signal_scale.view(-1, 1, 1, 1)
        noise_scale = noise_scale.view(-1, 1, 1, 1)

        noisy_images = signal_scale * images + noise_scale * noise
        return noisy_images, noise

    def forward_trajectory(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return clean images and their forward-diffusion states.

        ``images`` has shape ``[B, C, H, W]`` and ``timesteps`` is a one-
        dimensional ``torch.long`` tensor of schedule indices. The returned
        trajectory has shape ``[B, T + 1, C, H, W]``: column zero contains the
        clean images, followed by one column for each requested timestep. The
        same Gaussian noise is used at every timestep so the columns show the
        progressive corruption of each image. The returned timesteps are on
        the same device as ``images``.
        """
        if images.ndim != 4 or not images.is_floating_point():
            raise ValueError(
                "images must be a floating-point tensor shaped [B, C, H, W]."
            )
        if timesteps.ndim != 1 or timesteps.dtype != torch.long:
            raise ValueError(
                "timesteps must be a one-dimensional torch.long tensor.")
        if timesteps.numel() == 0:
            raise ValueError(
                "timesteps must contain at least one schedule index.")
        if ((timesteps < 0) | (timesteps >= self.num_steps)).any():
            raise ValueError("timesteps must be between 0 and num_steps - 1.")

        if noise is None:
            noise = torch.randn_like(images)

        trajectory = [images]
        device_timesteps = timesteps.to(images.device)
        for timestep in device_timesteps:
            batch_timesteps = timestep.expand(images.shape[0])
            noisy_images, _ = self.add_noise(
                images, batch_timesteps, noise=noise)
            trajectory.append(noisy_images)

        return torch.stack(trajectory, dim=1), device_timesteps

    def reverse_step(
        self,
        model: nn.Module,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict and sample the images at the preceding diffusion step."""
        if noisy_images.ndim != 4 or not noisy_images.is_floating_point():
            raise ValueError(
                "noisy_images must be a floating-point tensor shaped [B, C, H, W]."
            )
        if (
            timesteps.shape != (noisy_images.shape[0],)
            or timesteps.dtype != torch.long
        ):
            raise ValueError(
                "timesteps must have dtype torch.long and shape [B].")
        if ((timesteps < 0) | (timesteps >= self.num_steps)).any():
            raise ValueError("timesteps must be between 0 and num_steps - 1.")

        predicted_noise = model(noisy_images, timesteps)
        if predicted_noise.shape != noisy_images.shape:
            raise ValueError(
                "model output must have the same shape as noisy_images.")

        coefficient_indices = timesteps.to(self.betas.device)
        mean_scale = self.sqrt_reciprocal_alphas[coefficient_indices]
        noise_scale = (
            self.betas[coefficient_indices]
            / self.sqrt_one_minus_alphas_cumprod[coefficient_indices]
        )
        variance = self.posterior_variance[coefficient_indices]

        mean_scale = mean_scale.to(noisy_images).view(-1, 1, 1, 1)
        noise_scale = noise_scale.to(noisy_images).view(-1, 1, 1, 1)
        variance = variance.to(noisy_images).view(-1, 1, 1, 1)

        model_mean = mean_scale * \
            (noisy_images - noise_scale * predicted_noise)
        random_noise = torch.randn_like(noisy_images)
        nonzero_mask = (timesteps != 0).to(noisy_images).view(-1, 1, 1, 1)
        return model_mean + nonzero_mask * variance.sqrt() * random_noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device | str | None = None,
        *,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Generate images by repeatedly applying the learned reverse process.

        ``on_step(completed_steps, images)`` optionally observes the initial
        noise (step zero) and each reverse step. Observers must not modify the
        images. The return value remains the final image batch.
        """
        if len(shape) != 4 or any(size < 1 for size in shape):
            raise ValueError(
                "shape must contain four positive dimensions [B, C, H, W]."
            )
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration as exc:
                raise ValueError(
                    "model must have parameters or device must be provided.") from exc

        images = torch.randn(shape, device=device)
        was_training = model.training
        model.eval()

        try:
            if on_step is not None:
                on_step(0, images)
            for completed_steps, timestep in tqdm(
                enumerate(reversed(range(self.num_steps)), start=1),
                total=self.num_steps,
                desc="Sampling",
                unit="step",
            ):
                timesteps = torch.full(
                    (shape[0],),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )
                images = self.reverse_step(model, images, timesteps)
                if on_step is not None:
                    on_step(completed_steps, images)
        finally:
            model.train(was_training)

        return images
