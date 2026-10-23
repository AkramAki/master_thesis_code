"""Lightning module for training a DDPM noise predictor."""

from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import nn

from radio_diffusion.diffusion.ddpm import GaussianDiffusion
from radio_diffusion.training.losses import WeightedLoss


class DDPMModule(LightningModule):
    """Train a model to predict the noise added at random diffusion steps."""

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        validation_seed: int | None = None,
        loss_function: WeightedLoss | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if optimizer_name not in ("Adam", "AdamW"):
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.model = model
        self.diffusion = diffusion
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.validation_seed = validation_seed
        self.loss_function = (
            loss_function if loss_function is not None else WeightedLoss(["MSELoss"], [1.0])
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(noisy_images, timesteps)

    def _shared_step(
        self,
        images: torch.Tensor,
        stage: str,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        generator = None
        if stage == "val" and self.validation_seed is not None:
            generator = torch.Generator(device=images.device)
            generator.manual_seed(self.validation_seed + batch_idx)

        timesteps = torch.randint(
            0,
            self.diffusion.num_steps,
            (len(images),),
            device=images.device,
            generator=generator,
        )
        noise = torch.randn(
            images.shape,
            dtype=images.dtype,
            device=images.device,
            generator=generator,
        )
        noisy_images, noise = self.diffusion.add_noise(
            images, timesteps, noise=noise
        )
        predicted_noise = self(noisy_images, timesteps)
        loss, loss_terms = self.loss_function.compute(predicted_noise, noise)

        for name, term in loss_terms.items():
            self.log(
                f"{stage}_loss_{name}",
                term,
                on_step=stage == "train",
                on_epoch=True,
                batch_size=len(images),
            )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return loss

    def training_step(self, images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(images, "train", batch_idx)

    def validation_step(self, images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(images, "val", batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_type = getattr(torch.optim, self.optimizer_name)
        return optimizer_type(
            self.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwargs,
        )
