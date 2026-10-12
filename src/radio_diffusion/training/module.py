"""Lightning module for training a DDPM noise predictor."""

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.nn import functional as F

from radio_diffusion.diffusion.ddpm import GaussianDiffusion


class DDPMModule(LightningModule):
    """Train a model to predict the noise added at random diffusion steps."""

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.model = model
        self.diffusion = diffusion
        self.learning_rate = learning_rate

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(noisy_images, timesteps)

    def _shared_step(self, images: torch.Tensor, stage: str) -> torch.Tensor:
        timesteps = torch.randint(
            0,
            self.diffusion.num_steps,
            (len(images),),
            device=images.device,
        )
        noisy_images, noise = self.diffusion.add_noise(images, timesteps)
        predicted_noise = self(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)

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
        return self._shared_step(images, "train")

    def validation_step(self, images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(images, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
