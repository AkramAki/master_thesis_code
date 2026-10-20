"""Callbacks for reviewing generated samples during training."""

from pathlib import Path

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch import LightningModule
from torchvision.utils import save_image


class SampleGridCallback(Callback):
    """Save fixed-seed DDPM samples as a grid and a raw tensor periodically."""

    def __init__(
        self,
        output_dir: str | Path,
        shape: tuple[int, int, int, int],
        every_n_epochs: int = 10,
        seed: int = 42,
        nrow: int = 4,
    ) -> None:
        super().__init__()
        if len(shape) != 4 or any(size < 1 for size in shape):
            raise ValueError("shape must contain four positive dimensions [B, C, H, W].")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive.")
        if nrow < 1:
            raise ValueError("nrow must be positive.")

        self.output_dir = Path(output_dir)
        self.shape = shape
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.nrow = nrow

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        epoch_number = trainer.current_epoch + 1
        if epoch_number % self.every_n_epochs != 0 or not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        device = pl_module.device
        cuda_devices = (
            [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
            if device.type == "cuda"
            else []
        )
        # Sampling uses torch's RNG; restore the training RNG after using the
        # same initial latent noise at every preview interval.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(self.seed)
            for cuda_device in cuda_devices:
                torch.cuda.default_generators[cuda_device].manual_seed(self.seed)
            samples = pl_module.diffusion.sample(
                model=pl_module.model,
                shape=self.shape,
                device=device,
            )

        samples = samples.detach().cpu()
        filename = f"epoch={epoch_number:04d}-step={trainer.global_step:08d}"
        torch.save(samples, self.output_dir / f"{filename}.pt")
        display_samples = ((samples + 1) / 2).clamp(0, 1)
        save_image(
            display_samples,
            self.output_dir / f"{filename}.png",
            nrow=self.nrow,
        )
