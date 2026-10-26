"""Callbacks for reviewing generated samples during training."""

from math import ceil
from pathlib import Path

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import MLFlowLogger

from radio_diffusion.plotting import save_sample_grid


class SampleGridCallback(Callback):
    """Save fixed-seed final samples and intermediate reverse-sampling grids."""

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
        if (
            trainer.sanity_checking
            or epoch_number % self.every_n_epochs != 0
            or not trainer.is_global_zero
        ):
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        device = pl_module.device
        total_steps = pl_module.diffusion.num_steps
        snapshot_steps = {
            ceil(total_steps * fraction) for fraction in (0, 0.25, 0.5, 0.75)
        }
        snapshots = {}

        def capture_snapshot(completed_steps: int, images: torch.Tensor) -> None:
            if completed_steps in snapshot_steps and completed_steps < total_steps:
                snapshots[completed_steps] = images.detach().cpu().clone()

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
                on_step=capture_snapshot,
            )

        samples = samples.detach().cpu()
        filename = f"epoch={epoch_number:04d}-step={trainer.global_step:08d}"
        torch.save(samples, self.output_dir / f"{filename}.pt")
        final_path = self.output_dir / f"{filename}.png"
        save_sample_grid(
            samples,
            final_path,
            nrow=self.nrow,
            title=(
                f"Epoch {epoch_number} — 100% sampling steps completed "
                f"({total_steps}/{total_steps})"
            ),
        )
        grid_paths = [final_path]
        progress_dir = self.output_dir / "denoising" / filename
        for completed_steps, images in sorted(snapshots.items()):
            percent = 100 * completed_steps / total_steps
            path = progress_dir / f"sampling-step={completed_steps:05d}-of-{total_steps:05d}.png"
            save_sample_grid(
                images,
                path,
                nrow=self.nrow,
                title=(
                    f"Epoch {epoch_number} — {percent:g}% sampling steps completed "
                    f"({completed_steps}/{total_steps})"
                ),
            )
            grid_paths.append(path)

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                for path in grid_paths:
                    relative_parent = path.parent.relative_to(self.output_dir)
                    artifact_path = (Path("samples") / relative_parent).as_posix()
                    logger.experiment.log_artifact(
                        logger.run_id, str(path), artifact_path=artifact_path,
                    )
