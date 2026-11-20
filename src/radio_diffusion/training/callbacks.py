"""Callbacks for reviewing generated samples during training."""

import json
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from radio_diffusion.plotting import save_sample_grid


_CONDITIONING_LABELS = {
    "noise_rms_jy_per_beam": "Noise RMS (Jy/beam)",
    "peak_brightness_jy_per_beam": "Peak (Jy/beam)",
    "jet_pa_deg": "Jet PA (deg)",
    "jet_radial_extent_mas": "Jet extent (mas)",
    "ridgeline_arc_clipped": "Ridge arc clipped",
    "cropped_image_pixel_maximum": "Crop maximum (Jy/beam)",
    "cropped_image_pixel_minimum": "Crop minimum (Jy/beam)",
    "cropped_image_pixel_mean": "Crop mean (Jy/beam)",
    "cropped_image_pixel_std": "Crop std (Jy/beam)",
}


class SampleGridCallback(Callback):
    """Save fixed-seed final grids and four-example denoising trajectories.

    Conditioning is either explicit scalar/list values or seeded complete rows
    drawn once from the training split. Every artifact uses the same generated
    batch. Final images use physical intensities; denoising snapshots retain
    model coordinates. Local generators preserve the training RNG state.
    """

    def __init__(
        self,
        output_dir: str | Path,
        shape: tuple[int, int, int, int],
        every_n_epochs: int = 10,
        seed: int = 42,
        nrow: int = 4,
        save_final_grid: bool = True,
        save_denoising_grids: bool = True,
        conditioning_params: list[str] | None = None,
        conditioning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if len(shape) != 4 or any(size < 1 for size in shape):
            raise ValueError("shape must contain four positive dimensions [B, C, H, W].")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive.")
        if nrow < 1:
            raise ValueError("nrow must be positive.")
        if not isinstance(save_final_grid, bool) or not isinstance(save_denoising_grids, bool):
            raise ValueError("Sample grid switches must be true or false.")
        if conditioning is not None and not isinstance(conditioning, dict):
            raise ValueError("samples.conditioning must be a table.")

        self.output_dir = Path(output_dir)
        self.shape = shape
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        self.nrow = nrow
        self.save_final_grid = save_final_grid
        self.save_denoising_grids = save_denoising_grids
        self.conditioning_params = tuple(conditioning_params or ())
        settings = dict(conditioning or {})
        self.conditioning_mode = settings.pop("mode", "manual")
        if self.conditioning_mode not in ("manual", "auto"):
            raise ValueError("samples.conditioning.mode must be 'manual' or 'auto'.")
        self.conditioning_seed = settings.pop("seed", seed)
        if isinstance(self.conditioning_seed, bool) or not isinstance(self.conditioning_seed, int):
            raise ValueError("samples.conditioning.seed must be an integer.")
        if not self.conditioning_params:
            if conditioning:
                rank_zero_warn(
                    "Omitting samples.conditioning because no model conditioning fields are active.",
                    stacklevel=1,
                )
            self.conditioning_mode = "unconditional"
            settings = {}
        elif self.conditioning_mode == "auto" and settings:
            raise ValueError("Automatic sample conditioning does not accept manual field values.")
        elif self.conditioning_mode == "manual" and (
            self.save_final_grid or self.save_denoising_grids
        ):
            if set(settings) != set(self.conditioning_params):
                raise ValueError(
                    "Manual samples.conditioning must supply exactly these fields: "
                    + ", ".join(self.conditioning_params)
                )
        self.conditioning_values = settings
        self._conditioning_batch: dict | None = None

    def _prepare_conditioning(self, trainer: Trainer) -> None:
        if self._conditioning_batch is not None:
            return
        if self.conditioning_mode == "auto":
            data = trainer.datamodule
            if data is None or not hasattr(data, "sample_conditioning"):
                raise ValueError("Automatic sample conditioning requires a training data module.")
            self._conditioning_batch = data.sample_conditioning(
                self.shape[0], self.conditioning_seed
            )
            return

        metadata = {}
        for field, configured in self.conditioning_values.items():
            try:
                value = torch.as_tensor(configured)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"Sample conditioning {field!r} must be numerical or boolean."
                ) from error
            if value.ndim == 0:
                value = value.expand(self.shape[0]).clone()
            if value.is_complex() or tuple(value.shape) != (self.shape[0],):
                raise ValueError(
                    f"Sample conditioning {field!r} must be a scalar or a list "
                    f"of exactly {self.shape[0]} values."
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"Sample conditioning {field!r} must contain finite values.")
            metadata[field] = value
        self._conditioning_batch = {
            "metadata": metadata,
            "metadata_mask": {
                field: torch.ones(self.shape[0], dtype=torch.bool) for field in metadata
            },
            "source_observations": [None] * self.shape[0],
        }

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.is_global_zero and (self.save_final_grid or self.save_denoising_grids):
            self._prepare_conditioning(trainer)

    def _sample_records(self) -> list[dict]:
        batch = self._conditioning_batch
        return [
            {
                "sample_id": f"S{index + 1:02d}",
                "conditioning": {
                    field: value[index].item() for field, value in batch["metadata"].items()
                },
                "validity": {
                    field: bool(value[index])
                    for field, value in batch["metadata_mask"].items()
                },
                "source_observation": batch["source_observations"][index],
            }
            for index in range(self.shape[0])
        ]

    @staticmethod
    def _sample_labels(records: list[dict]) -> list[str]:
        labels = []
        for record in records:
            lines = [record["sample_id"]]
            for field, value in record["conditioning"].items():
                name = _CONDITIONING_LABELS.get(field, field)
                formatted = str(value) if isinstance(value, bool) else f"{value:.4g}"
                if not record["validity"][field]:
                    formatted = f"missing → {formatted}"
                lines.append(f"{name}: {formatted}")
            labels.append("\n".join(lines))
        return labels

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
            or not (self.save_final_grid or self.save_denoising_grids)
        ):
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        samples, snapshots, stage_steps, use_masks = self._generate_samples(trainer, pl_module)
        samples, normalization = self._restore_sample_scale(trainer, samples)
        filename = f"epoch={epoch_number:04d}-step={trainer.global_step:08d}"
        torch.save(samples, self.output_dir / f"{filename}.pt")
        records = self._sample_records()
        manifest_path = self._write_sample_manifest(
            trainer, pl_module, filename, records, stage_steps, use_masks, normalization,
        )
        grid_paths = [manifest_path] + self._save_sample_grids(
            samples, snapshots, records, filename, epoch_number,
            stage_steps, pl_module.diffusion.num_steps, use_masks,
        )
        print(
            f"Saved {self.shape[0]} samples for epoch {epoch_number}: {self.output_dir}\n"
            f"Final grid: {self.save_final_grid}; denoising grids: {self.save_denoising_grids}",
            flush=True,
        )
        self._log_artifacts(trainer, grid_paths)

    def _generate_samples(self, trainer: Trainer, pl_module: LightningModule):
        """Generate samples and retain denoising snapshots in model coordinates."""
        device = pl_module.device
        total_steps = pl_module.diffusion.num_steps
        stage_steps = [round(total_steps * index / 3) for index in range(4)]
        snapshots = {}
        self._prepare_conditioning(trainer)
        batch = self._conditioning_batch

        def capture_snapshot(completed_steps: int, images: torch.Tensor) -> None:
            if completed_steps in stage_steps:
                snapshots[completed_steps] = images.detach().cpu().clone()

        use_masks = getattr(pl_module.model, "use_validity_masks", False)
        samples = pl_module.diffusion.sample(
            model=pl_module.model,
            shape=self.shape,
            device=device,
            metadata=batch["metadata"] or None,
            metadata_mask=(batch["metadata_mask"] if use_masks else None),
            generator=torch.Generator(device=device).manual_seed(self.seed),
            on_step=capture_snapshot if self.save_denoising_grids else None,
        )

        return samples.detach().cpu(), snapshots, stage_steps, use_masks

    def _restore_sample_scale(self, trainer: Trainer, samples: torch.Tensor):
        """Use the data module's mapping and physical per-image conditioning."""
        normalizer = getattr(trainer.datamodule, "normalizer", None)
        if normalizer is None:
            return samples, None
        metadata = {**trainer.datamodule.dataset_metadata, **self._conditioning_batch["metadata"]}
        return normalizer.inverse(samples, metadata), trainer.datamodule.state_dict()["normalization"]

    def _write_sample_manifest(
        self, trainer, pl_module, filename, records, stage_steps, use_masks, normalization,
    ) -> Path:
        """Record the actual inverse mapping alongside physical final samples."""
        manifest_path = self.output_dir / f"{filename}.json"
        manifest = {
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "noise_seed": self.seed,
            "conditioning_mode": self.conditioning_mode,
            "conditioning_seed": self.conditioning_seed,
            "use_validity_masks": use_masks,
            "total_diffusion_steps": pl_module.diffusion.num_steps,
            "denoising_completed_steps": stage_steps if self.save_denoising_grids else [],
            "eligible_observations": self._conditioning_batch.get("eligible_observations"),
            "with_replacement": self._conditioning_batch.get("with_replacement", False),
            "normalization": normalization,
            "sample_coordinate_space": "physical",
            "denoising_coordinate_space": "model",
            "samples": records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest_path

    def _save_sample_grids(
        self, samples, snapshots, records, filename, epoch_number,
        stage_steps, total_steps, use_masks,
    ) -> list[Path]:
        """Plot physical final samples and the model-space denoising trajectory."""
        sample_labels = self._sample_labels(records)
        conditioning_title = (
            "Unconditional" if not self.conditioning_params
            else f"Conditioning: {self.conditioning_mode}; validity masks: "
            + ("on" if use_masks else "off")
        )
        grid_paths = []
        if self.save_final_grid:
            final_path = self.output_dir / f"{filename}.png"
            save_sample_grid(
                samples, final_path, nrow=self.nrow,
                title=f"Epoch {epoch_number} — final samples\n{conditioning_title}",
                sample_labels=sample_labels,
            )
            grid_paths.append(final_path)
        if self.save_denoising_grids:
            grid_paths.extend(self._save_denoising_grids(
                snapshots, filename, sample_labels, epoch_number,
                conditioning_title, stage_steps, total_steps,
            ))
        return grid_paths

    def _save_denoising_grids(
        self, snapshots, filename, sample_labels, epoch_number,
        conditioning_title, stage_steps, total_steps,
    ) -> list[Path]:
        """Save four-sample groups without inverse-transforming noisy states."""
        progress_dir = self.output_dir / "denoising" / filename
        column_labels = [
            (
                "Initial noise" if step == 0 else "Final" if step == total_steps
                else f"{total_steps - step} noise steps remaining"
            ) + f"\n{step}/{total_steps} reverse steps"
            for step in stage_steps
        ]
        grid_paths = []
        for start in range(0, self.shape[0], 4):
            stop = min(start + 4, self.shape[0])
            trajectory = torch.stack([snapshots[step][start:stop] for step in stage_steps], dim=1)
            path = progress_dir / f"samples={start + 1:02d}-{stop:02d}.png"
            save_sample_grid(
                trajectory.flatten(0, 1), path, nrow=4,
                title=(
                    f"Epoch {epoch_number} — samples {start + 1}–{stop} denoising\n"
                    f"{conditioning_title}"
                ),
                row_labels=sample_labels[start:stop], column_labels=column_labels,
            )
            grid_paths.append(path)
        return grid_paths

    def _log_artifacts(self, trainer: Trainer, grid_paths: list[Path]) -> None:
        """Upload the same locally saved artifacts when MLflow logging is active."""
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                for path in grid_paths:
                    relative_parent = path.parent.relative_to(self.output_dir)
                    artifact_path = (Path("samples") / relative_parent).as_posix()
                    logger.experiment.log_artifact(
                        logger.run_id, str(path), artifact_path=artifact_path,
                    )
