"""Lightning module for training DDPM noise or clean-image predictors."""

import os
from typing import Any

from radio_diffusion.paths import load_env

load_env()

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from torch import nn

from radio_diffusion.diffusion.ddpm import GaussianDiffusion
from radio_diffusion.load_config import NormalizationSettings
from radio_diffusion.normalization import ImageIntensityTransform
from radio_diffusion.training.losses import WeightedLoss


class DDPMModule(LightningModule):
    """Train against the configured noise or clean-image target."""

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        validation_seed: int | None = None,
        loss_function: WeightedLoss | None = None,
        ssim_before_transform: bool = False,
        normalization: NormalizationSettings | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if optimizer_name not in ("Adam", "AdamW"):
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        if not isinstance(ssim_before_transform, bool):
            raise ValueError("ssim_before_transform must be a boolean.")

        self.model = model
        self._train_forward = self.model
        self._warned_ignored_conditioning = False
        self.diffusion = diffusion
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.validation_seed = validation_seed
        self.loss_function = (
            loss_function if loss_function is not None else WeightedLoss(["MSELoss"], [1.0])
        )
        self.ssim_before_transform = ssim_before_transform
        self.intensity_transform = ImageIntensityTransform(
            normalization if normalization is not None else NormalizationSettings()
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if rank_zero_only.rank == 0:
                print(
                    "Compiling training model with TorchInductor "
                    f"(TORCHINDUCTOR_COMPILE_THREADS={os.environ.get('TORCHINDUCTOR_COMPILE_THREADS', 'default')}).",
                    flush=True,
                )
            object.__setattr__(self, "_train_forward", torch.compile(self.model))

    # def forward(
    #     self,
    #     noisy_images: torch.Tensor,
    #     timesteps: torch.Tensor,
    # ) -> torch.Tensor:
    #     return self.compiled_model(noisy_images, timesteps)

    def _shared_step(
        self,
        images: torch.Tensor | dict[str, Any],
        stage: str,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        metadata = None
        metadata_mask = None
        if isinstance(images, dict):
            metadata = images.get("metadata")
            metadata_mask = images.get("metadata_mask")
            images = images["image"]

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
        if getattr(self.model, "supports_conditioning", False):
            conditioning = {"metadata": metadata}
            if getattr(self.model, "use_validity_masks", False):
                conditioning["metadata_mask"] = metadata_mask
            prediction = self._train_forward(
                noisy_images, timesteps, **conditioning
            )
        else:
            if metadata and not self._warned_ignored_conditioning:
                rank_zero_warn(
                    f"{type(self.model).__name__} does not support conditioning; "
                    f"omitting metadata fields: {', '.join(metadata)}."
                )
                self._warned_ignored_conditioning = True
            prediction = self._train_forward(noisy_images, timesteps)
        target = noise if self.diffusion.prediction_target == "noise" else images
        image_inputs = {}
        if self.loss_function.requires_images:
            image_inputs = self._ssim_inputs(
                images, noisy_images, timesteps, prediction, metadata
            )
        loss, loss_terms = self.loss_function.compute(prediction, target, **image_inputs)

        for name, term in loss_terms.items():
            self.log(
                f"{stage}_loss_{name}",
                term,
                on_step=stage == "train",
                on_epoch=True,
                batch_size=len(images),
                sync_dist=True,
            )

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
            sync_dist=True,
        )
        return loss

    def _ssim_inputs(
        self,
        images: torch.Tensor,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
        metadata: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor | float]:
        """Prepare the image pair and reference range without detaching predictions.

        Normalized comparisons use range one. Physical comparisons use the
        original bounds; with normalization disabled these are measured from
        each reference crop. Constant reference ranges use one, matching the
        intensity transform's convention for constant images.
        """
        dtype = (
            torch.float64
            if torch.float64 in (images.dtype, noisy_images.dtype, prediction.dtype)
            else torch.float32
        )
        with torch.autocast(device_type=images.device.type, enabled=False):
            image_prediction = self.diffusion.estimate_clean_image(
                noisy_images.to(dtype=dtype), timesteps, prediction.to(dtype=dtype)
            )
            image_target = images.to(dtype=dtype)
            settings = self.intensity_transform.settings
            if settings.enabled and not self.ssim_before_transform:
                data_range = 1.0
            elif settings.enabled:
                bounds = dict(metadata or {})
                if settings.bounds == "shared":
                    data_module = getattr(self._trainer, "datamodule", None)
                    bounds.update(getattr(data_module, "dataset_metadata", {}))
                image_prediction = self.intensity_transform.inverse(image_prediction, bounds)
                image_target = self.intensity_transform.inverse(image_target, bounds)
                minimum = torch.as_tensor(
                    bounds[self.intensity_transform.minimum_field],
                    dtype=dtype, device=images.device,
                )
                maximum = torch.as_tensor(
                    bounds[self.intensity_transform.maximum_field],
                    dtype=dtype, device=images.device,
                )
                data_range = maximum - minimum
                data_range = (
                    data_range.reshape(()) if data_range.numel() == 1
                    else data_range.reshape(-1)
                )
            else:
                reference = image_target.detach().flatten(start_dim=1)
                data_range = reference.amax(dim=1) - reference.amin(dim=1)

            if isinstance(data_range, torch.Tensor):
                data_range = torch.where(
                    data_range == 0, torch.ones_like(data_range), data_range
                )
            return {
                "image_prediction": image_prediction,
                "image_target": image_target,
                "ssim_data_range": data_range,
            }

    def training_step(
        self, images: torch.Tensor | dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(images, "train", batch_idx)

    def validation_step(
        self, images: torch.Tensor | dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(images, "val", batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_type = getattr(torch.optim, self.optimizer_name)
        return optimizer_type(
            self.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwargs,
        )
