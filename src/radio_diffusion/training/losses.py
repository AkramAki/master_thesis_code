"""Validated, weighted regression and image losses for diffusion models."""

import math

import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure

# Pytorch supported losses
_REGRESSION_LOSS_NAMES = ("MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss")
_LOSS_NAMES = (*_REGRESSION_LOSS_NAMES, "SSIMLoss")


class SSIMLoss(nn.Module):
    """Return mean one-minus-SSIM for images in the same comparison space."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        data_range: float | torch.Tensor,
    ) -> torch.Tensor:
        """Use a positive scalar range or one range per image, shaped [B].

        Dividing both images by their supplied range is equivalent to scaling
        SSIM's stabilizing constants by that range. Image values are not clipped.
        """
        if (
            prediction.ndim != 4
            or prediction.shape != target.shape
            or prediction.numel() == 0
            or not prediction.is_floating_point()
            or not target.is_floating_point()
            or prediction.device != target.device
        ):
            raise ValueError(
                "SSIMLoss requires nonempty floating-point images with matching "
                "[B, C, H, W] shapes and devices."
            )
        if isinstance(data_range, bool) or (
            isinstance(data_range, torch.Tensor)
            and (data_range.dtype == torch.bool or data_range.is_complex())
        ):
            raise ValueError("SSIMLoss data_range must contain finite, positive numbers.")

        dtype = (
            torch.float64
            if torch.float64 in (prediction.dtype, target.dtype)
            else torch.float32
        )
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            scale = torch.as_tensor(data_range, device=prediction.device, dtype=dtype)
            if scale.ndim == 1 and scale.shape == (len(prediction),):
                scale = scale.view(-1, 1, 1, 1)
            elif scale.ndim != 0:
                raise ValueError(
                    "SSIMLoss data_range must be a scalar or a tensor shaped [B]."
                )
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("SSIMLoss data_range must contain finite, positive numbers.")
            scores = structural_similarity_index_measure(
                prediction.to(dtype=dtype) / scale,
                target.to(dtype=dtype) / scale,
                gaussian_kernel=True,
                sigma=1.5,
                k1=0.01,
                k2=0.03,
                data_range=1.0,
                reduction="none",
            )
            return (1 - scores).mean()


class WeightedLoss(nn.Module):
    """Combine regression and image losses with weights summing to one."""

    def __init__(self, loss_names: list[str], weights: list[float]) -> None:
        super().__init__()
        if (
            not isinstance(loss_names, list)
            or not loss_names
            or any(not isinstance(name, str) for name in loss_names)
        ):
            raise ValueError("loss.loss must be a nonempty list of loss names.")
        if not isinstance(weights, list) or len(weights) != len(loss_names):
            raise ValueError("loss.weights must have one weight per loss in loss.loss.")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights
        ):
            raise ValueError("loss.weights must contain finite, nonnegative numbers.")
        try:
            total = math.fsum(weights)
        except OverflowError:
            total = math.inf
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"loss.weights must sum to 1; got {total}.")
        if len(set(loss_names)) != len(loss_names):
            raise ValueError("loss.loss must not contain duplicate loss names.")

        losses = []
        for name in loss_names:
            if name not in _LOSS_NAMES:
                raise ValueError(
                    f"Unknown or unsupported loss {name!r}. "
                    f"Expected one of: {', '.join(_LOSS_NAMES)}."
                )
            losses.append(SSIMLoss() if name == "SSIMLoss" else getattr(nn, name)())

        self.loss_names = tuple(loss_names)
        self.weights = tuple(float(weight) for weight in weights)
        self.losses = nn.ModuleList(losses)

    @property
    def requires_images(self) -> bool:
        """Whether computing the selected losses requires a clean-image pair."""
        return "SSIMLoss" in self.loss_names

    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        image_prediction: torch.Tensor | None = None,
        image_target: torch.Tensor | None = None,
        ssim_data_range: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted total and unweighted terms for logging."""
        if self.requires_images and (
            image_prediction is None or image_target is None or ssim_data_range is None
        ):
            raise ValueError(
                "SSIMLoss requires image_prediction, image_target, and ssim_data_range."
            )
        terms = {}
        for name, loss in zip(self.loss_names, self.losses):
            terms[name] = (
                loss(image_prediction, image_target, ssim_data_range)
                if name == "SSIMLoss"
                else loss(prediction, target)
            )
        total = sum(
            weight * terms[name]
            for name, weight in zip(self.loss_names, self.weights)
        )
        return total, terms

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        image_prediction: torch.Tensor | None = None,
        image_target: torch.Tensor | None = None,
        ssim_data_range: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the weighted scalar loss."""
        return self.compute(
            prediction,
            target,
            image_prediction=image_prediction,
            image_target=image_target,
            ssim_data_range=ssim_data_range,
        )[0]
