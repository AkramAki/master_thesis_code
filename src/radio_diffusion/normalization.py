"""Reversible intensity transforms followed by min/max rescaling to [0, 1]."""

import math

import torch

from radio_diffusion.load_config import NormalizationSettings


class ImageIntensityTransform:
    """Apply the selected intensity transformation, then min–max scaling.

    When enabled, normalize applies both operations in that order, and inverse
    reverses them. When disabled, normalize and inverse bypass both operations.
    Inverse inputs outside [0, 1] are clipped or extrapolated linearly according
    to the required inverse_out_of_bounds setting.
    """

    def __init__(self, settings: NormalizationSettings) -> None:
        self.settings = settings
        prefix = "dataset_pixel" if settings.bounds == "shared" else "cropped_image_pixel"
        self.minimum_field = f"{prefix}_minimum"
        self.maximum_field = f"{prefix}_maximum"
        self.normalize = self._normalize_image if settings.enabled else self._identity
        self.inverse = self._inverse_image if settings.enabled else self._identity

    @staticmethod
    def _identity(image: torch.Tensor, metadata: dict) -> torch.Tensor:
        return image

    @staticmethod
    def _tensor_summary(images: torch.Tensor) -> str:
        """Describe a failed tensor without including its full contents."""
        if not isinstance(images, torch.Tensor):
            return f"type={type(images).__name__}"
        summary = f"dtype={images.dtype}, shape={tuple(images.shape)}, device={images.device}"
        if not images.is_floating_point():
            return summary
        values = images.detach()
        finite = values[torch.isfinite(values)]
        finite_range = (
            f"[{finite.min().item():.6g}, {finite.max().item():.6g}]"
            if finite.numel() else "none"
        )
        return (
            f"{summary}, finite_range={finite_range}, "
            f"nan_count={torch.isnan(values).sum().item()}, "
            f"posinf_count={torch.isposinf(values).sum().item()}, "
            f"neginf_count={torch.isneginf(values).sum().item()}"
        )

    @classmethod
    def _validate_images(
        cls, images: torch.Tensor, *, stage: str = "normalization",
        source: torch.Tensor | None = None,
    ) -> None:
        if (
            not isinstance(images, torch.Tensor)
            or not images.is_floating_point()
            or images.numel() == 0
            or not torch.isfinite(images).all()
        ):
            message = (
                f"Invalid tensor at {stage}: expected a nonempty, finite floating-point tensor. "
                f"Received: {cls._tensor_summary(images)}."
            )
            if source is not None:
                message += f" Input to this operation: {cls._tensor_summary(source)}."
            raise ValueError(message)

    def _transform(self, images: torch.Tensor) -> torch.Tensor:
        settings = self.settings
        if settings.transform == "linear":
            return images
        scaled = images / settings.log_scale
        if settings.transform == "log1p":
            if (scaled <= -1).any():
                raise ValueError("log1p normalization requires every pixel > -log_scale.")
            return torch.log1p(scaled)
        return torch.sign(scaled) * torch.log1p(scaled.abs())

    def _inverse_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Scale exponentials before they grow and preserve the derivative at zero."""
        settings = self.settings
        if settings.transform == "linear":
            return images
        dtype = images.dtype
        images = images.to(dtype=torch.float64)
        positive = images >= 0
        positive_images = torch.where(positive, images, torch.zeros_like(images))
        negative_images = torch.where(positive, torch.zeros_like(images), images)
        if settings.transform == "log1p":
            positive_result = (
                torch.exp(positive_images + math.log(settings.log_scale))
                * -torch.expm1(-positive_images)
            )
            negative_result = torch.expm1(negative_images) * settings.log_scale
            return torch.where(positive, positive_result, negative_result).to(dtype=dtype)
        scale = torch.exp(images.abs() + math.log(settings.log_scale))
        result = scale * torch.where(
            positive, -torch.expm1(-positive_images), torch.expm1(negative_images),
        )
        return result.to(dtype=dtype)

    def _bounds(self, images: torch.Tensor, metadata: dict):
        """Transform the saved extrema and broadcast them over each image."""
        values = []
        for field in (self.minimum_field, self.maximum_field):
            if field not in metadata:
                raise ValueError(f"Normalization requires metadata field {field!r}.")
            values.append(torch.as_tensor(metadata[field], device=images.device, dtype=images.dtype))
        minimum, maximum = torch.broadcast_tensors(*values)
        bounds = self._transform(torch.stack((minimum, maximum)))
        self._validate_images(bounds, stage="transformed normalization metadata bounds")
        if (bounds[0] > bounds[1]).any():
            raise ValueError("Normalization requires transformed minimum <= maximum.")
        self._validate_images(bounds[1] - bounds[0], stage="normalization metadata span")
        if minimum.numel() == 1:
            return bounds.reshape(2).unbind()
        if images.ndim >= 4 and minimum.shape in ((len(images),), (len(images), 1)):
            return bounds.reshape(2, len(images), *([1] * (images.ndim - 1))).unbind()
        raise ValueError("Supply one minimum and maximum per image, or one shared pair.")

    def _normalize_image(self, images: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Map transformed min/max to zero/one; constant images map to zero."""
        self._validate_images(images)
        minimum, maximum = self._bounds(images, metadata)
        span = maximum - minimum
        denominator = torch.where(span == 0, torch.ones_like(span), span)
        result = (self._transform(images) - minimum) / denominator
        self._validate_images(result)
        return result

    def _inverse_image(self, images: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Invert [0, 1] exactly; apply the selected policy outside that interval."""
        settings = (
            f"transform={self.settings.transform}, bounds={self.settings.bounds}, "
            f"log_scale={self.settings.log_scale}, "
            f"inverse_out_of_bounds={self.settings.inverse_out_of_bounds}"
        )
        self._validate_images(images, stage=f"inverse normalization input ({settings})")
        # Keep inverse arithmetic in float64, then validate the original output dtype.
        values = images.to(dtype=torch.float64)
        try:
            minimum, maximum = self._bounds(values, metadata)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Inverse normalization metadata bounds ({settings}): {error}") from error
        below, above = values < 0, values > 1
        if self.settings.inverse_out_of_bounds == "clip":
            bounded = values.clamp(0, 1)
        else:
            # Strict comparisons retain the exact inverse derivative at the endpoints.
            bounded = torch.where(
                below, torch.zeros_like(values),
                torch.where(above, torch.ones_like(values), values),
            )
        span = maximum - minimum
        transformed = bounded * span + minimum
        self._validate_images(
            transformed, stage=f"inverse min/max rescaling output ({settings})", source=bounded,
        )
        result = self._inverse_transform(transformed)
        self._validate_images(
            result, stage=f"inverse intensity transform output ({settings})", source=transformed,
        )
        if self.settings.inverse_out_of_bounds == "linear":
            if self.settings.transform == "linear":
                lower_slope = upper_slope = span
            else:
                endpoints = torch.stack((minimum, maximum))
                if self.settings.transform == "signed_log1p":
                    endpoints = endpoints.abs()
                lower_slope, upper_slope = (
                    torch.exp(endpoints + math.log(self.settings.log_scale)) * span
                ).unbind()
            lower_delta = torch.where(below, values, torch.zeros_like(values))
            upper_delta = torch.where(above, values - 1, torch.zeros_like(values))
            result = result + lower_delta * lower_slope + upper_delta * upper_slope
            self._validate_images(
                result, stage=f"inverse linear extrapolation output ({settings})", source=values,
            )
        output = result.to(dtype=images.dtype)
        self._validate_images(
            output, stage=f"inverse output conversion to {images.dtype} ({settings})", source=result,
        )
        return output
