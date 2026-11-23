"""Reversible intensity transforms followed by min/max rescaling to [0, 1]."""

import torch

from radio_diffusion.load_config import NormalizationSettings


class ImageIntensityTransform:
    """Apply the selected intensity transformation, then min–max scaling.

    When enabled, normalize applies both operations in that order, and inverse
    reverses them. When disabled, normalize and inverse bypass both operations.
    Settings are validated before construction.
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
    def _validate_images(images: torch.Tensor) -> None:
        if (
            not isinstance(images, torch.Tensor)
            or not images.is_floating_point()
            or images.numel() == 0
            or not torch.isfinite(images).all()
        ):
            raise ValueError("Normalization requires nonempty, finite floating-point tensors.")

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
        settings = self.settings
        if settings.transform == "linear":
            return images
        if settings.transform == "log1p":
            return torch.expm1(images) * settings.log_scale
        return torch.sign(images) * torch.expm1(images.abs()) * settings.log_scale

    def _bounds(self, images: torch.Tensor, metadata: dict):
        """Transform the saved extrema and broadcast them over each image."""
        values = []
        for field in (self.minimum_field, self.maximum_field):
            if field not in metadata:
                raise ValueError(f"Normalization requires metadata field {field!r}.")
            values.append(torch.as_tensor(metadata[field], device=images.device, dtype=images.dtype))
        minimum, maximum = torch.broadcast_tensors(*values)
        bounds = self._transform(torch.stack((minimum, maximum)))
        self._validate_images(bounds)
        if (bounds[0] > bounds[1]).any():
            raise ValueError("Normalization requires transformed minimum <= maximum.")
        self._validate_images(bounds[1] - bounds[0])
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
        """Undo min/max rescaling, then undo the intensity transformation."""
        self._validate_images(images)
        minimum, maximum = self._bounds(images, metadata)
        transformed = images * (maximum - minimum) + minimum
        result = self._inverse_transform(transformed)
        self._validate_images(result)
        return result
