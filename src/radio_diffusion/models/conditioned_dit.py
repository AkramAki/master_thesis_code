"""Condition a regular DiT with one learned token per scalar metadata field."""

import torch
from torch import nn

from radio_diffusion.models.dit import DiT


class ConditionedDiT(nn.Module):
    """Encode numerical scalars and booleans as extra self-attention tokens.

    Each field has its own projection, so tokens retain field identity. The
    inner DiT owns image processing; this wrapper owns metadata encoding.
    Missing values use the dataset's substitutions. When use_validity_masks is
    enabled, each token encodes both its value and validity flag. An empty
    selection runs the inner DiT without extra tokens.
    """

    supports_conditioning = True

    def __init__(
        self,
        conditioning_params: list[str],
        use_validity_masks: bool = False,
        **dit_kwargs,
    ) -> None:
        super().__init__()
        if not isinstance(conditioning_params, list) or any(
            not isinstance(field, str) or not field.strip()
            for field in conditioning_params
        ):
            raise ValueError(
                "conditioning_params must be a list of nonempty field names."
            )
        if len(conditioning_params) != len(set(conditioning_params)):
            raise ValueError("conditioning_params must contain unique field names.")
        if not isinstance(use_validity_masks, bool):
            raise ValueError("use_validity_masks must be true or false.")

        self.dit = DiT(**dit_kwargs)
        self.conditioning_params = tuple(conditioning_params)
        self.use_validity_masks = use_validity_masks
        self.metadata_encoders = nn.ModuleList(
            nn.Linear(2 if use_validity_masks else 1, self.dit.hidden_size)
            for _ in self.conditioning_params
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
        metadata_mask: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.conditioning_params:
            return self.dit(noisy_images, timesteps)
        if not isinstance(metadata, dict):
            raise ValueError("ConditionedDiT requires a dictionary of selected metadata.")
        missing = [field for field in self.conditioning_params if field not in metadata]
        if missing:
            raise ValueError(f"Missing conditioning fields: {', '.join(missing)}")
        if self.use_validity_masks:
            if not isinstance(metadata_mask, dict):
                raise ValueError("ConditionedDiT requires configured validity masks.")
            missing_masks = [
                field for field in self.conditioning_params if field not in metadata_mask
            ]
            if missing_masks:
                raise ValueError(f"Missing validity masks: {', '.join(missing_masks)}")
        if noisy_images.ndim != 4:
            raise ValueError("noisy_images must be shaped [B, in_channels, H, W].")

        batch_size = noisy_images.shape[0]
        tokens = []
        for field, encoder in zip(self.conditioning_params, self.metadata_encoders):
            value = metadata[field]
            if (
                not isinstance(value, torch.Tensor)
                or value.is_complex()
                or tuple(value.shape) not in ((batch_size,), (batch_size, 1))
            ):
                raise ValueError(
                    f"Conditioning field {field!r} must be a numerical scalar or "
                    "boolean tensor shaped [B] or [B, 1]. Strings and vectors "
                    "are not supported by ConditionedDiT."
                )
            value = value.to(device=noisy_images.device, dtype=encoder.weight.dtype)
            features = value.reshape(batch_size, 1)
            if self.use_validity_masks:
                valid = metadata_mask[field]
                if (
                    not isinstance(valid, torch.Tensor)
                    or valid.dtype != torch.bool
                    or tuple(valid.shape) not in ((batch_size,), (batch_size, 1))
                ):
                    raise ValueError(
                        f"Validity mask {field!r} must be a boolean tensor "
                        "shaped [B] or [B, 1]."
                    )
                valid = valid.to(device=noisy_images.device).reshape(batch_size, 1)
                features = torch.cat(
                    (
                        torch.where(valid, features, torch.zeros_like(features)),
                        valid.to(dtype=features.dtype),
                    ),
                    dim=1,
                )
            tokens.append(encoder(features))

        return self.dit._forward_with_tokens(
            noisy_images, timesteps, torch.stack(tokens, dim=1)
        )
