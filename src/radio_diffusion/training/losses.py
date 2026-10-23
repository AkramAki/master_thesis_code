"""Validated, weighted losses for diffusion noise prediction."""

import math

import torch
from torch import nn

# Pytorch supported losses
_NOISE_LOSS_NAMES = ("MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss")


class WeightedLoss(nn.Module):
    """Combine noise-regression losses with nonnegative weights summing to one."""

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
            if name not in _NOISE_LOSS_NAMES:
                raise ValueError(
                    f"Unknown or unsupported noise-prediction loss {name!r}. "
                    f"Expected one of: {', '.join(_NOISE_LOSS_NAMES)}."
                )
            losses.append(getattr(nn, name)())

        self.loss_names = tuple(loss_names)
        self.weights = tuple(float(weight) for weight in weights)
        self.losses = nn.ModuleList(losses)

    def compute(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted total and unweighted terms for logging."""
        terms = {
            name: loss(prediction, target)
            for name, loss in zip(self.loss_names, self.losses)
        }
        total = sum(
            weight * terms[name]
            for name, weight in zip(self.loss_names, self.weights)
        )
        return total, terms

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the weighted scalar loss."""
        return self.compute(prediction, target)[0]
