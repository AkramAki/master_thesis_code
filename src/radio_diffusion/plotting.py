"""Shared raw-intensity display scaling for sample previews and review plots."""

from math import isfinite
from pathlib import Path

from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
import torch


def shared_intensity_norm(batches, gamma=0.5, limits=None):
    """Use shared raw-intensity limits, either automatic or an explicit pair."""
    if limits is None:
        low = min(float(batch.min()) for batch in batches)
        high = max(float(batch.max()) for batch in batches)
        if high <= low:
            high = low + 1
    else:
        low, high = limits
        if not (isfinite(low) and isfinite(high) and low < high):
            raise ValueError("limits must be two finite values with minimum < maximum.")
    return PowerNorm(gamma=gamma, vmin=low, vmax=high)


def save_sample_grid(
    images: torch.Tensor,
    path: str | Path,
    *,
    nrow: int = 4,
    title: str = "Generated samples",
) -> None:
    """Save a grayscale grid using the review plots' gamma and inferno colormap.

    One raw-intensity colorbar serves the whole grid. Each saved grid has its
    own automatic limits; the model output and saved tensors are unchanged.
    """
    images = images.detach().float().cpu()
    if images.ndim != 4 or images.shape[1] != 1 or len(images) == 0:
        raise ValueError("Sample grids require images shaped [batch, 1, height, width].")
    if nrow < 1:
        raise ValueError("nrow must be positive.")

    columns = min(nrow, len(images))
    rows = (len(images) + columns - 1) // columns
    figure = Figure(figsize=(2.5 * columns + 1, 2.5 * rows + 0.5), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    norm = shared_intensity_norm([images])
    for index, ax in enumerate(axes.flat):
        if index >= len(images):
            ax.axis("off")
            continue
        artist = ax.imshow(
            images[index, 0].numpy(), cmap="inferno", norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sample {index + 1}", fontsize=9)
    figure.suptitle(title)
    figure.colorbar(artist, ax=list(axes.flat), shrink=0.8, label="Raw intensity")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=120)
    finally:
        figure.clear()
