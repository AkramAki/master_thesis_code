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
    sample_labels: list[str] | None = None,
    row_labels: list[str] | None = None,
    column_labels: list[str] | None = None,
) -> None:
    """Save a grayscale grid using the review plots' gamma and inferno colormap.

    One raw-intensity colorbar serves the whole grid. Each saved grid has its
    own automatic limits; the model output and saved tensors are unchanged.
    Final grids can label each sample; trajectory grids can label examples by
    row and denoising stages by column.
    """
    images = images.detach().float().cpu()
    if images.ndim != 4 or images.shape[1] != 1 or len(images) == 0:
        raise ValueError("Sample grids require images shaped [batch, 1, height, width].")
    if nrow < 1:
        raise ValueError("nrow must be positive.")

    columns = min(nrow, len(images))
    rows = (len(images) + columns - 1) // columns
    if sample_labels is not None and len(sample_labels) != len(images):
        raise ValueError("sample_labels must contain one label per image.")
    if row_labels is not None and len(row_labels) != rows:
        raise ValueError("row_labels must contain one label per grid row.")
    if column_labels is not None and len(column_labels) != columns:
        raise ValueError("column_labels must contain one label per grid column.")
    label_width = 2.5 if row_labels is not None else 0
    figure = Figure(
        figsize=(2.5 * columns + 1 + label_width, 2.5 * rows + 0.5),
        layout="constrained",
    )
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
        row, column = divmod(index, columns)
        if sample_labels is not None:
            ax.set_title(sample_labels[index], fontsize=8)
        elif column_labels is not None:
            if row == 0:
                ax.set_title(column_labels[column], fontsize=9)
        elif row_labels is None:
            ax.set_title(f"Sample {index + 1}", fontsize=9)
        if row_labels is not None and column == 0:
            ax.set_ylabel(
                row_labels[row], rotation=0, ha="right", va="center",
                labelpad=12, fontsize=8,
            )
    figure.suptitle(title)
    figure.colorbar(artist, ax=list(axes.flat), shrink=0.8, label="Raw intensity")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=120)
    finally:
        figure.clear()
