"""Reusable saved-run discovery, inference, and plots for review notebooks.

Notebooks choose settings and arrange the review. This module provides the
shared operations; experiment.py builds components and training/ owns fitting.
"""

import csv
from copy import deepcopy
from dataclasses import dataclass
from html import escape
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from radio_diffusion.experiment import build_data_module, load_config
from radio_diffusion.plotting import shared_intensity_norm as _shared_norm

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"


@dataclass
class SavedRun:
    """A timestamped training run, rather than a numbered model version."""

    path: Path
    config: dict

    @property
    def name(self) -> str:
        return self.config["run"]["name"]

    def checkpoint(self, choice: str = "auto") -> Path:
        """Resolve an explicit filename, or prefer best then last automatically."""
        if choice == "auto":
            # Lightning can append version suffixes such as best-v1.ckpt.
            for prefix in ("best", "last"):
                candidates = [
                    path for path in (self.path / "checkpoints").glob("*.ckpt")
                    if re.fullmatch(rf"{prefix}(?:-v\d+)?", path.stem)
                ]
                if candidates:
                    return max(candidates, key=lambda path: path.stat().st_mtime)
            return max(
                (self.path / "checkpoints").glob("*.ckpt"),
                key=lambda path: path.stat().st_mtime,
            )
        name = choice if choice.endswith(".ckpt") else f"{choice}.ckpt"
        candidate = self.path / "checkpoints" / name
        if Path(name).name != name or not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint is unavailable: {candidate}")
        return candidate


def discover_runs(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> list[SavedRun]:
    """Find all checkpointed runs; sort by name, then newest run ID first."""
    runs = [
        SavedRun(config_file.parent, load_config(config_file))
        for config_file in Path(output_root).expanduser().rglob("config.toml")
        if any((config_file.parent / "checkpoints").glob("*.ckpt"))
    ]
    runs.sort(key=lambda run: (run.path.name, str(run.path)), reverse=True)
    runs.sort(key=lambda run: run.name.casefold())
    if not runs:
        raise FileNotFoundError(
            f"No checkpointed runs found under {output_root}. "
            "Set output_root to the directory containing your saved runs."
        )
    return runs


def run_table(runs: list[SavedRun]) -> str:
    """Return a notebook HTML table without adding a pandas dependency."""
    rows = []
    for index, run in enumerate(runs):
        model = run.config["model"]
        summary = f"channels={model.get('channels', 'default')}"
        if model.get("attention_levels"):
            summary += f"; attention={model['attention_levels']}"
        values = (
            index,
            run.name,
            run.path.name,
            ", ".join(
                path.name
                for path in sorted((run.path / "checkpoints").glob("*.ckpt"))
            ),
            len(list((run.path / "samples").glob("epoch=*-step=*.pt"))),
            summary,
        )
        columns = "".join(f"<td>{escape(str(value))}</td>" for value in values)
        rows.append(f"<tr>{columns}</tr>")
    headers = (
        "Index", "Model / experiment", "Run ID (date + ID)",
        "Checkpoints", "Saved epochs", "Architecture",
    )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def check_comparison(runs, modules) -> tuple[int, int, int]:
    """Require the same image shape and actual saved diffusion schedules."""
    shapes = [
        tuple(
            run.config.get("samples", {}).get(
                "shape", (4, run.config["model"].get("in_channels", 1), 256, 256)
            )
        )[1:]
        for run in runs
    ]
    if len(set(shapes)) != 1 or len(shapes[0]) != 3 or shapes[0][0] != 1:
        raise ValueError("Choose models with the same grayscale image shape [1, H, W].")
    for module in modules:
        if not torch.equal(modules[0].diffusion.betas.cpu(), module.diffusion.betas.cpu()):
            raise ValueError("Shared-input comparison requires matching diffusion schedules.")
        if (
            module.model.in_channels != shapes[0][0]
            or module.model.out_channels != shapes[0][0]
        ):
            raise ValueError(
                "The saved sample shape must match the model's input and output channels."
            )
        factor = 2 ** (len(module.model.channels) - 1)
        if any(size % factor for size in shapes[0][1:]):
            raise ValueError(f"Image dimensions must be divisible by {factor} for both models.")
    return shapes[0]


def seeded_noise(shape, seed: int) -> torch.Tensor:
    """Create shared CPU inputs without changing the notebook's global RNG."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def denoise(
    module, images, *, seed, device="cpu", batch_size=4,
    start_timestep=None, label="Denoising",
):
    """Reverse from supplied inputs, returning CPU images and preserving RNG state.

    Fixed seeds, inputs, batch size, and device reproduce both initial and
    reverse-step randomness. Process small batches and restore the model's device
    so two large models can share one GPU.
    """
    if batch_size < 1 or len(images) < 1:
        raise ValueError("batch_size and image count must be positive.")
    start = module.diffusion.num_steps - 1 if start_timestep is None else start_timestep
    if not isinstance(start, int) or not 0 <= start < module.diffusion.num_steps:
        raise ValueError("start_timestep must be between 0 and num_steps - 1.")
    device = torch.device(device)
    original_device = next(module.parameters()).device
    was_training = module.training
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    results = []
    try:
        module.to(device).eval()
        batches = (len(images) + batch_size - 1) // batch_size
        with torch.random.fork_rng(devices=cuda_devices), tqdm(
            total=batches * (start + 1), desc=label, leave=False,
        ) as progress:
            torch.random.default_generator.manual_seed(seed)
            for cuda_device in cuda_devices:
                torch.cuda.default_generators[cuda_device].manual_seed(seed)
            for offset in range(0, len(images), batch_size):
                generated = images[offset:offset + batch_size].to(device).clone()
                for timestep in reversed(range(start + 1)):
                    timesteps = torch.full(
                        (len(generated),), timestep, device=device, dtype=torch.long,
                    )
                    generated = module.diffusion.reverse_step(
                        module.model, generated, timesteps,
                    )
                    progress.update()
                results.append(generated.cpu())
    finally:
        module.to(original_device).train(was_training)
    return torch.cat(results)


def common_validation_images(runs, *, count=4, seed=42):
    """Pick shared validation files using the original splits of both runs.

    Only reads local data; preserves training split seeds and never starts rsync.
    """
    if count < 1:
        raise ValueError("count must be positive.")
    datasets, lookups = [], []
    for run in runs:
        config = deepcopy(run.config)
        config["data"]["num_workers"] = 0
        data = build_data_module(config)
        data.setup("validate")
        datasets.append(data.val_dataset)
        lookups.append({
            Path(path).resolve(): index
            for index, path in enumerate(data.val_dataset.paths)
        })
    paths = sorted(set.intersection(*(set(lookup) for lookup in lookups)))
    if not paths:
        raise ValueError(
            "These runs have no validation files in common. "
            "Choose runs with overlapping held-out splits."
        )
    indices = torch.randperm(
        len(paths), generator=torch.Generator().manual_seed(seed),
    )[:count].tolist()
    selected_paths = [paths[index] for index in indices]
    images = torch.stack([datasets[0][lookups[0][path]] for path in selected_paths])
    return images, selected_paths


def _image_with_colorbar(figure, ax, image, gamma):
    """Draw one independently scaled image with a white outline and its own bar."""
    artist = ax.imshow(
        image[0].numpy(), cmap="inferno", norm=_shared_norm([image], gamma),
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.5)
    colorbar = figure.colorbar(
        artist, ax=ax, fraction=0.05, pad=0.03, shrink=0.8,
        ticks=MaxNLocator(nbins=3),
    )
    colorbar.ax.tick_params(labelsize=7)
    colorbar.ax.yaxis.get_offset_text().set_fontsize(7)
    return artist


def show_grids(
    batches, titles, *, nrow=4, gamma=0.5, limits=None,
    independent_colorbars=False,
):
    """Show white-bordered grids with shared limits or independently scaled tiles.

    With independent_colorbars=True, each tile has its own automatic range and
    colorbar; limits is ignored. Model grids remain adjacent with paired positions.
    """
    batches = [batch.detach().cpu() for batch in batches]
    if independent_colorbars:
        if nrow < 1:
            raise ValueError("nrow must be positive.")
        image_count = max(len(batch) for batch in batches)
        columns = min(nrow, image_count)
        rows = (image_count + columns - 1) // columns
        fig = plt.figure(
            figsize=(9 * len(batches), 2.8 * rows), layout="constrained",
        )
        panels = fig.subfigures(1, len(batches), squeeze=False)
        for panel, batch, title in zip(panels[0], batches, titles):
            panel.suptitle(f"{title}\nIndependent image scales")
            axes = panel.subplots(rows, columns, squeeze=False)
            for index, ax in enumerate(axes.flat):
                if index >= len(batch):
                    ax.axis("off")
                    continue
                _image_with_colorbar(panel, ax, batch[index], gamma)
                ax.set_title(f"Image {index + 1}", fontsize=9)
        plt.show()
        return

    norm = _shared_norm(batches, gamma, limits=limits)
    cmap = plt.get_cmap("inferno").with_extremes(bad="white")
    fig, axes = plt.subplots(
        1, len(batches), figsize=(7 * len(batches), 7),
        squeeze=False, layout="constrained",
    )
    for ax, batch, title in zip(axes[0], batches, titles):
        # Give each tile a white border without assigning it an intensity value.
        # Borders are added only to the display copy, after computing the scale.
        bordered = torch.nn.functional.pad(batch, (2, 2, 2, 2), value=float("nan"))
        grid = make_grid(bordered, nrow=nrow, padding=0, pad_value=float("nan"))
        artist = ax.imshow(
            grid[0].numpy(), cmap=cmap, norm=norm, interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(artist, ax=list(axes[0]), shrink=0.7, label="Raw intensity (shared scale)")
    plt.show()


def show_reconstructions(
    clean, noisy, reconstructed, labels, *, gamma=0.5, limits=None,
    independent_colorbars=False,
):
    """Compare originals and reconstructions with shared or per-image colorbars.

    Independent colorbars use each image's automatic range and ignore limits.
    """
    batches = [batch.detach().cpu() for batch in (clean, noisy, *reconstructed)]
    # Keep the original and reconstructed intensities comparable. Gaussian noise
    # has a separate, explicitly labelled scale so it cannot flatten radio detail.
    if not independent_colorbars:
        norm = _shared_norm([batches[0], *batches[2:]], gamma, limits=limits)
        noise_norm = _shared_norm([batches[1]], 1)
    noise_title = (
        "Shared noisy input" if independent_colorbars
        else "Shared noisy input\n(separate scale)"
    )
    titles = ["Original", noise_title, *labels]
    fig, axes = plt.subplots(
        len(clean), len(batches), figsize=(12, 2.7 * len(clean)),
        squeeze=False, layout="constrained",
    )
    if independent_colorbars:
        fig.suptitle("Each image uses its own intensity scale")
    for column, (batch, title) in enumerate(zip(batches, titles)):
        for row, image in enumerate(batch):
            if independent_colorbars:
                _image_with_colorbar(
                    fig, axes[row, column], image, 1 if column == 1 else gamma,
                )
            else:
                artist = axes[row, column].imshow(
                    image[0].numpy(), cmap="inferno",
                    norm=noise_norm if column == 1 else norm,
                )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(title)
            if column == 0:
                axes[row, column].set_ylabel(f"Image {row + 1}")
    if not independent_colorbars:
        fig.colorbar(
            artist, ax=list(axes[:, [0, *range(2, len(batches))]].flat),
            shrink=0.6, label="Raw intensity (original + reconstructions)",
        )
        fig.colorbar(
            axes[0, 1].images[0], ax=list(axes[:, 1]),
            shrink=0.6, label="Noisy input intensity",
        )
    plt.show()


def _snapshot_epoch(path):
    return int(re.search(r"epoch=(\d+)", path.name).group(1))


def show_training_progress(
    runs, labels, *, max_snapshots=6, sample_count=2, gamma=0.5, limits=None,
    independent_colorbars=False,
):
    """Show fixed samples across epochs with shared or independent image scales.

    Independent colorbars use each image's automatic range and ignore limits.
    """
    if max_snapshots < 2 or sample_count < 1:
        raise ValueError("Use at least two snapshot columns and one sample row.")
    histories = []
    for run, label in zip(runs, labels):
        files = sorted((run.path / "samples").glob("epoch=*-step=*.pt"), key=_snapshot_epoch)
        if not files:
            print(f"{label}: no periodic sample tensors were saved.")
            continue
        if len(files) == 1:
            print(f"{label}: only one saved epoch; a progression needs more snapshots.")
        indices = torch.linspace(
            0, len(files) - 1, min(max_snapshots, len(files)),
        ).round().long().tolist()
        files = [files[index] for index in indices]
        batches = [
            torch.load(path, map_location="cpu", weights_only=True)[:sample_count]
            for path in files
        ]
        histories.append((run, label, files, batches))
    if not histories:
        return
    if not independent_colorbars:
        norm = _shared_norm(
            [batch for _, _, _, batches in histories for batch in batches], gamma,
            limits=limits,
        )
    for run, label, files, batches in histories:
        rows = min(len(batch) for batch in batches)
        fig, axes = plt.subplots(
            rows, len(files), figsize=(2.6 * len(files), 2.6 * rows),
            squeeze=False, layout="constrained",
        )
        for column, (path, batch) in enumerate(zip(files, batches)):
            for row in range(rows):
                if independent_colorbars:
                    _image_with_colorbar(fig, axes[row, column], batch[row], gamma)
                else:
                    artist = axes[row, column].imshow(
                        batch[row, 0].numpy(), cmap="inferno", norm=norm,
                    )
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if row == 0:
                    axes[row, column].set_title(f"Epoch {_snapshot_epoch(path)}")
                if column == 0:
                    axes[row, column].set_ylabel(f"Saved sample {row + 1}")
        saved_seed = run.config.get("samples", {}).get("seed", run.config["run"]["seed"])
        title = f"{label} — training progression (saved seed {saved_seed})"
        if independent_colorbars:
            title += "\nIndependent image scales"
        fig.suptitle(title)
        if not independent_colorbars:
            fig.colorbar(
                artist, ax=list(axes.flat), shrink=0.6,
                label="Raw intensity (shared across both models)",
            )
        plt.show()


def show_loss_curves(runs, labels):
    """Plot epoch losses from Lightning CSV logs when available."""
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for run, label in zip(runs, labels):
        files = list((run.path / "logs").rglob("metrics.csv"))
        if not files:
            print(f"{label}: no CSV loss history was saved.")
            continue
        path = max(files, key=lambda file: file.stat().st_mtime)
        with path.open(newline="", encoding="utf-8") as file:
            records = list(csv.DictReader(file))
        for field, style in (("train_loss_epoch", "--"), ("val_loss", "-")):
            values = {
                int(float(record["epoch"])) + 1: float(record[field])
                for record in records
                if record.get("epoch") and record.get(field)
            }
            if values:
                epochs = sorted(values)
                stage = "validation" if field == "val_loss" else "train"
                ax.plot(
                    epochs, [values[epoch] for epoch in epochs], style,
                    label=f"{label}: {stage}",
                )
    if not ax.lines:
        plt.close(fig)
        return
    ax.set(xlabel="Epoch", ylabel="Noise-prediction loss", title="Training and validation loss")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.2)
    plt.show()
