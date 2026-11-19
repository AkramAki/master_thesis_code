"""Prepare preflight training checks and report GPU memory use."""

from typing import Any

from radio_diffusion.load_config import ExperimentConfig
from radio_diffusion.training.train import build_training_module, seed_experiment

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
from torch.utils.data import DataLoader, Dataset


class _SyntheticImageDataset(Dataset):
    """Provide synthetic images and metadata for the configured training module."""

    def __init__(
        self,
        count: int,
        image_shape: tuple[int, ...],
        conditioning_params: list[str],
    ) -> None:
        self.count = count
        self.image_shape = image_shape
        self.conditioning_params = tuple(conditioning_params)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor | dict[str, Any]:
        image = torch.zeros(self.image_shape)
        if not self.conditioning_params:
            return image
        return {
            "image": image,
            "metadata": {name: torch.tensor(0.0) for name in self.conditioning_params},
            "metadata_mask": {name: True for name in self.conditioning_params},
        }


class _MemoryCallback(Callback):
    """Collect peak CUDA memory for the current training process."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__()
        self.report = report

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        device = trainer.strategy.root_device
        self.report.update(device=str(device), rank=trainer.global_rank)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        device = trainer.strategy.root_device
        self.report["completed_steps"] = trainer.global_step
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            self.report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            self.report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)


def _parameter_summary(module: LightningModule) -> dict[str, int]:
    """Count parameters and their current storage, excluding buffers."""
    parameters = list(module.parameters())
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    }


def _preflight_devices(config: ExperimentConfig) -> int | list[int]:
    """Convert the configured GPU selection into the preflight interface."""
    devices = config.trainer.devices
    if devices in ("auto", "-1", -1):
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("GPU preflight requested, but no CUDA GPUs are visible.")
        return count
    if isinstance(devices, str):
        return [int(index) for index in devices.split(",")]
    return devices


def _build_preflight_dataloader(
    config: ExperimentConfig, device_count: int
) -> DataLoader:
    """Build enough synthetic images for one full batch on every GPU."""
    dataset = _SyntheticImageDataset(
        count=config.data.batch_size * device_count,
        image_shape=config.sample_shape[1:],
        conditioning_params=config.conditioning_params,
    )
    return DataLoader(dataset, batch_size=config.data.batch_size, num_workers=0)


def _format_memory(value: int | None) -> str:
    """Format measured bytes as MiB, including unavailable measurements."""
    return "not collected" if value is None else f"{value / 1024**2:.1f} MiB"


@rank_zero_only
def print_preflight_report(config: ExperimentConfig, report: dict[str, Any]) -> None:
    """Print the returned preflight measurements together after the check."""
    status = "PASS" if report["success"] else "FAIL"
    device = report["device"] if report["device"] is not None else "not available"
    rank = report["rank"] if report["rank"] is not None else "not available"
    lines = [
        f"Preflight result: {config.source_path.name}: {status}",
        f"  Batch size per device: {config.data.batch_size}",
        f"  Configured GPU devices: {config.trainer.devices}",
        f"  Parameters: {report['total_parameters']:,} total; "
        f"{report['trainable_parameters']:,} trainable",
        f"  Parameter storage: {_format_memory(report['parameter_bytes'])}",
        f"  GPU measurements: rank {rank}, device {device}",
        f"    Peak allocated: {_format_memory(report['peak_allocated_bytes'])}",
        f"    Peak reserved: {_format_memory(report['peak_reserved_bytes'])}",
        f"  Completed optimizer steps: {report['completed_steps']}",
    ]
    if report["error"]:
        lines.append(f"  Error: {report['error']}")
    print("\n".join(lines), flush=True)


def run_preflight(
    module: LightningModule,
    dataloader: DataLoader,
    *,
    devices: int | list[int] = 1,
    steps: int = 1,
) -> dict[str, Any]:
    """Simulate GPU training and measure memory use.

    module: The Lightning module used to simulate training. Its weights change.
    dataloader: The batches used during the training check.
    devices: The number of GPUs or a list of GPU indices to use.
    steps: The number of training steps to run. Defaults to one.

    Returns success, an error message, parameter counts, and memory use in bytes
    for the current rank. Unavailable memory measurements are None.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    report: dict[str, Any] = {
        "success": False,
        "error": None,
        **_parameter_summary(module),
        "device": None,
        "rank": None,
        "completed_steps": 0,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        max_steps=steps,  # Stop after the requested optimizer steps.
        limit_val_batches=0,  # Skip validation.
        logger=False,  # Disable experiment logging.
        callbacks=[_MemoryCallback(report)],  # Collect peak GPU memory.
        enable_checkpointing=False,  # Skip checkpoint files.
        enable_progress_bar=False,  # Hide the progress bar.
        enable_model_summary=False,  # Hide the model summary.
    )
    try:
        trainer.fit(module, train_dataloaders=dataloader)
    except torch.cuda.OutOfMemoryError as exc:
        report.update(error=str(exc), completed_steps=trainer.global_step)
    else:
        report["success"] = True
    return report


def run_config_preflight(config: ExperimentConfig) -> dict[str, Any]:
    """Prepare and run preflight, print its measurements, and return the report.

    config: The validated experiment settings used for the training check.
    """
    devices = _preflight_devices(config)
    device_count = devices if isinstance(devices, int) else len(devices)
    seed_experiment(config)
    module = build_training_module(config)
    dataloader = _build_preflight_dataloader(config, device_count)
    report = run_preflight(module, dataloader, devices=devices)
    print_preflight_report(config, report)
    return report
