"""Build training components and run one validated experiment."""

from dataclasses import asdict
from datetime import datetime
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4

from radio_diffusion.paths import REPOSITORY_ROOT, get_path, load_env

load_env()

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch

from radio_diffusion.data.mojave import MojaveDataModule
from radio_diffusion.diffusion.ddpm import GaussianDiffusion
from radio_diffusion.load_config import ExperimentConfig
from radio_diffusion.models.conditioned_dit import ConditionedDiT
from radio_diffusion.models.dit import DiT
from radio_diffusion.models.unet import UNet
from radio_diffusion.training.callbacks import SampleGridCallback
from radio_diffusion.training.losses import WeightedLoss
from radio_diffusion.training.module import DDPMModule
from radio_diffusion.training.tracking import MLflowTrackingCallback


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    """Build the noise-prediction model from its validated settings."""
    settings = asdict(config.model)
    model_type = config.model.type
    settings.pop("type")
    settings.pop("conditioning_params")
    settings.pop("conditioning_settings")
    if model_type == "unet":
        for name in ("channels", "blocks_per_level", "attention_levels"):
            if isinstance(settings[name], list):
                settings[name] = tuple(settings[name])
        return UNet(**settings)
    if model_type == "dit":
        return DiT(**settings)
    if model_type == "conditioned_dit":
        return ConditionedDiT(
            conditioning_params=config.conditioning_params,
            **config.conditioning_settings,
            **settings,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def build_diffusion(config: ExperimentConfig) -> GaussianDiffusion:
    """Build the configured diffusion process."""
    return GaussianDiffusion(**asdict(config.diffusion))


def build_loss(config: ExperimentConfig) -> WeightedLoss:
    """Build the configured weighted noise-prediction loss."""
    return WeightedLoss(config.loss.loss, config.loss.weights)


def build_training_module(config: ExperimentConfig) -> DDPMModule:
    """Build the Lightning module with its model, diffusion, loss, and optimizer."""
    optimizer = asdict(config.optimizer)
    optimizer_name = optimizer.pop("type")
    learning_rate = optimizer.pop("learning_rate")
    optimizer["betas"] = tuple(optimizer["betas"])
    return DDPMModule(
        model=build_model(config),
        diffusion=build_diffusion(config),
        loss_function=build_loss(config),
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer,
        validation_seed=config.run.seed,
    )


def build_data_module(config: ExperimentConfig) -> MojaveDataModule:
    """Build the data module with the configured image shape and metadata fields."""
    settings = vars(config.data).copy()
    settings.pop("type")
    if settings["data_dir"] is not None:
        data_dir = Path(settings["data_dir"]).expanduser()
        if not data_dir.is_absolute():
            data_dir = REPOSITORY_ROOT / data_dir
        settings["data_dir"] = data_dir
    settings["expected_image_shape"] = config.sample_shape[1:]
    settings["metadata_fields"] = config.conditioning_params
    return MojaveDataModule(**settings)


def build_sample_callback(config: ExperimentConfig, run_dir: Path) -> SampleGridCallback:
    """Build sample generation with the configured frequency and conditioning."""
    samples = config.samples
    conditioning = samples.conditioning
    values = None
    if conditioning is not None:
        values = {
            "mode": conditioning.mode,
            "seed": conditioning.seed,
            **conditioning.values,
        }
    return SampleGridCallback(
        output_dir=run_dir / "samples",
        shape=config.sample_shape,
        every_n_epochs=samples.every_n_epochs,
        seed=samples.seed,
        nrow=samples.nrow,
        save_final_grid=samples.save_final_grid,
        save_denoising_grids=samples.save_denoising_grids,
        conditioning_params=config.conditioning_params,
        conditioning=values,
    )


def build_callbacks(
    config: ExperimentConfig,
    run_dir: Path,
    tracking_uri: str | None,
) -> list[Callback]:
    """Build checkpoint, sample, and optional MLflow callbacks."""
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=False,
        )
    ]
    if config.samples.enabled:
        callbacks.append(build_sample_callback(config, run_dir))
    if tracking_uri is not None:
        callbacks.append(
            MLflowTrackingCallback(
                config=_config_values(config),
                run_dir=run_dir,
                tracking_uri=tracking_uri,
                study=config.tracking.study,
            )
        )
    return callbacks


def build_trainer(config: ExperimentConfig) -> Trainer:
    """Build the Trainer, run directory, loggers, and training callbacks."""
    tracking_uri = _get_tracking_uri(config)
    run_dir = _create_run_directory(config)
    callbacks = build_callbacks(config, run_dir, tracking_uri)
    return Trainer(
        **asdict(config.trainer),
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=str(run_dir), name="logs"),
        callbacks=callbacks,
    )


def _config_values(config: ExperimentConfig) -> dict[str, Any]:
    """Convert settings into the dictionary format used by MLflow tracking."""
    values = asdict(config)
    values.pop("source_path")
    conditioning = values["samples"]["conditioning"]
    if conditioning is not None:
        metadata = conditioning.pop("values")
        values["samples"]["conditioning"] = {**conditioning, **metadata}
    return values


def _get_tracking_uri(config: ExperimentConfig) -> str | None:
    """Check optional MLflow availability and return its server address."""
    if not config.tracking.enabled:
        return None
    if find_spec("mlflow") is None:
        raise ModuleNotFoundError(
            "MLflow tracking is enabled but mlflow is not installed. "
            "Run: python -m pip install -e '.[tracking]'"
        )
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    endpoint = urlsplit(tracking_uri)
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
        raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) tracking server URL.")
    return tracking_uri


def _create_run_directory(config: ExperimentConfig) -> Path:
    """Choose a run directory and save the source config on rank zero."""
    source_path = config.source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {source_path}")
    height, width = config.sample_shape[2:]
    image_size = str(height) if height == width else f"{height}x{width}"
    output_root = get_path("OUTPUT_ROOT") / f"{config.data.type}_{image_size}" / config.run.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / f"{timestamp}_{uuid4().hex[:6]}"
    # DDP workers resolve the shared checkpoint directory during Trainer setup.
    if rank_zero_only.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_path, run_dir / "config.toml")
        print(
            f"Run directory: {run_dir}\n"
            f"Checkpoints: {run_dir / 'checkpoints'}\n"
            f"CSV metrics: {run_dir / 'logs'}",
            flush=True,
        )
    return run_dir


def seed_experiment(config: ExperimentConfig) -> None:
    """Seed model initialization and dataloader workers."""
    seed_everything(config.run.seed, workers=True)


def format_model_size_summary(model: torch.nn.Module) -> str:
    """Describe parameter counts and approximate FP32 training memory."""
    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    storage = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    mib = 1024**2
    # FP32 parameters, gradients, and two Adam moment buffers; no activations.
    adam_storage = trainable * 4 * 4
    return "\n".join(
        (
            f"Parameters: {total:,} total; {trainable:,} trainable",
            f"Parameter storage: {storage / mib:.1f} MiB",
            f"Approx. FP32 parameters + gradients + Adam state: "
            f"{adam_storage / mib:.1f} MiB (per model replica; excludes activations)",
        )
    )


def _format_batch_summary(config: ExperimentConfig) -> str:
    """Describe the batch size per device and across explicitly selected devices."""
    batch_size = config.data.batch_size
    devices = config.trainer.devices
    device_count = None
    if isinstance(devices, int) and devices > 0:
        device_count = devices
    elif isinstance(devices, list):
        device_count = len(devices)
    elif isinstance(devices, str) and devices not in ("auto", "-1"):
        device_count = len(devices.split(","))
    summary = f"DataLoader batch size per device: {batch_size}"
    if device_count is not None:
        summary += (
            f"\nEffective batch size across {device_count} devices: "
            f"{batch_size * device_count} (assuming distributed data parallel)"
        )
    return summary


def _prepare_training_outputs(
    config: ExperimentConfig,
    module: DDPMModule,
    trainer: Trainer,
    on_run_created: Callable[[Path], None] | None,
) -> None:
    """Print and save the summary, then report the new run directory."""
    if rank_zero_only.rank != 0:
        return
    summary = format_model_size_summary(module.model) + "\n" + _format_batch_summary(config)
    run_dir = Path(trainer.checkpoint_callback.dirpath).parent
    print(f"Model size:\n{summary}", flush=True)
    (run_dir / "model_summary.txt").write_text(summary + "\n", encoding="utf-8")
    if on_run_created is not None:
        on_run_created(run_dir)


def _training_result(trainer: Trainer) -> dict[str, str]:
    """Return the run and checkpoint paths after distributed setup has resolved them."""
    checkpoint = trainer.checkpoint_callback
    run_dir = Path(checkpoint.dirpath).parent
    if trainer.is_global_zero:
        print(f"Best checkpoint: {checkpoint.best_model_path}", flush=True)
        print(f"Last checkpoint: {checkpoint.last_model_path}", flush=True)
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": checkpoint.best_model_path,
        "last_checkpoint": checkpoint.last_model_path,
    }


def run_training(
    config: ExperimentConfig,
    on_run_created: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Build and train one experiment, returning its output paths.

    config: The validated experiment settings, including the source config path.
    on_run_created: Optional function called with the run directory before training.
    """
    seed_experiment(config)
    module = build_training_module(config)
    data = build_data_module(config)
    trainer = build_trainer(config)
    _prepare_training_outputs(config, module, trainer, on_run_created)
    trainer.fit(module, datamodule=data)
    return _training_result(trainer)
