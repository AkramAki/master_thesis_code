"""Load experiment configuration and construct its model components."""

import tomllib
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torch
from torch.utils.data import DataLoader, Dataset

from radio_diffusion.data.mojave import MojaveDataModule
from radio_diffusion.diffusion.ddpm import GaussianDiffusion
from radio_diffusion.models.unet import UNet
from radio_diffusion.training.callbacks import SampleGridCallback
from radio_diffusion.training.losses import WeightedLoss
from radio_diffusion.training.module import DDPMModule
from radio_diffusion.training.tracking import MLflowTrackingCallback

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ExperimentConfig(dict[str, Any]):
    """Configuration mapping that remembers the TOML file it came from."""

    def __init__(self, values: dict[str, Any], source_path: Path) -> None:
        super().__init__(values)
        self.source_path = source_path


@dataclass
class TrainerSettings:
    """Settings that control the length and runtime of a training run."""

    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int | str = "auto"
    log_every_n_steps: int = 50
    check_val_every_n_epoch: int = 1

    def to_trainer_kwargs(self) -> dict[str, Any]:
        """Return the fields accepted by Lightning's Trainer."""
        return {
            "max_epochs": self.max_epochs,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "log_every_n_steps": self.log_every_n_steps,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
        }


@dataclass
class TrackingSettings:
    """Optional MLflow tracking; the endpoint is set through the environment."""

    enabled: bool = False
    study: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("tracking.enabled must be true or false.")
        if self.study is not None and (
            not isinstance(self.study, str) or not self.study.strip()
        ):
            raise ValueError("tracking.study must be a nonempty string when provided.")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a TOML experiment configuration and validate its training settings."""
    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as config_file:
        config = ExperimentConfig(tomllib.load(config_file), source_path)
    build_loss(config)
    get_sample_shape(config)
    TrackingSettings(**config.get("tracking", {}))
    return config


def seed_experiment(config: dict[str, Any]) -> None:
    """Seed model initialization and dataloader workers from the config."""
    seed_everything(config["run"]["seed"], workers=True)


def build_model(config: dict[str, Any]) -> UNet:
    """Build the configured noise-prediction model."""
    model_config = dict(config["model"])
    model_type = model_config.pop("type")
    if model_type != "unet":
        raise ValueError(f"Unsupported model type: {model_type}")

    for key in ("channels", "blocks_per_level", "attention_levels"):
        if key in model_config:
            model_config[key] = tuple(model_config[key])

    return UNet(**model_config)


def build_diffusion(config: dict[str, Any]) -> GaussianDiffusion:
    """Build the configured Gaussian diffusion process."""
    return GaussianDiffusion(**config["diffusion"])


def get_sample_shape(config: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return the shared image shape used for sampling and dataset validation."""
    channels = config["model"].get("in_channels", 1)
    shape = config.get("samples", {}).get("shape", (4, channels, 256, 256))
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 4
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in shape
        )
    ):
        raise ValueError(
            "samples.shape must contain four positive integers "
            "[batch, channels, height, width]."
        )
    if shape[1] != channels:
        raise ValueError("samples.shape channels must match model.in_channels.")
    return tuple(shape)


def build_data_module(config: dict[str, Any]) -> MojaveDataModule:
    """Build the configured data module, resolving relative paths at repo root."""
    data_config = dict(config["data"])
    data_type = data_config.pop("type")
    if data_type != "mojave":
        raise ValueError(f"Unsupported data module type: {data_type}")

    data_dir = Path(data_config["data_dir"]).expanduser()
    if not data_dir.is_absolute():
        data_dir = _REPOSITORY_ROOT / data_dir
    data_config["data_dir"] = data_dir
    data_config.setdefault("seed", config["run"]["seed"])
    data_config["expected_image_shape"] = get_sample_shape(config)[1:]

    return MojaveDataModule(**data_config)


def build_loss(config: dict[str, Any]) -> WeightedLoss:
    """Build validated noise losses; older configs retain unweighted MSE."""
    if "loss" not in config:
        return WeightedLoss(["MSELoss"], [1.0])
    loss_config = config["loss"]
    if not isinstance(loss_config, dict) or set(loss_config) != {"loss", "weights"}:
        raise ValueError("[loss] must contain only the loss and weights lists.")
    return WeightedLoss(loss_config["loss"], loss_config["weights"])


def build_training_module(config: dict[str, Any]) -> DDPMModule:
    """Build the Lightning module, including its configured optimizer and losses."""
    loss_function = build_loss(config)
    optimizer_config = dict(config["optimizer"])
    optimizer_name = optimizer_config.pop("type")
    learning_rate = optimizer_config.pop("learning_rate")

    return DDPMModule(
        model=build_model(config),
        diffusion=build_diffusion(config),
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_config,
        validation_seed=config["run"]["seed"],
        loss_function=loss_function,
    )


def format_model_size_summary(model: torch.nn.Module) -> str:
    """Describe parameter count and rough FP32 training-memory requirements."""
    parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    mib = 1024**2
    # FP32 parameters + gradients + two Adam moment buffers; excludes activations.
    adam_training_bytes = trainable_parameters * 4 * 4

    return "\n".join(
        (
            f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable",
            f"Parameter storage: {parameter_bytes / mib:.1f} MiB",
            f"Approx. FP32 parameters + gradients + Adam state: "
            f"{adam_training_bytes / mib:.1f} MiB (per model replica; excludes activations)",
        )
    )


class _SyntheticImageDataset(Dataset):
    """Small CPU dataset used to check a configured per-device batch."""

    def __init__(self, count: int, image_shape: tuple[int, ...]) -> None:
        self.count = count
        self.image_shape = image_shape

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        del index
        return torch.zeros(self.image_shape)


class _PreflightMemoryCallback(Callback):
    """Report each worker's peak CUDA memory after its preflight step."""

    def on_train_start(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        del pl_module
        device = trainer.strategy.root_device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def on_train_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        del pl_module
        device = trainer.strategy.root_device
        if device.type != "cuda":
            return
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        print(
            f"GPU preflight rank {trainer.global_rank} ({device}): "
            f"peak allocated {allocated:.0f} MiB, peak reserved {reserved:.0f} MiB",
            flush=True,
        )


def preflight_training(
    config: dict[str, Any], training_module: DDPMModule | None = None
) -> None:
    """Run one configured training batch on the configured device setup."""
    settings = TrainerSettings(**config.get("trainer", {}))
    accelerator = settings.accelerator
    devices = settings.devices
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU preflight requested, but CUDA is unavailable.")
        available_devices = torch.cuda.device_count()
        if isinstance(devices, int):
            requested_devices = devices
        elif isinstance(devices, (list, tuple)):
            requested_devices = len(devices)
        else:
            requested_devices = available_devices
        if requested_devices > available_devices:
            raise RuntimeError(
                f"Config requests {requested_devices} GPUs, but only "
                f"{available_devices} are visible."
            )
        device_count = requested_devices
    elif isinstance(devices, int):
        device_count = devices
    elif isinstance(devices, (list, tuple)):
        device_count = len(devices)
    else:
        device_count = 1

    image_shape = get_sample_shape(config)[1:]
    per_device_batch = config["data"]["batch_size"]
    dataset = _SyntheticImageDataset(
        count=per_device_batch * device_count,
        image_shape=image_shape,
    )
    dataloader = DataLoader(dataset, batch_size=per_device_batch, num_workers=0)

    trainer_kwargs = settings.to_trainer_kwargs()
    trainer_kwargs.update(
        max_epochs=1,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=[_PreflightMemoryCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(
        training_module or build_training_module(config),
        train_dataloaders=dataloader,
    )


def build_trainer(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> Trainer:
    """Build a Trainer with checkpointing and configured review callbacks."""
    trainer_settings = TrainerSettings(**config.get("trainer", {}))
    tracking_settings = TrackingSettings(**config.get("tracking", {}))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if tracking_settings.enabled:
        if find_spec("mlflow") is None:
            raise ModuleNotFoundError(
                "MLflow tracking is enabled but mlflow is not installed. "
                "Run: python -m pip install -e '.[tracking]'"
            )
        endpoint = urlsplit(tracking_uri)
        if endpoint.scheme not in ("http", "https") or not endpoint.netloc:
            raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) tracking server URL.")
    run_config = config["run"]
    output_root = Path(
        run_config.get("output_dir", f"outputs/{run_config['name']}")
    ).expanduser()
    if not output_root.is_absolute():
        output_root = _REPOSITORY_ROOT / output_root

    source_path = config_path or getattr(config, "source_path", None)
    if source_path is None:
        raise ValueError(
            "build_trainer needs the source TOML path so it can save a copy. "
            "Load the config with load_config() or pass config_path."
        )
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {source_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / f"{timestamp}_{uuid4().hex[:6]}"
    # DDP re-executes this builder in each worker. Only rank zero creates files;
    # ModelCheckpoint broadcasts its directory when distributed setup completes.
    if rank_zero_only.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_path, run_dir / "config.toml")

    callbacks = [
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

    sample_config = config.get("samples", {})
    if sample_config.get("enabled", True):
        shape = get_sample_shape(config)
        callbacks.append(
            SampleGridCallback(
                output_dir=run_dir / "samples",
                shape=shape,
                every_n_epochs=sample_config.get("every_n_epochs", 10),
                seed=sample_config.get("seed", run_config["seed"]),
                nrow=sample_config.get("nrow", 4),
            )
        )

    early_stopping_config = config.get("early_stopping", {})
    if early_stopping_config.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stopping_config.get("patience", 20),
                min_delta=early_stopping_config.get("min_delta", 0.001),
            )
        )

    if tracking_settings.enabled:
        callbacks.append(
            MLflowTrackingCallback(
                config=config,
                run_dir=run_dir,
                tracking_uri=tracking_uri,
                study=tracking_settings.study,
            )
        )

    return Trainer(
        **trainer_settings.to_trainer_kwargs(),
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=str(run_dir), name="logs"),
        callbacks=callbacks,
    )


def load_experiment_checkpoint(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> DDPMModule:
    """Load a run checkpoint, building its model from the saved TOML config.

    ``checkpoint_path`` may be a checkpoint file or a run directory. For a run
    directory, ``best.ckpt`` is preferred and ``last.ckpt`` is the fallback.
    The config is found beside the run directory unless ``config_path`` is
    provided explicitly.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_dir():
        checkpoints_dir = path / "checkpoints"
        candidates = (
            checkpoints_dir / "best.ckpt",
            checkpoints_dir / "last.ckpt",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    if config_path is None:
        config_file = next(
            (parent / "config.toml" for parent in path.parents
             if (parent / "config.toml").is_file()),
            None,
        )
        if config_file is None:
            raise FileNotFoundError(
                f"Could not find config.toml in the checkpoint's parent folders: {path}"
            )
    else:
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            raise FileNotFoundError(f"Experiment config does not exist: {config_file}")

    config = load_config(config_file)
    module = build_training_module(config)
    loaded_module = DDPMModule.load_from_checkpoint(
        path,
        model=module.model,
        diffusion=module.diffusion,
        learning_rate=module.learning_rate,
        optimizer_name=module.optimizer_name,
        optimizer_kwargs=module.optimizer_kwargs,
        validation_seed=module.validation_seed,
        loss_function=module.loss_function,
        map_location=map_location,
    )
    loaded_module.to(map_location)
    if eval_mode:
        loaded_module.eval()
    return loaded_module
