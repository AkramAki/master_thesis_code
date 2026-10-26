"""MLflow run metadata and deferred logger setup after dataset validation."""

from copy import deepcopy
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import lightning
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _flatten_parameters(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Keep TOML section names and encode lists as readable JSON parameters."""
    parameters = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            parameters.update(_flatten_parameters(value, name))
        elif isinstance(value, (list, tuple)):
            parameters[name] = json.dumps(value, default=str)
        else:
            parameters[name] = value
    return parameters


def _git_tags() -> dict[str, str]:
    """Record the source revision when this checkout has Git metadata."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"git.commit": revision, "git.dirty": str(bool(status)).lower()}


class MLflowTrackingCallback(Callback):
    """Add MLflow after data.setup() provides a verified experiment name.

    Lightning handles metric forwarding and successful/failed run finalization.
    CSV stays the first logger so existing local review tools continue to work.
    """

    def __init__(
        self,
        config: dict[str, Any],
        run_dir: Path,
        tracking_uri: str,
        study: str | None = None,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.run_dir = run_dir
        self.tracking_uri = tracking_uri
        self.study = study
        self.logger: MLFlowLogger | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        if self.logger is not None:
            trainer.loggers = [
                logger for logger in trainer.loggers if logger is not self.logger
            ]

        data = trainer.datamodule
        if data is None or getattr(data, "image_shape", None) is None:
            raise ValueError("MLflow tracking requires a configured, verified data module.")

        tags = {
            "dataset.variant": data.variant,
            "run_dir": str(self.run_dir),
            "python.version": platform.python_version(),
            "torch.version": str(torch.__version__),
            "lightning.version": lightning.__version__,
        }
        if self.study is not None:
            tags["study"] = self.study
        if trainer.is_global_zero:
            tags.update(_git_tags())

        self.logger = MLFlowLogger(
            experiment_name=data.experiment_name,
            run_name=f"{self.config['run']['name']}_{self.run_dir.name}",
            tracking_uri=self.tracking_uri,
            tags=tags,
            log_model=False,
            synchronous=True,
        )
        trainer.loggers = [*trainer.loggers, self.logger]
        if not trainer.is_global_zero:
            return

        resolved = deepcopy(self.config)
        resolved["data"].update(
            variant=data.variant,
            data_dir=str(data.data_dir),
            seed=data.seed,
            image_shape=list(data.image_shape),
            train_samples=len(data.train_dataset),
            val_samples=len(data.val_dataset),
        )
        resolved.setdefault("loss", {
            "loss": list(pl_module.loss_function.loss_names),
            "weights": list(pl_module.loss_function.weights),
        })
        parameters = _flatten_parameters(resolved)
        parameters["runtime.world_size"] = trainer.world_size
        parameters["runtime.model_parameters"] = sum(
            parameter.numel() for parameter in pl_module.model.parameters()
        )
        self.logger.log_hyperparams(parameters)
        print(
            f"MLflow experiment: {data.experiment_name}\n"
            f"MLflow run ID: {self.logger.run_id}\n"
            f"MLflow server: {self.tracking_uri}",
            flush=True,
        )
