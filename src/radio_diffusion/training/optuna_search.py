"""Search model settings, keeping configs and sample grids but no model weights.

Run from the repository root in the project environment:
    python -m radio_diffusion.training.optuna_search --study dit-search --hours 72
    python -m radio_diffusion.training.optuna_search --study dit-search --preflight

Dashboard, using the storage path printed at startup:
    optuna-dashboard /path/to/OUTPUT_ROOT/optuna_runs/optuna.journal

Edit CONFIG, SEARCH_SPACE, CONDITIONING_CHOICES, SAMPLER, and PRUNER below.
Each study is one MLflow experiment; each trial is one MLflow run. Trial TOMLs
and sample grids live under OUTPUT_ROOT/optuna_runs/<study>/ and are uploaded
to MLflow. OUTPUT_ROOT is read from .env; the shared journal lives alongside
the study directories. --output and --storage override these locations.
One independent worker uses each GPU in SEARCH['gpus']; --gpu selects just one.
Copy best.toml for normal training after the search. Study history persists,
but interrupted model training is not resumed because weights are not saved.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gc
from itertools import product
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, sleep

from radio_diffusion.paths import get_path, load_env

load_env()

from lightning.pytorch import Callback, Trainer
from mlflow.tracking import MlflowClient
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import tomli_w
import torch

from radio_diffusion.load_config import ExperimentConfig, load_config
from radio_diffusion.training.preflight import run_config_preflight
from radio_diffusion.training.tracking import MLflowTrackingCallback
from radio_diffusion.training.train import (
    _config_values,
    _get_tracking_uri,
    build_data_module,
    build_sample_callback,
    build_training_module,
    seed_experiment,
)


# Fixed settings, with the same sections as an experiment TOML. A setting in
# SEARCH_SPACE replaces its CONFIG value. Removing it from SEARCH_SPACE fixes it.
CONFIG = {
    "run": {"name": "optuna", "seed": 42},
    "tracking": {"enabled": True},
    "trainer": {
        "max_epochs": 200,
        "accelerator": "gpu",
        "devices": [1],  # Each worker replaces this with its selected GPU.
        "log_every_n_steps": 50,
        "check_val_every_n_epoch": 1,
    },
    "data": {
        "type": "mojave",
        "image_size": 256,
        "batch_size": 16,
        "num_workers": 10,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 42,
        "normalization": {
            "enabled": True,
            "transform": "linear",
            "bounds": "per_image",
            "log_scale": 1.0,
        },
    },
    "model": {
        "type": "conditioned_dit",
        "in_channels": 1,
        "out_channels": 1,
        "patch_size": 16,
        "hidden_size": 780,
        "depth": 20,
        "num_heads": 10,
        "mlp_ratio": 4.0,
        "conditioning_params": [
            "jet_pa_deg", "jet_radial_extent_mas",
            "peak_brightness_jy_per_beam",
            "cropped_image_pixel_minimum", "cropped_image_pixel_maximum",
        ],
        "conditioning_settings": {"use_validity_masks": True},
    },
    "diffusion": {"num_steps": 1000, "beta_start": 0.0001, "beta_end": 0.02},
    "loss": {"loss": ["MSELoss"], "weights": [1.0]},
    "optimizer": {
        "type": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 0.00000001,
        "amsgrad": False,
    },
    "samples": {
        "enabled": True,
        "num_samples": 4,
        "every_n_epochs": 25,
        "seed": 42,
        "nrow": 2,
        "save_final_grid": True,
        "save_denoising_grids": False,
        "conditioning": {"mode": "auto"},
    },
}

# Each key is a dotted CONFIG path; each list contains candidate values.
# Add another scalar setting, e.g. "optimizer.learning_rate": [0.0001, 0.0003].
# Widths must divide by heads; patch sizes must divide both image dimensions.
# These example ranges have not been checked against the server's GPU memory.
SEARCH_SPACE = {
    "model.patch_size": [8, 16, 32],
    "model.hidden_size": [512, 768, 1024],
    "model.depth": [8, 12, 16],
    "model.mlp_ratio": [2.0, 4.0],
    "model.num_heads": [4, 8, 16],
    # Signed log1p preserves negative pixels; log_scale stays fixed in CONFIG.
    "data.normalization.transform": ["linear", "signed_log1p"],
}

# Named combinations are one additional categorical parameter. Use {} to keep
# CONFIG's conditioning fixed. Per-image normalization requires both scale fields.
CONDITIONING_CHOICES = {
    "pixel_scale": [
        "cropped_image_pixel_minimum", "cropped_image_pixel_maximum",
    ],
    "jet": [
        "jet_pa_deg", "jet_radial_extent_mas",
        "cropped_image_pixel_minimum", "cropped_image_pixel_maximum",
    ],
    "jet_and_peak": [
        "jet_pa_deg", "jet_radial_extent_mas",
        "peak_brightness_jy_per_beam",
        "cropped_image_pixel_minimum", "cropped_image_pixel_maximum",
    ],
}

# TPE learns from previous trials; random samples independently; grid tries all
# listed combinations. startup_trials applies only to TPE's initial random trials.
SAMPLER = {"name": "tpe", "seed": 42, "startup_trials": 10}

# Pruning stops weak trials. min_epochs counts completed epochs, starting at one.
# Supported names: successive_halving, median, none. Median uses startup_trials.
PRUNER = {
    "name": "successive_halving", "min_epochs": 35,
    "reduction_factor": 2, "startup_trials": 5,
}

SEARCH = {
    "study": "dit-search", "hours": 72, "storage": "optuna.journal",
    "gpus": [1, 2, 3],
    "output_dir": "optuna_runs", "preflight": False,
}


def parse_args() -> argparse.Namespace:
    """Read search controls; experiment settings stay in CONFIG."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--study", default=SEARCH["study"])
    parser.add_argument("--experiment", help="MLflow experiment; defaults to --study.")
    parser.add_argument("--storage", type=Path, help="Journal path; defaults to OUTPUT_ROOT/optuna_runs/optuna.journal.")
    parser.add_argument("--gpu", type=int, help="Run one worker on this visible GPU index.")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to OUTPUT_ROOT/optuna_runs.")
    parser.add_argument(
        "--hours", type=float, default=SEARCH["hours"], help="Search budget per launch (default: 72).",
    )
    parser.add_argument("--trials", type=int, help="Optional trial limit per worker per launch.")
    parser.add_argument("--epochs", type=int, default=CONFIG["trainer"]["max_epochs"])
    parser.add_argument("--sample-every", type=int, default=CONFIG["samples"]["every_n_epochs"])
    parser.add_argument("--sampler", choices=("tpe", "random", "grid"), default=SAMPLER["name"])
    parser.add_argument(
        "--pruner", choices=("successive_halving", "median", "none"), default=PRUNER["name"],
    )
    parser.add_argument("--prune-after", type=int, default=PRUNER["min_epochs"])
    parser.add_argument(
        "--preflight", action=argparse.BooleanOptionalAction, default=SEARCH["preflight"],
        help="Check one synthetic GPU optimizer step before each trial; excludes sampling.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be finite and positive.")
    if args.epochs < 1 or args.sample_every < 1:
        parser.error("--epochs and --sample-every must be positive.")
    if args.pruner != "none" and not 1 <= args.prune_after < args.epochs:
        parser.error("Require 1 <= --prune-after < --epochs, or use --pruner none.")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be positive.")
    args.output = (args.output or get_path("OUTPUT_ROOT") / SEARCH["output_dir"]).expanduser().resolve()
    args.storage = (args.storage or args.output / SEARCH["storage"]).expanduser().resolve()
    args.experiment = args.experiment or args.study
    return args


def search_choices() -> dict[str, list]:
    """Collect scalar choices and optional named conditioning combinations."""
    choices = deepcopy(SEARCH_SPACE)
    if "model.type" in choices:
        raise ValueError("Keep model.type fixed in CONFIG.")
    managed = {"trainer.max_epochs", "samples.every_n_epochs", "tracking.study"}
    if managed.intersection(choices):
        raise ValueError("Set epochs, sample frequency, and study through CONFIG/CLI, not SEARCH_SPACE.")
    if CONDITIONING_CHOICES:
        if CONFIG["model"]["type"] != "conditioned_dit":
            raise ValueError("Set CONDITIONING_CHOICES = {} for an unconditioned model.")
        if "model.conditioning_params" in choices:
            raise ValueError("Define conditioning combinations in CONDITIONING_CHOICES.")
        choices["conditioning"] = list(CONDITIONING_CHOICES)
    if not choices or any(not isinstance(items, list) or not items for items in choices.values()):
        raise ValueError("Define at least one search parameter with a nonempty candidate list.")
    return choices


def trial_config(trial, args: argparse.Namespace) -> ExperimentConfig:
    """Apply sampled values to a fresh copy of the fixed settings."""
    values = deepcopy(CONFIG)
    for path, choices in search_choices().items():
        selected = trial.suggest_categorical(path, choices)
        if path == "conditioning":
            values["model"]["conditioning_params"] = list(CONDITIONING_CHOICES[selected])
            continue
        *parents, key = path.split(".")
        section = values
        for parent in parents:
            section = section.setdefault(parent, {})
        section[key] = selected
    values["run"]["name"] += f"_trial_{trial.number:05d}"
    values["trainer"]["max_epochs"] = args.epochs
    if getattr(args, "gpu", None) is not None:
        values["trainer"]["devices"] = [args.gpu]
    values["tracking"]["study"] = args.study
    values["samples"]["every_n_epochs"] = args.sample_every
    return ExperimentConfig.from_dict(values, Path(__file__).resolve())


def validate_search(args: argparse.Namespace) -> ExperimentConfig:
    """Check candidate combinations before spending training time."""
    choices = search_choices()
    for candidate in product(*choices.values()):
        parameters = dict(zip(choices, candidate))
        try:
            config = trial_config(optuna.trial.FixedTrial(parameters), args)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid search combination {parameters}: {exc}") from exc
        devices = config.trainer.devices
        if devices != 1 and not (isinstance(devices, list) and len(devices) == 1):
            raise ValueError("Search trials support one device; use devices=1 or devices=[index].")
        if not config.tracking.enabled:
            raise ValueError("Enable CONFIG['tracking']['enabled'] for the search.")
    _get_tracking_uri(config)
    if args.preflight and not torch.cuda.is_available():
        raise RuntimeError("--preflight requires a visible CUDA GPU.")
    return config


def make_sampler(args: argparse.Namespace):
    """Choose the algorithm that proposes trial parameters."""
    seed = SAMPLER["seed"] + (getattr(args, "gpu", None) or 0)
    if args.sampler == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=SAMPLER["startup_trials"], constant_liar=True,
        )
    if args.sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.GridSampler(search_choices(), seed=SAMPLER["seed"])


def make_pruner(args: argparse.Namespace):
    """Choose whether and when unpromising trials stop early."""
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER["startup_trials"], n_warmup_steps=args.prune_after,
        )
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=args.prune_after, reduction_factor=PRUNER["reduction_factor"],
    )


def create_study(args: argparse.Namespace) -> optuna.Study:
    """Resume a study only when its experiment and search settings match."""
    settings = {
        "config": CONFIG, "search_space": SEARCH_SPACE,
        "conditioning": CONDITIONING_CHOICES,
        "sampler": {**SAMPLER, "name": args.sampler},
        "pruner": {**PRUNER, "name": args.pruner, "min_epochs": args.prune_after},
        "epochs": args.epochs, "sample_every": args.sample_every,
        "experiment": args.experiment, "preflight": args.preflight,
    }
    args.storage.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study, storage=JournalStorage(JournalFileBackend(str(args.storage))),
        direction="minimize", sampler=make_sampler(args), pruner=make_pruner(args),
        load_if_exists=True,
    )
    previous = study.user_attrs.get("search_settings")
    if previous is not None and previous != settings:
        raise ValueError("Search settings changed. Choose a new --study name.")
    study.set_user_attr("search_settings", settings)
    return study


def without_none(values: dict) -> dict:
    """Omit optional unset values, which TOML cannot represent."""
    return {
        key: without_none(value) if isinstance(value, dict) else value
        for key, value in values.items() if value is not None
    }


def save_trial_config(trial, config: ExperimentConfig, args: argparse.Namespace):
    """Write a complete config that normal training can load directly."""
    run_dir = args.output / args.study / f"trial_{trial.number:05d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "config.toml"
    with path.open("wb") as destination:
        tomli_w.dump(without_none(_config_values(config)), destination)
    config = load_config(path)
    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("config_path", str(path))
    return config, run_dir


def start_mlflow_run(trial, config: ExperimentConfig, args: argparse.Namespace):
    """Give every tested model a run, including models rejected by preflight."""
    client = MlflowClient(tracking_uri=_get_tracking_uri(config))
    experiment = client.get_experiment_by_name(args.experiment)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(args.experiment)
    run = client.create_run(
        experiment_id, run_name=f"trial_{trial.number:05d}",
        tags={
            "optuna.study": args.study, "optuna.trial": str(trial.number),
            "optuna.state": "RUNNING", "optuna.sampler": args.sampler,
            "optuna.pruner": args.pruner,
            "optuna.gpu": str(getattr(args, "gpu", None)),
        },
    )
    trial.set_user_attr("mlflow_run_id", run.info.run_id)
    trial.set_user_attr("mlflow_experiment_id", experiment_id)
    return client, run.info.run_id


def cleanup_memory() -> None:
    """Release discarded model cycles and compilation caches between trials."""
    torch.compiler.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_preflight(trial, config: ExperimentConfig, run_dir: Path, client, run_id: str):
    """Reuse the synthetic training check with a separate, disposable model."""
    try:
        report = run_config_preflight(config)
    finally:
        cleanup_memory()
    trial.set_user_attr("preflight", report)
    path = run_dir / "preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    client.log_artifact(run_id, str(path), artifact_path="preflight")
    if not report["success"]:
        raise torch.cuda.OutOfMemoryError(report["error"] or "GPU preflight failed.")


class TrialFeedback(Callback):
    """Keep the best score in memory and report validation progress to Optuna."""

    def __init__(self, trial) -> None:
        self.trial = trial
        self.best_score = math.inf

    def on_validation_end(self, trainer: Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get("val_loss")
        if score is None:
            raise RuntimeError("Validation did not log val_loss.")
        value = float(score)
        if not math.isfinite(value):
            raise FloatingPointError("Validation loss is not finite.")
        self.best_score = min(self.best_score, value)
        self.trial.set_user_attr("best_val_loss", self.best_score)
        self.trial.report(value, step=trainer.current_epoch + 1)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Stopped after {trainer.current_epoch + 1} epochs.")


def build_search_trainer(config, args, run_dir: Path, run_id: str, feedback):
    """Reuse tracking and sample callbacks, without any checkpoint callback."""
    tracking = MLflowTrackingCallback(
        config=_config_values(config), run_dir=run_dir,
        tracking_uri=_get_tracking_uri(config), study=args.study,
        experiment_name=args.experiment, run_id=run_id,
    )
    callbacks = [tracking, feedback]
    if config.samples.enabled:
        callbacks.append(build_sample_callback(config, run_dir))
    return Trainer(
        **asdict(config.trainer), default_root_dir=run_dir,
        logger=False, callbacks=callbacks, enable_checkpointing=False,
        num_sanity_val_steps=0, enable_progress_bar=False,
    )


def fit_trial(trial, config, args, run_dir: Path, run_id: str) -> float:
    """Train fresh weights and return only the best validation score."""
    seed_experiment(config)
    module = build_training_module(config)
    data = build_data_module(config)
    feedback = TrialFeedback(trial)
    trainer = build_search_trainer(config, args, run_dir, run_id, feedback)
    trial.set_user_attr("model_parameters", sum(p.numel() for p in module.model.parameters()))
    trainer.fit(module, datamodule=data)
    if not math.isfinite(feedback.best_score):
        raise FloatingPointError("Training produced no finite validation score.")
    return feedback.best_score


def finish_mlflow_run(client, run_id: str, trial, state: str) -> None:
    """Distinguish pruned trials from failed training without saving models."""
    try:
        client.set_tag(run_id, "optuna.state", state)
        best = trial.user_attrs.get("best_val_loss")
        if best is not None:
            client.log_metric(run_id, "best_val_loss", best)
        client.set_terminated(run_id, status="FAILED" if state == "FAIL" else "FINISHED")
    except Exception as exc:
        print(f"Trial {trial.number}: could not finalize MLflow run {run_id}: {exc}", flush=True)


def objective(trial, args: argparse.Namespace) -> float:
    """Save the trial config, optionally check memory, and train without weights on disk."""
    config, run_dir = save_trial_config(trial, trial_config(trial, args), args)
    client, run_id = start_mlflow_run(trial, config, args)
    state, started = "FAIL", monotonic()
    try:
        client.log_artifact(run_id, str(config.source_path), artifact_path="config")
        for name, value in trial.params.items():
            client.log_param(run_id, name, value)
        if args.preflight:
            check_preflight(trial, config, run_dir, client, run_id)
        score = fit_trial(trial, config, args, run_dir, run_id)
        state = "COMPLETE"
        return score
    except optuna.TrialPruned:
        state = "PRUNED"
        raise
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        raise
    finally:
        trial.set_user_attr("elapsed_seconds", monotonic() - started)
        finish_mlflow_run(client, run_id, trial, state)


def after_trial(study, trial) -> None:
    """Keep a copy of the winning config and free each discarded model."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        folder = Path(trial.user_attrs["config_path"]).parent.parent
        # Other GPUs may finish concurrently. Choose the winner while locked.
        with (folder / ".best.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            winner = study.best_trial
            temporary = folder / "best.toml.tmp"
            shutil.copyfile(winner.user_attrs["config_path"], temporary)
            temporary.replace(folder / "best.toml")
        if winner.number == trial.number:
            print(f"Best config: {folder / 'best.toml'}", flush=True)
    cleanup_memory()


def optimize_worker(study, args: argparse.Namespace) -> None:
    """Run sequential trials on one GPU, sharing study history with the others."""
    print(f"Worker GPU: {args.gpu}; study: {args.study}", flush=True)
    study.optimize(
        lambda trial: objective(trial, args), n_trials=args.trials,
        timeout=args.hours * 3600, n_jobs=1, gc_after_trial=True,
        catch=(torch.cuda.OutOfMemoryError, FloatingPointError), callbacks=[after_trial],
    )


def launch_workers(args: argparse.Namespace, config: ExperimentConfig) -> None:
    """Start one process per selected GPU and stop the others on a fatal error."""
    gpus = SEARCH["gpus"]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("SEARCH['gpus'] must contain distinct GPU indices.")
    if any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in gpus):
        raise ValueError(f"GPU indices {gpus} are not all visible to PyTorch.")
    client = MlflowClient(tracking_uri=_get_tracking_uri(config))
    if client.get_experiment_by_name(args.experiment) is None:
        client.create_experiment(args.experiment)
    processes = {}
    try:
        for gpu in gpus:
            command = [
                sys.executable, "-m", "radio_diffusion.training.optuna_search",
                *sys.argv[1:], "--gpu", str(gpu),
            ]
            processes[subprocess.Popen(command)] = gpu
        while processes:
            for process, gpu in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                del processes[process]
                if code:
                    raise RuntimeError(f"Worker on GPU {gpu} exited with status {code}.")
            if processes:
                sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


def print_result(study) -> int:
    """Report the best completed trial across all workers."""
    if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        print("No completed trials. Review failed/pruned trials in Optuna and MLflow.")
        return 1
    best = study.best_trial
    print(f"Best trial: {best.number}\nValidation loss: {best.value}")
    print(json.dumps(best.params, indent=2))
    print(f"Config: {best.user_attrs['config_path']}")
    return 0


def main() -> int:
    """Launch the GPU pool, or run a single worker selected with --gpu."""
    args = parse_args()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    config = validate_search(args)
    study = create_study(args)
    print(
        f"Study: {study.study_name}\nMLflow experiment: {args.experiment}\n"
        f"Storage: {args.storage}\nOutput directory: {args.output / args.study}", flush=True,
    )
    print("The budget stops new trials; active trials can finish afterward.", flush=True)
    if args.gpu is None and config.trainer.accelerator != "cpu":
        launch_workers(args, config)
    else:
        optimize_worker(study, args)
        if args.gpu is not None:
            return 0
    return print_result(study)


if __name__ == "__main__":
    raise SystemExit(main())
