"""Run experiment configs sequentially from a simple text queue."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import sys
import traceback
from typing import Callable
from uuid import uuid4

from radio_diffusion.experiment import (
    build_data_module,
    build_trainer,
    build_training_module,
    format_model_size_summary,
    load_config,
    preflight_training,
    seed_experiment,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SUMMARY_FIELDS = (
    "index",
    "config",
    "status",
    "started_at",
    "finished_at",
    "run_dir",
    "best_checkpoint",
    "last_checkpoint",
    "error",
)


def read_queue(queue_path: str | Path) -> list[Path]:
    """Read ordered config paths, relative to the queue file's directory."""
    queue_path = Path(queue_path).expanduser().resolve()
    if not queue_path.is_file():
        raise FileNotFoundError(f"Queue file does not exist: {queue_path}")

    config_paths = []
    for line_number, raw_line in enumerate(
        queue_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        config_entry = raw_line.split("#", maxsplit=1)[0].strip()
        if not config_entry:
            continue
        config_path = Path(config_entry).expanduser()
        if not config_path.is_absolute():
            config_path = queue_path.parent / config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config on queue line {line_number} does not exist: {config_path}"
            )
        config_paths.append(config_path)

    if not config_paths:
        raise ValueError(f"Queue has no config paths: {queue_path}")
    return config_paths


def train_config(
    config_path: Path,
    on_run_created: Callable[[Path], None] | None = None,
) -> dict[str, str]:
    """Build and train one experiment, returning its run directory."""
    config = load_config(config_path)
    seed_experiment(config)
    training_module = build_training_module(config)
    model_summary = format_model_size_summary(training_module.model)
    batch_size = config["data"]["batch_size"]
    devices = config.get("trainer", {}).get("devices", "auto")
    batch_summary = f"DataLoader batch size per device: {batch_size}"
    if isinstance(devices, int):
        batch_summary += (
            f"\nEffective batch size across {devices} devices: "
            f"{batch_size * devices} (assuming distributed data parallel)"
        )
    print(f"Model size:\n{model_summary}\n{batch_summary}", flush=True)

    data = build_data_module(config)
    data.prepare_data()

    trainer = build_trainer(config)
    run_dir = Path(trainer.checkpoint_callback.dirpath).parent
    (run_dir / "model_summary.txt").write_text(
        model_summary + "\n" + batch_summary + "\n", encoding="utf-8"
    )
    print(f"Run directory: {run_dir}", flush=True)
    if on_run_created is not None:
        on_run_created(run_dir)
    trainer.fit(training_module, datamodule=data)

    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}", flush=True)
    print(f"Last checkpoint: {trainer.checkpoint_callback.last_model_path}", flush=True)
    return {
        "run_dir": str(run_dir),
        "best_checkpoint": trainer.checkpoint_callback.best_model_path,
        "last_checkpoint": trainer.checkpoint_callback.last_model_path,
    }


def save_summary(summary_path: Path, records: list[dict[str, str]]) -> None:
    """Write the queue's current status table to CSV."""
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train experiment configs one at a time in queue-file order."
    )
    parser.add_argument(
        "queue_file",
        type=Path,
        help="Text file with one TOML config path per line.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Check one training step for each config using its batch size and "
            "GPU setup, then exit without full training."
        ),
    )
    args = parser.parse_args(argv)

    try:
        config_paths = read_queue(args.queue_file)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    log_path = None
    summary_path = None
    if not args.preflight:
        queue_file = args.queue_file.expanduser().resolve()
        queue_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{uuid4().hex[:6]}"
        queue_dir = _REPOSITORY_ROOT / "outputs" / "queues" / queue_id
        queue_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(queue_file, queue_dir / "queue.txt")
        log_path = queue_dir / "queue.log"
        summary_path = queue_dir / "summary.csv"

    def log(message: str) -> None:
        if log_path is not None:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp} {message}\n")
        print(message, flush=True)

    records = [
        {
            "index": str(index),
            "config": str(config_path),
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "run_dir": "",
            "best_checkpoint": "",
            "last_checkpoint": "",
            "error": "",
        }
        for index, config_path in enumerate(config_paths, start=1)
    ]
    if summary_path is not None:
        save_summary(summary_path, records)

    total = len(config_paths)
    for index, config_path in enumerate(config_paths, start=1):
        record = records[index - 1]
        record["status"] = "running"
        record["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if summary_path is not None:
            save_summary(summary_path, records)
        mode = "PREFLIGHT" if args.preflight else "START"
        log(f"[{index}/{total}] {mode} {config_path}")

        def record_run_directory(run_dir: Path) -> None:
            record["run_dir"] = str(run_dir)
            record["best_checkpoint"] = str(run_dir / "checkpoints" / "best.ckpt")
            record["last_checkpoint"] = str(run_dir / "checkpoints" / "last.ckpt")
            if summary_path is not None:
                save_summary(summary_path, records)

        try:
            if args.preflight:
                config = load_config(config_path)
                seed_experiment(config)
                batch_size = config["data"]["batch_size"]
                devices = config.get("trainer", {}).get("devices", "auto")
                training_module = build_training_module(config)
                log(
                    "Model size:\n"
                    f"{format_model_size_summary(training_module.model)}"
                )
                log(f"Configured per-device batch size: {batch_size}; devices: {devices}")
                preflight_training(config, training_module)
                record["status"] = "preflight_passed"
            else:
                result = train_config(config_path, on_run_created=record_run_directory)
                record.update(result)
                record["status"] = "completed"
        except Exception as exc:
            error_trace = traceback.format_exc()
            record["status"] = "preflight_failed" if args.preflight else "failed"
            record["error"] = str(exc)
            log(error_trace.rstrip())
            label = "PREFLIGHT FAILED" if args.preflight else "FAILED"
            log(f"[{index}/{total}] {label}; continuing to the next config: {config_path}")

        record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if summary_path is not None:
            save_summary(summary_path, records)
        if record["status"] == "completed":
            log(f"[{index}/{total}] COMPLETED {config_path}")
        elif record["status"] == "preflight_passed":
            log(f"[{index}/{total}] PREFLIGHT PASSED {config_path}")

    failure_status = "preflight_failed" if args.preflight else "failed"
    failures = [record for record in records if record["status"] == failure_status]
    if args.preflight:
        log("\nPreflight report")
        for record in records:
            status = "PASS" if record["status"] == "preflight_passed" else "FAIL"
            log(f"{record['index']}. {Path(record['config']).name}: {status}")
            if record["error"]:
                log(f"   {record['error']}")

    if failures:
        if args.preflight:
            log(f"Preflight finished: {total - len(failures)}/{total} configs passed, {len(failures)} failed.")
            return 1
        log(f"Queue finished: {total - len(failures)}/{total} succeeded, {len(failures)} failed.")
        log(f"Queue report: {summary_path}")
        log(f"Queue log: {log_path}")
        return 1

    if args.preflight:
        log(f"Preflight passed for all {total} configs.")
        return 0

    log(f"Queue finished successfully ({total}/{total} runs).")
    log(f"Queue report: {summary_path}")
    log(f"Queue log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
