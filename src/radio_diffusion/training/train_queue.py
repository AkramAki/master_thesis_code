"""Run queued experiment configs through preflight or normal training."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from time import monotonic
import traceback

from radio_diffusion.paths import load_env

load_env()

from lightning.pytorch.utilities.rank_zero import rank_zero_only

from radio_diffusion.load_config import load_config
from radio_diffusion.training.preflight import run_config_preflight
from radio_diffusion.training.train import run_training


def read_queue(queue_path: str | Path) -> list[Path]:
    """Read config paths in order, relative to the queue file's directory."""
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


@rank_zero_only
def log(message: str) -> None:
    """Print a timestamped queue message on rank zero."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _run_config(
    config_path: Path, index: int, total: int, preflight: bool
) -> dict[str, str]:
    """Load and execute one config, retaining failures for the queue report."""
    started = monotonic()
    record = {
        "index": str(index),
        "config": str(config_path),
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": "",
        "run_dir": "",
        "best_checkpoint": "",
        "last_checkpoint": "",
        "error": "",
        "elapsed": "",
    }
    mode = "PREFLIGHT" if preflight else "START"
    log(f"[{index}/{total}] {mode} {config_path}")

    def record_run_directory(run_dir: Path) -> None:
        record["run_dir"] = str(run_dir)
        record["best_checkpoint"] = str(run_dir / "checkpoints" / "best.ckpt")
        record["last_checkpoint"] = str(run_dir / "checkpoints" / "last.ckpt")

    try:
        config = load_config(config_path)
        if preflight:
            report = run_config_preflight(config)
            record["status"] = "preflight_passed" if report["success"] else "preflight_failed"
            record["error"] = report["error"] or ""
        else:
            record.update(run_training(config, on_run_created=record_run_directory))
            record["status"] = "completed"
    except Exception as exc:
        record["status"] = "preflight_failed" if preflight else "failed"
        record["error"] = str(exc)
        log(traceback.format_exc().rstrip())

    record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    record["elapsed"] = f"{monotonic() - started:.1f}s"
    label = record["status"].replace("_", " ").upper()
    log(f"[{index}/{total}] {label} {config_path}")
    if record["status"] in ("failed", "preflight_failed"):
        log("Continuing to the next config.")
    return record


def _print_queue_report(records: list[dict[str, str]], preflight: bool) -> int:
    """Print the final queue summary and return its exit code."""
    passed = 0
    log("Preflight report" if preflight else "Queue report")
    for record in records:
        succeeded = record["status"] in ("completed", "preflight_passed")
        passed += int(succeeded)
        status = "PASS" if succeeded else "FAIL"
        log(f"{record['index']}. {Path(record['config']).name}: {status} ({record['elapsed']})")
        if record["run_dir"]:
            log(f"   Run directory: {record['run_dir']}")
        if record["error"]:
            log(f"   {record['error']}")
    total = len(records)
    mode = "Preflight" if preflight else "Queue"
    description = "configs passed" if preflight else "runs completed"
    log(f"{mode} finished: {passed}/{total} {description}; {total - passed} failed.")
    return 0 if passed == total else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Read the command-line options and validate the queue paths."""
    parser = argparse.ArgumentParser(
        description="Run experiment configs one at a time in queue-file order."
    )
    parser.add_argument(
        "queue_file",
        type=Path,
        help="Text file with one TOML config path per line.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run one GPU training step per config with synthetic batches, then exit.",
    )
    args = parser.parse_args(argv)
    try:
        args.config_paths = read_queue(args.queue_file)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    """Run each queued config in the selected mode and summarize the results."""
    args = _parse_args(argv)
    records = []
    total = len(args.config_paths)
    for index, config_path in enumerate(args.config_paths, start=1):
        records.append(_run_config(config_path, index, total, args.preflight))
    return _print_queue_report(records, args.preflight)


if __name__ == "__main__":
    sys.exit(main())
