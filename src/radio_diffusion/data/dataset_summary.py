"""Describe preprocessing runs and aggregate saved, untransformed crop metadata.

Dataset length counts input observations; sample count counts successful saves.
Finished means every observation was attempted, even if some or all failed.
"""

import json
import math
from pathlib import Path


def initial_dataset_summary(
    dataset_length: int,
    preprocessing_hash: str,
    metadata_schema: dict,
    preprocessing_settings: dict,
) -> dict:
    """Build the unfinished summary written before processing starts."""
    return {
        "preprocessing_hash": preprocessing_hash,
        "preprocessing_finished": False,
        "dataset_length": dataset_length,
        "sample_count": 0,
        "sample_files": [],
        "failed_observations": [],
        "unexpected_sample_files": [],
        "metadata_schema": metadata_schema,
        "preprocessing_settings": preprocessing_settings,
        "cropped_image_pixels": None,
        "metadata_statistics": {},
    }


def write_dataset_summary(output_dir: Path, summary: dict) -> None:
    """Replace the JSON atomically so interruption cannot leave partial JSON."""
    path = output_dir / "dataset_summary.json"
    temporary_path = output_dir / "dataset_summary.json.tmp"
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def finish_dataset_summary(
    output_dir: Path, summary: dict, metadata_rows: dict, failures: list,
) -> None:
    """Save results after all observations were attempted, including failures."""
    summary["sample_files"] = sorted(metadata_rows)
    summary["sample_count"] = len(metadata_rows)
    summary["failed_observations"] = [
        {"filename": filename, "error": error} for filename, error in sorted(failures)
    ]
    rows = [metadata_rows[name] for name in summary["sample_files"]]
    summary.update(summarize_metadata(rows, summary["metadata_schema"]))
    saved_files = {path.name for path in output_dir.glob("*.icn.pt") if path.is_file()}
    summary["unexpected_sample_files"] = sorted(saved_files - metadata_rows.keys())
    summary["preprocessing_finished"] = True
    write_dataset_summary(output_dir, summary)


def _numeric_statistics(values: list[float], total_count: int) -> dict:
    """Summarize finite values with a population standard deviation."""
    count = len(values)
    mean = math.fsum(values) / count if count else None
    return {
        "valid_count": count,
        "missing_or_nonfinite_count": total_count - count,
        "minimum": min(values) if count else None,
        "maximum": max(values) if count else None,
        "mean": mean,
        "std": (
            math.sqrt(math.fsum((value - mean) ** 2 for value in values) / count)
            if count else None
        ),
    }


def summarize_metadata(rows: list[dict], metadata_schema: dict) -> dict:
    """Summarize finite scalar metrics; omit arithmetic averages of angles."""
    statistics = {}
    for field, definition in metadata_schema.items():
        if definition["type"] != "number" or field == "jet_pa_deg":
            continue
        values = [float(row[field]) for row in rows if math.isfinite(row[field])]
        statistics[field] = _numeric_statistics(values, len(rows))
    return {
        "cropped_image_pixels": _summarize_cropped_pixels(rows),
        "metadata_statistics": statistics,
    }


def _summarize_cropped_pixels(rows: list[dict]) -> dict | None:
    """Merge crop population variances with pixel counts, without averaging stds."""
    if not rows or not all(
        math.isfinite(row[field])
        for row in rows
        for field in (
            "cropped_image_pixel_minimum", "cropped_image_pixel_maximum",
            "cropped_image_pixel_mean", "cropped_image_pixel_std",
        )
    ):
        return None
    counts = [math.prod(row["image_size"]) for row in rows]
    count = sum(counts)
    mean = math.fsum(
        size * row["cropped_image_pixel_mean"] for size, row in zip(counts, rows)
    ) / count
    variance = math.fsum(
        size * (
            row["cropped_image_pixel_std"] ** 2
            + (row["cropped_image_pixel_mean"] - mean) ** 2
        )
        for size, row in zip(counts, rows)
    ) / count
    return {
        "count": count,
        "minimum": min(row["cropped_image_pixel_minimum"] for row in rows),
        "maximum": max(row["cropped_image_pixel_maximum"] for row in rows),
        "mean": mean,
        "std": math.sqrt(variance),
    }
