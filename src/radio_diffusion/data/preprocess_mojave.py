"""Preprocess MOJAVE observations into central crops and extracted metadata.

Stored pixels are cropped without intensity scaling or transformation.
"""

import argparse
import json
import warnings
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal

from radio_diffusion.paths import (
    get_mojave_data_dir,
    get_path,
    get_preprocess_cpu_cores,
    load_env,
)

load_env()

import torch
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

from radio_diffusion.data.dataset_summary import (
    finish_dataset_summary,
    initial_dataset_summary,
    write_dataset_summary,
)
from radio_diffusion.data.mojave_metadata import (
    METADATA_SCHEMA,
    JetOrientationSettings,
    RidgelineSettings,
    cropped_image_metadata,
    full_image_metadata,
)
from radio_diffusion.data.preprocessing_hash import (
    calculate_preprocessing_hash,
    compare_hash_to_current_hash,
)
from radio_diffusion.utils.timing import timelogger


IGNORED_WARNING_CATEGORIES = (FITSFixedWarning, VerifyWarning)


@dataclass(frozen=True)
class PreprocessingSettings:
    """Input/output locations, workers, crop size and metadata subsettings."""

    jet_orientation: JetOrientationSettings
    ridgeline: RidgelineSettings
    input_dir: str | Path | None = None
    uv_dir: str | Path | None = None
    output_dir: str | Path | None = None
    image_size: int = 256
    cpu_cores: int | None = None
    noise_method: Literal["header", "four_corner"] = "four_corner"


PREPROCESSING_SETTINGS = PreprocessingSettings(
    jet_orientation=JetOrientationSettings(
        inner_mas=0.2,
        outer_mas=3.0,
        sigma_cut=3.0,
    ),
    ridgeline=RidgelineSettings(
        radial_step_mas=0.05,
        max_radius_mas=None,
        half_width_deg=90.0,
        angle_step_deg=0.5,
        sigma_cut=8.0,
    ),
)


def iter_observations(input_dir: str | Path, uv_dir: str | Path) -> Iterator[tuple]:
    """Yield image paths paired with their UV path, or None if unavailable."""
    for image_path in sorted(Path(input_dir).glob("*.icn.fits.gz")):
        if not image_path.is_file():
            continue
        observation_name = image_path.name.removesuffix(".icn.fits.gz")
        uv_path = Path(uv_dir) / f"{observation_name}.uvf"
        yield image_path, uv_path if uv_path.is_file() else None


def process_observation(
    observation: tuple[Path, Path | None], settings: PreprocessingSettings,
) -> dict:
    """Open one FITS/UV pair, save the crop and return its saved metadata."""
    image_path, uv_path = observation
    with (
        fits.open(image_path) as image_hdul,
        (fits.open(uv_path) if uv_path is not None else nullcontext()) as uv_hdul,
    ):
        metadata = full_image_metadata(
            image_hdul, image_path.name.split(".", 1)[0], settings.noise_method,
            settings.jet_orientation, settings.ridgeline,
        )
        image = torch.from_numpy(image_hdul[0].data[0, 0].astype("float32"))
        cropped_image = center_crop(image, settings.image_size).clone()
        metadata.update(cropped_image_metadata(cropped_image))
        # Additional full-image features can use image_hdul and uv_hdul here.
        # uv_hdul is None when the observation has no matching UV file.
    output_name = image_path.name.removesuffix(".icn.fits.gz") + ".icn.pt"
    torch.save({"image": cropped_image, "metadata": metadata}, settings.output_dir / output_name)
    return metadata


def _process_observation_with_warning_counts(
    observation: tuple[Path, Path | None], settings: PreprocessingSettings,
) -> tuple[Counter[str], str | None, dict | None]:
    """Suppress and count selected warnings, including when processing fails."""
    warning_counts = Counter()
    error_message = None
    metadata = None
    with warnings.catch_warnings():
        showwarning = warnings.showwarning

        def count_warning(message, category, filename, lineno, file=None, line=None):
            for ignored_category in IGNORED_WARNING_CATEGORIES:
                if issubclass(category, ignored_category):
                    warning_counts[ignored_category.__name__] += 1
                    return
            showwarning(message, category, filename, lineno, file=file, line=line)

        for category in IGNORED_WARNING_CATEGORIES:
            warnings.simplefilter("always", category=category)
        warnings.showwarning = count_warning
        try:
            metadata = process_observation(observation, settings)
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
    return warning_counts, error_message, metadata


def _resolve_preprocessing_settings(settings: PreprocessingSettings) -> PreprocessingSettings:
    """Resolve environment defaults and paths once before starting workers."""
    cpu_cores = get_preprocess_cpu_cores() if settings.cpu_cores is None else settings.cpu_cores
    if settings.image_size < 1 or cpu_cores < 1:
        raise ValueError("Image size and CPU worker count must be positive.")
    input_dir = (
        get_path("MOJAVE_FITS_DIR") if settings.input_dir is None
        else Path(settings.input_dir).expanduser()
    )
    uv_dir = (
        get_path("MOJAVE_UV_DIR") if settings.uv_dir is None
        else Path(settings.uv_dir).expanduser()
    )
    output_dir = (
        get_mojave_data_dir(settings.image_size) if settings.output_dir is None
        else Path(settings.output_dir).expanduser()
    )
    return replace(
        settings, input_dir=input_dir, uv_dir=uv_dir, output_dir=output_dir, cpu_cores=cpu_cores,
    )


def _settings_values(settings: PreprocessingSettings) -> dict:
    """Serialize effective settings, including nested metadata settings, for JSON."""
    values = asdict(settings)
    for name in ("input_dir", "uv_dir", "output_dir"):
        if values[name] is not None:
            values[name] = str(values[name])
    return values


def _dataset_is_current(settings: PreprocessingSettings, observations: list[tuple]) -> bool:
    """Check the completed definition, processing parameters and saved file set."""
    summary_path = settings.output_dir / "dataset_summary.json"
    values = _settings_values(settings)
    parameters = {
        name: values[name]
        for name in ("input_dir", "uv_dir", "image_size", "noise_method", "jet_orientation", "ridgeline")
    }
    if not compare_hash_to_current_hash(
        summary_path, len(observations), preprocessing_settings=parameters,
    ):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample_files = summary.get("sample_files")
    if not isinstance(sample_files, list) or not sample_files:
        return False
    if not all(isinstance(name, str) for name in sample_files):
        return False
    failures = summary.get("failed_observations", [])
    if not isinstance(failures, list) or not all(
        isinstance(row, dict) and isinstance(row.get("filename"), str) for row in failures
    ):
        return False
    expected_files = set(sample_files)
    attempted = {name.removesuffix(".icn.pt") + ".icn.fits.gz" for name in sample_files}
    attempted.update(row["filename"] for row in failures)
    saved_files = {
        path.name for path in settings.output_dir.glob("*.icn.pt") if path.is_file()
    }
    return (
        summary.get("sample_count") == len(sample_files) == len(expected_files)
        and saved_files == expected_files
        and attempted == {image.name for image, _ in observations}
    )


def _print_preprocessing_start(settings: PreprocessingSettings) -> None:
    """Report input locations and effective worker settings."""
    print(
        f"MOJAVE preprocessing\nFITS directory: {settings.input_dir}\nUV directory: {settings.uv_dir}\n"
        f"Output directory: {settings.output_dir}\n"
        f"Image size: {settings.image_size}; CPU cores: {settings.cpu_cores}\n"
        "Ignoring warnings (occurrences will be counted): "
        + ", ".join(category.__name__ for category in IGNORED_WARNING_CATEGORIES),
        flush=True,
    )


def _collect_observation_result(future, image_path, metadata_rows, failures, warning_counts):
    """Record one completed future without treating interruption as a failure."""
    try:
        counts, error_message, metadata = future.result()
    except Exception as error:
        failures.append((image_path.name, f"{type(error).__name__}: {error}"))
        return
    warning_counts.update(counts)
    if error_message is not None:
        failures.append((image_path.name, error_message))
        return
    output_name = image_path.name.removesuffix(".icn.fits.gz") + ".icn.pt"
    metadata_rows[output_name] = metadata


def _process_observations(observations, settings: PreprocessingSettings) -> tuple:
    """Attempt every observation and collect saved metadata, failures and warnings."""
    process_image = partial(
        _process_observation_with_warning_counts, settings=settings,
    )
    failures = []
    metadata_rows = {}
    warning_counts = Counter()
    with ProcessPoolExecutor(
        max_workers=settings.cpu_cores, initializer=torch.set_num_threads, initargs=(1,),
    ) as executor, tqdm(
        total=len(observations), desc="Processing observations", unit="image",
    ) as progress:
        futures = {
            executor.submit(process_image, observation): observation[0]
            for observation in observations
        }
        for future in as_completed(futures):
            _collect_observation_result(
                future, futures[future], metadata_rows, failures, warning_counts,
            )
            progress.update(1)
    return metadata_rows, failures, warning_counts


def _print_preprocessing_report(observations, summary, failures, warning_counts) -> None:
    """Report all attempted observations and raise if none could be saved."""
    processed_count = summary["sample_count"]
    missing_uv = [image.name for image, uv in observations if uv is None]
    if summary["unexpected_sample_files"]:
        print(
            "Output directory contains image files not saved successfully "
            "by this run; see unexpected_sample_files in dataset_summary.json."
        )
    print(f"\nImages found: {len(observations)}")
    print(f"Images processed and saved: {processed_count}")
    print(f"Images without matching UV files: {len(missing_uv)}")
    print(f"Processing failures: {len(failures)}")
    print("Ignored warnings:")
    for category in IGNORED_WARNING_CATEGORIES:
        print(f"  {category.__name__}: {warning_counts[category.__name__]}")
    if missing_uv:
        print("Images without matching UV files:")
        for filename in missing_uv:
            print(f"  {filename}")
    if failures:
        print("Failed images:")
        for filename, error in sorted(failures):
            print(f"  {filename}: {error}")
        if processed_count == 0:
            raise RuntimeError(
                f"Preprocessing failed for all {len(failures)} images; "
                "no images were saved. See report above."
            )
    print(f"Completed: {processed_count} images saved; {len(failures)} observations failed.")


@timelogger
def preprocess_data(settings: PreprocessingSettings | None = None) -> None:
    """Reuse a current dataset or preprocess raw inputs, recording completion."""
    settings = _resolve_preprocessing_settings(
        PREPROCESSING_SETTINGS if settings is None else settings,
    )
    _print_preprocessing_start(settings)
    observations = list(iter_observations(settings.input_dir, settings.uv_dir))
    if not observations:
        raise FileNotFoundError(f"No FITS files found in {settings.input_dir}")
    if settings.output_dir.exists() and not settings.output_dir.is_dir():
        raise NotADirectoryError(f"Mojave data path is not a directory: {settings.output_dir}")
    if _dataset_is_current(settings, observations):
        print(f"MOJAVE dataset is up to date: {settings.output_dir}", flush=True)
        return
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary = initial_dataset_summary(
        len(observations), calculate_preprocessing_hash(len(observations)),
        METADATA_SCHEMA, _settings_values(settings),
    )
    # Invalidate an earlier success before overwriting any image files.
    write_dataset_summary(settings.output_dir, summary)
    metadata_rows, failures, warning_counts = _process_observations(observations, settings)
    finish_dataset_summary(settings.output_dir, summary, metadata_rows, failures)
    _print_preprocessing_report(observations, summary, failures, warning_counts)
    if summary["unexpected_sample_files"]:
        raise RuntimeError(
            f"Preprocessing output contains stale image files: {settings.output_dir}. "
            "See unexpected_sample_files in dataset_summary.json; use a clean output directory."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", default=None, help="FITS directory; defaults to MOJAVE_FITS_DIR in .env.",
    )
    parser.add_argument(
        "--uv_dir", default=None, help="UV directory; defaults to MOJAVE_UV_DIR in .env.",
    )
    parser.add_argument("--output_dir", default=None, help="Defaults to the size-specific data directory.")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--cpu_cores", type=int, default=None, help="Defaults to MOJAVE_PREPROCESS_CPUS.")
    args = parser.parse_args()
    if (args.image_size is not None and args.image_size < 1) or (
        args.cpu_cores is not None and args.cpu_cores < 1
    ):
        parser.error("--image_size and --cpu_cores must be positive.")
    overrides = {name: value for name, value in vars(args).items() if value is not None}
    preprocess_data(replace(PREPROCESSING_SETTINGS, **overrides))


if __name__ == "__main__":
    main()
