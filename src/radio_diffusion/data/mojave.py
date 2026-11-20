"""Mojave images with optional metadata for conditioning."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import re

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch.utils.data import DataLoader, Dataset, random_split

from radio_diffusion.load_config import NormalizationSettings
from radio_diffusion.paths import get_mojave_data_dir
from radio_diffusion.normalization import ImageNormalizer
from radio_diffusion.data.preprocess_mojave import (
    PREPROCESSING_SETTINGS,
    preprocess_data,
)


_METADATA_DEFAULTS = {
    "source_name": "",
    "image_size": (0, 0),
    "noise_rms_jy_per_beam": 0.0,
    "noise_method": "",
    "peak_brightness_jy_per_beam": 0.0,
    "jet_pa_deg": 0.0,
    "jet_radial_extent_mas": 0.0,
    "ridgeline_stop_reason": "",
    "ridgeline_arc_clipped": False,
    "cropped_image_pixel_maximum": 0.0,
    "cropped_image_pixel_minimum": 0.0,
    "cropped_image_pixel_mean": 0.0,
    "cropped_image_pixel_std": 0.0,
}


def _validate_metadata_fields(metadata_fields: list[str] | None) -> list[str]:
    if metadata_fields is None:
        return []
    if not isinstance(metadata_fields, list) or any(
        not isinstance(field, str) or not field.strip() for field in metadata_fields
    ):
        raise ValueError("metadata_fields must be a list of nonempty field names.")
    if len(metadata_fields) != len(set(metadata_fields)):
        raise ValueError("metadata_fields must contain unique field names.")
    unknown = set(metadata_fields) - _METADATA_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown Mojave metadata fields: {', '.join(sorted(unknown))}")
    return list(metadata_fields)


def _validate_data_dimensions(variant, expected_image_shape, image_size):
    """Validate the dataset label and resolve the expected crop dimensions."""
    if not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", variant):
        raise ValueError(
            "data.variant must start with a letter or digit and contain only "
            "letters, digits, underscores, or hyphens."
        )
    if expected_image_shape is not None:
        if len(expected_image_shape) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in expected_image_shape
        ):
            raise ValueError(
                "expected_image_shape must be [channels, height, width] "
                "with positive integer dimensions."
            )
        expected_image_shape = tuple(expected_image_shape)
    if image_size is None:
        image_size = expected_image_shape[1] if expected_image_shape else 256
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer.")
    return expected_image_shape, image_size


class MojaveDataset(Dataset):
    """Return images alone, or image/metadata/metadata_mask dictionaries.

    Numeric and boolean metadata is returned as tensors; categories remain
    strings. True mask entries indicate present, finite values. Missing numbers
    and image dimensions use zero, booleans use False, and strings use "".
    """

    def __init__(
        self, paths, expected_image_shape=None, metadata_fields=None,
        normalization: NormalizationSettings | None = None,
        dataset_metadata: dict | None = None,
    ):
        self.paths = paths
        self.expected_image_shape = expected_image_shape
        self.metadata_fields = _validate_metadata_fields(metadata_fields)
        self.dataset_metadata = dataset_metadata if dataset_metadata is not None else {}
        settings = normalization if normalization is not None else NormalizationSettings()
        self.normalizer = ImageNormalizer(settings)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sample = torch.load(path, map_location="cpu", weights_only=True)
        image = self._load_image(sample, path)
        metadata = self._load_metadata(sample, path)
        image = self.normalizer.normalize(image, metadata)
        if not self.metadata_fields:
            return image
        selected, mask = self._select_metadata(metadata, path)
        return {"image": image, "metadata": selected, "metadata_mask": mask}

    def _load_image(self, sample, path):
        """Validate a saved crop before adding its channel dimension."""
        if not isinstance(sample, dict) or "image" not in sample:
            raise ValueError(f"Expected a saved dictionary containing 'image': {path}")
        image = sample["image"]
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 2
            or not image.is_floating_point()
            or any(size < 1 for size in image.shape)
        ):
            raise ValueError(
                f"Expected a nonempty floating-point image shaped [height, width]: {path}"
            )
        image = image.unsqueeze(0)
        if (
            self.expected_image_shape is not None
            and tuple(image.shape) != self.expected_image_shape
        ):
            raise ValueError(
                f"Image {path} has shape {tuple(image.shape)}; expected "
                f"{self.expected_image_shape} [channels, height, width]. "
                "Check the dataset and samples.shape."
            )
        return image

    def _load_metadata(self, sample, path):
        metadata = sample.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a metadata dictionary: {path}")
        return {**metadata, **self.dataset_metadata}

    def _select_metadata(self, metadata, path):
        """Keep physical metadata values and their original validity masks."""
        selected = {}
        mask = {}
        for field in self.metadata_fields:
            value, valid = self._metadata_value(field, metadata.get(field), path)
            selected[field] = value
            mask[field] = valid
        return selected, mask

    @staticmethod
    def _metadata_value(field, value, path):
        """Validate one selected metadata field and fill missing values."""
        default = _METADATA_DEFAULTS[field]
        if isinstance(default, str):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Expected string metadata for {field!r}: {path}")
            valid = value is not None and bool(value.strip())
            return value if valid else default, valid
        valid = value is not None
        if valid:
            value = torch.as_tensor(value)
            expected_shape = (len(default),) if isinstance(default, tuple) else ()
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"Metadata {field!r} has shape {tuple(value.shape)}; "
                    f"expected {expected_shape}: {path}"
                )
            valid = bool(torch.isfinite(value).all())
        dtype = (
            torch.bool if isinstance(default, bool)
            else torch.float32 if isinstance(default, float) else torch.long
        )
        return torch.as_tensor(value if valid else default, dtype=dtype), valid


class MojaveDataModule(LightningDataModule):
    """Mojave data module with optional metadata for conditioning.

    Returns images with shape [B, C, H, W] when metadata_fields is empty.
    Otherwise returns image, metadata and metadata_mask dictionaries. Missing
    values use the placeholders documented in MojaveDataset. During setup,
    metadata_missing_counts records substitutions across the entire dataset
    and rank zero emits one summary warning if any were needed. This requires
    reading every file when metadata is requested.

    Args: 
        data_dir: Optional directory containing .icn.pt files. Defaults to the
            size-specific directory under MOJAVE_DATA_ROOT from .env.
        remote_source: Ignored legacy argument. Raw data syncing is manual.
        batch_size: Number of images per batch.
        num_workers: Number of workers for data loading.
        val_fraction: Fraction of data used for validation.
        test_fraction: Fraction of data used for testing.
        seed: Random seed for reproducible splitting.
        variant: Legacy preprocessing label retained as tracking metadata.
        expected_image_shape: Optional [channels, height, width] from samples.shape.
        image_size: Preprocessing image size; defaults to the expected shape's
            height, or 256 when no expected shape is supplied.
        metadata_fields: Optional ordered selection of saved metadata field names.
        normalization: Optional reversible image transform and range settings.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        remote_source: str | None = None,
        variant: str = "mojave_unspecified",
        expected_image_shape: tuple[int, int, int] | None = None,
        image_size: int | None = None,
        metadata_fields: list[str] | None = None,
        normalization: NormalizationSettings | None = None,
    ) -> None:
        super().__init__()
        expected_image_shape, image_size = _validate_data_dimensions(
            variant, expected_image_shape, image_size,
        )
        self.image_size = image_size
        self.data_dir = (
            get_mojave_data_dir(image_size)
            if data_dir is None else Path(data_dir).expanduser()
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.variant = variant
        self.expected_image_shape = expected_image_shape
        self.metadata_fields = _validate_metadata_fields(metadata_fields)
        self.normalization = normalization if normalization is not None else NormalizationSettings()
        self.dataset_metadata: dict = {}
        self.normalizer: ImageNormalizer | None = None
        self.metadata_missing_counts: dict[str, int] = {}
        self._complete_conditioning_rows: dict[Path, dict] = {}
        self.image_shape: tuple[int, int, int] | None = None

    @property
    def experiment_name(self) -> str:
        """Group MOJAVE runs by actual image dimensions, after setup()."""
        if self.image_shape is None:
            raise RuntimeError("Call data.setup() before reading experiment_name.")
        _, height, width = self.image_shape
        return f"mojave_{height}x{width}"

    def prepare_data(self) -> None:
        """Delegate dataset reuse and raw-data preprocessing to preprocess_data."""
        if self.expected_image_shape is not None and self.expected_image_shape != (
            1, self.image_size, self.image_size,
        ):
            raise ValueError(
                "MOJAVE preprocessing produces [1, image_size, image_size] images; "
                "expected_image_shape must match."
            )

        preprocess_data(replace(
            PREPROCESSING_SETTINGS, image_size=self.image_size, output_dir=self.data_dir,
        ))

    @staticmethod
    def compute_stats(paths):
        total = total_sq = count = 0

        dataset = MojaveDataset(paths)
        for index in range(len(dataset)):
            image = dataset[index]
            total += image.sum().item()
            total_sq += image.square().sum().item()
            count += image.numel()

        mean = total / count
        std = max(total_sq / count - mean**2, 0) ** 0.5

        return mean, std

    def setup(self, stage: str | None = None) -> None:
        """Resolve normalization once and apply it consistently to every split."""
        self.image_shape = None
        self.metadata_missing_counts = {}
        self._complete_conditioning_rows = {}
        paths = self._dataset_paths()
        if self.normalizer is None:
            self.dataset_metadata = self._read_dataset_metadata()
            self.normalizer = ImageNormalizer(self.normalization)
        dataset = MojaveDataset(paths, self.expected_image_shape, self.metadata_fields)
        first_sample = dataset[0]
        first_image = first_sample["image"] if self.metadata_fields else first_sample
        self.image_shape = tuple(first_image.shape)
        print(f"Dataset group: {self.experiment_name}")
        self._scan_metadata(dataset, paths, first_sample)
        self._build_split_datasets(paths, stage)

    def _read_dataset_metadata(self) -> dict:
        """Read shared cropped-pixel extrema once from the saved summary."""
        if not self.normalization.enabled or self.normalization.bounds != "shared":
            return {}
        path = self.data_dir / "dataset_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["preprocessing_finished"] is not True:
            raise ValueError(f"Normalization requires finished preprocessing: {path}")
        pixels = summary["cropped_image_pixels"]
        if not isinstance(pixels, dict):
            raise ValueError(f"Normalization requires finite dataset pixel statistics: {path}")
        return {
            "dataset_pixel_minimum": pixels["minimum"],
            "dataset_pixel_maximum": pixels["maximum"],
        }

    def _dataset_paths(self):
        """Find saved crops and reject missing or empty dataset directories."""
        print(f"Mojave input directory: {self.data_dir}")
        if not self.data_dir.exists():
            print(f"Mojave input directory does not exist: {self.data_dir}")
            raise FileNotFoundError(
                f"Mojave input directory does not exist: {self.data_dir}"
            )
        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Mojave input path is not a directory: {self.data_dir}"
            )

        paths = sorted(self.data_dir.glob("*.icn.pt"))
        if not paths:
            raise FileNotFoundError(
                f"No .icn.pt files found in Mojave input directory: {self.data_dir}"
            )
        return paths

    def _scan_metadata(self, dataset, paths, first_sample) -> None:
        """Collect raw conditioning rows and report substituted metadata."""
        if not self.metadata_fields:
            return
        self.metadata_missing_counts = {field: 0 for field in self.metadata_fields}
        for index in range(len(dataset)):
            sample = first_sample if index == 0 else dataset[index]
            for field, valid in sample["metadata_mask"].items():
                self.metadata_missing_counts[field] += int(not valid)
            if all(sample["metadata_mask"].values()):
                self._complete_conditioning_rows[paths[index]] = sample["metadata"]
        substitutions = [
            f"{field}: {count}/{len(dataset)} replaced with {_METADATA_DEFAULTS[field]!r}"
            for field, count in self.metadata_missing_counts.items() if count
        ]
        if substitutions:
            rank_zero_warn(
                "Mojave metadata substitutions: " + "; ".join(substitutions)
                + ". Substituted values have metadata_mask=False."
            )

    def _build_split_datasets(self, paths, stage) -> None:
        """Shares one normalization mapping across train, validation and test."""
        splits = random_split(
            paths,
            [
                1 - self.val_fraction - self.test_fraction,
                self.val_fraction,
                self.test_fraction,
            ],
            generator=torch.Generator().manual_seed(self.seed),
        )
        stages = (
            ("train", (None, "fit")),
            ("val", (None, "fit", "validate")),
            ("test", (None, "test")),
        )
        for split_paths, (name, supported_stages) in zip(splits, stages):
            if stage in supported_stages:
                dataset = MojaveDataset(
                    split_paths, self.image_shape, self.metadata_fields,
                    normalization=self.normalization,
                    dataset_metadata=self.dataset_metadata,
                )
                setattr(self, f"{name}_dataset", dataset)

    def state_dict(self) -> dict:
        """Include fitted normalization in Lightning checkpoints."""
        if self.normalizer is None:
            return {}
        return {"normalization": {
            "settings": asdict(self.normalization),
            "dataset_metadata": self.dataset_metadata,
        }}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the original mapping instead of recomputing its shared scale."""
        if "normalization" not in state_dict:
            return
        state = state_dict["normalization"]
        settings = NormalizationSettings(**state["settings"])
        if settings != self.normalization:
            raise ValueError("Checkpoint normalization settings do not match data.normalization.")
        self.normalization = settings
        self.dataset_metadata = state["dataset_metadata"]
        self.normalizer = ImageNormalizer(settings)
        for name in ("train_dataset", "val_dataset", "test_dataset"):
            if hasattr(self, name):
                dataset = getattr(self, name)
                dataset.normalizer = ImageNormalizer(settings)
                dataset.dataset_metadata = self.dataset_metadata

    def sample_conditioning(self, count: int, seed: int) -> dict:
        """Draw complete metadata rows from the training split with a local RNG.

        Fields are sampled together to preserve their observed relationships.
        Draws use replacement only when fewer than count observations qualify.
        Validation/test observations and substituted values are excluded.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Conditioning sample count must be a positive integer.")
        if not self.metadata_fields:
            raise ValueError("Automatic conditioning requires selected metadata fields.")
        if not hasattr(self, "train_dataset"):
            raise RuntimeError("Call data.setup('fit') before sampling conditioning.")
        eligible = [
            path for path in self.train_dataset.paths
            if path in self._complete_conditioning_rows
        ]
        if not eligible:
            raise ValueError(
                "No training observations have valid values for every selected "
                "conditioning field. Supply manual sample conditioning or "
                "change the selected fields."
            )
        generator = torch.Generator().manual_seed(seed)
        replacement = len(eligible) < count
        indices = (
            torch.randint(len(eligible), (count,), generator=generator)
            if replacement else torch.randperm(len(eligible), generator=generator)[:count]
        )
        paths = [eligible[index] for index in indices.tolist()]
        metadata = {}
        for field in self.metadata_fields:
            values = [self._complete_conditioning_rows[path][field] for path in paths]
            if any(not isinstance(value, torch.Tensor) or value.ndim != 0 for value in values):
                raise ValueError(
                    f"Automatic conditioning field {field!r} must contain "
                    "numerical scalars or booleans."
                )
            metadata[field] = torch.stack(values)
        return {
            "metadata": metadata,
            "metadata_mask": {
                field: torch.ones(count, dtype=torch.bool) for field in self.metadata_fields
            },
            "source_observations": [str(path) for path in paths],
            "eligible_observations": len(eligible),
            "with_replacement": replacement,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
