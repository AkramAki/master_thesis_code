"""Mojave images with optional metadata for conditioning."""

from pathlib import Path
import re

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from torch.utils.data import DataLoader, Dataset, random_split

from radio_diffusion.paths import (
    REPOSITORY_ROOT,
    get_mojave_data_dir,
    get_path,
    get_preprocess_cpu_cores,
)
from radio_diffusion.data.preprocess_mojave import preprocess_data


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


class MojaveDataset(Dataset):
    """Return images alone, or image/metadata/metadata_mask dictionaries.

    Numeric and boolean metadata is returned as tensors; categories remain
    strings. True mask entries indicate present, finite values. Missing numbers
    and image dimensions use zero, booleans use False, and strings use "".
    """

    def __init__(self, paths, expected_image_shape=None, metadata_fields=None):
        self.paths = paths
        self.expected_image_shape = expected_image_shape
        self.metadata_fields = _validate_metadata_fields(metadata_fields)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        sample = torch.load(path, map_location="cpu", weights_only=True)
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
        if not self.metadata_fields:
            return image

        metadata = sample.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected a metadata dictionary: {path}")
        selected = {}
        mask = {}
        for field in self.metadata_fields:
            default = _METADATA_DEFAULTS[field]
            value = metadata.get(field)
            if isinstance(default, str):
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"Expected string metadata for {field!r}: {path}")
                valid = value is not None and bool(value.strip())
                value = value if valid else default
            else:
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
                    else torch.float32 if isinstance(default, float)
                    else torch.long
                )
                value = torch.as_tensor(value if valid else default, dtype=dtype)
            selected[field] = value
            mask[field] = valid

        return {"image": image, "metadata": selected, "metadata_mask": mask}


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
    ) -> None:
        super().__init__()
        if not isinstance(variant, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", variant
        ):
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
        """Reuse processed images or preprocess local raw data, without syncing."""
        if self.data_dir.exists() and not self.data_dir.is_dir():
            raise NotADirectoryError(f"Mojave data path is not a directory: {self.data_dir}")
        if any(path.is_file() for path in self.data_dir.glob("*.icn.pt")):
            print(
                f"Using existing preprocessed MOJAVE data: {self.data_dir}",
                flush=True,
            )
            return

        print(
            f"Preprocessed data not available at {self.data_dir}; checking raw data.",
            flush=True,
        )
        image_dir = get_path("MOJAVE_FITS_DIR")
        uv_dir = get_path("MOJAVE_UV_DIR")
        sync_hint = (
            f"Run `python {REPOSITORY_ROOT / 'scripts/sync_examples.py'}` "
            "to sync raw FITS/UV examples, then retry."
        )
        print(f"Raw FITS directory: {image_dir}\nRaw UV directory: {uv_dir}", flush=True)
        for label, directory in (("FITS", image_dir), ("UV", uv_dir)):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Raw {label} directory not available: {directory}. {sync_hint}"
                )

        image_paths = sorted(path for path in image_dir.glob("*.icn.fits.gz") if path.is_file())
        if not image_paths:
            raise FileNotFoundError(f"No raw .icn.fits.gz files in {image_dir}. {sync_hint}")
        missing_uv = [
            image.name.removesuffix(".icn.fits.gz") + ".uvf"
            for image in image_paths
            if not (uv_dir / (image.name.removesuffix(".icn.fits.gz") + ".uvf")).is_file()
        ]
        if missing_uv:
            raise FileNotFoundError(
                f"Missing {len(missing_uv)} matching UV files in {uv_dir} "
                f"(first: {missing_uv[0]}). {sync_hint}"
            )
        if self.expected_image_shape is not None and self.expected_image_shape != (
            1, self.image_size, self.image_size,
        ):
            raise ValueError(
                "MOJAVE preprocessing produces [1, image_size, image_size] images; "
                "expected_image_shape must match."
            )

        cpu_cores = get_preprocess_cpu_cores()
        print(
            f"Raw data available ({len(image_paths)} FITS/UV pairs); running preprocessing.\n"
            f"Image size: {self.image_size}; CPU cores: {cpu_cores}\n"
            f"Output directory: {self.data_dir}",
            flush=True,
        )
        preprocess_data(
            input_dir=str(image_dir),
            uv_dir=str(uv_dir),
            img_size=self.image_size,
            cpu_cores=cpu_cores,
            output_dir=str(self.data_dir),
        )
        if not any(path.is_file() for path in self.data_dir.glob("*.icn.pt")):
            raise RuntimeError(f"MOJAVE preprocessing produced no .icn.pt files in {self.data_dir}")
        print(f"MOJAVE preprocessing complete: {self.data_dir}", flush=True)

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
        self.image_shape = None
        self.metadata_missing_counts = {}
        self._complete_conditioning_rows = {}
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

        generator = torch.Generator().manual_seed(self.seed)
        dataset = MojaveDataset(
            paths, self.expected_image_shape, self.metadata_fields
        )
        first_sample = dataset[0]
        first_image = first_sample["image"] if self.metadata_fields else first_sample
        self.image_shape = tuple(first_image.shape)
        print(f"Dataset group: {self.experiment_name}")

        if self.metadata_fields:
            self.metadata_missing_counts = {field: 0 for field in self.metadata_fields}
            for index in range(len(dataset)):
                sample = first_sample if index == 0 else dataset[index]
                for field, valid in sample["metadata_mask"].items():
                    self.metadata_missing_counts[field] += int(not valid)
                if all(sample["metadata_mask"].values()):
                    self._complete_conditioning_rows[paths[index]] = sample["metadata"]
            substitutions = [
                f"{field}: {count}/{len(dataset)} replaced with {_METADATA_DEFAULTS[field]!r}"
                for field, count in self.metadata_missing_counts.items()
                if count
            ]
            if substitutions:
                rank_zero_warn(
                    "Mojave metadata substitutions: " + "; ".join(substitutions)
                    + ". Substituted values have metadata_mask=False."
                )

        train, val, test = random_split(
            paths,
            [
                1 - self.val_fraction - self.test_fraction,
                self.val_fraction,
                self.test_fraction,
            ],
            generator=generator,
        )

        if stage in (None, "fit"):
            self.train_dataset = MojaveDataset(
                train, self.image_shape, self.metadata_fields
            )

        if stage in (None, "fit", "validate"):
            self.val_dataset = MojaveDataset(
                val, self.image_shape, self.metadata_fields
            )

        if stage in (None, "test"):
            self.test_dataset = MojaveDataset(
                test, self.image_shape, self.metadata_fields
            )

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
