"""Mojave images for unconditional generation."""

import shutil
import subprocess
from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class MojaveDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx], map_location="cpu").unsqueeze(0)


class MojaveDataModule(LightningDataModule):
    """Mojave data module for unconditional image generation.

    Returns images with shape [B, C, W, H]

    Args: 
        data_dir: Directory as Path or String containing the .icn.pt files created with preprocess_mojave.py.
        remote_source: Optional SSH rsync source, for example user@host:/path/to/mojave.
        batch_size: Number of images per batch.
        num_workers: Number of workers for data loading.
        val_fraction: Fraction of data used for validation.
        test_fraction: Fraction of data used for testing.
        seed: Random seed for reproducible splitting.
        max_samples: Optional cap on images before train/validation/test splitting.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 16,
        num_workers: int = 0,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        remote_source: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        if max_samples is not None and (
            not isinstance(max_samples, int)
            or isinstance(max_samples, bool)
            or max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer or None.")

        self.remote_source = remote_source
        configured_data_dir = Path(data_dir).expanduser()
        mounted_remote_dir = self._local_path_for_remote_source(remote_source)
        if mounted_remote_dir is not None and mounted_remote_dir.is_dir():
            self.data_dir = mounted_remote_dir
        else:
            self.data_dir = configured_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.max_samples = max_samples

    @staticmethod
    def _local_path_for_remote_source(
        remote_source: str | None,
    ) -> Path | None:
        """Get the absolute path portion of an rsync ``host:/path`` source."""
        if remote_source is None:
            return None
        _, separator, remote_path = remote_source.partition(":")
        if not separator or not remote_path.startswith("/"):
            return None
        return Path(remote_path)

    def prepare_data(self) -> None:
        """Optionally sync the dataset from a remote SSH source with rsync."""
        if self.remote_source is None:
            print("No remote_source configured; skipping Mojave data sync.")
            return
        if self.data_dir.exists():
            print(
                f"Using existing Mojave data directory {self.data_dir}; "
                "skipping remote sync."
            )
            return

        if shutil.which("rsync") is None:
            raise FileNotFoundError(
                "rsync is required to sync remote Mojave data.")

        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = self.data_dir.with_name(
            f".{self.data_dir.name}.rsync-partial")
        source = self.remote_source.rstrip("/") + "/"
        print(
            f"Syncing Mojave data from {source} to {self.data_dir}...", flush=True)
        subprocess.run(
            ["rsync", "-a", "--partial", "--info=progress2",
                "--", source, f"{staging_dir}/"],
            check=True,
        )
        staging_dir.rename(self.data_dir)
        print("Mojave data sync complete.")

    @staticmethod
    def compute_stats(paths):
        total = total_sq = count = 0

        for path in paths:
            image = torch.load(path, map_location="cpu")
            total += image.sum().item()
            total_sq += image.square().sum().item()
            count += image.numel()

        mean = total / count
        std = max(total_sq / count - mean**2, 0) ** 0.5

        return mean, std

    def setup(self, stage: str | None = None) -> None:
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
        if self.max_samples is not None and self.max_samples < len(paths):
            subset_indices = torch.randperm(
                len(paths), generator=generator
            )[:self.max_samples].tolist()
            paths = [paths[index] for index in subset_indices]
            print(
                f"Using a reproducible subset of {self.max_samples} images "
                f"from {self.data_dir}."
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
            self.train_dataset = MojaveDataset(train)

        if stage in (None, "fit", "validate"):
            self.val_dataset = MojaveDataset(val)

        if stage in (None, "test"):
            self.test_dataset = MojaveDataset(test)

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
