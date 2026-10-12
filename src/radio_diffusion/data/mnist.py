"""MNIST images for unconditional generation."""

from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class _MNISTWithoutLabels(Dataset):
    """Keep digit labels out of the generative model's batch format."""

    def __init__(self, dataset: MNIST) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[index]
        return image


class MNISTDataModule(LightningDataModule):
    """MNIST data module for unconditional image generation.

    Returns images normalized to [-1, 1] with shape [B, 1, 28, 28].

    Args:
        data_dir: Directory containing the MNIST dataset (or where it should be downloaded).
        batch_size: Number of images per batch.
        num_workers: Number of workers for data loading.
        val_fraction: Fraction of training data used for validation.
        seed: Random seed for reproducible splitting.
        download: Whether to download MNIST if missing.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 0,
        val_fraction: float = 0.1,
        seed: int = 42,
        download: bool = False,
    ) -> None:
        super().__init__()
        if batch_size < 1 or num_workers < 0 or val_fraction <= 0 or val_fraction >= 1:
            raise ValueError("batch_size must be positive; num_workers must be >= 0; val_fraction must be between 0 and 1.")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.download = download
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def _load_mnist(self, train: bool, download: bool = False) -> MNIST:
        try:
            return MNIST(
                root=str(self.data_dir), train=train,
                transform=self.transform, download=download,
            )
        except RuntimeError as exc:
            # Do not mistake unrelated loading or download failures for missing data.
            if "Dataset not found" not in str(exc):
                raise
            raise FileNotFoundError(
                f"MNIST files are missing or incomplete in {self.data_dir / 'MNIST'}. "
                "Choose the correct data_dir or explicitly enable downloading."
            ) from exc

    def prepare_data(self) -> None:
        """Download MNIST files if they are not already present in the data_dir."""
        self._load_mnist(train=True, download=self.download)
        self._load_mnist(train=False, download=self.download)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            dataset = _MNISTWithoutLabels(self._load_mnist(train=True))

            val_size = int(len(dataset) * self.val_fraction)

            if not 0 < val_size < len(dataset):
                raise ValueError("val_fraction must produce a non-empty train and validation split.")

            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [len(dataset) - val_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage in (None, "test"):
            self.test_dataset = _MNISTWithoutLabels(self._load_mnist(train=False))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
