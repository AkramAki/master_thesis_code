"""MNIST images for unconditional generation."""

from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class _ImagesOnly(Dataset):
    """Keep digit labels out of the generative model's batch format."""

    def __init__(self, dataset: MNIST) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[index]
        return image


class MNISTDataModule(LightningDataModule):
    """Return float32 batches of shape [B, 1, 28, 28], scaled to [-1, 1].

    Downloads are disabled by default. Only ``prepare_data`` may download;
    ``setup`` loads local files and creates a seeded train/validation split.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 0,
        val_size: int = 5000,
        seed: int = 42,
        download: bool = False,
    ) -> None:
        super().__init__()
        if batch_size < 1 or num_workers < 0 or val_size < 1:
            raise ValueError("batch_size and val_size must be positive; num_workers must be >= 0.")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.seed = seed
        self.download = download
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def _dataset(self, train: bool, download: bool = False) -> MNIST:
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
        """Check both official splits, downloading only when authorized."""
        self._dataset(train=True, download=self.download)
        self._dataset(train=False, download=self.download)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            dataset = _ImagesOnly(self._dataset(train=True))
            if self.val_size >= len(dataset):
                raise ValueError("val_size must be smaller than the training dataset.")
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [len(dataset) - self.val_size, self.val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        if stage in (None, "test"):
            self.test_dataset = _ImagesOnly(self._dataset(train=False))

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
