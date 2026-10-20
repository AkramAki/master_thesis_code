"""Mojave images for unconditional generation."""

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
        batch_size: Number of images per batch.
        num_workers: Number of workers for data loading.
        val_fraction: Fraction of data used for validation.
        test_fraction: Fraction of data used for testing.
        seed: Random seed for reproducible splitting.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 16,
        num_workers: int = 0,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

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
        paths = sorted(self.data_dir.glob("*.icn.pt"))
    
        train, val, test = random_split(
            paths,
            [
                1 - self.val_fraction - self.test_fraction,
                self.val_fraction,
                self.test_fraction,
            ],
            generator=torch.Generator().manual_seed(self.seed),
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
        