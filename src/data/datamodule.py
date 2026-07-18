from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import NMRGraphDataset, collate_nmr_graph


class NMRGraphDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_path: str,
            val_path: str,
            test_path: Optional[str] = None,
            train_batch_size: int = 32,
            val_batch_size: int = 64,
            num_workers: int = 4,
            pin_memory: bool = True,
            transform: Optional[Any] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NMRGraphDataset(
            self.hparams.train_path, transform=self.transform
        )
        self.val_dataset = NMRGraphDataset(
            self.hparams.val_path, transform=self.transform
        )
        self.test_dataset = (
            NMRGraphDataset(self.hparams.test_path, transform=self.transform)
            if self.hparams.test_path
            else None
        )

    def _loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_nmr_graph,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.hparams.train_batch_size, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.hparams.val_batch_size, False)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return self._loader(self.test_dataset, self.hparams.val_batch_size, False)
