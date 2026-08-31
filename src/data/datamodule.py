from typing import Any, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import NMRGraphDataset, TransformingCollator
import time


class IndexedSubset(Dataset):
    """Subset that exposes a stable position for DDP metric/table ordering."""

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        sample = self.dataset[self.indices[position]]
        # GraphSample is a regular dataclass, so this validation-only metadata
        # does not alter serialized training examples or the model input path.
        sample.validation_index = int(position)
        return sample


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
            batch_transform: Optional[Any] = None,
            val_generation_size: int = 1024,
            val_generation_seed: int = 0,
            inference_only_validation: bool = False,
    ):
        super().__init__()
        if transform is not None and batch_transform is not None:
            raise ValueError("Specify batch_transform; transform is a legacy alias")
        self.save_hyperparameters(ignore=["transform", "batch_transform"])
        self.batch_transform = (
            batch_transform if batch_transform is not None else transform
        )
        self.collator = TransformingCollator(self.batch_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in [None, "fit"]:
            print("Start loading dataset...")
            start_time = time.time()
            
            self.val_dataset = NMRGraphDataset(
                self.hparams.val_path
            )
            print(f"Done loading val dataset: {len(self.val_dataset)} Time taken: {time.time() - start_time:.2f}s")

            start_time = time.time()
            generation_size = min(
                self.hparams.val_generation_size, len(self.val_dataset)
            )
            generator = torch.Generator().manual_seed(
                self.hparams.val_generation_seed
            )
            generation_indices = torch.randperm(
                len(self.val_dataset), generator=generator
            )[:generation_size].tolist()
            self.val_generation_dataset = IndexedSubset(
                self.val_dataset, generation_indices
            )
            print(f"Done loading val generation dataset: {len(self.val_generation_dataset)} Time taken: {time.time() - start_time:.2f}s")

            start_time = time.time()
            self.train_dataset = NMRGraphDataset(
                self.hparams.train_path
            )
            print(f"Done loading train dataset: {len(self.train_dataset)} Time taken: {time.time() - start_time:.2f}s")
        if stage in [None, "test"]:
            self.test_dataset = (
                NMRGraphDataset(self.hparams.test_path)
                if self.hparams.test_path
                else None
            )
            if self.test_dataset is not None:
                print("Done loading test dataset: ", len(self.test_dataset))

    def _loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.hparams.train_batch_size, True)

    def val_dataloader(self):
        full = self._loader(self.val_dataset, self.hparams.val_batch_size, False)
        if len(self.val_generation_dataset) == 0:
            if self.hparams.inference_only_validation:
                raise ValueError(
                    "inference_only_validation requires "
                    "val_generation_size > 0"
                )
            return full
        generation = self._loader(
            self.val_generation_dataset, self.hparams.val_batch_size, False
        )
        return [full, generation]

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return self._loader(self.test_dataset, self.hparams.val_batch_size, False)
