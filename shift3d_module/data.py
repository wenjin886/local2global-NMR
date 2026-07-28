"""In-memory split dataset and Lightning data module for 3D2Shift."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


IN_MEMORY_DATASET_VERSION = 4


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ShiftDataset(Dataset):
    """Load one complete, offline-expanded split into CPU memory once."""

    def __init__(
        self,
        root: str | Path,
        split: str,
    ) -> None:
        self.path = Path(root) / f"{split}.pt"
        size_gib = self.path.stat().st_size / (1024**3)
        started = time.perf_counter()
        print(
            f"Loading complete {split} split from {self.path} "
            f"({size_gib:.2f} GiB) into CPU memory..."
        )
        payload = _load_torch(self.path)
        elapsed = time.perf_counter() - started
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"{self.path} is not a versioned 3D2Shift split. Rebuild it."
            )
        version = int(payload.get("version", 0))
        if version < IN_MEMORY_DATASET_VERSION:
            raise RuntimeError(
                f"{self.path} uses 3D2Shift dataset version {version}; re-run "
                "`python -m preprocess.uspto_3d_nmr build` to create one "
                "in-memory PT file per split."
            )
        if payload.get("split") != split:
            raise RuntimeError(
                f"{self.path} contains split {payload.get('split')!r}, "
                f"expected {split!r}"
            )
        self.samples: Sequence[Mapping[str, Any]] = payload["samples"]
        print(
            f"Loaded {len(self.samples):,} {split} conformer samples "
            f"in {elapsed:.1f}s; no further dataset disk reads are required."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        source = self.samples[index]
        positions = source["positions"]
        if positions.ndim != 2 or positions.size(-1) != 3:
            raise RuntimeError(
                f"Sample {source.get('id', index)!r} contains positions with "
                f"shape {tuple(positions.shape)}; expected an offline "
                "conformer with shape [N, 3]. Rebuild the dataset."
            )
        # Collation copies tensors into a new batch, so cloning every tensor
        # here only adds CPU work and memory traffic.
        return source


def collate_shift_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    batch_size = len(samples)
    max_atoms = max(int(sample["atomic_numbers"].numel()) for sample in samples)
    max_h = max(int(sample["h_peak_shifts"].numel()) for sample in samples)
    max_c = max(int(sample["c_peak_shifts"].numel()) for sample in samples)
    atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    positions = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    environment_ids = torch.full((batch_size, max_atoms), -1, dtype=torch.long)
    h_peak_shifts = torch.zeros((batch_size, max_h), dtype=torch.float32)
    h_peak_mask = torch.zeros((batch_size, max_h), dtype=torch.bool)
    h_peak_integrations = torch.zeros((batch_size, max_h), dtype=torch.float32)
    h_peak_integration_mask = torch.zeros(
        (batch_size, max_h), dtype=torch.bool
    )
    c_peak_shifts = torch.zeros((batch_size, max_c), dtype=torch.float32)
    c_peak_mask = torch.zeros((batch_size, max_c), dtype=torch.bool)
    for row, sample in enumerate(samples):
        atoms = int(sample["atomic_numbers"].numel())
        h_count = int(sample["h_peak_shifts"].numel())
        c_count = int(sample["c_peak_shifts"].numel())
        atomic_numbers[row, :atoms] = sample["atomic_numbers"]
        positions[row, :atoms] = sample["positions"]
        atom_mask[row, :atoms] = True
        environment_ids[row, :atoms] = sample["environment_ids"]
        h_peak_shifts[row, :h_count] = sample["h_peak_shifts"]
        h_peak_mask[row, :h_count] = True
        h_peak_integrations[row, :h_count] = sample["h_peak_integrations"]
        h_peak_integration_mask[row, :h_count] = sample[
            "h_peak_integration_mask"
        ]
        c_peak_shifts[row, :c_count] = sample["c_peak_shifts"]
        c_peak_mask[row, :c_count] = True
    return {
        "id": [sample["id"] for sample in samples],
        "smiles": [sample["smiles"] for sample in samples],
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "atom_mask": atom_mask,
        "environment_ids": environment_ids,
        "h_peak_shifts": h_peak_shifts,
        "h_peak_mask": h_peak_mask,
        "h_peak_integrations": h_peak_integrations,
        "h_peak_integration_mask": h_peak_integration_mask,
        "c_peak_shifts": c_peak_shifts,
        "c_peak_mask": c_peak_mask,
    }


class Shift3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle_seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.datasets: Dict[str, ShiftDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.datasets["train"] = ShiftDataset(
                self.hparams.data_dir,
                "train",
            )
            self.datasets["val"] = ShiftDataset(
                self.hparams.data_dir,
                "val",
            )
        if stage in (None, "test", "predict"):
            self.datasets["test"] = ShiftDataset(
                self.hparams.data_dir,
                "test",
            )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.hparams.num_workers)
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(self.hparams.shuffle_seed))
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.hparams.batch_size),
            shuffle=shuffle,
            generator=generator,
            num_workers=workers,
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=(
                bool(self.hparams.persistent_workers) and workers > 0
            ),
            collate_fn=collate_shift_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
