"""Sharded dataset and Lightning data module for 3D2Shift."""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ShardedShiftDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        random_conformer: bool = False,
        conformer_index: int = 0,
    ) -> None:
        self.directory = Path(root) / split
        manifest = json.loads(
            (self.directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.shards = list(manifest["shards"])
        self.ends: List[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.ends.append(total)
        self.random_conformer = random_conformer
        self.conformer_index = conformer_index
        self._cached_shard = -1
        self._cached_samples: Sequence[Mapping[str, Any]] = []

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        shard_index = bisect.bisect_right(self.ends, index)
        if shard_index >= len(self.shards):
            raise IndexError(index)
        offset = 0 if shard_index == 0 else self.ends[shard_index - 1]
        if shard_index != self._cached_shard:
            self._cached_samples = _load_torch(
                self.directory / self.shards[shard_index]["path"]
            )
            self._cached_shard = shard_index
        source = self._cached_samples[index - offset]
        conformers = source["positions"]
        if self.random_conformer and conformers.size(0) > 1:
            conformer_index = int(
                torch.randint(conformers.size(0), size=()).item()
            )
        else:
            conformer_index = min(self.conformer_index, conformers.size(0) - 1)
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in source.items()
            if key != "positions"
        } | {"positions": conformers[conformer_index].clone()}


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
        train_random_conformer: bool = True,
        eval_conformer_index: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.datasets: Dict[str, ShardedShiftDataset] = {}

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.datasets["train"] = ShardedShiftDataset(
                self.hparams.data_dir,
                "train",
                random_conformer=self.hparams.train_random_conformer,
            )
            self.datasets["val"] = ShardedShiftDataset(
                self.hparams.data_dir,
                "val",
                conformer_index=self.hparams.eval_conformer_index,
            )
        if stage in (None, "test", "predict"):
            self.datasets["test"] = ShardedShiftDataset(
                self.hparams.data_dir,
                "test",
                conformer_index=self.hparams.eval_conformer_index,
            )

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        workers = int(self.hparams.num_workers)
        return DataLoader(
            self.datasets[split],
            batch_size=int(self.hparams.batch_size),
            shuffle=shuffle,
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
