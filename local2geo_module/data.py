from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence

import pytorch_lightning as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch.utils.data import DataLoader, Dataset

from .constants import (
    AROMATIC,
    DOUBLE,
    GEOMETRY_TO_INDEX,
    SINGLE,
    TRIPLE,
)


def _load_torch(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _bond_type_index(bond: rdchem.Bond) -> int:
    if bond.GetIsAromatic():
        return AROMATIC
    mapping = {
        rdchem.BondType.SINGLE: SINGLE,
        rdchem.BondType.DOUBLE: DOUBLE,
        rdchem.BondType.TRIPLE: TRIPLE,
    }
    if bond.GetBondType() not in mapping:
        raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
    return mapping[bond.GetBondType()]


def _geometry_class(atom: rdchem.Atom) -> int:
    degree = atom.GetDegree()
    hybridization = atom.GetHybridization()
    atomic_number = atom.GetAtomicNum()
    if degree <= 1:
        name = "terminal"
    elif hybridization == rdchem.HybridizationType.SP:
        name = "linear"
    elif atom.GetIsAromatic() or hybridization == rdchem.HybridizationType.SP2:
        name = "trigonal_planar"
    elif degree == 2 and atomic_number in {8, 16}:
        name = "bent"
    elif degree == 3 and atomic_number in {7, 15}:
        name = "trigonal_pyramidal"
    elif degree <= 4:
        name = "tetrahedral"
    else:
        name = "other"
    return GEOMETRY_TO_INDEX[name]


@lru_cache(maxsize=4096)
def graph_from_smiles(smiles: str) -> Dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical))
    canonical_ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    heavy = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    heavy.sort(key=lambda index: (
        molecule.GetAtomWithIdx(index).GetAtomicNum(), canonical_ranks[index]
    ))
    old_to_new = {old: new for new, old in enumerate(heavy)}
    num_atoms = len(heavy)
    atomic_numbers = torch.tensor([
        molecule.GetAtomWithIdx(index).GetAtomicNum() for index in heavy
    ], dtype=torch.long)
    formal_charges = torch.tensor([
        molecule.GetAtomWithIdx(index).GetFormalCharge() for index in heavy
    ], dtype=torch.float)
    hydrogen_counts = torch.tensor([
        sum(neighbor.GetAtomicNum() == 1
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors())
        for index in heavy
    ], dtype=torch.float)
    geometry_classes = torch.tensor([
        _geometry_class(molecule.GetAtomWithIdx(index)) for index in heavy
    ], dtype=torch.long)
    bond_types = torch.zeros((num_atoms, num_atoms), dtype=torch.long)
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in old_to_new or right not in old_to_new:
            continue
        i, j = old_to_new[left], old_to_new[right]
        bond_types[i, j] = bond_types[j, i] = _bond_type_index(bond)

    table = Chem.GetPeriodicTable()
    covalent_radii = torch.tensor([
        table.GetRcovalent(int(value)) for value in atomic_numbers
    ], dtype=torch.float)
    vdw_radii = torch.tensor([
        table.GetRvdw(int(value)) for value in atomic_numbers
    ], dtype=torch.float)
    return {
        "smiles": canonical,
        "atomic_numbers": atomic_numbers,
        "formal_charges": formal_charges,
        "hydrogen_counts": hydrogen_counts,
        "geometry_classes": geometry_classes,
        "bond_types": bond_types,
        "covalent_radii": covalent_radii,
        "vdw_radii": vdw_radii,
    }


class Local2GeoDataset(Dataset):
    """Read SMILES from a local2global-NMR preprocessed .pt split."""

    def __init__(
        self,
        path: str,
        max_heavy_atoms: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        items = _load_torch(path)
        if not isinstance(items, Sequence):
            raise TypeError(f"Expected a sequence in {path}, got {type(items)!r}")
        self.smiles = []
        for item in items:
            value = _get(item, "isomeric_smiles", _get(item, "smiles"))
            if not value:
                continue
            if max_heavy_atoms is not None:
                stored_atoms = _get(item, "h")
                if stored_atoms is not None:
                    heavy_count = int(torch.as_tensor(stored_atoms).ne(1).sum())
                    if heavy_count > max_heavy_atoms:
                        continue
            self.smiles.append(str(value))
            if limit is not None and len(self.smiles) >= limit:
                break
        if not self.smiles:
            raise ValueError(f"No usable SMILES found in {path}")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return graph_from_smiles(self.smiles[index])


def collate_local2geo(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(samples)
    max_atoms = max(sample["atomic_numbers"].numel() for sample in samples)
    atomic_numbers = torch.zeros((batch_size, max_atoms), dtype=torch.long)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool)
    formal_charges = torch.zeros((batch_size, max_atoms), dtype=torch.float)
    hydrogen_counts = torch.zeros((batch_size, max_atoms), dtype=torch.float)
    geometry_classes = torch.full((batch_size, max_atoms), -100, dtype=torch.long)
    covalent_radii = torch.zeros((batch_size, max_atoms), dtype=torch.float)
    vdw_radii = torch.zeros((batch_size, max_atoms), dtype=torch.float)
    bond_types = torch.full(
        (batch_size, max_atoms, max_atoms), -100, dtype=torch.long
    )
    for batch_index, sample in enumerate(samples):
        size = sample["atomic_numbers"].numel()
        atomic_numbers[batch_index, :size] = sample["atomic_numbers"]
        atom_mask[batch_index, :size] = True
        formal_charges[batch_index, :size] = sample["formal_charges"]
        hydrogen_counts[batch_index, :size] = sample["hydrogen_counts"]
        geometry_classes[batch_index, :size] = sample["geometry_classes"]
        covalent_radii[batch_index, :size] = sample["covalent_radii"]
        vdw_radii[batch_index, :size] = sample["vdw_radii"]
        bond_types[batch_index, :size, :size] = sample["bond_types"]
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    diagonal = torch.eye(max_atoms, dtype=torch.bool)[None]
    pair_mask &= ~diagonal
    return {
        "smiles": [sample["smiles"] for sample in samples],
        "atomic_numbers": atomic_numbers,
        "atom_mask": atom_mask,
        "pair_mask": pair_mask,
        "formal_charges": formal_charges,
        "hydrogen_counts": hydrogen_counts,
        "geometry_classes": geometry_classes,
        "covalent_radii": covalent_radii,
        "vdw_radii": vdw_radii,
        "bond_types": bond_types,
    }


class Local2GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        train_batch_size: int = 16,
        val_batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_heavy_atoms: Optional[int] = 192,
        train_limit: Optional[int] = None,
        val_limit: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Local2GeoDataset(
                self.hparams.train_path,
                self.hparams.max_heavy_atoms,
                self.hparams.train_limit,
            )
            self.val_dataset = Local2GeoDataset(
                self.hparams.val_path,
                self.hparams.max_heavy_atoms,
                self.hparams.val_limit,
            )
        if stage in (None, "test") and self.hparams.test_path:
            self.test_dataset = Local2GeoDataset(
                self.hparams.test_path, self.hparams.max_heavy_atoms
            )

    def _loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        workers = int(self.hparams.num_workers)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=bool(self.hparams.persistent_workers and workers),
            collate_fn=collate_local2geo,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self.hparams.train_batch_size, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, self.hparams.val_batch_size, False)

    def test_dataloader(self) -> Optional[DataLoader]:
        dataset = getattr(self, "test_dataset", None)
        return None if dataset is None else self._loader(
            dataset, self.hparams.val_batch_size, False
        )
