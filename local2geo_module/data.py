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
    """Build an explicit-H graph in the same slot order as NMRToGraph.

    Hydrogen slots come first and are exchangeable. Heavy slots are grouped by
    element, with canonical rank breaking ties within an element.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    Chem.RemoveStereochemistry(molecule)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical))
    canonical_ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))

    hydrogen = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 1
    ]
    heavy = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    hydrogen.sort(key=lambda index: canonical_ranks[index])
    heavy.sort(key=lambda index: (
        molecule.GetAtomWithIdx(index).GetAtomicNum(), canonical_ranks[index]
    ))
    ordered = hydrogen + heavy
    old_to_new = {old: new for new, old in enumerate(ordered)}
    num_hydrogens = len(hydrogen)
    num_atoms = len(ordered)

    atomic_numbers = torch.tensor([
        molecule.GetAtomWithIdx(index).GetAtomicNum() for index in ordered
    ], dtype=torch.long)
    formal_charges = torch.tensor([
        molecule.GetAtomWithIdx(index).GetFormalCharge() for index in ordered
    ], dtype=torch.float)
    hydrogen_counts = torch.tensor([
        (
            sum(
                neighbor.GetAtomicNum() == 1
                for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            )
            if molecule.GetAtomWithIdx(index).GetAtomicNum() != 1 else 0
        )
        for index in ordered
    ], dtype=torch.float)
    geometry_classes = torch.tensor([
        _geometry_class(molecule.GetAtomWithIdx(index)) for index in ordered
    ], dtype=torch.long)

    bond_types = torch.zeros((num_atoms, num_atoms), dtype=torch.long)
    h_attachment = torch.full((num_atoms,), -100, dtype=torch.long)
    for bond in molecule.GetBonds():
        left = old_to_new[bond.GetBeginAtomIdx()]
        right = old_to_new[bond.GetEndAtomIdx()]
        bond_type = _bond_type_index(bond)
        bond_types[left, right] = bond_types[right, left] = bond_type
        if left < num_hydrogens:
            h_attachment[left] = right
        if right < num_hydrogens:
            h_attachment[right] = left

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
        "h_attachment": h_attachment,
        "num_hydrogens": num_hydrogens,
        "num_heavy_atoms": len(heavy),
        "covalent_radii": covalent_radii,
        "vdw_radii": vdw_radii,
    }


class Local2GeoDataset(Dataset):
    """Read SMILES from a local2global-NMR preprocessed .pt split."""

    def __init__(
        self,
        path: str,
        max_heavy_atoms: Optional[int] = None,
        max_total_atoms: Optional[int] = None,
        limit: Optional[int] = None,
        permute_hydrogens: bool = False,
    ):
        items = _load_torch(path)
        if not isinstance(items, Sequence):
            raise TypeError(f"Expected a sequence in {path}, got {type(items)!r}")
        self.smiles = []
        self.permute_hydrogens = permute_hydrogens
        for item in items:
            value = _get(item, "isomeric_smiles", _get(item, "smiles"))
            if not value:
                continue
            stored_atoms = _get(item, "h")
            if max_heavy_atoms is not None and stored_atoms is not None:
                heavy_count = int(torch.as_tensor(stored_atoms).ne(1).sum())
                if heavy_count > max_heavy_atoms:
                    continue
            if max_total_atoms is not None and stored_atoms is not None:
                if int(torch.as_tensor(stored_atoms).numel()) > max_total_atoms:
                    continue
            # Stored atom counts are preferred for filtering because this keeps
            # dataset initialization cheap on the full USPTO split.
            if stored_atoms is None and (
                max_heavy_atoms is not None or max_total_atoms is not None
            ):
                graph = graph_from_smiles(str(value))
                if (
                    max_heavy_atoms is not None
                    and graph["num_heavy_atoms"] > max_heavy_atoms
                ):
                    continue
                if (
                    max_total_atoms is not None
                    and graph["atomic_numbers"].numel() > max_total_atoms
                ):
                    continue
            self.smiles.append(str(value))
            if limit is not None and len(self.smiles) >= limit:
                break
        if not self.smiles:
            raise ValueError(f"No usable SMILES found in {path}")

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        graph = graph_from_smiles(self.smiles[index])
        if not self.permute_hydrogens or graph["num_hydrogens"] < 2:
            return graph
        num_hydrogens = graph["num_hydrogens"]
        num_atoms = graph["atomic_numbers"].numel()
        permutation = torch.cat([
            torch.randperm(num_hydrogens),
            torch.arange(num_hydrogens, num_atoms),
        ])
        permuted = dict(graph)
        for key in (
            "atomic_numbers",
            "formal_charges",
            "hydrogen_counts",
            "geometry_classes",
            "h_attachment",
            "covalent_radii",
            "vdw_radii",
        ):
            permuted[key] = graph[key][permutation]
        permuted["bond_types"] = graph["bond_types"][
            permutation[:, None], permutation[None, :]
        ]
        return permuted


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
    h_attachment = torch.full((batch_size, max_atoms), -100, dtype=torch.long)
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
        h_attachment[batch_index, :size] = sample["h_attachment"]
        bond_types[batch_index, :size, :size] = sample["bond_types"]

    hydrogen_mask = atom_mask & atomic_numbers.eq(1)
    heavy_mask = atom_mask & atomic_numbers.ne(1)
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    diagonal = torch.eye(max_atoms, dtype=torch.bool)[None]
    pair_mask &= ~diagonal
    heavy_pair_mask = (
        heavy_mask[:, :, None] & heavy_mask[:, None, :] & ~diagonal
    )
    attachment_mask = hydrogen_mask[:, :, None] & heavy_mask[:, None, :]
    return {
        "smiles": [sample["smiles"] for sample in samples],
        "atomic_numbers": atomic_numbers,
        "atom_mask": atom_mask,
        "hydrogen_mask": hydrogen_mask,
        "heavy_mask": heavy_mask,
        "pair_mask": pair_mask,
        "heavy_pair_mask": heavy_pair_mask,
        "attachment_mask": attachment_mask,
        "formal_charges": formal_charges,
        "hydrogen_counts": hydrogen_counts,
        "geometry_classes": geometry_classes,
        "covalent_radii": covalent_radii,
        "vdw_radii": vdw_radii,
        "bond_types": bond_types,
        "h_attachment": h_attachment,
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
        max_total_atoms: Optional[int] = 256,
        train_limit: Optional[int] = None,
        val_limit: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        arguments = (
            self.hparams.max_heavy_atoms,
            self.hparams.max_total_atoms,
        )
        if stage in (None, "fit"):
            self.train_dataset = Local2GeoDataset(
                self.hparams.train_path,
                *arguments,
                self.hparams.train_limit,
                permute_hydrogens=True,
            )
            self.val_dataset = Local2GeoDataset(
                self.hparams.val_path, *arguments, self.hparams.val_limit
            )
        if stage in (None, "test") and self.hparams.test_path:
            self.test_dataset = Local2GeoDataset(
                self.hparams.test_path, *arguments
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
