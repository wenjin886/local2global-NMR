from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch.utils.data import DataLoader, Dataset

import os
from .constants import (
    AROMATIC,
    BOND_LENGTH_SCALES,
    DOUBLE,
    GEOMETRY_COSINES,
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


def _analytic_local_targets(
    molecule: Chem.Mol,
    ordered: Sequence[int],
    old_to_new: Mapping[int, int],
    bond_types: torch.Tensor,
    geometry_classes: torch.Tensor,
    covalent_radii: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Derive local distance supervision from a clean 2D graph only.

    The targets deliberately contain no conformer information. 1--3 distances
    follow the law of cosines using the VSEPR class of the centre atom. 1--4
    distances use a deterministic torsion prior: planar for conjugated central
    bonds, gauche-like for rings, and anti for acyclic heavy-atom chains.
    Distances are stored as log-ratios to the endpoint covalent-radius sum.
    """
    atoms = len(ordered)
    adjacency = bond_types.gt(0)
    distance = torch.full((atoms, atoms), atoms + 1, dtype=torch.long)
    distance.fill_diagonal_(0)
    distance[adjacency] = 1
    for middle in range(atoms):
        distance = torch.minimum(
            distance,
            distance[:, middle, None] + distance[None, middle, :],
        )

    one_three = distance.eq(2)
    one_four = distance.eq(3)
    log_ratio_13 = torch.zeros((atoms, atoms), dtype=torch.float)
    log_ratio_14 = torch.zeros((atoms, atoms), dtype=torch.float)
    torsion_classes = torch.full((atoms, atoms), -100, dtype=torch.long)
    ring_bonds = torch.zeros((atoms, atoms), dtype=torch.float)
    conjugated_bonds = torch.zeros((atoms, atoms), dtype=torch.float)

    for bond in molecule.GetBonds():
        left = old_to_new[bond.GetBeginAtomIdx()]
        right = old_to_new[bond.GetEndAtomIdx()]
        ring_bonds[left, right] = ring_bonds[right, left] = float(
            bond.IsInRing()
        )
        conjugated_bonds[left, right] = conjugated_bonds[right, left] = float(
            bond.GetIsConjugated() or bond.GetIsAromatic()
        )

    scales = torch.tensor(BOND_LENGTH_SCALES, dtype=torch.float)
    cosines = torch.tensor(GEOMETRY_COSINES, dtype=torch.float)

    def bond_length(left: int, right: int) -> torch.Tensor:
        return (
            covalent_radii[left] + covalent_radii[right]
        ) * scales[bond_types[left, right]]

    def endpoint_log_ratio(
        target: torch.Tensor,
        left: int,
        right: int,
    ) -> torch.Tensor:
        baseline = (
            covalent_radii[left] + covalent_radii[right]
        ).clamp_min(0.5)
        return torch.log(target.clamp_min(0.5) / baseline)

    for left in range(atoms):
        for right in range(left + 1, atoms):
            if one_three[left, right]:
                path = Chem.GetShortestPath(
                    molecule, int(ordered[left]), int(ordered[right])
                )
                if len(path) != 3:
                    continue
                centre = old_to_new[path[1]]
                first = bond_length(left, centre)
                second = bond_length(centre, right)
                cosine = cosines[geometry_classes[centre]].clamp(-1.0, 1.0)
                chord = (
                    first.square()
                    + second.square()
                    - 2.0 * first * second * cosine
                ).clamp_min(1e-4).sqrt()
                value = endpoint_log_ratio(chord, left, right)
                log_ratio_13[left, right] = log_ratio_13[right, left] = value

            if not one_four[left, right]:
                continue
            path = Chem.GetShortestPath(
                molecule, int(ordered[left]), int(ordered[right])
            )
            if len(path) != 4:
                continue
            path_new = [old_to_new[index] for index in path]
            i, j, k, l = path_new
            central_bond = molecule.GetBondBetweenAtoms(path[1], path[2])
            conjugated = bool(
                central_bond.GetIsConjugated()
                or central_bond.GetIsAromatic()
            )
            in_ring = bool(central_bond.IsInRing())
            heavy_path = all(
                molecule.GetAtomWithIdx(index).GetAtomicNum() != 1
                for index in path
            )
            if not heavy_path:
                # Hydrogen-containing 1--4 paths are not independently
                # rotatable. They still supervise membership, but do not
                # receive an incompatible torsion/distance target.
                continue
            if conjugated:
                torsion_class, torsion_cosine = 0, 1.0
            elif in_ring:
                torsion_class, torsion_cosine = 1, 0.5
            else:
                torsion_class, torsion_cosine = 2, -1.0

            d_ij = bond_length(i, j)
            d_jk = bond_length(j, k)
            d_kl = bond_length(k, l)
            cos_j = cosines[geometry_classes[j]].clamp(-1.0, 1.0)
            cos_k = cosines[geometry_classes[k]].clamp(-1.0, 1.0)
            sin_j = (1.0 - cos_j.square()).clamp_min(0.0).sqrt()
            sin_k = (1.0 - cos_k.square()).clamp_min(0.0).sqrt()
            squared = (
                d_ij.square() + d_jk.square() + d_kl.square()
                - 2.0 * d_ij * d_jk * cos_j
                - 2.0 * d_jk * d_kl * cos_k
                + 2.0 * d_ij * d_kl * cos_j * cos_k
                - 2.0
                * d_ij
                * d_kl
                * sin_j
                * sin_k
                * torsion_cosine
            )
            chord = squared.clamp_min(1e-4).sqrt()
            value = endpoint_log_ratio(chord, left, right)
            log_ratio_14[left, right] = log_ratio_14[right, left] = value
            torsion_classes[left, right] = torsion_classes[
                right, left
            ] = torsion_class

    return {
        "one_three_targets": one_three.to(torch.float),
        "one_four_targets": one_four.to(torch.float),
        "one_three_log_ratio": log_ratio_13,
        "one_four_log_ratio": log_ratio_14,
        "torsion_classes": torsion_classes,
        "ring_bonds": ring_bonds,
        "conjugated_bonds": conjugated_bonds,
    }


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
    graph = {
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
    graph.update(_analytic_local_targets(
        molecule=molecule,
        ordered=ordered,
        old_to_new=old_to_new,
        bond_types=bond_types,
        geometry_classes=geometry_classes,
        covalent_radii=covalent_radii,
    ))
    return graph


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
        for key in (
            "one_three_targets",
            "one_four_targets",
            "one_three_log_ratio",
            "one_four_log_ratio",
            "torsion_classes",
            "ring_bonds",
            "conjugated_bonds",
        ):
            permuted[key] = graph[key][
                permutation[:, None], permutation[None, :]
            ]
        return permuted


def _expand_parquet_paths(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
) -> Sequence[Path]:
    if os.path.isdir(paths):
        paths = [os.path.join(paths, file) for file in os.listdir(paths)]
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.parquet")))
        elif any(character in str(path) for character in "*?[]"):
            expanded.extend(sorted(path.parent.glob(path.name)))
        else:
            expanded.append(path)
    missing = [path for path in expanded if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing parquet input(s): "
            + ", ".join(str(path) for path in missing)
        )
    if not expanded:
        raise ValueError("No parquet files were found")
    return expanded


def _canonical_smiles(value: object) -> Optional[str]:
    if value is None:
        return None
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=False
    )


def _read_parquet_smiles(
    paths: Sequence[Path],
    smiles_column: str,
) -> Sequence[str]:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Reading the example parquet data requires pandas and pyarrow. "
            "Install the project dependencies with `pip install -e .`."
        ) from error
    values = []
    for path in paths:
        try:
            frame = pd.read_parquet(path, columns=[smiles_column])
        except ImportError as error:
            raise ImportError(
                "No parquet engine is installed. Install pyarrow (declared "
                "in pyproject.toml) before training local2geo."
            ) from error
        if smiles_column not in frame:
            raise KeyError(
                f"{path} does not contain SMILES column {smiles_column!r}"
            )
        values.extend(frame[smiles_column].dropna().astype(str).tolist())
    canonical = filter(None, (_canonical_smiles(value) for value in values))
    # Preserve one instance of each 2D molecule. Duplicate spectra must not
    # leak the same graph across train/validation/test.
    return sorted(set(canonical))


def _split_smiles(
    smiles: Sequence[str],
    split: str,
    ratios: Sequence[float],
    seed: int,
) -> Sequence[str]:
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train, val, test, or all")
    if split == "all":
        return list(smiles)
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split_ratios must contain three values summing to 1")

    def key(value: str) -> str:
        return hashlib.sha1(
            f"{seed}:{value}".encode("utf-8")
        ).hexdigest()

    ordered = sorted(smiles, key=key)
    size = len(ordered)
    train_end = int(ratios[0] * size)
    val_end = train_end + int(ratios[1] * size)
    if size >= 3:
        train_end = min(max(train_end, 1), size - 2)
        val_end = min(max(val_end, train_end + 1), size - 1)
    ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, size),
    }
    start, stop = ranges[split]
    return ordered[start:stop]


class ParquetSmilesDataset(Dataset):
    """Explicit-H clean graphs read directly from example parquet SMILES."""

    def __init__(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        split: str,
        smiles_column: str = "smiles",
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        split_seed: int = 1729,
        max_heavy_atoms: Optional[int] = 128,
        max_total_atoms: Optional[int] = 192,
        limit: Optional[int] = None,
        permute_hydrogens: bool = False,
        preloaded_smiles: Optional[Sequence[str]] = None,
    ) -> None:
        all_smiles = preloaded_smiles
        if all_smiles is None:
            parquet_paths = _expand_parquet_paths(paths)
            all_smiles = _read_parquet_smiles(
                parquet_paths, smiles_column
            )
        values = _split_smiles(
            all_smiles,
            split=split,
            ratios=split_ratios,
            seed=split_seed,
        )
        self.smiles = []
        self.permute_hydrogens = permute_hydrogens
        for value in values:
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                continue
            heavy_atoms = molecule.GetNumHeavyAtoms()
            total_atoms = Chem.AddHs(molecule).GetNumAtoms()
            if (
                max_heavy_atoms is not None
                and heavy_atoms > max_heavy_atoms
            ):
                continue
            if (
                max_total_atoms is not None
                and total_atoms > max_total_atoms
            ):
                continue
            self.smiles.append(value)
            if limit is not None and len(self.smiles) >= limit:
                break
        if not self.smiles:
            raise ValueError(
                f"No usable molecules remained in parquet split {split!r}"
            )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        graph = graph_from_smiles(self.smiles[index])
        if not self.permute_hydrogens or graph["num_hydrogens"] < 2:
            return graph
        num_hydrogens = graph["num_hydrogens"]
        atoms = graph["atomic_numbers"].numel()
        permutation = torch.cat([
            torch.randperm(num_hydrogens),
            torch.arange(num_hydrogens, atoms),
        ])
        sample = dict(graph)
        for key in (
            "atomic_numbers",
            "formal_charges",
            "hydrogen_counts",
            "geometry_classes",
            "h_attachment",
            "covalent_radii",
            "vdw_radii",
        ):
            sample[key] = graph[key][permutation]
        for key in (
            "bond_types",
            "one_three_targets",
            "one_four_targets",
            "one_three_log_ratio",
            "one_four_log_ratio",
            "torsion_classes",
            "ring_bonds",
            "conjugated_bonds",
        ):
            sample[key] = graph[key][
                permutation[:, None], permutation[None, :]
            ]
        return sample


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
    pair_float_keys = (
        "one_three_targets",
        "one_four_targets",
        "one_three_log_ratio",
        "one_four_log_ratio",
        "ring_bonds",
        "conjugated_bonds",
    )
    pair_values = {
        key: torch.zeros(
            (batch_size, max_atoms, max_atoms), dtype=torch.float
        )
        for key in pair_float_keys
    }
    torsion_classes = torch.full(
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
        for key in pair_float_keys:
            pair_values[key][batch_index, :size, :size] = sample[key]
        torsion_classes[batch_index, :size, :size] = sample[
            "torsion_classes"
        ]

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
        "torsion_classes": torsion_classes,
        **pair_values,
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


class ParquetLocal2GeoDataModule(pl.LightningDataModule):
    """Standalone parquet datamodule used by the hybrid topology prior."""

    def __init__(
        self,
        parquet_paths: Union[
            str, Path, Sequence[Union[str, Path]]
        ],
        smiles_column: str = "smiles",
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        split_seed: int = 1729,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_heavy_atoms: Optional[int] = 96,
        max_total_atoms: Optional[int] = 128,
        train_limit: Optional[int] = None,
        val_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _dataset(
        self,
        split: str,
        limit: Optional[int],
        permute_hydrogens: bool = False,
    ) -> ParquetSmilesDataset:
        return ParquetSmilesDataset(
            paths=self.hparams.parquet_paths,
            split=split,
            smiles_column=self.hparams.smiles_column,
            split_ratios=self.hparams.split_ratios,
            split_seed=self.hparams.split_seed,
            max_heavy_atoms=self.hparams.max_heavy_atoms,
            max_total_atoms=self.hparams.max_total_atoms,
            limit=limit,
            permute_hydrogens=permute_hydrogens,
            preloaded_smiles=self._all_smiles,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "_all_smiles"):
            self._all_smiles = _read_parquet_smiles(
                _expand_parquet_paths(self.hparams.parquet_paths),
                self.hparams.smiles_column,
            )
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(
                "train", self.hparams.train_limit, permute_hydrogens=True
            )
            self.val_dataset = self._dataset(
                "val", self.hparams.val_limit
            )
        if stage in (None, "test"):
            self.test_dataset = self._dataset(
                "test", self.hparams.test_limit
            )

    def _loader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        workers = int(self.hparams.num_workers)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=bool(
                self.hparams.persistent_workers and workers > 0
            ),
            collate_fn=collate_local2geo,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset, self.hparams.train_batch_size, True
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset, self.hparams.val_batch_size, False
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset, self.hparams.val_batch_size, False
        )
