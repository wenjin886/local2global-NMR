import copy
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset


BOND_TYPE_TO_INDEX = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_1d_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=dtype)
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(dtype=dtype).reshape(-1)


def load_local_vocab(path: Optional[str]) -> Dict[int, Dict[str, int]]:
    if path is None:
        return {}
    with open(path, "r") as handle:
        raw_vocab = json.load(handle)
    return {
        int(atomic_number): {
            label: index for index, label in enumerate(labels.keys())
        }
        for atomic_number, labels in raw_vocab.items()
    }


def local_environment_label(atom: Any, molecule: Any) -> str:
    is_aromatic = int(atom.GetIsAromatic())
    num_neighbor_h = 0
    neighbors = []
    for neighbor in atom.GetNeighbors():
        atomic_number = neighbor.GetAtomicNum()
        if atomic_number == 1:
            num_neighbor_h += 1
            continue
        bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        bond_type = BOND_TYPE_TO_INDEX[str(bond.GetBondType())]
        neighbors.append((atomic_number, bond_type))
    neighbors.sort()
    label = "%d%d" % (is_aromatic, num_neighbor_h)
    return label + "".join("%d%d" % pair for pair in neighbors)


def graph_targets_from_smiles(
        smiles: str,
        local_vocab: Optional[Mapping[int, Mapping[str, int]]] = None,
) -> Dict[str, torch.Tensor]:
    """Build atom slots and graph targets in one explicit-H RDKit ordering."""
    try:
        from rdkit import Chem
    except ImportError as error:
        raise ImportError(
            "RDKit is required to create graph targets from SMILES. "
            "Install project dependencies or preprocess targets separately."
        ) from error

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES: %s" % smiles)
    molecule = Chem.AddHs(molecule)

    atom_types = torch.tensor(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=torch.long,
    )
    num_atoms = atom_types.numel()
    bond_types = torch.zeros((num_atoms, num_atoms), dtype=torch.long)
    h_attachment = torch.full((num_atoms,), -100, dtype=torch.long)
    local_labels = torch.full((num_atoms,), -100, dtype=torch.long)

    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        destination = bond.GetEndAtomIdx()
        bond_index = BOND_TYPE_TO_INDEX[str(bond.GetBondType())]
        bond_types[source, destination] = bond_index
        bond_types[destination, source] = bond_index

    for atom in molecule.GetAtoms():
        atom_index = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        if atomic_number == 1:
            heavy_neighbors = [
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() != 1
            ]
            if len(heavy_neighbors) != 1:
                raise ValueError(
                    "Expected H atom %d to have one heavy neighbor, got %d"
                    % (atom_index, len(heavy_neighbors))
                )
            h_attachment[atom_index] = heavy_neighbors[0]
        elif local_vocab and atomic_number in local_vocab:
            label = local_environment_label(atom, molecule)
            local_labels[atom_index] = local_vocab[atomic_number].get(label, -100)

    return {
        "h": atom_types,
        "bond_types": bond_types,
        "h_attachment": h_attachment,
        "local_labels": local_labels,
    }


@dataclass
class GraphSample:
    h: torch.Tensor
    h_nmr: torch.Tensor
    c_nmr: torch.Tensor
    h_nmr_integration: torch.Tensor
    bond_types: torch.Tensor
    h_attachment: torch.Tensor
    local_labels: torch.Tensor
    smiles: str = ""


@dataclass
class GraphBatch:
    atom_types: torch.Tensor
    atom_mask: torch.Tensor
    h_nmr: torch.Tensor
    h_nmr_mask: torch.Tensor
    h_nmr_integration: torch.Tensor
    c_nmr: torch.Tensor
    c_nmr_mask: torch.Tensor
    bond_types: torch.Tensor
    h_attachment: torch.Tensor
    local_labels: torch.Tensor
    smiles: List[str]

    def to(self, device: torch.device) -> "GraphBatch":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if torch.is_tensor(value) else value
        return GraphBatch(**values)

    def model_inputs(self) -> Dict[str, torch.Tensor]:
        return {
            "atom_types": self.atom_types,
            "atom_mask": self.atom_mask,
            "h_nmr": self.h_nmr,
            "h_nmr_mask": self.h_nmr_mask,
            "h_nmr_integration": self.h_nmr_integration,
            "c_nmr": self.c_nmr,
            "c_nmr_mask": self.c_nmr_mask,
        }


class NMRGraphDataset(Dataset):
    """Load NMR samples and construct aligned explicit-H graph targets."""

    def __init__(
            self,
            path: str,
            local_vocab_path: Optional[str] = None,
            transform: Optional[Any] = None,
    ):
        super().__init__()
        self.path = str(path)
        self.items = torch.load(self.path)
        self.local_vocab = load_local_vocab(local_vocab_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> GraphSample:
        item = copy.copy(self.items[index])
        smiles = _get_value(item, "smiles", "")

        existing_bonds = _get_value(item, "bond_types")
        existing_attachment = _get_value(item, "h_attachment")
        if existing_bonds is None or existing_attachment is None:
            targets = graph_targets_from_smiles(smiles, self.local_vocab)
        else:
            atom_types = _as_1d_tensor(_get_value(item, "h"), torch.long)
            local_labels = _get_value(item, "local_labels")
            if local_labels is None:
                local_labels = torch.full_like(atom_types, fill_value=-100)
            targets = {
                "h": atom_types,
                "bond_types": torch.as_tensor(existing_bonds, dtype=torch.long),
                "h_attachment": _as_1d_tensor(existing_attachment, torch.long),
                "local_labels": _as_1d_tensor(local_labels, torch.long),
            }

        h_nmr = _as_1d_tensor(_get_value(item, "h_nmr"), torch.float)
        integration = _get_value(item, "h_nmr_integration")
        if integration is None:
            integration = torch.ones_like(h_nmr)

        sample = GraphSample(
            h=targets["h"],
            h_nmr=h_nmr,
            c_nmr=_as_1d_tensor(_get_value(item, "c_nmr"), torch.float),
            h_nmr_integration=_as_1d_tensor(integration, torch.float),
            bond_types=targets["bond_types"],
            h_attachment=targets["h_attachment"],
            local_labels=targets["local_labels"],
            smiles=smiles,
        )
        return self.transform(sample) if self.transform is not None else sample


def _pad_1d(
        values: Sequence[torch.Tensor],
        padding_value: float,
        dtype: torch.dtype,
) -> torch.Tensor:
    max_length = max((value.numel() for value in values), default=0)
    output = torch.full(
        (len(values), max_length),
        fill_value=padding_value,
        dtype=dtype,
    )
    for index, value in enumerate(values):
        output[index, :value.numel()] = value.to(dtype=dtype)
    return output


def collate_nmr_graph(samples: Sequence[GraphSample]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    atom_types = _pad_1d([sample.h for sample in samples], 0, torch.long)
    atom_mask = atom_types.ne(0)
    h_nmr = _pad_1d([sample.h_nmr for sample in samples], 0.0, torch.float)
    c_nmr = _pad_1d([sample.c_nmr for sample in samples], 0.0, torch.float)
    h_nmr_integration = _pad_1d(
        [sample.h_nmr_integration for sample in samples],
        0.0,
        torch.float,
    )
    h_nmr_mask = torch.zeros_like(h_nmr, dtype=torch.bool)
    c_nmr_mask = torch.zeros_like(c_nmr, dtype=torch.bool)
    for index, sample in enumerate(samples):
        h_nmr_mask[index, :sample.h_nmr.numel()] = True
        c_nmr_mask[index, :sample.c_nmr.numel()] = True

    num_atoms = atom_types.size(1)
    bond_types = torch.full(
        (len(samples), num_atoms, num_atoms),
        fill_value=-100,
        dtype=torch.long,
    )
    h_attachment = torch.full_like(atom_types, fill_value=-100)
    local_labels = torch.full_like(atom_types, fill_value=-100)
    for index, sample in enumerate(samples):
        size = sample.h.numel()
        bond_types[index, :size, :size] = sample.bond_types
        h_attachment[index, :size] = sample.h_attachment
        local_labels[index, :size] = sample.local_labels

    return GraphBatch(
        atom_types=atom_types,
        atom_mask=atom_mask,
        h_nmr=h_nmr,
        h_nmr_mask=h_nmr_mask,
        h_nmr_integration=h_nmr_integration,
        c_nmr=c_nmr,
        c_nmr_mask=c_nmr_mask,
        bond_types=bond_types,
        h_attachment=h_attachment,
        local_labels=local_labels,
        smiles=[sample.smiles for sample in samples],
    )
