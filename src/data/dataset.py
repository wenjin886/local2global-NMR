import copy
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .constants import (
    BOND_TYPE_CANDIDATES,
    MAX_J_VALUES,
    MULTIPLICITY_MISSING_INDEX,
)


BOND_TYPE_TO_INDEX = {
    "SINGLE": 1,
    "DOUBLE": 2,
    "TRIPLE": 3,
    "AROMATIC": 4,
}
CANDIDATE_TO_INDEX = {
    candidate: index for index, candidate in enumerate(BOND_TYPE_CANDIDATES)
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


def _fragment_count(atom: Any, molecule: Any) -> torch.Tensor:
    counts = torch.zeros(len(BOND_TYPE_CANDIDATES), dtype=torch.long)
    for neighbor in atom.GetNeighbors():
        bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        candidate = "%d-%d" % (
            neighbor.GetAtomicNum(),
            BOND_TYPE_TO_INDEX[str(bond.GetBondType())],
        )
        if candidate not in CANDIDATE_TO_INDEX:
            raise ValueError(
                "Unsupported fragment candidate %s. Add it to "
                "BOND_TYPE_CANDIDATES after checking its dataset frequency."
                % candidate
            )
        counts[CANDIDATE_TO_INDEX[candidate]] += 1
    return counts


def graph_targets_from_smiles(smiles: str) -> Dict[str, torch.Tensor]:
    """Build element-grouped canonical heavy queries and explicit-H targets.

    Input slots are sorted by element. Heavy targets use the same element groups,
    with RDKit canonical ranks breaking ties inside each element. Hydrogen rows
    are exchangeable and are handled by permutation-invariant losses.
    """
    try:
        from rdkit import Chem
    except ImportError as error:
        raise ImportError(
            "RDKit is required to create graph targets from SMILES. "
            "Install project dependencies or materialize targets elsewhere."
        ) from error

    molecule_without_h = Chem.MolFromSmiles(smiles)
    if molecule_without_h is None:
        raise ValueError("Invalid SMILES: %s" % smiles)
    canonical_smiles = Chem.MolToSmiles(
        molecule_without_h,
        canonical=True,
        isomericSmiles=False,
    )
    molecule = Chem.AddHs(Chem.MolFromSmiles(canonical_smiles))
    canonical_ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))

    heavy_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    hydrogen_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 1
    ]
    heavy_indices.sort(key=lambda index: (
        molecule.GetAtomWithIdx(index).GetAtomicNum(),
        canonical_ranks[index],
    ))
    hydrogen_indices.sort(key=lambda index: canonical_ranks[index])

    num_hydrogens = len(hydrogen_indices)
    num_atoms = molecule.GetNumAtoms()
    old_to_slot = {}
    for slot, old_index in enumerate(hydrogen_indices):
        old_to_slot[old_index] = slot
    for heavy_rank, old_index in enumerate(heavy_indices):
        old_to_slot[old_index] = num_hydrogens + heavy_rank

    atom_types = torch.tensor(
        [1] * num_hydrogens
        + [molecule.GetAtomWithIdx(index).GetAtomicNum() for index in heavy_indices],
        dtype=torch.long,
    )
    bond_types = torch.zeros((num_atoms, num_atoms), dtype=torch.long)
    h_attachment = torch.full((num_atoms,), -100, dtype=torch.long)
    heavy_fragment_labels = torch.full(
        (num_atoms, len(BOND_TYPE_CANDIDATES)),
        fill_value=-100,
        dtype=torch.long,
    )
    h_parent_fragment_labels = torch.full_like(
        heavy_fragment_labels,
        fill_value=-100,
    )
    h_parent_types = torch.full((num_atoms,), -100, dtype=torch.long)

    fragment_by_old_index = {
        old_index: _fragment_count(molecule.GetAtomWithIdx(old_index), molecule)
        for old_index in heavy_indices
    }
    for old_index in heavy_indices:
        slot = old_to_slot[old_index]
        heavy_fragment_labels[slot] = fragment_by_old_index[old_index]

    for old_index in hydrogen_indices:
        slot = old_to_slot[old_index]
        neighbors = list(molecule.GetAtomWithIdx(old_index).GetNeighbors())
        if len(neighbors) != 1 or neighbors[0].GetAtomicNum() == 1:
            raise ValueError("Every explicit H must have exactly one heavy parent")
        parent_old_index = neighbors[0].GetIdx()
        parent_slot = old_to_slot[parent_old_index]
        h_attachment[slot] = parent_slot
        h_parent_types[slot] = neighbors[0].GetAtomicNum()
        h_parent_fragment_labels[slot] = fragment_by_old_index[parent_old_index]

    for bond in molecule.GetBonds():
        source_old = bond.GetBeginAtomIdx()
        destination_old = bond.GetEndAtomIdx()
        if (
            molecule.GetAtomWithIdx(source_old).GetAtomicNum() == 1
            or molecule.GetAtomWithIdx(destination_old).GetAtomicNum() == 1
        ):
            continue
        source = old_to_slot[source_old]
        destination = old_to_slot[destination_old]
        bond_index = BOND_TYPE_TO_INDEX[str(bond.GetBondType())]
        bond_types[source, destination] = bond_index
        bond_types[destination, source] = bond_index

    return {
        "h": atom_types,
        "bond_types": bond_types,
        "h_attachment": h_attachment,
        "heavy_fragment_labels": heavy_fragment_labels,
        "h_parent_fragment_labels": h_parent_fragment_labels,
        "h_parent_types": h_parent_types,
    }


@dataclass
class GraphSample:
    h: torch.Tensor
    h_nmr: torch.Tensor
    c_nmr: torch.Tensor
    h_nmr_integration: torch.Tensor
    h_nmr_integration_mask: torch.Tensor
    h_nmr_multiplicity: torch.Tensor
    h_nmr_multiplicity_mask: torch.Tensor
    h_nmr_j: torch.Tensor
    h_nmr_j_mask: torch.Tensor
    bond_types: torch.Tensor
    h_attachment: torch.Tensor
    heavy_fragment_labels: torch.Tensor
    h_parent_fragment_labels: torch.Tensor
    h_parent_types: torch.Tensor
    smiles: str = ""


@dataclass
class GraphBatch:
    atom_types: torch.Tensor
    atom_mask: torch.Tensor
    h_nmr: torch.Tensor
    h_nmr_mask: torch.Tensor
    h_nmr_integration: torch.Tensor
    h_nmr_integration_mask: torch.Tensor
    h_nmr_multiplicity: torch.Tensor
    h_nmr_multiplicity_mask: torch.Tensor
    h_nmr_j: torch.Tensor
    h_nmr_j_mask: torch.Tensor
    c_nmr: torch.Tensor
    c_nmr_mask: torch.Tensor
    bond_types: torch.Tensor
    h_attachment: torch.Tensor
    heavy_fragment_labels: torch.Tensor
    h_parent_fragment_labels: torch.Tensor
    h_parent_types: torch.Tensor
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
            "h_nmr_integration_mask": self.h_nmr_integration_mask,
            "h_nmr_multiplicity": self.h_nmr_multiplicity,
            "h_nmr_multiplicity_mask": self.h_nmr_multiplicity_mask,
            "h_nmr_j": self.h_nmr_j,
            "h_nmr_j_mask": self.h_nmr_j_mask,
            "c_nmr": self.c_nmr,
            "c_nmr_mask": self.c_nmr_mask,
        }


class NMRGraphDataset(Dataset):
    """Load NMR samples and align all targets to ordered output queries."""

    REQUIRED_TARGETS = (
        "bond_types",
        "h_attachment",
        "heavy_fragment_labels",
        "h_parent_fragment_labels",
        "h_parent_types",
    )

    def __init__(self, path: str, transform: Optional[Any] = None):
        super().__init__()
        self.path = str(path)
        self.items = torch.load(self.path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> GraphSample:
        item = copy.copy(self.items[index])
        smiles = _get_value(item, "smiles", "")
        if all(_get_value(item, key) is not None for key in self.REQUIRED_TARGETS):
            targets = {
                "h": _as_1d_tensor(_get_value(item, "h"), torch.long),
                "bond_types": torch.as_tensor(
                    _get_value(item, "bond_types"), dtype=torch.long
                ),
                "h_attachment": _as_1d_tensor(
                    _get_value(item, "h_attachment"), torch.long
                ),
                "heavy_fragment_labels": torch.as_tensor(
                    _get_value(item, "heavy_fragment_labels"), dtype=torch.long
                ),
                "h_parent_fragment_labels": torch.as_tensor(
                    _get_value(item, "h_parent_fragment_labels"), dtype=torch.long
                ),
                "h_parent_types": _as_1d_tensor(
                    _get_value(item, "h_parent_types"), torch.long
                ),
            }
        else:
            targets = graph_targets_from_smiles(smiles)

        h_nmr = _as_1d_tensor(_get_value(item, "h_nmr"), torch.float)
        integration = _get_value(item, "h_nmr_integration")
        integration_is_available = integration is not None
        if integration is None:
            integration = torch.zeros_like(h_nmr)
        integration_mask = _get_value(item, "h_nmr_integration_mask")
        if integration_mask is None:
            integration_mask = torch.full_like(
                h_nmr, integration_is_available, dtype=torch.bool
            )
        multiplicity = _get_value(item, "h_nmr_multiplicity")
        multiplicity_is_available = multiplicity is not None
        if multiplicity is None:
            multiplicity = torch.full_like(
                h_nmr, MULTIPLICITY_MISSING_INDEX, dtype=torch.long
            )
        multiplicity_mask = _get_value(item, "h_nmr_multiplicity_mask")
        if multiplicity_mask is None:
            multiplicity_mask = torch.full_like(
                h_nmr, multiplicity_is_available, dtype=torch.bool
            )
        j_values = _get_value(item, "h_nmr_j")
        if j_values is None:
            j_values = torch.zeros((h_nmr.numel(), MAX_J_VALUES), dtype=torch.float)
        j_values = torch.as_tensor(j_values, dtype=torch.float)
        if j_values.ndim != 2 or j_values.shape[0] != h_nmr.numel():
            raise ValueError("h_nmr_j must have shape [num_h_peaks, num_j_slots]")
        j_mask = _get_value(item, "h_nmr_j_mask")
        if j_mask is None:
            j_mask = j_values.ne(0)
        sample = GraphSample(
            h=targets["h"],
            h_nmr=h_nmr,
            c_nmr=_as_1d_tensor(_get_value(item, "c_nmr"), torch.float),
            h_nmr_integration=_as_1d_tensor(integration, torch.float),
            h_nmr_integration_mask=_as_1d_tensor(integration_mask, torch.bool),
            h_nmr_multiplicity=_as_1d_tensor(multiplicity, torch.long),
            h_nmr_multiplicity_mask=_as_1d_tensor(multiplicity_mask, torch.bool),
            h_nmr_j=j_values,
            h_nmr_j_mask=torch.as_tensor(j_mask, dtype=torch.bool),
            bond_types=targets["bond_types"],
            h_attachment=targets["h_attachment"],
            heavy_fragment_labels=targets["heavy_fragment_labels"],
            h_parent_fragment_labels=targets["h_parent_fragment_labels"],
            h_parent_types=targets["h_parent_types"],
            smiles=smiles,
        )
        return self.transform(sample) if self.transform is not None else sample


def _pad_1d(
        values: Sequence[torch.Tensor],
        padding_value: float,
        dtype: torch.dtype,
) -> torch.Tensor:
    max_length = max((value.numel() for value in values), default=0)
    output = torch.full((len(values), max_length), padding_value, dtype=dtype)
    for index, value in enumerate(values):
        output[index, :value.numel()] = value.to(dtype=dtype)
    return output


def _pad_2d(values: Sequence[torch.Tensor], padding_value, dtype) -> torch.Tensor:
    max_rows = max((value.shape[0] for value in values), default=0)
    max_columns = max((value.shape[1] for value in values), default=0)
    output = torch.full(
        (len(values), max_rows, max_columns), padding_value, dtype=dtype
    )
    for index, value in enumerate(values):
        output[index, :value.shape[0], :value.shape[1]] = value.to(dtype=dtype)
    return output


def collate_nmr_graph(samples: Sequence[GraphSample]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    atom_types = _pad_1d([sample.h for sample in samples], 0, torch.long)
    atom_mask = atom_types.ne(0)
    h_nmr = _pad_1d([sample.h_nmr for sample in samples], 0.0, torch.float)
    c_nmr = _pad_1d([sample.c_nmr for sample in samples], 0.0, torch.float)
    h_nmr_integration = _pad_1d(
        [sample.h_nmr_integration for sample in samples], 0.0, torch.float
    )
    h_nmr_integration_mask = _pad_1d(
        [sample.h_nmr_integration_mask for sample in samples], False, torch.bool
    )
    h_nmr_multiplicity = _pad_1d(
        [sample.h_nmr_multiplicity for sample in samples], 0, torch.long
    )
    h_nmr_multiplicity_mask = _pad_1d(
        [sample.h_nmr_multiplicity_mask for sample in samples], False, torch.bool
    )
    h_nmr_j = _pad_2d([sample.h_nmr_j for sample in samples], 0.0, torch.float)
    h_nmr_j_mask = _pad_2d(
        [sample.h_nmr_j_mask for sample in samples], False, torch.bool
    )
    h_nmr_mask = torch.zeros_like(h_nmr, dtype=torch.bool)
    c_nmr_mask = torch.zeros_like(c_nmr, dtype=torch.bool)
    for index, sample in enumerate(samples):
        h_nmr_mask[index, :sample.h_nmr.numel()] = True
        c_nmr_mask[index, :sample.c_nmr.numel()] = True

    batch_size, num_atoms = atom_types.shape
    num_fragments = len(BOND_TYPE_CANDIDATES)
    bond_types = torch.full(
        (batch_size, num_atoms, num_atoms), -100, dtype=torch.long
    )
    h_attachment = torch.full_like(atom_types, -100)
    heavy_fragment_labels = torch.full(
        (batch_size, num_atoms, num_fragments), -100, dtype=torch.long
    )
    h_parent_fragment_labels = torch.full_like(heavy_fragment_labels, -100)
    h_parent_types = torch.full_like(atom_types, -100)
    for index, sample in enumerate(samples):
        size = sample.h.numel()
        bond_types[index, :size, :size] = sample.bond_types
        h_attachment[index, :size] = sample.h_attachment
        heavy_fragment_labels[index, :size] = sample.heavy_fragment_labels
        h_parent_fragment_labels[index, :size] = sample.h_parent_fragment_labels
        h_parent_types[index, :size] = sample.h_parent_types

    return GraphBatch(
        atom_types=atom_types,
        atom_mask=atom_mask,
        h_nmr=h_nmr,
        h_nmr_mask=h_nmr_mask,
        h_nmr_integration=h_nmr_integration,
        h_nmr_integration_mask=h_nmr_integration_mask,
        h_nmr_multiplicity=h_nmr_multiplicity,
        h_nmr_multiplicity_mask=h_nmr_multiplicity_mask,
        h_nmr_j=h_nmr_j,
        h_nmr_j_mask=h_nmr_j_mask,
        c_nmr=c_nmr,
        c_nmr_mask=c_nmr_mask,
        bond_types=bond_types,
        h_attachment=h_attachment,
        heavy_fragment_labels=heavy_fragment_labels,
        h_parent_fragment_labels=h_parent_fragment_labels,
        h_parent_types=h_parent_types,
        smiles=[sample.smiles for sample in samples],
    )
