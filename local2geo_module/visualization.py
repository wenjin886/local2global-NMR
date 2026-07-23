from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Geometry import Point3D

from .constants import AROMATIC, DOUBLE, NONE, SINGLE, TRIPLE


RDKIT_BOND_TYPES = {
    SINGLE: Chem.BondType.SINGLE,
    DOUBLE: Chem.BondType.DOUBLE,
    TRIPLE: Chem.BondType.TRIPLE,
    AROMATIC: Chem.BondType.AROMATIC,
}


def molecule_from_graph(
    atomic_numbers: torch.Tensor,
    formal_charges: torch.Tensor,
    bond_types: torch.Tensor,
    coordinates: Optional[torch.Tensor] = None,
) -> Chem.Mol:
    editable = Chem.RWMol()
    for atomic_number, charge in zip(
        atomic_numbers.tolist(), formal_charges.tolist()
    ):
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(round(charge)))
        editable.AddAtom(atom)
    atom_count = atomic_numbers.numel()
    for left in range(atom_count):
        for right in range(left + 1, atom_count):
            bond_type = int(bond_types[left, right])
            if bond_type == NONE:
                continue
            editable.AddBond(left, right, RDKIT_BOND_TYPES[bond_type])
            if bond_type == AROMATIC:
                editable.GetAtomWithIdx(left).SetIsAromatic(True)
                editable.GetAtomWithIdx(right).SetIsAromatic(True)
    molecule = editable.GetMol()
    if coordinates is not None:
        conformer = Chem.Conformer(atom_count)
        for index, (x, y, z) in enumerate(coordinates.tolist()):
            conformer.SetAtomPosition(
                index, Point3D(float(x), float(y), float(z))
            )
        molecule.RemoveAllConformers()
        molecule.AddConformer(conformer, assignId=True)
    return molecule


def graph_image(
    atomic_numbers: torch.Tensor,
    formal_charges: torch.Tensor,
    bond_types: torch.Tensor,
):
    molecule = molecule_from_graph(
        atomic_numbers, formal_charges, bond_types
    )
    rdDepictor.Compute2DCoords(molecule)
    return Draw.MolToImage(
        molecule,
        size=(700, 500),
        kekulize=False,
        wedgeBonds=False,
    )


def write_sdf(
    path: Path,
    atomic_numbers: torch.Tensor,
    formal_charges: torch.Tensor,
    bond_types: torch.Tensor,
    coordinates: torch.Tensor,
    name: str,
) -> None:
    molecule = molecule_from_graph(
        atomic_numbers, formal_charges, bond_types, coordinates
    )
    molecule.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)
    writer.write(molecule)
    writer.close()


def sample_geometry_summary(
    sample: Dict[str, torch.Tensor],
    coordinates: torch.Tensor,
    target_lengths: torch.Tensor,
) -> Dict[str, float]:
    bond_types = sample["bond_types"]
    atom_mask = sample["atom_mask"]
    pair_mask = atom_mask[:, None] & atom_mask[None, :]
    upper = torch.triu(pair_mask, diagonal=1)
    edge = bond_types.ne(NONE) & upper
    vector = coordinates[None, :, :] - coordinates[:, None, :]
    distance = torch.sqrt(vector.square().sum(dim=-1) + 1e-8)
    bond_mae = (
        float((distance[edge] - target_lengths[edge]).abs().mean())
        if edge.any() else 0.0
    )
    adjacency = bond_types.ne(NONE) & pair_mask
    two_hop = (adjacency.float() @ adjacency.float()).gt(0)
    nonlocal_mask = upper & ~adjacency & ~two_hop
    radii_sum = sample["vdw_radii"][:, None] + sample["vdw_radii"][None, :]
    ratio = distance / radii_sum.clamp_min(1e-8)
    min_nonbond_ratio = (
        float(ratio[nonlocal_mask].min()) if nonlocal_mask.any() else 1.0
    )
    return {
        "bond_mae_angstrom": bond_mae,
        "min_nonbond_vdw_ratio": min_nonbond_ratio,
    }
