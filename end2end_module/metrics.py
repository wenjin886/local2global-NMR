"""Target-free quality metrics and serialization for generated molecules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from local2geo_module.constants import BOND_LENGTH_SCALES
from local2geo_module.visualization import molecule_from_graph


def graph_to_canonical_smiles(
    atomic_numbers: torch.Tensor,
    bond_types: torch.Tensor,
) -> Optional[str]:
    """Convert an explicit-H graph to canonical, non-stereo SMILES.

    GraphBatch does not expose formal charge, so this is deliberately a
    neutral-graph diagnostic. Invalid predicted chemistry returns ``None``.
    """
    from rdkit import Chem

    try:
        molecule = molecule_from_graph(
            atomic_numbers.cpu(),
            torch.zeros_like(atomic_numbers, device="cpu"),
            bond_types.cpu(),
        )
        Chem.SanitizeMol(molecule)
        molecule = Chem.RemoveHs(molecule)
        return Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=False
        )
    except Exception:
        return None


def rdkit_graph_quality(
    atomic_numbers: torch.Tensor,
    bond_types: torch.Tensor,
) -> Dict[str, float]:
    """RDKit validity plus explicit-valence atom/molecule stability.

    Unlike the original QM9 distance-to-bond heuristic, this uses the model's
    corrected categorical graph and RDKit's element-aware valence model. This
    supports the wider USPTO element set and aromatic bonds more safely.
    """
    from rdkit import Chem

    molecule = molecule_from_graph(
        atomic_numbers.cpu(),
        torch.zeros_like(atomic_numbers, device="cpu"),
        bond_types.cpu(),
    )
    connected = float(len(Chem.GetMolFrags(molecule)) == 1)
    try:
        molecule.UpdatePropertyCache(strict=False)
        periodic_table = Chem.GetPeriodicTable()
        stable = []
        for atom in molecule.GetAtoms():
            allowed = {
                int(value) for value in periodic_table.GetValenceList(
                    atom.GetAtomicNum()
                ) if int(value) >= 0
            }
            # Explicit H atoms are present in the graph, so total valence is
            # the relevant quantity. Aromatic valence is handled by RDKit.
            valence = int(round(float(atom.GetExplicitValence())))
            stable.append(not allowed or valence in allowed)
        atom_stability = sum(stable) / max(1, len(stable))
        molecule_stability = float(bool(stable) and all(stable))
    except Exception:
        atom_stability = 0.0
        molecule_stability = 0.0
    try:
        Chem.SanitizeMol(molecule)
        validity = 1.0
    except Exception:
        validity = 0.0
    return {
        "validity": validity,
        "connected": connected,
        "atom_stability": atom_stability,
        "molecule_stability": molecule_stability,
    }


def geometry_quality(
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
    bond_types: torch.Tensor,
    covalent_radii: torch.Tensor,
    vdw_radii: torch.Tensor,
    clash_ratio_threshold: float = 0.75,
) -> Dict[str, float]:
    """Target-free bond-length and non-local clash diagnostics."""
    atom_count = atomic_numbers.numel()
    finite = torch.isfinite(coordinates).all(dim=-1)
    finite_fraction = float(finite.float().mean()) if atom_count else 0.0
    if atom_count < 2 or not bool(finite.all()):
        return {
            "finite_coordinate_fraction": finite_fraction,
            "bond_length_mae_angstrom": float("nan"),
            "min_nonbond_vdw_ratio": float("nan"),
            "clash_free": 0.0,
        }
    distance = torch.cdist(coordinates.float(), coordinates.float())
    upper = torch.triu(
        torch.ones((atom_count, atom_count), dtype=torch.bool), diagonal=1
    )
    bonded = bond_types.gt(0)
    bond_mask = upper & bonded
    if bond_mask.any():
        scales = coordinates.new_tensor(BOND_LENGTH_SCALES)[bond_types]
        target = (
            covalent_radii[:, None] + covalent_radii[None, :]
        ) * scales
        bond_mae = float((distance[bond_mask] - target[bond_mask]).abs().mean())
    else:
        bond_mae = float("nan")

    # Exclude bonded and 1--3 pairs; they are allowed inside the vdW envelope.
    two_hop = (bonded.float() @ bonded.float()).gt(0)
    nonlocal_mask = upper & ~bonded & ~two_hop
    if nonlocal_mask.any():
        radii_sum = vdw_radii[:, None] + vdw_radii[None, :]
        ratios = distance[nonlocal_mask] / radii_sum[nonlocal_mask].clamp_min(1e-8)
        minimum = float(ratios.min())
        clash_free = float(bool((ratios >= clash_ratio_threshold).all()))
    else:
        minimum = 1.0
        clash_free = 1.0
    return {
        "finite_coordinate_fraction": finite_fraction,
        "bond_length_mae_angstrom": bond_mae,
        "min_nonbond_vdw_ratio": minimum,
        "clash_free": clash_free,
    }


def write_xyz(
    path: Path,
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
    comment: str,
) -> None:
    """Write one generated structure in standard XYZ format."""
    from rdkit import Chem

    path.parent.mkdir(parents=True, exist_ok=True)
    periodic_table = Chem.GetPeriodicTable()
    lines = [str(atomic_numbers.numel()), comment.replace("\n", " ")]
    for atomic_number, xyz in zip(atomic_numbers.tolist(), coordinates.tolist()):
        symbol = periodic_table.GetElementSymbol(int(atomic_number))
        lines.append(
            f"{symbol:<3s} {float(xyz[0]): .8f} {float(xyz[1]): .8f} "
            f"{float(xyz[2]): .8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
