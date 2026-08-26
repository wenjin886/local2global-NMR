"""Target-free quality metrics and serialization for generated molecules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from local2geo_module.constants import BOND_LENGTH_SCALES
from local2geo_module.visualization import molecule_from_graph


def graph_exact_match_vectors(
    atom_types: torch.Tensor,
    atom_mask: torch.Tensor,
    target_bond_types: torch.Tensor,
    target_h_attachment: torch.Tensor,
    predicted_edge_logits: torch.Tensor,
    predicted_h_attachment_probabilities: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return per-molecule exact-match vectors with exchangeable hydrogens.

    Heavy-atom connectivity and typed bonds are evaluated on aligned atom
    slots. Explicit hydrogen identities are permutation invariant: only the
    number of hydrogens assigned to each heavy atom must match.
    """
    predicted_bond_types = predicted_edge_logits.argmax(dim=-1)
    predicted_parents = predicted_h_attachment_probabilities.argmax(dim=-1)
    heavy_mask = atom_mask & atom_types.ne(1)
    hydrogen_mask = atom_mask & atom_types.eq(1)
    pair_mask = heavy_mask[:, :, None] & heavy_mask[:, None, :]
    pair_mask = pair_mask & torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    pair_mask = pair_mask & target_bond_types.ge(0)

    target_h_counts = predicted_h_attachment_probabilities.new_zeros(
        atom_types.shape
    )
    predicted_h_counts = torch.zeros_like(target_h_counts)
    for sample_index in range(atom_types.size(0)):
        valid_target_h = hydrogen_mask[sample_index] & target_h_attachment[
            sample_index
        ].ge(0)
        target_parents = target_h_attachment[
            sample_index, valid_target_h
        ].long()
        if target_parents.numel():
            target_h_counts[sample_index].scatter_add_(
                0,
                target_parents,
                torch.ones_like(target_parents, dtype=target_h_counts.dtype),
            )
        sample_predicted_parents = predicted_parents[
            sample_index, hydrogen_mask[sample_index]
        ]
        if sample_predicted_parents.numel():
            predicted_h_counts[sample_index].scatter_add_(
                0,
                sample_predicted_parents,
                torch.ones_like(
                    sample_predicted_parents,
                    dtype=predicted_h_counts.dtype,
                ),
            )

    typed_exact = []
    connectivity_exact = []
    edge_accuracies = []
    connectivity_accuracies = []
    h_count_exact = []
    for sample_index in range(atom_types.size(0)):
        sample_pairs = pair_mask[sample_index]
        predicted = predicted_bond_types[sample_index, sample_pairs]
        target = target_bond_types[sample_index, sample_pairs]
        typed_equal = predicted.eq(target)
        connectivity_equal = predicted.ne(0).eq(target.ne(0))
        sample_h_exact = predicted_h_counts[
            sample_index, heavy_mask[sample_index]
        ].eq(target_h_counts[sample_index, heavy_mask[sample_index]]).all()
        typed_edges_exact = (
            typed_equal.all()
            if typed_equal.numel()
            else torch.ones((), dtype=torch.bool, device=atom_types.device)
        )
        connectivity_edges_exact = (
            connectivity_equal.all()
            if connectivity_equal.numel()
            else torch.ones((), dtype=torch.bool, device=atom_types.device)
        )
        typed_exact.append((typed_edges_exact & sample_h_exact).float())
        connectivity_exact.append(
            (connectivity_edges_exact & sample_h_exact).float()
        )
        h_count_exact.append(sample_h_exact.float())
        edge_accuracies.append(
            typed_equal.float().mean()
            if typed_equal.numel()
            else target_h_counts.new_tensor(1.0)
        )
        connectivity_accuracies.append(
            connectivity_equal.float().mean()
            if connectivity_equal.numel()
            else target_h_counts.new_tensor(1.0)
        )

    return {
        "typed_exact": torch.stack(typed_exact),
        "connectivity_exact": torch.stack(connectivity_exact),
        "h_count_exact": torch.stack(h_count_exact),
        "edge_accuracy": torch.stack(edge_accuracies),
        "connectivity_accuracy": torch.stack(connectivity_accuracies),
    }


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
