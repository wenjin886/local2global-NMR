"""A training-free local-geometry initializer.

RDKit is used only to parse SMILES, add explicit hydrogens, and expose 2D
chemical annotations.  No RDKit conformer generation, force field, or 3D
coordinates are used.  A soft categorical graph is simulated from the parsed
graph, projected to a working neighborhood, assembled from local templates,
and relaxed with differentiable PyTorch chemical-prior terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem


NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC = range(5)
NUM_BOND_TYPES = 5
BOND_LENGTH_SCALES = torch.tensor(
    [1.0, 1.0, 0.90, 0.85, 0.93], dtype=torch.float64
)


@dataclass
class SoftMolecularGraph:
    molecule: rdchem.Mol
    symbols: List[str]
    atomic_numbers: torch.Tensor
    edge_probabilities: torch.Tensor
    projected_bond_types: torch.Tensor
    projected_adjacency: torch.Tensor


@dataclass
class GeometryResult:
    smiles: str
    symbols: List[str]
    coordinates: np.ndarray
    diagnostics: Dict[str, Union[int, float]]
    soft_graph: SoftMolecularGraph


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


def _validate_probability(value: float, name: str, inclusive_zero: bool = True) -> None:
    lower_ok = value >= 0.0 if inclusive_zero else value > 0.0
    if not lower_ok or value >= 1.0:
        relation = "[0, 1)" if inclusive_zero else "(0, 1)"
        raise ValueError(f"{name} must be in {relation}; got {value}")


def simulate_soft_graph(
    molecule: rdchem.Mol,
    edge_confidence: float = 0.97,
    nonedge_bond_probability: float = 0.002,
    logit_noise: float = 0.0,
    edge_threshold: float = 0.5,
    seed: int = 0,
) -> SoftMolecularGraph:
    """Convert a fixed molecular graph into symmetric noisy edge probabilities."""
    _validate_probability(edge_confidence, "edge_confidence", inclusive_zero=False)
    _validate_probability(
        nonedge_bond_probability, "nonedge_bond_probability"
    )
    _validate_probability(edge_threshold, "edge_threshold")
    if logit_noise < 0:
        raise ValueError("logit_noise must be non-negative")

    num_atoms = molecule.GetNumAtoms()
    probabilities = torch.zeros(
        (num_atoms, num_atoms, NUM_BOND_TYPES), dtype=torch.float64
    )
    probabilities[..., NONE] = 1.0 - nonedge_bond_probability
    probabilities[..., 1:] = nonedge_bond_probability / (NUM_BOND_TYPES - 1)
    diagonal = torch.arange(num_atoms)
    probabilities[diagonal, diagonal] = 0.0
    probabilities[diagonal, diagonal, NONE] = 1.0

    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        target = _bond_type_index(bond)
        row = torch.full(
            (NUM_BOND_TYPES,),
            (1.0 - edge_confidence) / (NUM_BOND_TYPES - 1),
            dtype=torch.float64,
        )
        row[target] = edge_confidence
        probabilities[i, j] = row
        probabilities[j, i] = row

    if logit_noise:
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(
            (num_atoms, num_atoms, NUM_BOND_TYPES),
            generator=generator,
            dtype=torch.float64,
        )
        noise = 0.5 * (noise + noise.transpose(0, 1))
        noise[diagonal, diagonal] = 0.0
        logits = probabilities.clamp_min(1e-12).log() + logit_noise * noise
        probabilities = torch.softmax(logits, dim=-1)
        probabilities[diagonal, diagonal] = 0.0
        probabilities[diagonal, diagonal, NONE] = 1.0

    bonded_probability = 1.0 - probabilities[..., NONE]
    projected_types = probabilities.argmax(dim=-1)
    adjacency = projected_types.ne(NONE) & bonded_probability.ge(edge_threshold)
    adjacency.fill_diagonal_(False)
    adjacency = adjacency | adjacency.T
    projected_types = torch.where(
        adjacency, projected_types, torch.zeros_like(projected_types)
    )
    projected_types = torch.maximum(projected_types, projected_types.T)

    return SoftMolecularGraph(
        molecule=molecule,
        symbols=[atom.GetSymbol() for atom in molecule.GetAtoms()],
        atomic_numbers=torch.tensor(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.long
        ),
        edge_probabilities=probabilities,
        projected_bond_types=projected_types,
        projected_adjacency=adjacency,
    )


def _target_angle_degrees(atom: rdchem.Atom, degree: int) -> float:
    if degree < 2:
        return 0.0
    hybridization = atom.GetHybridization()
    if atom.GetIsAromatic() or hybridization == rdchem.HybridizationType.SP2:
        return 120.0
    if hybridization == rdchem.HybridizationType.SP:
        return 180.0
    atomic_number = atom.GetAtomicNum()
    if degree == 2 and atomic_number in {8, 16}:
        return 104.5
    if degree == 3 and atomic_number in {7, 15}:
        return 107.0
    if degree <= 4:
        return 109.47122063449069
    if degree == 5:
        return 90.0
    return 90.0


def _orthonormal_perpendicular(axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.eye(3, dtype=axis.dtype)
    reference = candidates[torch.argmin(torch.abs(candidates @ axis))]
    first = torch.linalg.cross(axis, reference)
    first = first / first.norm().clamp_min(1e-12)
    second = torch.linalg.cross(axis, first)
    second = second / second.norm().clamp_min(1e-12)
    return first, second


def _root_directions(degree: int, angle_degrees: float) -> torch.Tensor:
    if degree == 0:
        return torch.empty((0, 3), dtype=torch.float64)
    if degree == 1:
        return torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    if degree == 2:
        half = np.deg2rad(angle_degrees) / 2.0
        return torch.tensor([
            [np.cos(half), np.sin(half), 0.0],
            [np.cos(half), -np.sin(half), 0.0],
        ], dtype=torch.float64)
    if degree == 3 and abs(angle_degrees - 120.0) < 3.0:
        phases = torch.arange(3, dtype=torch.float64) * (2.0 * torch.pi / 3.0)
        return torch.stack([
            torch.cos(phases), torch.sin(phases), torch.zeros_like(phases)
        ], dim=-1)
    if degree == 3:
        z = -1.0 / 3.0
        radial = np.sqrt(1.0 - z * z)
        phases = torch.arange(3, dtype=torch.float64) * (2.0 * torch.pi / 3.0)
        return torch.stack([
            radial * torch.cos(phases),
            radial * torch.sin(phases),
            torch.full_like(phases, z),
        ], dim=-1)
    if degree == 4:
        directions = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=torch.float64)
        return directions / directions.norm(dim=-1, keepdim=True)

    # Deterministic spherical fallback for uncommon hypervalent centers.
    indices = torch.arange(degree, dtype=torch.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / degree
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    phase = indices * (torch.pi * (3.0 - np.sqrt(5.0)))
    return torch.stack([radius * torch.cos(phase), radius * torch.sin(phase), z], -1)


def _connected_components(adjacency: torch.Tensor) -> List[List[int]]:
    unseen = set(range(adjacency.size(0)))
    components: List[List[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue = [root]
        component = []
        while queue:
            node = queue.pop(0)
            component.append(node)
            neighbors = torch.where(adjacency[node])[0].tolist()
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _bond_targets(graph: SoftMolecularGraph) -> torch.Tensor:
    table = Chem.GetPeriodicTable()
    radii = torch.tensor([
        table.GetRcovalent(int(z)) for z in graph.atomic_numbers.tolist()
    ], dtype=torch.float64)
    radius_sum = radii[:, None] + radii[None, :]
    conditional = graph.edge_probabilities[..., 1:]
    conditional = conditional / conditional.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (conditional * BOND_LENGTH_SCALES[1:]).sum(dim=-1)
    return radius_sum * scale


def _local_template_seed(
    graph: SoftMolecularGraph,
    target_lengths: torch.Tensor,
) -> Tuple[torch.Tensor, List[List[int]]]:
    adjacency = graph.projected_adjacency
    components = _connected_components(adjacency)
    positions = torch.full(
        (adjacency.size(0), 3), float("nan"), dtype=torch.float64
    )

    for component_index, component in enumerate(components):
        root = max(component, key=lambda i: (int(adjacency[i].sum()), -i))
        positions[root] = torch.tensor(
            [component_index * 6.0, 0.0, 0.0], dtype=torch.float64
        )
        root_neighbors = torch.where(adjacency[root])[0].tolist()
        root_angle = _target_angle_degrees(
            graph.molecule.GetAtomWithIdx(root), len(root_neighbors)
        )
        for neighbor, direction in zip(
            root_neighbors, _root_directions(len(root_neighbors), root_angle)
        ):
            if neighbor in component:
                positions[neighbor] = positions[root] + target_lengths[root, neighbor] * direction

        parent = {root: -1}
        for neighbor in root_neighbors:
            parent.setdefault(neighbor, root)
        queue = list(root_neighbors)
        while queue:
            center = queue.pop(0)
            center_neighbors = torch.where(adjacency[center])[0].tolist()
            children = [
                node for node in center_neighbors
                if torch.isnan(positions[node]).any()
            ]
            if not children:
                continue
            parent_node = parent.get(center, root)
            toward_parent = positions[parent_node] - positions[center]
            toward_parent = toward_parent / toward_parent.norm().clamp_min(1e-12)
            perpendicular, binormal = _orthonormal_perpendicular(toward_parent)
            angle = np.deg2rad(_target_angle_degrees(
                graph.molecule.GetAtomWithIdx(center), len(center_neighbors)
            ))
            # A deterministic staggered offset prevents long chains from starting eclipsed.
            phase_offset = (center % 2) * (np.pi / 3.0)
            for child_index, child in enumerate(children):
                phase = phase_offset + 2.0 * np.pi * child_index / len(children)
                direction = (
                    np.cos(angle) * toward_parent
                    + np.sin(angle) * (
                        np.cos(phase) * perpendicular
                        + np.sin(phase) * binormal
                    )
                )
                positions[child] = (
                    positions[center] + target_lengths[center, child] * direction
                )
                parent[child] = center
                queue.append(child)

        # False-negative soft edges can leave isolated atoms in the same parsed molecule.
        for offset, node in enumerate(component):
            if torch.isnan(positions[node]).any():
                positions[node] = positions[root] + torch.tensor(
                    [0.0, 0.0, 2.5 + offset], dtype=torch.float64
                )

    positions = torch.nan_to_num(positions)
    positions -= positions.mean(dim=0, keepdim=True)
    return positions, components


def _constraint_lists(graph: SoftMolecularGraph):
    adjacency = graph.projected_adjacency
    edges = [(i, j) for i in range(adjacency.size(0))
             for j in range(i + 1, adjacency.size(0)) if adjacency[i, j]]
    angles = []
    planar_centers = []
    for center in range(adjacency.size(0)):
        neighbors = torch.where(adjacency[center])[0].tolist()
        target = _target_angle_degrees(
            graph.molecule.GetAtomWithIdx(center), len(neighbors)
        )
        for left, right in combinations(neighbors, 2):
            angles.append((left, center, right, np.deg2rad(target)))
        atom = graph.molecule.GetAtomWithIdx(center)
        if len(neighbors) >= 3 and (
            atom.GetIsAromatic()
            or atom.GetHybridization() == rdchem.HybridizationType.SP2
        ):
            planar_centers.append((center, neighbors[:3]))

    planar_bonds = []
    for i, j in edges:
        bond_type = int(graph.projected_bond_types[i, j])
        parsed_bond = graph.molecule.GetBondBetweenAtoms(i, j)
        is_conjugated = parsed_bond is not None and parsed_bond.GetIsConjugated()
        if bond_type not in {DOUBLE, AROMATIC} and not is_conjugated:
            continue
        left = [n for n in torch.where(adjacency[i])[0].tolist() if n != j]
        right = [n for n in torch.where(adjacency[j])[0].tolist() if n != i]
        for a in left:
            for b in right:
                planar_bonds.append((a, i, j, b))
    return edges, angles, planar_centers, planar_bonds


def _prior_terms(
    positions: torch.Tensor,
    graph: SoftMolecularGraph,
    target_lengths: torch.Tensor,
    constraints,
) -> Dict[str, torch.Tensor]:
    edges, angles, planar_centers, planar_bonds = constraints
    zero = positions.sum() * 0.0

    bond_values = []
    for i, j in edges:
        distance = torch.linalg.vector_norm(positions[i] - positions[j]).clamp_min(1e-8)
        q = 1.0 - graph.edge_probabilities[i, j, NONE]
        bond_values.append(q * torch.log(distance / target_lengths[i, j]).square())
    bond = torch.stack(bond_values).mean() if bond_values else zero

    angle_values = []
    for left, center, right, target in angles:
        u = positions[left] - positions[center]
        v = positions[right] - positions[center]
        cosine = torch.dot(u, v) / (
            u.norm().clamp_min(1e-8) * v.norm().clamp_min(1e-8)
        )
        q = (
            (1.0 - graph.edge_probabilities[left, center, NONE])
            * (1.0 - graph.edge_probabilities[right, center, NONE])
        )
        angle_values.append(q * (cosine - np.cos(target)) ** 2)
    angle = torch.stack(angle_values).mean() if angle_values else zero

    planar_values = []
    for center, neighbors in planar_centers:
        vectors = [positions[n] - positions[center] for n in neighbors]
        volume = torch.dot(torch.linalg.cross(vectors[0], vectors[1]), vectors[2])
        denominator = torch.prod(torch.stack([v.norm() for v in vectors])).clamp_min(1e-8)
        planar_values.append((volume / denominator).square())
    for a, i, j, b in planar_bonds:
        first = torch.linalg.cross(positions[a] - positions[i], positions[j] - positions[i])
        second = torch.linalg.cross(positions[i] - positions[j], positions[b] - positions[j])
        cosine = torch.dot(first, second) / (
            first.norm().clamp_min(1e-8) * second.norm().clamp_min(1e-8)
        )
        planar_values.append(1.0 - cosine.square())
    planar = torch.stack(planar_values).mean() if planar_values else zero

    table = Chem.GetPeriodicTable()
    vdw = torch.tensor([
        table.GetRvdw(int(z)) for z in graph.atomic_numbers.tolist()
    ], dtype=positions.dtype, device=positions.device)
    excluded = graph.projected_adjacency.clone()
    excluded |= (graph.projected_adjacency.to(torch.int64)
                 @ graph.projected_adjacency.to(torch.int64)).bool()
    excluded.fill_diagonal_(True)
    clash_values = []
    for i in range(positions.size(0)):
        for j in range(i + 1, positions.size(0)):
            if excluded[i, j]:
                continue
            distance = torch.linalg.vector_norm(positions[i] - positions[j])
            minimum = 0.62 * (vdw[i] + vdw[j])
            clash_values.append(F.softplus((minimum - distance) / 0.10).square())
    clash = torch.stack(clash_values).mean() if clash_values else zero
    return {"bond": bond, "angle": angle, "planar": planar, "clash": clash}


def _relax(
    initial_positions: torch.Tensor,
    graph: SoftMolecularGraph,
    target_lengths: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, float, float]:
    if steps < 0:
        raise ValueError("relaxation_steps must be non-negative")
    constraints = _constraint_lists(graph)
    positions = initial_positions.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([positions], lr=0.025)

    def total_energy() -> torch.Tensor:
        terms = _prior_terms(positions, graph, target_lengths, constraints)
        return (
            100.0 * terms["bond"]
            + 20.0 * terms["angle"]
            + 5.0 * terms["planar"]
            + 0.02 * terms["clash"]
        )

    initial_energy = float(total_energy().detach())
    for _ in range(steps):
        optimizer.zero_grad()
        energy = total_energy()
        energy.backward()
        torch.nn.utils.clip_grad_norm_([positions], max_norm=20.0)
        optimizer.step()
        with torch.no_grad():
            positions -= positions.mean(dim=0, keepdim=True)
    final_energy = float(total_energy().detach())
    return positions.detach(), initial_energy, final_energy


def _diagnostics(
    positions: torch.Tensor,
    graph: SoftMolecularGraph,
    target_lengths: torch.Tensor,
    components: Sequence[Sequence[int]],
    initial_energy: float,
    final_energy: float,
) -> Dict[str, Union[int, float]]:
    edges, angles, _, _ = _constraint_lists(graph)
    bond_errors = []
    for i, j in edges:
        actual = torch.linalg.vector_norm(positions[i] - positions[j])
        bond_errors.append(float(torch.abs(actual - target_lengths[i, j])))
    angle_errors = []
    for left, center, right, target in angles:
        u = positions[left] - positions[center]
        v = positions[right] - positions[center]
        cosine = torch.dot(u, v) / (u.norm().clamp_min(1e-8) * v.norm().clamp_min(1e-8))
        actual = torch.acos(cosine.clamp(-1.0, 1.0))
        angle_errors.append(abs(float(torch.rad2deg(actual)) - np.rad2deg(target)))

    table = Chem.GetPeriodicTable()
    vdw = [table.GetRvdw(int(z)) for z in graph.atomic_numbers.tolist()]
    excluded = graph.projected_adjacency.clone()
    excluded |= (graph.projected_adjacency.to(torch.int64)
                 @ graph.projected_adjacency.to(torch.int64)).bool()
    excluded.fill_diagonal_(True)
    clash_count = 0
    for i in range(positions.size(0)):
        for j in range(i + 1, positions.size(0)):
            if excluded[i, j]:
                continue
            distance = float(torch.linalg.vector_norm(positions[i] - positions[j]))
            if distance < 0.62 * (vdw[i] + vdw[j]):
                clash_count += 1

    return {
        "num_projected_edges": len(edges),
        "num_components": len(components),
        "initial_prior_energy": initial_energy,
        "final_prior_energy": final_energy,
        "bond_mae_angstrom": float(np.mean(bond_errors)) if bond_errors else 0.0,
        "max_bond_error_angstrom": max(bond_errors, default=0.0),
        "angle_mae_degrees": float(np.mean(angle_errors)) if angle_errors else 0.0,
        "nonlocal_clash_count": clash_count,
    }


def generate_geometry(
    smiles: str,
    add_hydrogens: bool = True,
    edge_confidence: float = 0.97,
    nonedge_bond_probability: float = 0.002,
    logit_noise: float = 0.0,
    edge_threshold: float = 0.5,
    relaxation_steps: int = 600,
    seed: int = 0,
) -> GeometryResult:
    """Generate a locally constrained geometry without any 3D supervision."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if add_hydrogens:
        molecule = Chem.AddHs(molecule)
    graph = simulate_soft_graph(
        molecule=molecule,
        edge_confidence=edge_confidence,
        nonedge_bond_probability=nonedge_bond_probability,
        logit_noise=logit_noise,
        edge_threshold=edge_threshold,
        seed=seed,
    )
    target_lengths = _bond_targets(graph)
    initial, components = _local_template_seed(graph, target_lengths)
    positions, initial_energy, final_energy = _relax(
        initial, graph, target_lengths, relaxation_steps
    )
    diagnostics = _diagnostics(
        positions,
        graph,
        target_lengths,
        components,
        initial_energy,
        final_energy,
    )
    return GeometryResult(
        smiles=smiles,
        symbols=graph.symbols,
        coordinates=positions.cpu().numpy(),
        diagnostics=diagnostics,
        soft_graph=graph,
    )


def write_xyz(path: Union[str, Path], result: GeometryResult) -> None:
    path = Path(path)
    if path.suffix.lower() != ".xyz":
        raise ValueError(f"Output path must end in .xyz: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(result.symbols)), f"SMILES={result.smiles} local2geo_demo"]
    for symbol, coordinate in zip(result.symbols, result.coordinates):
        lines.append(
            f"{symbol:<2s} {coordinate[0]: .8f} {coordinate[1]: .8f} {coordinate[2]: .8f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
