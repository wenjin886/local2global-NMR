from __future__ import annotations

from itertools import combinations
from typing import Tuple

import torch
import torch.nn.functional as F


def graph_smoothed_seed(
    atom_mask: torch.Tensor,
    bond_probability: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Original fully differentiable deterministic seed."""
    batch, atoms = atom_mask.shape
    dtype = bond_probability.dtype
    index = torch.arange(
        atoms, device=atom_mask.device, dtype=dtype
    ) + 1.0
    count = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
    golden = torch.pi * (3.0 - 5.0 ** 0.5)
    phase = index[None, :] * golden
    z = 1.0 - 2.0 * (index[None, :] - 0.5) / count
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    anchors = torch.stack(
        (
            radius * torch.cos(phase),
            radius * torch.sin(phase),
            z,
        ),
        dim=-1,
    )
    scale = 1.5 * count.pow(1.0 / 3.0)
    anchors = anchors * scale.unsqueeze(-1) * atom_mask.unsqueeze(-1)

    degree = bond_probability.sum(dim=-1)
    laplacian = torch.diag_embed(degree) - bond_probability
    identity = torch.eye(
        atoms, device=bond_probability.device, dtype=dtype
    ).unsqueeze(0)
    system = identity + smoothing * laplacian
    seed = torch.linalg.solve(system, anchors)
    seed = seed * atom_mask.unsqueeze(-1)
    center = seed.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
    return (seed - center) * atom_mask.unsqueeze(-1)


def _floyd_warshall(edge_lengths: torch.Tensor) -> torch.Tensor:
    distance = edge_lengths.clone()
    for middle in range(distance.size(0)):
        through_middle = (
            distance[:, middle, None] + distance[None, middle, :]
        )
        distance = torch.minimum(distance, through_middle)
    return distance


def _classical_mds(distance: torch.Tensor) -> torch.Tensor:
    atoms = distance.size(0)
    identity = torch.eye(
        atoms, device=distance.device, dtype=distance.dtype
    )
    centering = identity - torch.ones_like(identity) / max(atoms, 1)
    gram = -0.5 * centering @ distance.square() @ centering
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    take = min(3, atoms)
    indices = torch.arange(
        atoms - 1,
        atoms - take - 1,
        -1,
        device=distance.device,
    )
    values = eigenvalues[indices].clamp_min(0.0)
    coordinates = eigenvectors[:, indices] * torch.sqrt(values)[None, :]
    if take < 3:
        coordinates = torch.nn.functional.pad(
            coordinates, (0, 3 - take)
        )

    # Eigenvectors are sign-ambiguous. Canonicalizing each sign makes repeated
    # eval calls deterministic without changing any pairwise distance.
    for axis in range(3):
        pivot = coordinates[:, axis].abs().argmax()
        sign = torch.where(
            coordinates[pivot, axis] < 0,
            coordinates.new_tensor(-1.0),
            coordinates.new_tensor(1.0),
        )
        coordinates[:, axis] = coordinates[:, axis] * sign
    return coordinates


def _apply_local_one_three_bounds(
    distance: torch.Tensor,
    hard_types: torch.Tensor,
    bond_targets: torch.Tensor,
    target_cosines: torch.Tensor,
) -> torch.Tensor:
    """Replace path-sum 1--3 distances with VSEPR-consistent chords."""
    bounded = distance.clone()
    bonded = hard_types.ne(0)
    for center in range(distance.size(0)):
        neighbors = bonded[center].nonzero(as_tuple=False).flatten().tolist()
        cosine = target_cosines[center].clamp(-1.0, 1.0)
        for first, second in combinations(neighbors, 2):
            left = bond_targets[first, center]
            right = bond_targets[center, second]
            chord = torch.sqrt(
                (
                    left.square()
                    + right.square()
                    - 2.0 * left * right * cosine
                ).clamp_min(1e-6)
            )
            bounded[first, second] = chord
            bounded[second, first] = chord
    return bounded


def _deterministic_jitter(
    atoms: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    index = torch.arange(atoms, device=device, dtype=dtype) + 1.0
    jitter = torch.stack(
        (
            torch.sin(index * 1.61803398875),
            torch.sin(index * 2.41421356237 + 0.7),
            torch.sin(index * 3.14159265359 + 1.3),
        ),
        dim=-1,
    )
    return jitter - jitter.mean(dim=0, keepdim=True)


def _break_linear_local_geometry(
    coordinates: torch.Tensor,
    hard_types: torch.Tensor,
    target_cosines: torch.Tensor,
    displacement: float = 0.45,
) -> torch.Tensor:
    """Give non-linear centers a transverse kick before local stress."""
    position = coordinates.clone()
    bonded = hard_types.ne(0)
    jitter = _deterministic_jitter(
        position.size(0), position.device, position.dtype
    )
    for center in range(position.size(0)):
        if target_cosines[center] < -0.8:
            continue
        neighbors = bonded[center].nonzero(as_tuple=False).flatten()
        if neighbors.numel() < 2:
            continue
        vectors = position[neighbors] - position[center]
        unit = vectors / torch.linalg.vector_norm(
            vectors, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        cosines = unit @ unit.transpose(0, 1)
        cosines.fill_diagonal_(1.0)
        flat = cosines.argmin()
        first = flat // cosines.size(1)
        second = flat % cosines.size(1)
        if cosines[first, second] > -0.90:
            continue
        axis = unit[first] - unit[second]
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-6)
        transverse = jitter[center] - (jitter[center] * axis).sum() * axis
        transverse = transverse / torch.linalg.vector_norm(
            transverse
        ).clamp_min(1e-6)
        position[center] = position[center] + displacement * transverse
    return position


def _seed_stress_refinement(
    coordinates: torch.Tensor,
    hard_types: torch.Tensor,
    bond_targets: torch.Tensor,
    graph_distances: torch.Tensor,
    target_cosines: torch.Tensor,
    planar_probabilities: torch.Tensor,
    num_steps: int,
    step_size: float,
) -> torch.Tensor:
    """Refine a detached MDS proposal with local chemistry constraints."""
    if num_steps <= 0 or coordinates.size(0) <= 1:
        return coordinates

    device = coordinates.device
    bonded = hard_types.ne(0)
    edge_indices = torch.triu(bonded, diagonal=1).nonzero(
        as_tuple=False
    )
    edge_target = (
        bond_targets[edge_indices[:, 0], edge_indices[:, 1]]
        if edge_indices.numel()
        else coordinates.new_empty(0)
    )

    one_three_pairs = []
    one_three_targets = []
    planar_triplets = []
    planar_weights = []
    for center in range(coordinates.size(0)):
        neighbors = bonded[center].nonzero(as_tuple=False).flatten().tolist()
        for first, second in combinations(neighbors, 2):
            left = bond_targets[first, center]
            right = bond_targets[center, second]
            cosine = target_cosines[center].clamp(-1.0, 1.0)
            target_squared = (
                left.square()
                + right.square()
                - 2.0 * left * right * cosine
            ).clamp_min(1e-6)
            one_three_pairs.append((first, second))
            one_three_targets.append(torch.sqrt(target_squared))
        if len(neighbors) >= 3 and planar_probabilities[center] > 0.05:
            for triplet in combinations(neighbors, 3):
                planar_triplets.append((center, *triplet))
                planar_weights.append(planar_probabilities[center])

    if one_three_pairs:
        one_three_indices = torch.tensor(
            one_three_pairs, device=device, dtype=torch.long
        )
        one_three_target = torch.stack(one_three_targets)
    else:
        one_three_indices = torch.empty(
            (0, 2), device=device, dtype=torch.long
        )
        one_three_target = coordinates.new_empty(0)
    if planar_triplets:
        planar_indices = torch.tensor(
            planar_triplets, device=device, dtype=torch.long
        )
        planar_weight = torch.stack(planar_weights)
    else:
        planar_indices = torch.empty(
            (0, 4), device=device, dtype=torch.long
        )
        planar_weight = coordinates.new_empty(0)

    # The global term is a lower bound, not a target distance. It preserves
    # the expanded topology supplied by MDS without forcing graph paths to be
    # perfectly straight. Local 1--3 constraints determine bond angles.
    hops = _floyd_warshall(
        torch.where(
            bonded,
            torch.ones_like(graph_distances),
            torch.full_like(graph_distances, torch.inf),
        ).fill_diagonal_(0.0)
    )
    global_mask = (
        torch.isfinite(graph_distances)
        & hops.ge(3.0)
        & torch.triu(torch.ones_like(bonded), diagonal=1)
    )
    global_indices = global_mask.nonzero(as_tuple=False)
    global_lower = (
        (0.72 * graph_distances[
            global_indices[:, 0], global_indices[:, 1]
        ]).clamp_max(4.5)
        if global_indices.numel()
        else coordinates.new_empty(0)
    )

    # Alternating distance projections are an effective symmetry breaker for
    # nearly collinear MDS chains: ordinary distance gradients have almost no
    # transverse component at 180 degrees. These are proposal-only updates.
    position = coordinates.detach().clone()
    with torch.no_grad():
        for _ in range(min(48, num_steps)):
            for indices, targets, fraction in (
                (edge_indices, edge_target, 0.35),
                (one_three_indices, one_three_target, 0.25),
            ):
                if not indices.numel():
                    continue
                first, second = indices.unbind(dim=-1)
                vector = position[first] - position[second]
                distance_value = torch.linalg.vector_norm(
                    vector, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                correction = (
                    fraction
                    * (
                        distance_value
                        - targets.unsqueeze(-1)
                    )
                    * vector
                    / distance_value
                )
                update = torch.zeros_like(position)
                update.index_add_(0, first, -0.5 * correction)
                update.index_add_(0, second, 0.5 * correction)
                position = position + update
            position = position - position.mean(dim=0, keepdim=True)

    first_moment = torch.zeros_like(position)
    second_moment = torch.zeros_like(position)
    beta1, beta2 = 0.9, 0.999
    with torch.enable_grad():
        for iteration in range(1, num_steps + 1):
            position = position.detach().requires_grad_(True)
            energy = position.sum() * 0.0
            if edge_indices.numel():
                edge_distance = torch.linalg.vector_norm(
                    position[edge_indices[:, 0]]
                    - position[edge_indices[:, 1]],
                    dim=-1,
                )
                energy = energy + 120.0 * torch.log(
                    edge_distance.clamp_min(1e-4)
                    / edge_target.clamp_min(1e-4)
                ).square().mean()
            if one_three_indices.numel():
                one_three_distance = torch.linalg.vector_norm(
                    position[one_three_indices[:, 0]]
                    - position[one_three_indices[:, 1]],
                    dim=-1,
                )
                energy = energy + 48.0 * torch.log(
                    one_three_distance.clamp_min(1e-4)
                    / one_three_target.clamp_min(1e-4)
                ).square().mean()
            if planar_indices.numel():
                center = position[planar_indices[:, 0]]
                vectors = [
                    position[planar_indices[:, index]] - center
                    for index in range(1, 4)
                ]
                unit = [
                    vector / torch.linalg.vector_norm(
                        vector, dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    for vector in vectors
                ]
                signed_volume = (
                    torch.linalg.cross(unit[0], unit[1], dim=-1)
                    * unit[2]
                ).sum(dim=-1)
                energy = energy + 2.0 * (
                    planar_weight * signed_volume.square()
                ).sum() / planar_weight.sum().clamp_min(1e-8)
            if global_indices.numel():
                global_distance = torch.linalg.vector_norm(
                    position[global_indices[:, 0]]
                    - position[global_indices[:, 1]],
                    dim=-1,
                )
                violation = F.softplus(
                    (global_lower - global_distance) / 0.10
                ) * 0.10
                energy = energy + 0.01 * violation.square().mean()

            gradient = torch.autograd.grad(energy, position)[0]
            gradient_norm = torch.linalg.vector_norm(
                gradient, dim=-1, keepdim=True
            )
            gradient = gradient / (1.0 + gradient_norm / 5.0)
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = (
                beta2 * second_moment
                + (1.0 - beta2) * gradient.square()
            )
            corrected_first = first_moment / (1.0 - beta1 ** iteration)
            corrected_second = second_moment / (1.0 - beta2 ** iteration)
            position = position - step_size * corrected_first / (
                corrected_second.sqrt() + 1e-8
            )
            position = position - position.mean(dim=0, keepdim=True)
    return position.detach()


def detached_graph_distance_mds_seed(
    atom_mask: torch.Tensor,
    probabilities: torch.Tensor,
    covalent_radii: torch.Tensor,
    bond_length_scales: torch.Tensor,
    geometry_probabilities: torch.Tensor | None = None,
    geometry_cosines: torch.Tensor | None = None,
    planar_geometry_index: int = 3,
    inflation: float = 1.15,
    jitter_scale: float = 0.08,
    disconnected_margin: float = 4.0,
    stress_steps: int = 384,
    stress_step_size: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard shortest-path MDS seed used only as a detached proposal.

    Returns the padded coordinates and hard bond-type matrix. Relaxation still
    consumes the original soft probabilities, so graph-logit gradients do not
    depend on this discrete seed construction.
    """
    batch, padded_atoms = atom_mask.shape
    seeds = torch.zeros(
        (batch, padded_atoms, 3),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    hard_types = torch.zeros(
        (batch, padded_atoms, padded_atoms),
        device=probabilities.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        for batch_index in range(batch):
            atoms = int(atom_mask[batch_index].sum())
            if atoms <= 1:
                continue
            sample_probabilities = probabilities[
                batch_index, :atoms, :atoms
            ]
            sample_types = sample_probabilities.argmax(dim=-1)
            sample_types = torch.triu(sample_types, diagonal=1)
            sample_types = sample_types + sample_types.transpose(0, 1)
            hard_types[
                batch_index, :atoms, :atoms
            ] = sample_types

            radii_sum = (
                covalent_radii[batch_index, :atoms, None]
                + covalent_radii[batch_index, None, :atoms]
            )
            target = radii_sum * bond_length_scales[sample_types]
            bonded = sample_types.ne(0)
            infinity = torch.full_like(target, torch.inf)
            edge_lengths = torch.where(bonded, target, infinity)
            edge_lengths.fill_diagonal_(0.0)
            distance = _floyd_warshall(edge_lengths)

            finite = torch.isfinite(distance)
            finite_max = distance[finite].max() if finite.any() else (
                distance.new_tensor(1.0)
            )
            distance = torch.where(
                finite,
                distance,
                finite_max + disconnected_margin,
            )
            sample_geometry = None
            target_cosines = None
            if (
                geometry_probabilities is not None
                and geometry_cosines is not None
            ):
                sample_geometry = geometry_probabilities[
                    batch_index, :atoms
                ].detach()
                target_cosines = (
                    sample_geometry * geometry_cosines.detach()
                ).sum(dim=-1)
                distance = _apply_local_one_three_bounds(
                    distance,
                    sample_types,
                    target,
                    target_cosines,
                )
            coordinates = _classical_mds(distance) * inflation
            coordinates = coordinates + jitter_scale * _deterministic_jitter(
                atoms, coordinates.device, coordinates.dtype
            )
            if (
                sample_geometry is not None
                and target_cosines is not None
                and stress_steps > 0
            ):
                coordinates = _break_linear_local_geometry(
                    coordinates,
                    sample_types,
                    target_cosines,
                )
                coordinates = _seed_stress_refinement(
                    coordinates,
                    sample_types,
                    target,
                    distance,
                    target_cosines,
                    sample_geometry[..., planar_geometry_index],
                    num_steps=stress_steps,
                    step_size=stress_step_size,
                )
            coordinates = coordinates - coordinates.mean(
                dim=0, keepdim=True
            )
            seeds[batch_index, :atoms] = coordinates
    return seeds.detach(), hard_types.detach()
