from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .constants import GEOMETRY_COSINES, NONE


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    return values[mask].mean() if mask.any() else zero


def _angle_mae(
    positions: torch.Tensor,
    target: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    neighbor_filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cosine_targets = torch.tensor(
        GEOMETRY_COSINES, device=positions.device, dtype=positions.dtype
    )
    errors = []
    for batch_index in range(target.size(0)):
        valid_atoms = batch["atom_mask"][batch_index].nonzero(
            as_tuple=False
        ).flatten()
        for center in valid_atoms.tolist():
            neighbors = (
                target[batch_index, center].ne(NONE)
                & batch["pair_mask"][batch_index, center]
            )
            if neighbor_filter is not None:
                neighbors &= neighbor_filter[batch_index]
            neighbor_indices = neighbors.nonzero(as_tuple=False).flatten()
            if neighbor_indices.numel() < 2:
                continue
            vectors = (
                positions[batch_index, neighbor_indices]
                - positions[batch_index, center]
            )
            vectors = F.normalize(vectors, dim=-1)
            matrix = vectors @ vectors.T
            pairs = torch.triu(
                torch.ones_like(matrix, dtype=torch.bool), diagonal=1
            )
            actual = torch.acos(matrix[pairs].clamp(-1.0, 1.0))
            geometry_index = batch["geometry_classes"][batch_index, center]
            target_angle = torch.acos(
                cosine_targets[geometry_index].clamp(-1.0, 1.0)
            )
            errors.append((actual - target_angle).abs())
    return (
        torch.cat(errors).mean()
        if errors else positions.sum() * 0.0
    )


def _clash_rate(
    distance: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    atom_selection: torch.Tensor,
    distance_scale: float,
) -> torch.Tensor:
    target = batch["bond_types"].clamp_min(0)
    adjacency = target.ne(NONE) & batch["pair_mask"]
    two_hop = torch.bmm(adjacency.float(), adjacency.float()).gt(0)
    upper = torch.triu(
        torch.ones_like(batch["pair_mask"], dtype=torch.bool), diagonal=1
    )
    selected_pairs = (
        atom_selection[:, :, None] & atom_selection[:, None, :]
    )
    nonlocal_mask = (
        batch["pair_mask"]
        & upper
        & selected_pairs
        & ~adjacency
        & ~two_hop
    )
    minimum = distance_scale * (
        batch["vdw_radii"][:, :, None]
        + batch["vdw_radii"][:, None, :]
    )
    return (
        distance[nonlocal_mask]
        .lt(minimum[nonlocal_mask])
        .float()
        .mean()
        if nonlocal_mask.any()
        else distance.sum() * 0.0
    )


@torch.no_grad()
def local2geo_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    target_lengths: torch.Tensor,
    clash_distance_scale: float = 0.80,
) -> Dict[str, torch.Tensor]:
    heavy_probabilities = outputs["projected_heavy_edge_probabilities"]
    heavy_prediction = heavy_probabilities.argmax(dim=-1)
    target = batch["bond_types"].clamp_min(0)
    heavy_upper = batch["heavy_pair_mask"] & torch.triu(
        torch.ones_like(batch["heavy_pair_mask"], dtype=torch.bool),
        diagonal=1,
    )
    pred_edge = heavy_prediction.ne(NONE) & heavy_upper
    target_heavy_edge = target.ne(NONE) & heavy_upper
    tp = (pred_edge & target_heavy_edge).sum().float()
    fp = (pred_edge & ~target_heavy_edge).sum().float()
    fn = (~pred_edge & target_heavy_edge).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    edge_f1 = (
        2.0 * precision * recall
        / (precision + recall).clamp_min(1e-8)
    )
    bond_type_accuracy = (
        heavy_prediction[target_heavy_edge]
        .eq(target[target_heavy_edge])
        .float()
        .mean()
        if target_heavy_edge.any() else tp * 0.0 + 1.0
    )
    exact = []
    for index in range(target.size(0)):
        valid = heavy_upper[index]
        exact.append(
            heavy_prediction[index, valid]
            .eq(target[index, valid])
            .all()
            .float()
        )
    graph_exact = torch.stack(exact).mean()
    entropy = -(
        heavy_probabilities.clamp_min(1e-8).log()
        * heavy_probabilities
    ).sum(dim=-1)
    entropy = (
        entropy[heavy_upper].mean()
        if heavy_upper.any() else entropy.sum() * 0.0
    )

    attachment = outputs["projected_h_attachment_probabilities"]
    predicted_counts = attachment.sum(dim=1)
    h_count_mae = (
        (
            predicted_counts[batch["heavy_mask"]]
            - batch["hydrogen_counts"][batch["heavy_mask"]]
        ).abs().mean()
        if batch["heavy_mask"].any()
        else attachment.sum() * 0.0
    )
    attachment_entropy_values = -(
        attachment.clamp_min(1e-12).log() * attachment
    ).sum(dim=-1)
    attachment_entropy = (
        attachment_entropy_values[batch["hydrogen_mask"]].mean()
        if batch["hydrogen_mask"].any()
        else attachment.sum() * 0.0
    )
    attachment_scores = []
    for index in range(target.size(0)):
        h_rows = batch["hydrogen_mask"][index]
        targets = batch["h_attachment"][index, h_rows]
        if targets.numel() == 0:
            continue
        predicted = attachment[index, h_rows].argmax(dim=-1)
        target_histogram = torch.bincount(
            targets, minlength=target.size(1)
        )
        predicted_histogram = torch.bincount(
            predicted, minlength=target.size(1)
        )
        attachment_scores.append(
            torch.minimum(target_histogram, predicted_histogram).sum().float()
            / targets.numel()
        )
    attachment_multiset_accuracy = (
        torch.stack(attachment_scores).mean()
        if attachment_scores else attachment.sum() * 0.0 + 1.0
    )

    positions = outputs["coordinates"]
    vector = positions[:, None, :, :] - positions[:, :, None, :]
    distance = torch.sqrt(vector.square().sum(dim=-1) + 1e-8)
    upper = batch["pair_mask"] & torch.triu(
        torch.ones_like(batch["pair_mask"], dtype=torch.bool), diagonal=1
    )
    all_edge = target.ne(NONE) & upper
    h_bond = all_edge & (
        batch["hydrogen_mask"][:, :, None]
        | batch["hydrogen_mask"][:, None, :]
    )
    heavy_bond = all_edge & (
        batch["heavy_mask"][:, :, None]
        & batch["heavy_mask"][:, None, :]
    )
    error = (distance - target_lengths).abs()
    zero = distance.sum() * 0.0
    bond_mae = _masked_mean(error, all_edge, zero)
    heavy_bond_mae = _masked_mean(error, heavy_bond, zero)
    h_bond_mae = _masked_mean(error, h_bond, zero)

    angle_mae = _angle_mae(positions, target, batch)
    heavy_neighbor_angle_mae = _angle_mae(
        positions, target, batch, neighbor_filter=batch["heavy_mask"]
    )
    clash_rate = _clash_rate(
        distance, batch, batch["atom_mask"], clash_distance_scale
    )
    heavy_clash_rate = _clash_rate(
        distance, batch, batch["heavy_mask"], clash_distance_scale
    )
    all_atom_local_score = torch.exp(
        -bond_mae / 0.15 - angle_mae / 0.35 - 5.0 * clash_rate
    )
    heavy_local_score = torch.exp(
        -heavy_bond_mae / 0.15
        - heavy_neighbor_angle_mae / 0.35
        - 5.0 * heavy_clash_rate
    )
    score = (
        edge_f1 * all_atom_local_score * heavy_local_score
    ).clamp_min(0.0).pow(1.0 / 3.0)
    geometry_prediction = outputs["geometry_logits"].argmax(dim=-1)
    geometry_accuracy = geometry_prediction.eq(
        batch["geometry_classes"]
    )[batch["atom_mask"]].float().mean()
    heavy_geometry_accuracy = geometry_prediction.eq(
        batch["geometry_classes"]
    )[batch["heavy_mask"]].float().mean()
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": edge_f1,
        "bond_type_accuracy": bond_type_accuracy,
        "graph_exact": graph_exact,
        "edge_entropy": entropy,
        "h_attachment_multiset_accuracy": attachment_multiset_accuracy,
        "h_count_mae": h_count_mae,
        "h_attachment_entropy": attachment_entropy,
        "geometry_class_accuracy": geometry_accuracy,
        "heavy_geometry_class_accuracy": heavy_geometry_accuracy,
        "bond_mae_angstrom": bond_mae,
        "heavy_bond_mae_angstrom": heavy_bond_mae,
        "h_bond_mae_angstrom": h_bond_mae,
        "angle_mae_radian": angle_mae,
        "heavy_neighbor_angle_mae_radian": heavy_neighbor_angle_mae,
        "clash_rate": clash_rate,
        "heavy_clash_rate": heavy_clash_rate,
        "local_score": all_atom_local_score,
        "heavy_local_score": heavy_local_score,
        "score": score,
    }
