from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .constants import GEOMETRY_COSINES, NONE


@torch.no_grad()
def local2geo_metrics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    target_lengths: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    probabilities = outputs["projected_edge_probabilities"]
    prediction = probabilities.argmax(dim=-1)
    target = batch["bond_types"].clamp_min(0)
    upper = batch["pair_mask"] & torch.triu(
        torch.ones_like(batch["pair_mask"], dtype=torch.bool), diagonal=1
    )
    pred_edge = prediction.ne(NONE) & upper
    target_edge = target.ne(NONE) & upper
    tp = (pred_edge & target_edge).sum().float()
    fp = (pred_edge & ~target_edge).sum().float()
    fn = (~pred_edge & target_edge).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    edge_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    bond_type_accuracy = (
        prediction[target_edge].eq(target[target_edge]).float().mean()
        if target_edge.any() else tp * 0.0 + 1.0
    )

    exact = []
    for index in range(target.size(0)):
        valid = upper[index]
        exact.append(prediction[index, valid].eq(target[index, valid]).all().float())
    graph_exact = torch.stack(exact).mean()
    entropy = -(
        probabilities.clamp_min(1e-8).log() * probabilities
    ).sum(dim=-1)
    entropy = entropy[upper].mean()

    positions = outputs["coordinates"]
    vector = positions[:, None, :, :] - positions[:, :, None, :]
    distance = torch.sqrt(vector.square().sum(dim=-1) + 1e-8)
    bond_mae = (
        (distance[target_edge] - target_lengths[target_edge]).abs().mean()
        if target_edge.any() else distance.sum() * 0.0
    )

    cosine_targets = torch.tensor(
        GEOMETRY_COSINES, device=positions.device, dtype=positions.dtype
    )
    angle_errors = []
    for batch_index in range(target.size(0)):
        size = int(batch["atom_mask"][batch_index].sum())
        for center in range(size):
            neighbors = torch.where(target_edge[batch_index, center] | (
                target[batch_index, center].ne(NONE) & batch["pair_mask"][batch_index, center]
            ))[0]
            # target_edge is upper-triangular; the second term restores both directions.
            neighbors = torch.where(
                target[batch_index, center].ne(NONE)
                & batch["pair_mask"][batch_index, center]
            )[0]
            if neighbors.numel() < 2:
                continue
            vectors = positions[batch_index, neighbors] - positions[batch_index, center]
            vectors = F.normalize(vectors, dim=-1)
            matrix = vectors @ vectors.T
            pairs = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
            actual = torch.acos(matrix[pairs].clamp(-1.0, 1.0))
            geometry_index = batch["geometry_classes"][batch_index, center]
            target_angle = torch.acos(cosine_targets[geometry_index].clamp(-1.0, 1.0))
            angle_errors.append((actual - target_angle).abs())
    angle_mae = (
        torch.cat(angle_errors).mean()
        if angle_errors else bond_mae * 0.0
    )

    adjacency = target.ne(NONE) & batch["pair_mask"]
    two_hop = torch.bmm(adjacency.float(), adjacency.float()).gt(0)
    excluded = adjacency | two_hop
    nonlocal_mask = upper & ~excluded
    minimum = 0.62 * (
        batch["vdw_radii"][:, :, None] + batch["vdw_radii"][:, None, :]
    )
    clash_rate = (
        distance[nonlocal_mask].lt(minimum[nonlocal_mask]).float().mean()
        if nonlocal_mask.any() else bond_mae * 0.0
    )
    local_score = torch.exp(
        -bond_mae / 0.15 - angle_mae / 0.35 - 5.0 * clash_rate
    )
    score = torch.sqrt((edge_f1 * local_score).clamp_min(0.0))
    geometry_accuracy = outputs["geometry_logits"].argmax(dim=-1).eq(
        batch["geometry_classes"]
    )[batch["atom_mask"]].float().mean()
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": edge_f1,
        "bond_type_accuracy": bond_type_accuracy,
        "graph_exact": graph_exact,
        "edge_entropy": entropy,
        "geometry_class_accuracy": geometry_accuracy,
        "bond_mae_angstrom": bond_mae,
        "angle_mae_radian": angle_mae,
        "clash_rate": clash_rate,
        "local_score": local_score,
        "score": score,
    }
