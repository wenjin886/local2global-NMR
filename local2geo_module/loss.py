from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .constants import BOND_ORDERS, NONE


class ProjectionGeometryLoss(nn.Module):
    def __init__(
        self,
        edge_weight: float = 1.0,
        margin_weight: float = 0.1,
        margin: float = 1.5,
        sharpness_weight: float = 0.005,
        degree_weight: float = 0.1,
        valence_weight: float = 0.1,
        geometry_class_weight: float = 0.25,
        residual_weight: float = 1e-4,
        local_bond_weight: float = 1.0,
        local_angle_weight: float = 0.5,
        local_planar_weight: float = 0.1,
        local_clash_weight: float = 0.01,
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.margin_weight = margin_weight
        self.margin = margin
        self.sharpness_weight = sharpness_weight
        self.degree_weight = degree_weight
        self.valence_weight = valence_weight
        self.geometry_class_weight = geometry_class_weight
        self.residual_weight = residual_weight
        self.local_bond_weight = local_bond_weight
        self.local_angle_weight = local_angle_weight
        self.local_planar_weight = local_planar_weight
        self.local_clash_weight = local_clash_weight
        self.register_buffer("bond_orders", torch.tensor(BOND_ORDERS))

    @staticmethod
    def _upper_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["pair_mask"] & torch.triu(
            torch.ones_like(batch["pair_mask"], dtype=torch.bool), diagonal=1
        )

    @staticmethod
    def _balanced_mean(
        values: torch.Tensor,
        positive: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        groups = []
        for mask in (valid & positive, valid & ~positive):
            if mask.any():
                groups.append(values[mask].mean())
        return torch.stack(groups).mean() if groups else values.sum() * 0.0

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        clean_geometry_terms: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = outputs["projected_edge_logits"]
        probabilities = outputs["projected_edge_probabilities"]
        targets = batch["bond_types"].clamp_min(0)
        upper = self._upper_mask(batch)
        positive = targets.ne(NONE)

        edge_per_pair = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        edge = self._balanced_mean(edge_per_pair, positive, upper)

        target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        competing = logits.masked_fill(
            F.one_hot(targets, logits.size(-1)).bool(), -torch.inf
        ).max(dim=-1).values
        margin_values = F.relu(self.margin - target_logit + competing)
        margin = self._balanced_mean(margin_values, positive, upper)

        q = 1.0 - probabilities[..., NONE]
        eps = 1e-8
        presence_entropy = -(
            q.clamp(eps, 1.0 - eps) * q.clamp(eps, 1.0 - eps).log()
            + (1.0 - q).clamp(eps, 1.0 - eps)
            * (1.0 - q).clamp(eps, 1.0 - eps).log()
        )
        presence_sharp = self._balanced_mean(presence_entropy, positive, upper)
        conditional = probabilities[..., 1:] / q.unsqueeze(-1).clamp_min(eps)
        type_entropy = -(
            conditional.clamp_min(eps).log() * conditional
        ).sum(dim=-1)
        type_mask = upper & positive
        type_sharp = (
            type_entropy[type_mask].mean()
            if type_mask.any() else type_entropy.sum() * 0.0
        )
        sharpness = presence_sharp + type_sharp

        target_degree = (positive & batch["pair_mask"]).sum(dim=-1).to(q.dtype)
        degree = F.smooth_l1_loss(
            outputs["expected_degree"][batch["atom_mask"]],
            target_degree[batch["atom_mask"]],
        )
        clean_one_hot = F.one_hot(targets, logits.size(-1)).to(logits.dtype)
        target_valence = (
            (clean_one_hot * self.bond_orders).sum(dim=-1) * batch["pair_mask"]
        ).sum(dim=-1)
        predicted_total_valence = outputs["expected_valence"] + batch["hydrogen_counts"]
        target_total_valence = target_valence + batch["hydrogen_counts"]
        valence = F.smooth_l1_loss(
            predicted_total_valence[batch["atom_mask"]],
            target_total_valence[batch["atom_mask"]],
        )
        geometry_class = F.cross_entropy(
            outputs["geometry_logits"].reshape(-1, outputs["geometry_logits"].size(-1)),
            batch["geometry_classes"].reshape(-1),
            ignore_index=-100,
        )
        residual = self._balanced_mean(
            outputs["projection_delta"].square().mean(dim=-1), positive, upper
        )

        losses = {
            "edge": edge,
            "margin": margin,
            "sharpness": sharpness,
            "degree": degree,
            "valence": valence,
            "geometry_class": geometry_class,
            "residual": residual,
            "local_bond": clean_geometry_terms["bond"],
            "local_angle": clean_geometry_terms["angle"],
            "local_planar": clean_geometry_terms["planar"],
            "local_clash": clean_geometry_terms["clash"],
        }
        total = (
            self.edge_weight * losses["edge"]
            + self.margin_weight * losses["margin"]
            + self.sharpness_weight * losses["sharpness"]
            + self.degree_weight * losses["degree"]
            + self.valence_weight * losses["valence"]
            + self.geometry_class_weight * losses["geometry_class"]
            + self.residual_weight * losses["residual"]
            + self.local_bond_weight * losses["local_bond"]
            + self.local_angle_weight * losses["local_angle"]
            + self.local_planar_weight * losses["local_planar"]
            + self.local_clash_weight * losses["local_clash"]
        )
        losses["weighted"] = total
        return total, losses
