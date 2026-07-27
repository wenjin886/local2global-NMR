"""Learned residual correction of a dense soft molecular graph.

The network never selects neighbours. Every pair remains in the computation,
and corrected logits retain an explicit identity path from the incoming graph:

    corrected_logits = raw_logits + bounded_residual

Consequently a downstream geometry/NMR loss always has a direct derivative
with respect to the NMR-to-graph logits, even before or after pretraining.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .constants import NUM_BOND_TYPES


def _masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    probabilities = torch.softmax(
        logits.masked_fill(~mask, -20.0), dim=dim
    ) * mask.to(logits.dtype)
    return probabilities / probabilities.sum(
        dim=dim, keepdim=True
    ).clamp_min(1e-8)


class DenseMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        pair_dim = 2 * hidden_dim + NUM_BOND_TYPES
        self.message = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        pair_probabilities: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        left = nodes[:, :, None, :].expand(-1, -1, nodes.size(1), -1)
        right = nodes[:, None, :, :].expand_as(left)
        features = torch.cat(
            [left + right, (left - right).abs(), pair_probabilities],
            dim=-1,
        )
        mask = pair_mask.unsqueeze(-1).to(nodes.dtype)
        messages = self.message(features)
        gates = torch.sigmoid(self.gate(features))
        count = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1).sqrt()
        aggregate = (messages * gates * mask).sum(dim=2) / count
        update = self.update(torch.cat([nodes, aggregate], dim=-1))
        return self.norm(nodes + update)


class SoftTopologyPrior(nn.Module):
    """Dense GNN that corrects graph logits and predicts local soft bounds."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.05,
        max_atomic_number: int = 118,
        residual_scale: float = 6.0,
        min_distance_ratio: float = 1.0,
        max_distance_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 16:
            raise ValueError("hidden_dim must be at least 16")
        self.residual_scale = residual_scale
        self.min_distance_ratio = min_distance_ratio
        self.max_distance_ratio = max_distance_ratio
        self.atom_embedding = nn.Embedding(
            max_atomic_number + 1, hidden_dim
        )
        self.charge_embedding = nn.Embedding(7, hidden_dim)
        self.hydrogen_embedding = nn.Embedding(2, hidden_dim)
        self.local_projection = nn.Linear(NUM_BOND_TYPES, hidden_dim)
        self.layers = nn.ModuleList([
            DenseMessageLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        symmetric_dim = 3 * hidden_dim + NUM_BOND_TYPES
        self.pair_trunk = nn.Sequential(
            nn.Linear(symmetric_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_residual_head = nn.Linear(hidden_dim, NUM_BOND_TYPES)
        self.attachment_residual_head = nn.Linear(hidden_dim, 1)
        self.one_three_head = nn.Linear(hidden_dim, 1)
        self.one_four_head = nn.Linear(hidden_dim, 1)
        self.one_three_ratio_head = nn.Linear(hidden_dim, 1)
        self.one_four_ratio_head = nn.Linear(hidden_dim, 1)
        self.ring_head = nn.Linear(hidden_dim, 1)
        self.conjugated_head = nn.Linear(hidden_dim, 1)
        self.torsion_head = nn.Linear(hidden_dim, 3)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )

    @staticmethod
    def _full_pair_probabilities(
        raw_heavy_logits: torch.Tensor,
        raw_attachment_logits: torch.Tensor,
        pair_mask: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        attachment_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        heavy = torch.softmax(
            raw_heavy_logits.masked_fill(
                ~heavy_pair_mask.unsqueeze(-1), -20.0
            ),
            dim=-1,
        )
        attachment = _masked_softmax(
            raw_attachment_logits, attachment_mask
        )
        probabilities = torch.zeros(
            (*pair_mask.shape, NUM_BOND_TYPES),
            dtype=raw_heavy_logits.dtype,
            device=raw_heavy_logits.device,
        )
        probabilities[..., 0] = 1.0
        probabilities = torch.where(
            heavy_pair_mask.unsqueeze(-1), heavy, probabilities
        )
        h_single = attachment + attachment.transpose(1, 2)
        h_pair_mask = attachment_mask | attachment_mask.transpose(1, 2)
        probabilities[..., 1] = torch.where(
            h_pair_mask, h_single, probabilities[..., 1]
        )
        probabilities[..., 0] = torch.where(
            h_pair_mask, 1.0 - h_single, probabilities[..., 0]
        )
        probabilities = probabilities * pair_mask.unsqueeze(-1)
        probabilities[..., 0] = torch.where(
            pair_mask,
            probabilities[..., 0],
            torch.ones_like(probabilities[..., 0]),
        )
        return {
            "heavy": heavy,
            "attachment": attachment,
            "full": probabilities,
        }

    def _distance_ratio(self, logits: torch.Tensor) -> torch.Tensor:
        span = self.max_distance_ratio - self.min_distance_ratio
        return self.min_distance_ratio + span * torch.sigmoid(logits)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        formal_charges: torch.Tensor,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        hydrogen_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        attachment_mask: torch.Tensor,
        raw_heavy_edge_logits: torch.Tensor,
        raw_h_attachment_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raw = self._full_pair_probabilities(
            raw_heavy_edge_logits,
            raw_h_attachment_logits,
            pair_mask,
            heavy_pair_mask,
            attachment_mask,
        )
        local = raw["full"].sum(dim=2)
        charge_index = formal_charges.round().long().clamp(-3, 3) + 3
        nodes = (
            self.atom_embedding(atomic_numbers.clamp_min(0))
            + self.charge_embedding(charge_index)
            + self.hydrogen_embedding(hydrogen_mask.long())
            + self.local_projection(local)
        )
        nodes = nodes * atom_mask.unsqueeze(-1)
        for layer in self.layers:
            nodes = layer(nodes, raw["full"], pair_mask)
            nodes = nodes * atom_mask.unsqueeze(-1)

        left = nodes[:, :, None, :]
        right = nodes[:, None, :, :]
        pair_features = torch.cat(
            [
                left + right,
                (left - right).abs(),
                left * right,
                raw["full"],
            ],
            dim=-1,
        )
        pair_hidden = self.pair_trunk(pair_features)

        # Both residual heads are additive. The derivative of every corrected
        # valid logit with respect to its raw counterpart therefore contains
        # an exact identity term, independently of the learned GNN path.
        edge_residual = self.residual_scale * torch.tanh(
            self.edge_residual_head(pair_hidden)
        )
        edge_residual = 0.5 * (
            edge_residual + edge_residual.transpose(1, 2)
        )
        corrected_heavy = raw_heavy_edge_logits + edge_residual
        corrected_heavy = corrected_heavy.masked_fill(
            ~heavy_pair_mask.unsqueeze(-1), -20.0
        )
        corrected_heavy[..., 0] = torch.where(
            heavy_pair_mask,
            corrected_heavy[..., 0],
            torch.full_like(corrected_heavy[..., 0], 20.0),
        )

        attachment_residual = self.residual_scale * torch.tanh(
            self.attachment_residual_head(pair_hidden).squeeze(-1)
        )
        corrected_attachment = (
            raw_h_attachment_logits + attachment_residual
        ).masked_fill(~attachment_mask, -20.0)

        one_three_logits = self.one_three_head(pair_hidden).squeeze(-1)
        one_four_logits = self.one_four_head(pair_hidden).squeeze(-1)
        ring_logits = self.ring_head(pair_hidden).squeeze(-1)
        conjugated_logits = self.conjugated_head(pair_hidden).squeeze(-1)
        torsion_logits = self.torsion_head(pair_hidden)
        # Symmetry is architectural rather than imposed by selecting one
        # triangle, so gradients remain available on both pair directions.
        one_three_logits = 0.5 * (
            one_three_logits + one_three_logits.transpose(1, 2)
        )
        one_four_logits = 0.5 * (
            one_four_logits + one_four_logits.transpose(1, 2)
        )
        ring_logits = 0.5 * (
            ring_logits + ring_logits.transpose(1, 2)
        )
        conjugated_logits = 0.5 * (
            conjugated_logits + conjugated_logits.transpose(1, 2)
        )
        torsion_logits = 0.5 * (
            torsion_logits + torsion_logits.transpose(1, 2)
        )

        ratio_13 = self._distance_ratio(
            self.one_three_ratio_head(pair_hidden).squeeze(-1)
        )
        ratio_14 = self._distance_ratio(
            self.one_four_ratio_head(pair_hidden).squeeze(-1)
        )
        ratio_13 = 0.5 * (ratio_13 + ratio_13.transpose(1, 2))
        ratio_14 = 0.5 * (ratio_14 + ratio_14.transpose(1, 2))

        return {
            "corrected_heavy_edge_logits": corrected_heavy,
            "corrected_h_attachment_logits": corrected_attachment,
            "edge_residual": edge_residual,
            "attachment_residual": attachment_residual,
            "geometry_logits": self.geometry_head(nodes),
            "one_three_logits": one_three_logits,
            "one_four_logits": one_four_logits,
            "one_three_probability": torch.sigmoid(one_three_logits),
            "one_four_probability": torch.sigmoid(one_four_logits),
            "one_three_distance_ratio": ratio_13,
            "one_four_distance_ratio": ratio_14,
            "ring_logits": ring_logits,
            "conjugated_logits": conjugated_logits,
            "torsion_logits": torsion_logits,
            "node_features": nodes,
        }
