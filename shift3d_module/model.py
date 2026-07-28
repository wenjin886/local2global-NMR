"""A dependency-light SchNet encoder with separate 1H and 13C heads."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class GaussianRBF(nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        centres = torch.linspace(0.0, cutoff, count)
        spacing = float(centres[1] - centres[0]) if count > 1 else cutoff
        self.register_buffer("centres", centres)
        self.gamma = 0.5 / max(spacing * spacing, 1e-8)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distances[..., None] - self.centres).square()
        )


class SchNetInteraction(nn.Module):
    def __init__(self, hidden_dim: int, rbf_count: int) -> None:
        super().__init__()
        self.atom_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.filter_network = nn.Sequential(
            nn.Linear(rbf_count, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        radial: torch.Tensor,
        pair_mask: torch.Tensor,
        cutoff_values: torch.Tensor,
    ) -> torch.Tensor:
        filters = self.filter_network(radial)
        filters = filters * cutoff_values[..., None]
        filters = filters * pair_mask[..., None]
        neighbour_features = self.atom_projection(features)[:, None, :, :]
        messages = (filters * neighbour_features).sum(dim=2)
        return self.normalization(features + self.update_network(messages))


class SchNetShiftModel(nn.Module):
    """Predict raw per-atom shifts using only atom types and coordinates.

    Distances make the representation E(3)-invariant, including reflection.
    The two heads avoid forcing proton and carbon ppm scales through one final
    readout while retaining a shared geometric encoder.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_interactions: int = 6,
        num_rbf: int = 64,
        cutoff: float = 6.0,
        max_atomic_number: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.embedding = nn.Embedding(
            max_atomic_number + 1, hidden_dim, padding_idx=0
        )
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.interactions = nn.ModuleList(
            [
                SchNetInteraction(hidden_dim, num_rbf)
                for _ in range(num_interactions)
            ]
        )

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.hydrogen_head = head()
        self.carbon_head = head()

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        displacement = positions[:, :, None, :] - positions[:, None, :, :]
        distances = displacement.square().sum(dim=-1).clamp_min(1e-12).sqrt()
        atoms = atom_mask.size(1)
        diagonal = torch.eye(
            atoms, device=atom_mask.device, dtype=torch.bool
        )[None]
        pair_mask = (
            atom_mask[:, :, None]
            & atom_mask[:, None, :]
            & ~diagonal
            & distances.lt(self.cutoff)
        )
        cutoff_values = 0.5 * (
            torch.cos(math.pi * distances / self.cutoff) + 1.0
        )
        cutoff_values = torch.where(
            distances.lt(self.cutoff), cutoff_values, 0.0
        )
        radial = self.rbf(distances)
        features = self.embedding(atomic_numbers)
        for interaction in self.interactions:
            features = interaction(
                features, radial, pair_mask, cutoff_values
            )
            features = features * atom_mask[..., None]
        return {
            "h_shifts": self.hydrogen_head(features).squeeze(-1),
            "c_shifts": self.carbon_head(features).squeeze(-1),
        }
