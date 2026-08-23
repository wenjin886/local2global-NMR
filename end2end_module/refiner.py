"""A compact spectrum-conditioned residual EGNN coordinate refiner."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ResidualEGNNLayer(nn.Module):
    """Dense E(3)-equivariant message passing with bounded coordinate steps."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_rbf: int,
        distance_cutoff: float,
        max_coordinate_step: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.distance_cutoff = float(distance_cutoff)
        self.max_coordinate_step = float(max_coordinate_step)
        centres = torch.linspace(0.0, distance_cutoff, num_rbf)
        spacing = float(centres[1] - centres[0]) if num_rbf > 1 else distance_cutoff
        self.register_buffer("rbf_centres", centres)
        self.rbf_gamma = 0.5 / max(spacing * spacing, 1e-8)
        message_input_dim = 2 * hidden_dim + edge_dim + num_rbf
        self.message = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate_gate = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.coordinate_gate.weight)
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

    def _radial(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.rbf_gamma
            * (distances.unsqueeze(-1) - self.rbf_centres).square()
        )

    def forward(
        self,
        node_features: torch.Tensor,
        coordinates: torch.Tensor,
        edge_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = atom_mask.size(1)
        diagonal = torch.eye(
            atoms, dtype=torch.bool, device=atom_mask.device
        )[None]
        pair_mask = (
            atom_mask[:, :, None] & atom_mask[:, None, :] & ~diagonal
        )
        relative = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        squared_distance = relative.square().sum(dim=-1).clamp_min(1e-12)
        distances = squared_distance.sqrt()
        left = node_features[:, :, None, :].expand(-1, -1, atoms, -1)
        right = node_features[:, None, :, :].expand_as(left)
        messages = self.message(
            torch.cat(
                [left + right, (left - right).abs(), edge_features, self._radial(distances)],
                dim=-1,
            )
        )
        pair_weights = pair_mask.to(messages.dtype).unsqueeze(-1)
        messages = messages * pair_weights
        neighbour_count = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)

        # All valid pairs communicate. The soft graph is a feature rather than
        # a cutoff, so a missed bond never makes two fragments invisible.
        direction = relative / distances.unsqueeze(-1).clamp_min(1e-4)
        coordinate_scale = torch.tanh(self.coordinate_gate(messages))
        coordinate_delta = (
            direction * coordinate_scale * pair_weights
        ).sum(dim=2) / neighbour_count.to(messages.dtype)
        coordinate_delta = self.max_coordinate_step * coordinate_delta
        coordinates = coordinates + coordinate_delta
        coordinates = coordinates * atom_mask.unsqueeze(-1)
        centre = coordinates.sum(dim=1, keepdim=True) / atom_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1).unsqueeze(-1)
        coordinates = (coordinates - centre) * atom_mask.unsqueeze(-1)

        aggregate = messages.sum(dim=2) / neighbour_count.to(messages.dtype)
        update = self.node_update(torch.cat([node_features, aggregate], dim=-1))
        node_features = self.node_norm(node_features + update)
        node_features = node_features * atom_mask.unsqueeze(-1)
        return node_features, coordinates


class SpectrumConditionedEGNNRefiner(nn.Module):
    """Refine generated coordinates without predicting or changing topology.

    The atom representation already contains atom-specific spectral context.
    A separately pooled H/C spectrum supplies a small global residual path.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        edge_dim: int = 5,
        num_layers: int = 4,
        num_rbf: int = 32,
        distance_cutoff: float = 8.0,
        max_coordinate_step: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.atom_projection = nn.Linear(input_dim, hidden_dim)
        self.spectrum_projection = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        # Start as an atom-feature-only refiner; the direct global spectral
        # bypass is learned in gradually instead of perturbing a checkpoint.
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.layers = nn.ModuleList(
            ResidualEGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_rbf=num_rbf,
                distance_cutoff=distance_cutoff,
                max_coordinate_step=max_coordinate_step / num_layers,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        graph_atom_features: torch.Tensor,
        h_peak_features: torch.Tensor,
        h_peak_mask: torch.Tensor,
        c_peak_features: torch.Tensor,
        c_peak_mask: torch.Tensor,
        edge_probabilities: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        initial_coordinates = coordinates.float()
        atom_state = self.atom_projection(graph_atom_features)
        global_spectrum = self.spectrum_projection(
            torch.cat(
                [
                    _masked_mean(h_peak_features, h_peak_mask),
                    _masked_mean(c_peak_features, c_peak_mask),
                ],
                dim=-1,
            )
        )
        broadcast_spectrum = global_spectrum[:, None, :].expand_as(atom_state)
        atom_state = self.fusion_norm(
            atom_state
            + self.fusion(torch.cat([atom_state, broadcast_spectrum], dim=-1))
        )
        atom_state = atom_state * atom_mask.unsqueeze(-1)
        refined = initial_coordinates
        for layer in self.layers:
            atom_state, refined = layer(
                atom_state,
                refined,
                edge_probabilities.to(atom_state.dtype),
                atom_mask,
            )
        displacement = (refined - initial_coordinates) * atom_mask.unsqueeze(-1)
        displacement_rms = torch.sqrt(
            displacement.square().sum(dim=-1).sum()
            / atom_mask.sum().clamp_min(1)
        )
        return {
            "coordinates": refined,
            "node_features": atom_state,
            "global_spectrum": global_spectrum,
            "displacement": displacement,
            "displacement_rms": displacement_rms,
        }
