from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .constants import BOND_ORDERS, GEOMETRY_NAMES, NONE, NUM_BOND_TYPES
from .geometry import DifferentiableLocalRelaxation


class ProjectionLayer(nn.Module):
    def __init__(self, hidden_dim: int, pair_hidden_dim: int):
        super().__init__()
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + NUM_BOND_TYPES + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.pair_update = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2 * NUM_BOND_TYPES, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, NUM_BOND_TYPES),
        )
        nn.init.zeros_(self.pair_update[-1].weight)
        nn.init.zeros_(self.pair_update[-1].bias)
        self.register_buffer("bond_orders", torch.tensor(BOND_ORDERS))

    def forward(
        self,
        node_features: torch.Tensor,
        raw_logits: torch.Tensor,
        current_logits: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        hydrogen_counts: torch.Tensor,
        formal_charges: torch.Tensor,
    ):
        probabilities = torch.softmax(current_logits, dim=-1)
        q = (1.0 - probabilities[..., NONE]) * pair_mask
        degree = q.sum(dim=-1)
        valence = (probabilities * self.bond_orders).sum(dim=-1)
        valence = (valence * pair_mask).sum(dim=-1)
        neighbor = torch.bmm(q, node_features) / degree.clamp_min(1.0).unsqueeze(-1)
        bond_histogram = (probabilities * pair_mask.unsqueeze(-1)).sum(dim=2)
        node_input = torch.cat([
            node_features,
            neighbor,
            bond_histogram,
            degree.unsqueeze(-1),
            valence.unsqueeze(-1),
            hydrogen_counts.unsqueeze(-1),
            formal_charges.unsqueeze(-1),
        ], dim=-1)
        updated = self.node_norm(node_features + self.node_update(node_input))
        updated = updated * atom_mask.unsqueeze(-1)
        left, right = updated[:, :, None, :], updated[:, None, :, :]
        pair_features = torch.cat([
            left + right,
            torch.abs(left - right),
            left * right,
            raw_logits,
            current_logits,
        ], dim=-1)
        delta = self.pair_update(pair_features)
        delta = 0.5 * (delta + delta.transpose(1, 2))
        return updated, delta


class SoftGraphProjector(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        pair_hidden_dim: int = 128,
        num_layers: int = 3,
        max_atomic_number: int = 118,
        geometry_temperature: float = 1.0,
        use_formal_charges: bool = False,
    ):
        super().__init__()
        self.element_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.charge_projection = nn.Linear(1, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            ProjectionLayer(hidden_dim, pair_hidden_dim) for _ in range(num_layers)
        ])
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(GEOMETRY_NAMES)),
        )
        self.geometry_temperature = geometry_temperature
        self.use_formal_charges = use_formal_charges
        self.register_buffer("bond_orders", torch.tensor(BOND_ORDERS))

    @staticmethod
    def _mask_logits(logits: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~pair_mask.unsqueeze(-1), -20.0)
        none = torch.where(
            pair_mask,
            masked[..., NONE],
            torch.full_like(masked[..., NONE], 20.0),
        )
        masked = masked.clone()
        masked[..., NONE] = none
        return masked

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        noisy_edge_logits: torch.Tensor,
        hydrogen_counts: torch.Tensor,
        formal_charges: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raw = self._mask_logits(
            0.5 * (noisy_edge_logits + noisy_edge_logits.transpose(1, 2)),
            pair_mask,
        )
        charges = (
            formal_charges
            if self.use_formal_charges else torch.zeros_like(formal_charges)
        )
        node = (
            self.element_embedding(atomic_numbers)
            + self.charge_projection(charges.unsqueeze(-1))
        ) * atom_mask.unsqueeze(-1)
        current = raw
        total_delta = torch.zeros_like(raw)
        for layer in self.layers:
            node, delta = layer(
                node,
                raw,
                current,
                atom_mask,
                pair_mask,
                hydrogen_counts,
                charges,
            )
            total_delta = total_delta + delta
            current = self._mask_logits(raw + total_delta, pair_mask)
        probabilities = torch.softmax(current, dim=-1)
        q = (1.0 - probabilities[..., NONE]) * pair_mask
        degree = q.sum(dim=-1)
        valence = ((probabilities * self.bond_orders).sum(dim=-1) * pair_mask).sum(-1)
        geometry_logits = self.geometry_head(torch.cat([
            node,
            degree.unsqueeze(-1),
            valence.unsqueeze(-1),
            hydrogen_counts.unsqueeze(-1),
        ], dim=-1))
        geometry_probabilities = torch.softmax(
            geometry_logits / self.geometry_temperature, dim=-1
        )
        return {
            "raw_edge_logits": raw,
            "projected_edge_logits": current,
            "projected_edge_probabilities": probabilities,
            "projection_delta": total_delta,
            "node_features": node,
            "geometry_logits": geometry_logits,
            "geometry_probabilities": geometry_probabilities,
            "expected_degree": degree,
            "expected_valence": valence,
        }


class LearnedCoordinateSeed(nn.Module):
    """Predict a canonical seed; 2D-prior losses, not 3D labels, train it."""

    def __init__(
        self,
        hidden_dim: int = 128,
        max_num_atoms: int = 192,
        coordinate_scale: float = 3.0,
    ):
        super().__init__()
        self.slot_embedding = nn.Embedding(max_num_atoms, hidden_dim)
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.coordinate_scale = coordinate_scale

    def forward(
        self,
        node_features: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        atoms = node_features.size(1)
        if atoms > self.slot_embedding.num_embeddings:
            raise ValueError(
                f"Batch has {atoms} atoms but seed supports only "
                f"{self.slot_embedding.num_embeddings}"
            )
        slots = torch.arange(atoms, device=node_features.device)
        coordinates = self.coordinate_scale * torch.tanh(
            self.readout(node_features + self.slot_embedding(slots)[None])
        )
        coordinates = coordinates * atom_mask.unsqueeze(-1)
        count = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(coordinates.dtype)
        center = coordinates.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        return (coordinates - center) * atom_mask.unsqueeze(-1)


class Local2GeoModel(nn.Module):
    def __init__(
        self,
        projector: SoftGraphProjector,
        coordinate_seed: LearnedCoordinateSeed,
        relaxation: DifferentiableLocalRelaxation,
        geometry_edge_temperature: float = 0.7,
    ):
        super().__init__()
        self.projector = projector
        self.coordinate_seed = coordinate_seed
        self.relaxation = relaxation
        self.geometry_edge_temperature = geometry_edge_temperature

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_edge_logits: torch.Tensor,
        differentiable_relaxation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.projector(
            atomic_numbers=batch["atomic_numbers"],
            atom_mask=batch["atom_mask"],
            pair_mask=batch["pair_mask"],
            noisy_edge_logits=noisy_edge_logits,
            hydrogen_counts=batch["hydrogen_counts"],
            formal_charges=batch["formal_charges"],
        )
        geometry_probabilities = torch.softmax(
            outputs["projected_edge_logits"] / self.geometry_edge_temperature,
            dim=-1,
        )
        seed_coordinates = self.coordinate_seed(
            outputs["node_features"], batch["atom_mask"]
        )
        coordinates, predicted_terms = self.relaxation(
            probabilities=geometry_probabilities,
            geometry_probabilities=outputs["geometry_probabilities"],
            atom_mask=batch["atom_mask"],
            pair_mask=batch["pair_mask"],
            covalent_radii=batch["covalent_radii"],
            vdw_radii=batch["vdw_radii"],
            initial_positions=seed_coordinates,
            differentiable=differentiable_relaxation,
        )
        outputs.update({
            "geometry_edge_probabilities": geometry_probabilities,
            "seed_coordinates": seed_coordinates,
            "coordinates": coordinates,
            "predicted_geometry_terms": predicted_terms,
        })
        return outputs

    def clean_geometry_terms(
        self,
        batch: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        clean_types = batch["bond_types"].clamp_min(0)
        clean_probabilities = torch.nn.functional.one_hot(
            clean_types, num_classes=NUM_BOND_TYPES
        ).to(coordinates.dtype)
        clean_geometry = torch.nn.functional.one_hot(
            batch["geometry_classes"].clamp_min(0),
            num_classes=len(GEOMETRY_NAMES),
        ).to(coordinates.dtype)
        clean_geometry = clean_geometry * batch["atom_mask"].unsqueeze(-1)
        return self.relaxation.terms(
            coordinates,
            clean_probabilities,
            clean_geometry,
            batch["atom_mask"],
            batch["pair_mask"],
            batch["covalent_radii"],
            batch["vdw_radii"],
        )
