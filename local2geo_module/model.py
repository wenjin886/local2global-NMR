from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .constants import (
    BOND_ORDERS,
    GEOMETRY_NAMES,
    NONE,
    NUM_BOND_TYPES,
    SINGLE,
)
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
    """Project heavy bonds and H attachments, then assemble all-atom edges."""

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
        self.attachment_update = nn.Sequential(
            nn.Linear(3 * hidden_dim + 2, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, 1),
        )
        nn.init.zeros_(self.attachment_update[-1].weight)
        nn.init.zeros_(self.attachment_update[-1].bias)
        self.all_atom_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.all_atom_norm = nn.LayerNorm(hidden_dim)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(GEOMETRY_NAMES)),
        )
        self.geometry_temperature = geometry_temperature
        self.use_formal_charges = use_formal_charges
        self.register_buffer("bond_orders", torch.tensor(BOND_ORDERS))

    @staticmethod
    def _mask_heavy_logits(
        logits: torch.Tensor, heavy_pair_mask: torch.Tensor
    ) -> torch.Tensor:
        masked = logits.masked_fill(~heavy_pair_mask.unsqueeze(-1), -20.0)
        masked = masked.clone()
        masked[..., NONE] = torch.where(
            heavy_pair_mask,
            masked[..., NONE],
            torch.full_like(masked[..., NONE], 20.0),
        )
        return masked

    @staticmethod
    def _attachment_probabilities(
        logits: torch.Tensor,
        attachment_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked = logits.masked_fill(~attachment_mask, -20.0)
        probabilities = torch.softmax(masked, dim=-1)
        probabilities = probabilities * attachment_mask.to(probabilities.dtype)
        normalizer = probabilities.sum(dim=-1, keepdim=True)
        return torch.where(
            normalizer > 0,
            probabilities / normalizer.clamp_min(1e-8),
            torch.zeros_like(probabilities),
        )

    @staticmethod
    def _assemble_all_atom_probabilities(
        heavy_probabilities: torch.Tensor,
        attachment_probabilities: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        attachment_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        shape = (*pair_mask.shape, NUM_BOND_TYPES)
        probabilities = torch.zeros(
            shape,
            dtype=heavy_probabilities.dtype,
            device=heavy_probabilities.device,
        )
        probabilities[..., NONE] = 1.0
        probabilities = torch.where(
            heavy_pair_mask.unsqueeze(-1),
            heavy_probabilities,
            probabilities,
        )
        h_single = attachment_probabilities * attachment_mask
        h_single = h_single + h_single.transpose(1, 2)
        probabilities[..., SINGLE] = torch.where(
            h_single.gt(0), h_single, probabilities[..., SINGLE]
        )
        probabilities[..., NONE] = torch.where(
            h_single.gt(0), 1.0 - h_single, probabilities[..., NONE]
        )
        probabilities = probabilities * pair_mask.unsqueeze(-1).to(
            probabilities.dtype
        )
        probabilities[..., NONE] = torch.where(
            pair_mask,
            probabilities[..., NONE],
            torch.ones_like(probabilities[..., NONE]),
        )
        return probabilities

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        attachment_mask: torch.Tensor,
        noisy_heavy_edge_logits: torch.Tensor,
        noisy_h_attachment_logits: torch.Tensor,
        hydrogen_counts: torch.Tensor,
        formal_charges: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        raw = self._mask_heavy_logits(
            0.5 * (
                noisy_heavy_edge_logits
                + noisy_heavy_edge_logits.transpose(1, 2)
            ),
            heavy_pair_mask,
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
                heavy_pair_mask,
                hydrogen_counts,
                charges,
            )
            total_delta = total_delta + delta
            current = self._mask_heavy_logits(
                raw + total_delta, heavy_pair_mask
            )
        heavy_probabilities = torch.softmax(current, dim=-1)

        left, right = node[:, :, None, :], node[:, None, :, :]
        attachment_features = torch.cat([
            left + right,
            torch.abs(left - right),
            left * right,
            noisy_h_attachment_logits.unsqueeze(-1),
            noisy_h_attachment_logits.sigmoid().unsqueeze(-1),
        ], dim=-1)
        attachment_delta = self.attachment_update(
            attachment_features
        ).squeeze(-1)
        attachment_logits = (
            noisy_h_attachment_logits + attachment_delta
        ).masked_fill(~attachment_mask, -20.0)
        attachment_probabilities = self._attachment_probabilities(
            attachment_logits, attachment_mask
        )
        probabilities = self._assemble_all_atom_probabilities(
            heavy_probabilities,
            attachment_probabilities,
            heavy_pair_mask,
            attachment_mask,
            pair_mask,
        )
        q = (1.0 - probabilities[..., NONE]) * pair_mask
        degree = q.sum(dim=-1)
        valence = (
            (probabilities * self.bond_orders).sum(dim=-1) * pair_mask
        ).sum(dim=-1)
        neighbor = torch.bmm(q, node) / degree.clamp_min(1.0).unsqueeze(-1)
        node = self.all_atom_norm(
            node + self.all_atom_update(torch.cat([
                node,
                neighbor,
                degree.unsqueeze(-1),
                valence.unsqueeze(-1),
            ], dim=-1))
        ) * atom_mask.unsqueeze(-1)
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
            "raw_heavy_edge_logits": raw,
            "projected_heavy_edge_logits": current,
            "projected_heavy_edge_probabilities": heavy_probabilities,
            "heavy_projection_delta": total_delta,
            "raw_h_attachment_logits": noisy_h_attachment_logits,
            "projected_h_attachment_logits": attachment_logits,
            "projected_h_attachment_probabilities": attachment_probabilities,
            "attachment_projection_delta": attachment_delta,
            "projected_edge_logits": probabilities.clamp_min(1e-8).log(),
            "projected_edge_probabilities": probabilities,
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
        max_num_atoms: int = 256,
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
        noisy_graph: Dict[str, torch.Tensor],
        differentiable_relaxation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.projector(
            atomic_numbers=batch["atomic_numbers"],
            atom_mask=batch["atom_mask"],
            pair_mask=batch["pair_mask"],
            heavy_pair_mask=batch["heavy_pair_mask"],
            attachment_mask=batch["attachment_mask"],
            noisy_heavy_edge_logits=noisy_graph["heavy_edge_logits"],
            noisy_h_attachment_logits=noisy_graph["h_attachment_logits"],
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
            reduction="mean",
        )
