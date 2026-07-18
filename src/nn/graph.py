from typing import Dict, Tuple

import torch
from torch import nn

from .attention import MaskedCrossAttentionBlock


class AtomInteractionBlock(nn.Module):
    """Bidirectional interaction between hydrogen and heavy-atom slots."""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.hydrogen_reads_heavy = MaskedCrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.heavy_reads_hydrogen = MaskedCrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
            self,
            atom_features: torch.Tensor,
            heavy_mask: torch.Tensor,
            hydrogen_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hydrogen_features, h_to_heavy_attention = self.hydrogen_reads_heavy(
            query=atom_features,
            context=atom_features,
            query_mask=hydrogen_mask,
            context_mask=heavy_mask,
        )
        mixed = torch.where(hydrogen_mask.unsqueeze(-1), hydrogen_features, atom_features)
        heavy_features, heavy_to_h_attention = self.heavy_reads_hydrogen(
            query=mixed,
            context=mixed,
            query_mask=heavy_mask,
            context_mask=hydrogen_mask,
        )
        mixed = torch.where(heavy_mask.unsqueeze(-1), heavy_features, mixed)
        return mixed, {
            "hydrogen_to_heavy": h_to_heavy_attention,
            "heavy_to_hydrogen": heavy_to_h_attention,
        }


class HeavyEdgeReadout(nn.Module):
    """Symmetric categorical readout for heavy-heavy bonds."""

    def __init__(
            self,
            hidden_dim: int,
            num_bond_types: int = 5,
            num_layers: int = 3,
    ):
        super().__init__()
        layers = []
        in_dim = 3 * hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_bond_types))
        self.mlp = nn.Sequential(*layers)

    def forward(
            self,
            atom_features: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left = atom_features[:, :, None, :]
        right = atom_features[:, None, :, :]
        pair_features = torch.cat([
            left + right,
            torch.abs(left - right),
            left * right,
        ], dim=-1)
        logits = self.mlp(pair_features)

        pair_mask = heavy_mask[:, :, None] & heavy_mask[:, None, :]
        diagonal = torch.eye(
            heavy_mask.size(1),
            dtype=torch.bool,
            device=heavy_mask.device,
        )[None, :, :]
        pair_mask = pair_mask & ~diagonal
        return logits, pair_mask


class FactorizedFragmentReadout(nn.Module):
    """Predict a categorical count for each neighbor-type/bond-type port."""

    def __init__(
            self,
            hidden_dim: int,
            num_fragment_types: int,
            max_fragment_count: int,
    ):
        super().__init__()
        self.num_fragment_types = num_fragment_types
        self.num_count_classes = max_fragment_count + 1
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_fragment_types * self.num_count_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.readout(features)
        return logits.view(
            *features.shape[:-1],
            self.num_fragment_types,
            self.num_count_classes,
        )


class HydrogenParentEnvironmentReadout(nn.Module):
    """Predict the element and factorized fragment of every H parent."""

    def __init__(
            self,
            hidden_dim: int,
            num_parent_types: int,
            num_fragment_types: int,
            max_fragment_count: int,
    ):
        super().__init__()
        self.parent_type_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_parent_types),
        )
        self.parent_fragment_readout = FactorizedFragmentReadout(
            hidden_dim=hidden_dim,
            num_fragment_types=num_fragment_types,
            max_fragment_count=max_fragment_count,
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.parent_type_readout(features), self.parent_fragment_readout(features)


class HydrogenAttachmentReadout(nn.Module):
    """Molecule-local H-to-heavy retrieval with projected cosine similarity."""

    def __init__(
            self,
            hidden_dim: int,
            attachment_dim: int = 128,
            temperature: float = 0.1,
    ):
        super().__init__()
        self.hydrogen_projection = nn.Linear(hidden_dim, attachment_dim)
        self.heavy_projection = nn.Linear(hidden_dim, attachment_dim)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(
            self,
            atom_features: torch.Tensor,
            hydrogen_mask: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hydrogen_embedding = nn.functional.normalize(
            self.hydrogen_projection(atom_features), dim=-1
        )
        heavy_embedding = nn.functional.normalize(
            self.heavy_projection(atom_features), dim=-1
        )
        temperature = self.log_temperature.exp().clamp(0.01, 1.0)
        logits = torch.matmul(
            hydrogen_embedding,
            heavy_embedding.transpose(1, 2),
        ) / temperature
        attachment_mask = hydrogen_mask[:, :, None] & heavy_mask[:, None, :]
        logits = logits.masked_fill(~attachment_mask, torch.finfo(logits.dtype).min)

        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities * attachment_mask.to(dtype=probabilities.dtype)
        normalizer = probabilities.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            normalizer > 0,
            probabilities / normalizer.clamp_min(torch.finfo(probabilities.dtype).eps),
            torch.zeros_like(probabilities),
        )
        return logits, probabilities, hydrogen_embedding, heavy_embedding


class FragmentConditioner(nn.Module):
    """Inject expected factorized fragment counts into heavy queries."""

    def __init__(self, num_fragment_types: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_fragment_types, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            atom_features: torch.Tensor,
            fragment_logits: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        count_values = torch.arange(
            fragment_logits.size(-1),
            dtype=fragment_logits.dtype,
            device=fragment_logits.device,
        )
        expected_counts = (
            torch.softmax(fragment_logits, dim=-1) * count_values
        ).sum(dim=-1)
        update = self.projection(expected_counts)
        conditioned = self.norm(atom_features + update)
        atom_features = torch.where(
            heavy_mask.unsqueeze(-1), conditioned, atom_features
        )
        return atom_features, expected_counts


class HydrogenContextAggregator(nn.Module):
    """Aggregate soft-assigned proton representations into heavy queries."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            atom_features: torch.Tensor,
            attachment_probabilities: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = self.value_projection(atom_features)
        context = torch.matmul(attachment_probabilities.transpose(1, 2), values)
        assigned_count = attachment_probabilities.sum(dim=1, keepdim=False).unsqueeze(-1)
        normalized_context = context / assigned_count.clamp_min(1.0)
        update = self.update(torch.cat([
            atom_features,
            normalized_context,
            assigned_count,
        ], dim=-1))
        refined = self.norm(atom_features + update)
        atom_features = torch.where(heavy_mask.unsqueeze(-1), refined, atom_features)
        return atom_features, context, assigned_count.squeeze(-1)
