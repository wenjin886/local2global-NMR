from typing import Dict, Mapping, Tuple

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


class HydrogenAttachmentReadout(nn.Module):
    """Predict one heavy-atom attachment distribution for every H slot."""

    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = 4 * hidden_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
            self,
            atom_features: torch.Tensor,
            hydrogen_mask: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hydrogen = atom_features[:, :, None, :]
        heavy = atom_features[:, None, :, :]
        pair_features = torch.cat([
            hydrogen.expand(-1, -1, atom_features.size(1), -1),
            heavy.expand(-1, atom_features.size(1), -1, -1),
            hydrogen * heavy,
            torch.abs(hydrogen - heavy),
        ], dim=-1)
        logits = self.mlp(pair_features).squeeze(-1)
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
        return logits, probabilities


class ElementWiseLocalReadout(nn.Module):
    """Separate local-environment classifiers for each configured element."""

    def __init__(self, hidden_dim: int, vocab_sizes: Mapping[int, int]):
        super().__init__()
        self.readouts = nn.ModuleDict({
            str(int(atomic_number)): nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, int(vocab_size)),
            )
            for atomic_number, vocab_size in vocab_sizes.items()
            if int(vocab_size) > 0
        })

    def forward(
            self,
            atom_features: torch.Tensor,
            atom_types: torch.Tensor,
            atom_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        outputs = {}
        for atomic_number, readout in self.readouts.items():
            mask = atom_mask & atom_types.eq(int(atomic_number))
            indices = mask.nonzero(as_tuple=False)
            features = atom_features[mask]
            outputs[atomic_number] = {
                "indices": indices,
                "logits": readout(features),
            }
        return outputs
