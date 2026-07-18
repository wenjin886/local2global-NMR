from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from src.data.constants import BOND_TYPE_CANDIDATES, HEAVY_ATOM_TYPES
from src.nn.attention import MaskedCrossAttentionBlock, MaskedSelfAttentionEncoder
from src.nn.embedding import AtomSlotEmbedding, NMRPeakEmbedding
from src.nn.graph import (
    AtomInteractionBlock,
    FactorizedFragmentReadout,
    FragmentConditioner,
    HeavyEdgeReadout,
    HydrogenContextAggregator,
    HydrogenAttachmentReadout,
    HydrogenParentEnvironmentReadout,
)


class NMRToGraph(nn.Module):
    """Predict molecular connectivity from atom slots and unassigned NMR peaks.

    All atoms share the same embedding and spectrum cross-attention.  Features
    are split into hydrogen and heavy-atom subsets only after NMR conditioning.
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            num_heads: int = 8,
            num_spectrum_layers: int = 3,
            num_atom_spectrum_layers: int = 3,
            num_atom_interaction_layers: int = 2,
            num_fourier_features: int = 64,
            max_atomic_number: int = 100,
            max_num_atoms: int = 192,
            num_bond_types: int = 5,
            fragment_candidates: Sequence[str] = BOND_TYPE_CANDIDATES,
            parent_atom_types: Sequence[int] = HEAVY_ATOM_TYPES,
            max_fragment_count: int = 4,
            attachment_dim: int = 128,
            attachment_temperature: float = 0.1,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.atom_embedding = AtomSlotEmbedding(
            hidden_dim=hidden_dim,
            max_atomic_number=max_atomic_number,
            max_num_atoms=max_num_atoms,
            dropout=dropout,
        )
        self.peak_embedding = NMRPeakEmbedding(
            hidden_dim=hidden_dim,
            num_fourier_features=num_fourier_features,
            dropout=dropout,
        )
        self.spectrum_encoder = MaskedSelfAttentionEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_spectrum_layers,
            dropout=dropout,
        )
        self.atom_spectrum_layers = nn.ModuleList([
            MaskedCrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_atom_spectrum_layers)
        ])
        self.heavy_query_embedding = nn.Embedding(max_num_atoms, hidden_dim)
        self.heavy_query_decoder = MaskedCrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.atom_interaction_layers = nn.ModuleList([
            AtomInteractionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_atom_interaction_layers)
        ])
        num_fragment_types = len(fragment_candidates)
        self.fragment_readout = FactorizedFragmentReadout(
            hidden_dim=hidden_dim,
            num_fragment_types=num_fragment_types,
            max_fragment_count=max_fragment_count,
        )
        self.h_parent_environment_readout = HydrogenParentEnvironmentReadout(
            hidden_dim=hidden_dim,
            num_parent_types=len(parent_atom_types),
            num_fragment_types=num_fragment_types,
            max_fragment_count=max_fragment_count,
        )
        self.attachment_readout = HydrogenAttachmentReadout(
            hidden_dim=hidden_dim,
            attachment_dim=attachment_dim,
            temperature=attachment_temperature,
        )
        self.fragment_conditioner = FragmentConditioner(
            num_fragment_types=num_fragment_types,
            hidden_dim=hidden_dim,
        )
        self.h_context_aggregator = HydrogenContextAggregator(hidden_dim=hidden_dim)
        self.edge_readout = HeavyEdgeReadout(
            hidden_dim=hidden_dim,
            num_bond_types=num_bond_types,
        )
        self.fragment_candidates = tuple(fragment_candidates)
        self.register_buffer(
            "parent_atom_types",
            torch.tensor(parent_atom_types, dtype=torch.long),
        )

    @staticmethod
    def _combine_spectra(
            h_nmr: torch.Tensor,
            h_nmr_mask: torch.Tensor,
            c_nmr: torch.Tensor,
            c_nmr_mask: torch.Tensor,
            h_nmr_integration: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_nmr_integration is None:
            h_nmr_integration = torch.ones_like(h_nmr)
        c_nmr_integration = torch.ones_like(c_nmr)

        shifts = torch.cat([h_nmr, c_nmr], dim=1)
        integrations = torch.cat([h_nmr_integration, c_nmr_integration], dim=1)
        peak_mask = torch.cat([h_nmr_mask, c_nmr_mask], dim=1)
        nucleus_types = torch.cat([
            torch.ones_like(h_nmr, dtype=torch.long),
            torch.full_like(c_nmr, fill_value=2, dtype=torch.long),
        ], dim=1)
        nucleus_types = nucleus_types * peak_mask.long()
        return shifts, nucleus_types, integrations, peak_mask

    def forward(
            self,
            atom_types: torch.Tensor,
            atom_mask: torch.Tensor,
            h_nmr: torch.Tensor,
            h_nmr_mask: torch.Tensor,
            c_nmr: torch.Tensor,
            c_nmr_mask: torch.Tensor,
            h_nmr_integration: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        atom_mask = atom_mask.bool()
        h_nmr_mask = h_nmr_mask.bool()
        c_nmr_mask = c_nmr_mask.bool()
        heavy_mask = atom_mask & atom_types.ne(1)
        hydrogen_mask = atom_mask & atom_types.eq(1)

        shifts, nucleus_types, integrations, peak_mask = self._combine_spectra(
            h_nmr=h_nmr,
            h_nmr_mask=h_nmr_mask,
            c_nmr=c_nmr,
            c_nmr_mask=c_nmr_mask,
            h_nmr_integration=h_nmr_integration,
        )
        peak_features = self.peak_embedding(
            shifts=shifts,
            nucleus_types=nucleus_types,
            integrations=integrations,
        )
        peak_features, spectrum_attention = self.spectrum_encoder(
            peak_features,
            peak_mask,
        )

        atom_features_pre_ca = self.atom_embedding(atom_types)
        atom_features = atom_features_pre_ca * atom_mask.unsqueeze(-1)
        atom_spectrum_attention = None
        for layer in self.atom_spectrum_layers:
            atom_features, atom_spectrum_attention = layer(
                query=atom_features,
                context=peak_features,
                query_mask=atom_mask,
                context_mask=peak_mask,
            )

        # ``data.h`` is element-sorted. The within-heavy rank therefore defines
        # stable element-grouped output queries used by every downstream stage.
        heavy_rank = heavy_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        heavy_query_seed = atom_features + self.heavy_query_embedding(heavy_rank)
        heavy_query_features, heavy_query_attention = self.heavy_query_decoder(
            query=heavy_query_seed,
            context=atom_features,
            query_mask=heavy_mask,
            context_mask=atom_mask,
        )
        atom_features = torch.where(
            heavy_mask.unsqueeze(-1),
            heavy_query_features,
            atom_features,
        )

        interaction_attention = []
        for layer in self.atom_interaction_layers:
            atom_features, attention = layer(
                atom_features=atom_features,
                heavy_mask=heavy_mask,
                hydrogen_mask=hydrogen_mask,
            )
            interaction_attention.append(attention)

        fragment_logits = self.fragment_readout(atom_features)
        h_parent_type_logits, h_parent_fragment_logits = (
            self.h_parent_environment_readout(atom_features)
        )
        (
            attachment_logits,
            attachment_probabilities,
            hydrogen_attachment_features,
            heavy_attachment_features,
        ) = self.attachment_readout(
            atom_features=atom_features,
            hydrogen_mask=hydrogen_mask,
            heavy_mask=heavy_mask,
        )
        fragment_conditioned_features, expected_fragment_counts = (
            self.fragment_conditioner(
                atom_features=atom_features,
                fragment_logits=fragment_logits,
                heavy_mask=heavy_mask,
            )
        )
        refined_atom_features, hydrogen_context, assigned_h_count = (
            self.h_context_aggregator(
                atom_features=fragment_conditioned_features,
                attachment_probabilities=attachment_probabilities,
                heavy_mask=heavy_mask,
            )
        )
        edge_logits, heavy_edge_mask = self.edge_readout(
            refined_atom_features,
            heavy_mask,
        )

        return {
            "atom_features_pre_ca": atom_features_pre_ca,
            "atom_features": atom_features,
            "heavy_query_features": atom_features,
            "graph_atom_features": refined_atom_features,
            "peak_features": peak_features,
            "heavy_mask": heavy_mask,
            "hydrogen_mask": hydrogen_mask,
            "heavy_edge_logits": edge_logits,
            "heavy_edge_mask": heavy_edge_mask,
            "h_attachment_logits": attachment_logits,
            "h_attachment_probabilities": attachment_probabilities,
            "hydrogen_attachment_features": hydrogen_attachment_features,
            "heavy_attachment_features": heavy_attachment_features,
            "fragment_logits": fragment_logits,
            "expected_fragment_counts": expected_fragment_counts,
            "h_parent_type_logits": h_parent_type_logits,
            "h_parent_fragment_logits": h_parent_fragment_logits,
            "parent_atom_types": self.parent_atom_types,
            "hydrogen_context": hydrogen_context,
            "assigned_h_count": assigned_h_count,
            "attention": {
                "spectrum": spectrum_attention,
                "atom_to_spectrum": atom_spectrum_attention,
                "heavy_query_to_atoms": heavy_query_attention,
                "atom_interaction": interaction_attention,
            },
        }
