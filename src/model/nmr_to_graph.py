import json
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from src.data.constants import BOND_TYPE_CANDIDATES, HEAVY_ATOM_TYPES
from src.nn.attention import MaskedCrossAttentionBlock, MaskedSelfAttentionEncoder
from src.nn.embedding import AtomSlotEmbedding, CNMRPeakEmbedding, HNMRPeakEmbedding
from src.nn.smiles import NMRToSMILESDecoder
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

    Atom slots, 1H peaks, and 13C peaks are concatenated and jointly encoded.
    Features are split into their original modalities only after this shared
    self-attention, while downstream queries can continue to read the complete
    joint memory.
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            num_heads: int = 8,
            num_joint_layers: int = 3,
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
            use_h_integration: bool = True,
            use_h_multiplicity: bool = True,
            use_h_j: bool = True,
            max_multiplicity_classes: int = 512,
            use_smiles_loss: bool = False,
            use_smiles_conditioning: bool = False,
            num_smiles_layers: int = 3,
            max_smiles_length: int = 256,
            smiles_vocab_path: Optional[str] = None,
            smiles_vocab_size: Optional[int] = None,
            teacher_force_smiles_during_eval: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.atom_embedding = AtomSlotEmbedding(
            hidden_dim=hidden_dim,
            max_atomic_number=max_atomic_number,
            max_num_atoms=max_num_atoms,
            dropout=dropout,
        )
        self.h_peak_embedding = HNMRPeakEmbedding(
            hidden_dim=hidden_dim,
            num_fourier_features=num_fourier_features,
            num_multiplicity_classes=max_multiplicity_classes,
            use_integration=use_h_integration,
            use_multiplicity=use_h_multiplicity,
            use_j=use_h_j,
            dropout=dropout,
        )
        self.c_peak_embedding = CNMRPeakEmbedding(
            hidden_dim=hidden_dim,
            num_fourier_features=num_fourier_features,
            dropout=dropout,
        )
        self.joint_encoder = MaskedSelfAttentionEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_joint_layers,
            dropout=dropout,
        )
        self.use_smiles_loss = use_smiles_loss
        self.use_smiles_conditioning = use_smiles_conditioning
        self.teacher_force_smiles_during_eval = teacher_force_smiles_during_eval
        self.max_smiles_length = max_smiles_length
        self.smiles_vocab = None
        if use_smiles_loss or use_smiles_conditioning:
            if smiles_vocab_path is not None:
                with open(smiles_vocab_path, encoding="utf-8") as handle:
                    self.smiles_vocab = json.load(handle)["smiles_vocab"]
                    smiles_vocab_size = len(self.smiles_vocab)
            if smiles_vocab_size is None:
                raise ValueError(
                    "SMILES decoder requires smiles_vocab_path or smiles_vocab_size"
                )
            self.smiles_decoder = NMRToSMILESDecoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_smiles_layers,
                vocab_size=smiles_vocab_size,
                max_length=max_smiles_length,
                dropout=dropout,
            )
        else:
            self.smiles_decoder = None
        self.atom_smiles_layer = (
            MaskedCrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            if use_smiles_conditioning else None
        )
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
            h_features: torch.Tensor,
            h_nmr_mask: torch.Tensor,
            c_features: torch.Tensor,
            c_nmr_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([h_features, c_features], dim=1)
        peak_mask = torch.cat([h_nmr_mask, c_nmr_mask], dim=1)
        return features, peak_mask

    def forward(
            self,
            atom_types: torch.Tensor,
            atom_mask: torch.Tensor,
            h_nmr: torch.Tensor,
            h_nmr_mask: torch.Tensor,
            c_nmr: torch.Tensor,
            c_nmr_mask: torch.Tensor,
            h_nmr_integration: Optional[torch.Tensor] = None,
            h_nmr_integration_mask: Optional[torch.Tensor] = None,
            h_nmr_multiplicity: Optional[torch.Tensor] = None,
            h_nmr_multiplicity_mask: Optional[torch.Tensor] = None,
            h_nmr_j: Optional[torch.Tensor] = None,
            h_nmr_j_mask: Optional[torch.Tensor] = None,
            smiles_input_ids: Optional[torch.Tensor] = None,
            smiles_input_mask: Optional[torch.Tensor] = None,
            teacher_force_smiles: Optional[bool] = None,
    ) -> Dict[str, object]:
        atom_mask = atom_mask.bool()
        h_nmr_mask = h_nmr_mask.bool()
        c_nmr_mask = c_nmr_mask.bool()
        heavy_mask = atom_mask & atom_types.ne(1)
        hydrogen_mask = atom_mask & atom_types.eq(1)

        h_peak_features = self.h_peak_embedding(
            shifts=h_nmr,
            peak_mask=h_nmr_mask,
            integrations=h_nmr_integration,
            integration_mask=h_nmr_integration_mask,
            multiplicities=h_nmr_multiplicity,
            multiplicity_mask=h_nmr_multiplicity_mask,
            j_values=h_nmr_j,
            j_mask=h_nmr_j_mask,
        )
        c_peak_features = self.c_peak_embedding(c_nmr, c_nmr_mask)
        atom_features_pre_joint = self.atom_embedding(atom_types)
        atom_features_pre_joint = (
            atom_features_pre_joint * atom_mask.unsqueeze(-1)
        )
        joint_features = torch.cat(
            [atom_features_pre_joint, h_peak_features, c_peak_features], dim=1
        )
        joint_mask = torch.cat(
            [atom_mask, h_nmr_mask, c_nmr_mask], dim=1
        )
        joint_features, joint_attention = self.joint_encoder(
            joint_features,
            joint_mask,
        )

        num_atoms = atom_types.size(1)
        num_h_peaks = h_nmr.size(1)
        atom_features = joint_features[:, :num_atoms]
        h_peak_features = joint_features[:, num_atoms:num_atoms + num_h_peaks]
        c_peak_features = joint_features[:, num_atoms + num_h_peaks:]
        peak_features, peak_mask = self._combine_spectra(
            h_features=h_peak_features,
            h_nmr_mask=h_nmr_mask,
            c_features=c_peak_features,
            c_nmr_mask=c_nmr_mask,
        )

        smiles_outputs = None
        if self.smiles_decoder is not None:
            if teacher_force_smiles is None:
                teacher_force_smiles = (
                    self.training or self.teacher_force_smiles_during_eval
                )
            if teacher_force_smiles:
                if smiles_input_ids is None or smiles_input_mask is None:
                    raise ValueError("Teacher forcing requires SMILES input tensors")
                smiles_outputs = self.smiles_decoder.teacher_force(
                    input_ids=smiles_input_ids,
                    input_mask=smiles_input_mask,
                    memory=joint_features,
                    memory_mask=joint_mask,
                )
            else:
                max_steps = (
                    smiles_input_ids.size(1)
                    if smiles_input_ids is not None
                    else self.max_smiles_length
                )
                smiles_outputs = self.smiles_decoder.generate(
                    memory=joint_features,
                    memory_mask=joint_mask,
                    max_steps=max_steps,
                )

        atom_smiles_attention = None
        if self.atom_smiles_layer is not None:
            atom_features, atom_smiles_attention = self.atom_smiles_layer(
                query=atom_features,
                context=smiles_outputs["hidden_states"],
                query_mask=atom_mask,
                context_mask=smiles_outputs["mask"],
            )

        # ``data.h`` is element-sorted. The within-heavy rank therefore defines
        # stable element-grouped output queries used by every downstream stage.
        heavy_rank = heavy_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        heavy_query_seed = atom_features + self.heavy_query_embedding(heavy_rank)
        heavy_query_features, heavy_query_attention = self.heavy_query_decoder(
            query=heavy_query_seed,
            context=joint_features,
            query_mask=heavy_mask,
            context_mask=joint_mask,
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
            "atom_features_pre_joint": atom_features_pre_joint,
            # Backward-compatible alias for existing representation scripts.
            "atom_features_pre_ca": atom_features_pre_joint,
            "atom_features": atom_features,
            "heavy_query_features": atom_features,
            "graph_atom_features": refined_atom_features,
            "joint_features": joint_features,
            "peak_features": peak_features,
            "h_peak_features": h_peak_features,
            "c_peak_features": c_peak_features,
            "smiles_logits": (
                smiles_outputs["logits"] if smiles_outputs is not None else None
            ),
            "smiles_hidden_states": (
                smiles_outputs["hidden_states"]
                if smiles_outputs is not None else None
            ),
            "smiles_token_ids": (
                smiles_outputs["token_ids"] if smiles_outputs is not None else None
            ),
            "smiles_token_mask": (
                smiles_outputs["mask"] if smiles_outputs is not None else None
            ),
            "smiles_teacher_forced": (
                smiles_outputs["teacher_forced"]
                if smiles_outputs is not None else None
            ),
            "use_smiles_loss": self.use_smiles_loss,
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
                "joint": joint_attention,
                "atom_to_smiles": atom_smiles_attention,
                "heavy_query_to_joint": heavy_query_attention,
                "atom_interaction": interaction_attention,
            },
        }
