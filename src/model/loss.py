from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.data.constants import parse_bond_type_candidates


class NMRGraphLoss(nn.Module):
    """Stage-wise local-to-global graph objective without a valence loss."""

    def __init__(
            self,
            heavy_fragment_weight: float = 1.0,
            heavy_fragment_presence_weight: float = 0.25,
            h_parent_fragment_weight: float = 1.0,
            h_parent_presence_weight: float = 0.25,
            h_parent_type_weight: float = 0.25,
            h_attachment_weight: float = 1.0,
            h_count_weight: float = 0.25,
            h_entropy_weight: float = 0.0,
            edge_weight: float = 1.0,
            fragment_edge_consistency_weight: float = 0.25,
            edge_class_weights: Optional[torch.Tensor] = None,
            permutation_invariant_hydrogens: bool = True,
    ):
        super().__init__()
        self.heavy_fragment_weight = heavy_fragment_weight
        self.heavy_fragment_presence_weight = heavy_fragment_presence_weight
        self.h_parent_fragment_weight = h_parent_fragment_weight
        self.h_parent_presence_weight = h_parent_presence_weight
        self.h_parent_type_weight = h_parent_type_weight
        self.h_attachment_weight = h_attachment_weight
        self.h_count_weight = h_count_weight
        self.h_entropy_weight = h_entropy_weight
        self.edge_weight = edge_weight
        self.fragment_edge_consistency_weight = fragment_edge_consistency_weight
        self.permutation_invariant_hydrogens = permutation_invariant_hydrogens
        if edge_class_weights is None:
            self.register_buffer("edge_class_weights", None)
        else:
            self.register_buffer("edge_class_weights", edge_class_weights.float())

    @staticmethod
    def _zero_like(outputs: Mapping[str, object]) -> torch.Tensor:
        return outputs["fragment_logits"].sum() * 0.0

    @staticmethod
    def fragment_count_loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
            node_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Zero vs non-zero is learned by the separate presence head. This CE is
        # conditional on a port being present, so the long zero tail cannot
        # dominate the count objective.
        valid = node_mask.unsqueeze(-1) & targets.gt(0)
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], targets[valid].long())

    @staticmethod
    def fragment_presence_loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
            node_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = node_mask.unsqueeze(-1) & targets.ge(0)
        if not valid.any():
            return logits.sum() * 0.0
        presence_logits = torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]
        presence_targets = targets.gt(0).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            presence_logits[valid], presence_targets[valid]
        )

    @staticmethod
    def _parent_type_indices(
            atomic_numbers: torch.Tensor,
            parent_atom_types: torch.Tensor,
    ) -> torch.Tensor:
        matches = atomic_numbers[:, None].eq(parent_atom_types[None, :])
        if not matches.any(dim=-1).all():
            unsupported = atomic_numbers[~matches.any(dim=-1)].unique().tolist()
            raise ValueError("Unsupported H parent atom types: %s" % unsupported)
        return matches.long().argmax(dim=-1)

    def permutation_invariant_h_environment_loss(
            self,
            parent_type_logits: torch.Tensor,
            parent_fragment_logits: torch.Tensor,
            hydrogen_mask: torch.Tensor,
            parent_types: torch.Tensor,
            parent_fragments: torch.Tensor,
            parent_atom_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fragment_losses = []
        type_losses = []
        presence_losses = []
        for batch_index in range(parent_type_logits.size(0)):
            predicted_rows = hydrogen_mask[batch_index]
            target_rows = hydrogen_mask[batch_index] & parent_types[batch_index].ge(0)
            predicted_type = parent_type_logits[batch_index, predicted_rows]
            predicted_fragment = parent_fragment_logits[batch_index, predicted_rows]
            target_type = parent_types[batch_index, target_rows]
            target_fragment = parent_fragments[batch_index, target_rows]
            if target_type.numel() == 0:
                continue

            target_type_index = self._parent_type_indices(
                target_type.long(), parent_atom_types.long()
            )
            type_log_prob = F.log_softmax(predicted_type, dim=-1)
            type_cost = -type_log_prob[:, target_type_index]

            fragment_log_prob = F.log_softmax(predicted_fragment, dim=-1)
            num_predictions = predicted_fragment.size(0)
            num_targets = target_fragment.size(0)
            num_fragment_types = target_fragment.size(1)
            expanded_log_prob = fragment_log_prob[:, None].expand(
                num_predictions,
                num_targets,
                fragment_log_prob.size(1),
                fragment_log_prob.size(2),
            )
            gather_index = target_fragment[None, :, :, None].expand(
                num_predictions,
                num_targets,
                num_fragment_types,
                1,
            )
            per_fragment_count_cost = -expanded_log_prob.gather(
                dim=-1, index=gather_index.long()
            ).squeeze(-1)
            positive_target = target_fragment.gt(0)[None, :, :]
            fragment_count_cost = (
                per_fragment_count_cost * positive_target
            ).sum(dim=-1) / positive_target.sum(dim=-1).clamp_min(1)

            presence_logits = (
                torch.logsumexp(predicted_fragment[..., 1:], dim=-1)
                - predicted_fragment[..., 0]
            )
            expanded_presence_logits = presence_logits[:, None, :].expand(
                num_predictions, num_targets, num_fragment_types
            )
            expanded_presence_targets = target_fragment.gt(0)[None, :, :].expand(
                num_predictions, num_targets, num_fragment_types
            ).to(dtype=expanded_presence_logits.dtype)
            fragment_presence_cost = F.binary_cross_entropy_with_logits(
                expanded_presence_logits,
                expanded_presence_targets,
                reduction="none",
            ).mean(dim=-1)

            matching_cost = fragment_count_cost + fragment_presence_cost + type_cost
            row_indices, column_indices = linear_sum_assignment(
                matching_cost.detach().cpu().numpy()
            )
            row_indices = torch.as_tensor(row_indices, device=matching_cost.device)
            column_indices = torch.as_tensor(column_indices, device=matching_cost.device)
            fragment_losses.append(
                fragment_count_cost[row_indices, column_indices].mean()
            )
            type_losses.append(type_cost[row_indices, column_indices].mean())

            presence_losses.append(
                fragment_presence_cost[row_indices, column_indices].mean()
            )

        zero = parent_fragment_logits.sum() * 0.0
        return (
            torch.stack(fragment_losses).mean() if fragment_losses else zero,
            torch.stack(type_losses).mean() if type_losses else zero,
            torch.stack(presence_losses).mean() if presence_losses else zero,
        )

    @staticmethod
    def _direct_attachment_loss(
            logits: torch.Tensor,
            hydrogen_mask: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        valid = hydrogen_mask & targets.ge(0)
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], targets[valid].long())

    @staticmethod
    def _permutation_invariant_attachment_loss(
            probabilities: torch.Tensor,
            hydrogen_mask: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        sample_losses = []
        for sample_index in range(probabilities.size(0)):
            valid = hydrogen_mask[sample_index] & targets[sample_index].ge(0)
            target_attachments = targets[sample_index, valid].long()
            predicted = probabilities[sample_index, hydrogen_mask[sample_index]]
            if target_attachments.numel() == 0:
                continue
            cost = -predicted.clamp_min(1e-12).log()[:, target_attachments]
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            rows = torch.as_tensor(rows, device=cost.device)
            columns = torch.as_tensor(columns, device=cost.device)
            sample_losses.append(cost[rows, columns].mean())
        return (
            torch.stack(sample_losses).mean()
            if sample_losses
            else probabilities.sum() * 0.0
        )

    @staticmethod
    def hydrogen_count_loss(
            probabilities: torch.Tensor,
            hydrogen_mask: torch.Tensor,
            heavy_mask: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        predicted_counts = probabilities.sum(dim=1)
        target_counts = torch.zeros_like(predicted_counts)
        for sample_index in range(probabilities.size(0)):
            valid = hydrogen_mask[sample_index] & targets[sample_index].ge(0)
            attachment_targets = targets[sample_index, valid].long()
            if attachment_targets.numel() > 0:
                target_counts[sample_index].scatter_add_(
                    0,
                    attachment_targets,
                    torch.ones_like(attachment_targets, dtype=target_counts.dtype),
                )
        if not heavy_mask.any():
            return probabilities.sum() * 0.0
        return F.smooth_l1_loss(
            predicted_counts[heavy_mask], target_counts[heavy_mask]
        )

    @staticmethod
    def hydrogen_entropy(
            probabilities: torch.Tensor,
            hydrogen_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not hydrogen_mask.any():
            return probabilities.sum() * 0.0
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
        return entropy[hydrogen_mask].mean()

    def edge_loss(
            self,
            outputs: Mapping[str, object],
            bond_types: torch.Tensor,
    ) -> torch.Tensor:
        logits = outputs["heavy_edge_logits"]
        mask = outputs["heavy_edge_mask"]
        mask = mask & torch.triu(torch.ones_like(mask), diagonal=1).bool()
        mask = mask & bond_types.ge(0)
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[mask], bond_types[mask].long(), weight=self.edge_class_weights
        )

    @staticmethod
    def realized_fragment_counts(
            outputs: Mapping[str, object],
            atom_types: torch.Tensor,
    ) -> torch.Tensor:
        edge_probabilities = torch.softmax(outputs["heavy_edge_logits"], dim=-1)
        attachment_probabilities = outputs["h_attachment_probabilities"]
        realized = []
        for neighbor_type, bond_type in parse_bond_type_candidates():
            if neighbor_type == 1 and bond_type == 1:
                count = attachment_probabilities.sum(dim=1)
            else:
                neighbor_mask = atom_types.eq(neighbor_type).unsqueeze(1)
                count = (
                    edge_probabilities[..., bond_type]
                    * neighbor_mask.to(dtype=edge_probabilities.dtype)
                ).sum(dim=-1)
            realized.append(count)
        return torch.stack(realized, dim=-1)

    def fragment_edge_consistency_loss(
            self,
            outputs: Mapping[str, object],
            atom_types: torch.Tensor,
    ) -> torch.Tensor:
        realized = self.realized_fragment_counts(outputs, atom_types)
        heavy_mask = outputs["heavy_mask"]
        if not heavy_mask.any():
            return realized.sum() * 0.0
        return F.smooth_l1_loss(
            realized[heavy_mask],
            outputs["expected_fragment_counts"][heavy_mask],
        )

    def forward(
            self,
            outputs: Mapping[str, object],
            atom_types: torch.Tensor,
            bond_types: torch.Tensor,
            h_attachment: torch.Tensor,
            heavy_fragment_labels: torch.Tensor,
            h_parent_fragment_labels: torch.Tensor,
            h_parent_types: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        losses["heavy_fragment"] = self.fragment_count_loss(
            outputs["fragment_logits"],
            heavy_fragment_labels,
            outputs["heavy_mask"],
        )
        losses["heavy_fragment_presence"] = self.fragment_presence_loss(
            outputs["fragment_logits"],
            heavy_fragment_labels,
            outputs["heavy_mask"],
        )
        (
            losses["h_parent_fragment"],
            losses["h_parent_type"],
            losses["h_parent_presence"],
        ) = self.permutation_invariant_h_environment_loss(
            parent_type_logits=outputs["h_parent_type_logits"],
            parent_fragment_logits=outputs["h_parent_fragment_logits"],
            hydrogen_mask=outputs["hydrogen_mask"],
            parent_types=h_parent_types,
            parent_fragments=h_parent_fragment_labels,
            parent_atom_types=outputs["parent_atom_types"],
        )
        zero = self._zero_like(outputs)
        if self.h_attachment_weight != 0:
            if self.permutation_invariant_hydrogens:
                losses["h_attachment"] = self._permutation_invariant_attachment_loss(
                    outputs["h_attachment_probabilities"],
                    outputs["hydrogen_mask"],
                    h_attachment,
                )
            else:
                losses["h_attachment"] = self._direct_attachment_loss(
                    outputs["h_attachment_logits"],
                    outputs["hydrogen_mask"],
                    h_attachment,
                )
        else:
            losses["h_attachment"] = zero
        losses["h_count"] = (
            self.hydrogen_count_loss(
                outputs["h_attachment_probabilities"],
                outputs["hydrogen_mask"],
                outputs["heavy_mask"],
                h_attachment,
            )
            if self.h_count_weight != 0
            else zero
        )
        losses["h_entropy"] = (
            self.hydrogen_entropy(
                outputs["h_attachment_probabilities"], outputs["hydrogen_mask"]
            )
            if self.h_entropy_weight != 0
            else zero
        )
        losses["edge"] = (
            self.edge_loss(outputs, bond_types) if self.edge_weight != 0 else zero
        )
        losses["fragment_edge_consistency"] = (
            self.fragment_edge_consistency_loss(outputs, atom_types)
            if self.fragment_edge_consistency_weight != 0
            else zero
        )

        total = (
            self.heavy_fragment_weight * losses["heavy_fragment"]
            + self.heavy_fragment_presence_weight * losses["heavy_fragment_presence"]
            + self.h_parent_fragment_weight * losses["h_parent_fragment"]
            + self.h_parent_presence_weight * losses["h_parent_presence"]
            + self.h_parent_type_weight * losses["h_parent_type"]
            + self.h_attachment_weight * losses["h_attachment"]
            + self.h_count_weight * losses["h_count"]
            + self.h_entropy_weight * losses["h_entropy"]
            + self.edge_weight * losses["edge"]
            + self.fragment_edge_consistency_weight
            * losses["fragment_edge_consistency"]
        )
        losses["weighted"] = total
        return total, losses
