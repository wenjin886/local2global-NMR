from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class NMRGraphLoss(nn.Module):
    """Supervised graph loss without an explicit valence penalty."""

    def __init__(
            self,
            edge_weight: float = 1.0,
            h_attachment_weight: float = 1.0,
            h_count_weight: float = 0.25,
            local_weight: float = 0.5,
            h_entropy_weight: float = 0.0,
            edge_class_weights: Optional[torch.Tensor] = None,
            permutation_invariant_hydrogens: bool = True,
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.h_attachment_weight = h_attachment_weight
        self.h_count_weight = h_count_weight
        self.local_weight = local_weight
        self.h_entropy_weight = h_entropy_weight
        self.permutation_invariant_hydrogens = permutation_invariant_hydrogens
        if edge_class_weights is None:
            self.register_buffer("edge_class_weights", None)
        else:
            self.register_buffer("edge_class_weights", edge_class_weights.float())

    @staticmethod
    def _zero_like(outputs: Mapping[str, object]) -> torch.Tensor:
        return outputs["heavy_edge_logits"].sum() * 0.0

    def edge_loss(
            self,
            outputs: Mapping[str, object],
            bond_types: torch.Tensor,
    ) -> torch.Tensor:
        logits = outputs["heavy_edge_logits"]
        mask = outputs["heavy_edge_mask"]
        upper_triangle = torch.triu(
            torch.ones_like(mask, dtype=torch.bool),
            diagonal=1,
        )
        mask = mask & upper_triangle & bond_types.ge(0)
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[mask],
            bond_types[mask].long(),
            weight=self.edge_class_weights,
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
            predicted = probabilities[sample_index, valid]
            if target_attachments.numel() == 0:
                continue

            log_probabilities = predicted.clamp_min(1e-12).log()
            cost = -log_probabilities[:, target_attachments]
            row_indices, column_indices = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )
            row_indices = torch.as_tensor(row_indices, device=cost.device)
            column_indices = torch.as_tensor(column_indices, device=cost.device)
            sample_losses.append(cost[row_indices, column_indices].mean())

        if not sample_losses:
            return probabilities.sum() * 0.0
        return torch.stack(sample_losses).mean()

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
            predicted_counts[heavy_mask],
            target_counts[heavy_mask],
        )

    @staticmethod
    def hydrogen_entropy(
            probabilities: torch.Tensor,
            hydrogen_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not hydrogen_mask.any():
            return probabilities.sum() * 0.0
        entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=-1)
        return entropy[hydrogen_mask].mean()

    @staticmethod
    def local_loss(
            local_outputs: Mapping[str, Mapping[str, torch.Tensor]],
            local_labels: torch.Tensor,
            zero: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for output in local_outputs.values():
            indices = output["indices"]
            logits = output["logits"]
            if indices.numel() == 0:
                continue
            targets = local_labels[indices[:, 0], indices[:, 1]].long()
            valid = targets.ge(0)
            if valid.any():
                losses.append(F.cross_entropy(logits[valid], targets[valid]))
        return torch.stack(losses).mean() if losses else zero

    def forward(
            self,
            outputs: Mapping[str, object],
            bond_types: torch.Tensor,
            h_attachment: torch.Tensor,
            local_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        losses["edge"] = self.edge_loss(outputs, bond_types)

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
        losses["h_count"] = self.hydrogen_count_loss(
            probabilities=outputs["h_attachment_probabilities"],
            hydrogen_mask=outputs["hydrogen_mask"],
            heavy_mask=outputs["heavy_mask"],
            targets=h_attachment,
        )
        losses["h_entropy"] = self.hydrogen_entropy(
            outputs["h_attachment_probabilities"],
            outputs["hydrogen_mask"],
        )

        zero = self._zero_like(outputs)
        losses["local"] = (
            self.local_loss(outputs["local_outputs"], local_labels, zero)
            if local_labels is not None
            else zero
        )
        total = (
            self.edge_weight * losses["edge"]
            + self.h_attachment_weight * losses["h_attachment"]
            + self.h_count_weight * losses["h_count"]
            + self.local_weight * losses["local"]
            + self.h_entropy_weight * losses["h_entropy"]
        )
        losses["weighted"] = total
        return total, losses
