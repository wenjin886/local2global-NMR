from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.data.constants import DEFAULT_MAX_NEIGHBOR_COUNTS, parse_bond_type_candidates
from src.data.constants import SMILES_PAD_INDEX


class NMRGraphLoss(nn.Module):
    """Stage-wise graph objective with a heavy-atom neighbor-count constraint."""

    def __init__(
            self,
            heavy_fragment_weight: float = 1.0,
            heavy_fragment_presence_weight: float = 0.25,
            heavy_neighbor_count_weight: Optional[float] = None,
            h_parent_fragment_weight: float = 1.0,
            h_parent_presence_weight: float = 0.25,
            h_parent_type_weight: float = 0.25,
            h_attachment_weight: float = 1.0,
            h_count_weight: float = 0.25,
            h_entropy_weight: float = 0.0,
            edge_weight: float = 1.0,
            edge_none_class_weight: float = 1.0,
            edge_bond_class_weight: float = 1.0,
            edge_total_neighbor_count_weight: float = 0.0,
            fragment_edge_consistency_weight: float = 0.25,
            smiles_weight: float = 0.0,
            edge_class_weights: Optional[torch.Tensor] = None,
            max_heavy_neighbor_counts: Optional[Mapping[int, int]] = None,
            permutation_invariant_hydrogens: bool = True,
            # Deprecated aliases retained so existing Hydra configs continue
            # to instantiate when resuming an older experiment.
            heavy_degree_weight: Optional[float] = None,
            max_heavy_degrees: Optional[Mapping[int, int]] = None,
    ):
        super().__init__()
        if heavy_neighbor_count_weight is None:
            heavy_neighbor_count_weight = (
                0.0 if heavy_degree_weight is None else heavy_degree_weight
            )
        elif (
                heavy_degree_weight is not None
                and heavy_degree_weight != heavy_neighbor_count_weight
        ):
            raise ValueError(
                "Conflicting heavy_neighbor_count_weight and deprecated "
                "heavy_degree_weight"
            )
        if max_heavy_neighbor_counts is None:
            max_heavy_neighbor_counts = max_heavy_degrees
        elif max_heavy_degrees is not None:
            raise ValueError(
                "Specify max_heavy_neighbor_counts or deprecated "
                "max_heavy_degrees, not both"
            )
        self.heavy_fragment_weight = heavy_fragment_weight
        self.heavy_fragment_presence_weight = heavy_fragment_presence_weight
        self.heavy_neighbor_count_weight = heavy_neighbor_count_weight
        self.h_parent_fragment_weight = h_parent_fragment_weight
        self.h_parent_presence_weight = h_parent_presence_weight
        self.h_parent_type_weight = h_parent_type_weight
        self.h_attachment_weight = h_attachment_weight
        self.h_count_weight = h_count_weight
        self.h_entropy_weight = h_entropy_weight
        self.edge_weight = edge_weight
        if edge_none_class_weight < 0 or edge_bond_class_weight < 0:
            raise ValueError("Edge class weights must be non-negative")
        if edge_none_class_weight == 0 and edge_bond_class_weight == 0:
            raise ValueError("At least one edge class weight must be positive")
        self.edge_none_class_weight = float(edge_none_class_weight)
        self.edge_bond_class_weight = float(edge_bond_class_weight)
        self.edge_total_neighbor_count_weight = (
            edge_total_neighbor_count_weight
        )
        self.fragment_edge_consistency_weight = fragment_edge_consistency_weight
        self.smiles_weight = smiles_weight
        self.permutation_invariant_hydrogens = permutation_invariant_hydrogens
        neighbor_count_limits = dict(DEFAULT_MAX_NEIGHBOR_COUNTS)
        if max_heavy_neighbor_counts is not None:
            neighbor_count_limits.update({
                int(atomic_number): int(max_neighbors)
                for atomic_number, max_neighbors
                in max_heavy_neighbor_counts.items()
            })
        neighbor_count_lookup = torch.full((119,), -1.0)
        for atomic_number, max_neighbors in neighbor_count_limits.items():
            if atomic_number < 0 or atomic_number >= neighbor_count_lookup.numel():
                raise ValueError(f"Unsupported atomic number: {atomic_number}")
            if max_neighbors <= 0:
                raise ValueError("Maximum neighbor counts must be positive")
            neighbor_count_lookup[atomic_number] = float(max_neighbors)
        # These dataset/config constants are not learned model state. Keeping
        # them out of state_dict lets checkpoints created before this loss was
        # introduced load strictly without a missing-buffer error.
        self.register_buffer(
            "max_neighbor_count_lookup",
            neighbor_count_lookup,
            persistent=False,
        )
        if edge_class_weights is not None:
            if (
                    self.edge_none_class_weight != 1.0
                    or self.edge_bond_class_weight != 1.0
            ):
                raise ValueError(
                    "Specify edge_class_weights or edge_none_class_weight/"
                    "edge_bond_class_weight, not both"
                )
            edge_class_weights = torch.as_tensor(
                edge_class_weights, dtype=torch.float
            )
            if edge_class_weights.ndim != 1:
                raise ValueError("edge_class_weights must be one-dimensional")
            if (edge_class_weights < 0).any() or not edge_class_weights.any():
                raise ValueError(
                    "edge_class_weights must be non-negative with at least "
                    "one positive entry"
                )
        # Scalar class weights are configuration constants and add no checkpoint
        # state. Preserve the old persistence behavior only when the deprecated
        # full-vector interface is explicitly used.
        self.register_buffer(
            "edge_class_weights",
            edge_class_weights,
            persistent=edge_class_weights is not None,
        )

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
    def expected_heavy_neighbor_count(logits: torch.Tensor) -> torch.Tensor:
        count_values = torch.arange(
            logits.size(-1), dtype=logits.dtype, device=logits.device
        )
        expected_counts = (
            torch.softmax(logits, dim=-1) * count_values
        ).sum(dim=-1)
        return expected_counts.sum(dim=-1)

    def heavy_neighbor_count_caps(self, atom_types: torch.Tensor) -> torch.Tensor:
        if (
                atom_types.numel()
                and atom_types.max() >= self.max_neighbor_count_lookup.numel()
        ):
            raise ValueError("Atomic number exceeds neighbor-count lookup size")
        return self.max_neighbor_count_lookup[atom_types.long()]

    def heavy_neighbor_count_overflow_loss(
            self,
            fragment_logits: torch.Tensor,
            atom_types: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> torch.Tensor:
        caps = self.heavy_neighbor_count_caps(atom_types)
        unsupported = heavy_mask & caps.le(0)
        if unsupported.any():
            atomic_numbers = atom_types[unsupported].unique().tolist()
            raise ValueError(
                "Missing maximum-neighbor-count limits for heavy atoms: "
                f"{atomic_numbers}"
            )
        valid = heavy_mask & caps.gt(0)
        if not valid.any():
            return fragment_logits.sum() * 0.0
        expected_neighbor_count = self.expected_heavy_neighbor_count(
            fragment_logits
        )
        normalized_overflow = (
            torch.relu(expected_neighbor_count - caps) / caps.clamp_min(1.0)
        )
        return normalized_overflow[valid].square().mean()

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
        if self.edge_class_weights is None:
            class_weights = logits.new_full(
                (logits.size(-1),), self.edge_bond_class_weight
            )
            class_weights[0] = self.edge_none_class_weight
        else:
            if self.edge_class_weights.numel() != logits.size(-1):
                raise ValueError(
                    "edge_class_weights length must match the number of edge "
                    f"classes ({logits.size(-1)})"
                )
            class_weights = self.edge_class_weights.to(
                device=logits.device, dtype=logits.dtype
            )
        return F.cross_entropy(
            logits[mask], bond_types[mask].long(), weight=class_weights
        )

    def edge_total_neighbor_count_overflow_loss(
            self,
            outputs: Mapping[str, object],
            atom_types: torch.Tensor,
    ) -> torch.Tensor:
        """Limit total neighbors realized by soft edges and H attachments."""
        edge_probabilities = torch.softmax(
            outputs["heavy_edge_logits"], dim=-1
        )
        edge_exists = 1.0 - edge_probabilities[..., 0]
        edge_exists = edge_exists * outputs["heavy_edge_mask"].to(
            dtype=edge_exists.dtype
        )
        expected_heavy_neighbors = edge_exists.sum(dim=-1)
        expected_h_neighbors = outputs[
            "h_attachment_probabilities"
        ].sum(dim=1)
        expected_total_neighbors = (
            expected_heavy_neighbors + expected_h_neighbors
        )

        heavy_mask = outputs["heavy_mask"]
        caps = self.heavy_neighbor_count_caps(atom_types)
        unsupported = heavy_mask & caps.le(0)
        if unsupported.any():
            atomic_numbers = atom_types[unsupported].unique().tolist()
            raise ValueError(
                "Missing maximum-neighbor-count limits for heavy atoms: "
                f"{atomic_numbers}"
            )
        valid = heavy_mask & caps.gt(0)
        if not valid.any():
            return outputs["heavy_edge_logits"].sum() * 0.0
        normalized_overflow = (
            torch.relu(expected_total_neighbors - caps)
            / caps.clamp_min(1.0)
        )
        return normalized_overflow[valid].square().mean()

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
            smiles_target_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        attachment_required = any(weight != 0 for weight in (
            self.h_attachment_weight,
            self.h_count_weight,
            self.h_entropy_weight,
            self.edge_total_neighbor_count_weight,
        ))
        if attachment_required and outputs.get("h_attachment_probabilities") is None:
            raise ValueError(
                "Attachment losses require model.predict_attachments=true"
            )
        edge_required = (
            self.edge_weight != 0
            or self.edge_total_neighbor_count_weight != 0
            or self.fragment_edge_consistency_weight != 0
        )
        if edge_required and outputs.get("heavy_edge_logits") is None:
            raise ValueError("Edge losses require model.predict_edges=true")

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
        losses["heavy_neighbor_count_overflow"] = (
            self.heavy_neighbor_count_overflow_loss(
                outputs["fragment_logits"], atom_types, outputs["heavy_mask"]
            )
            if self.heavy_neighbor_count_weight != 0
            else self._zero_like(outputs)
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
        losses["edge_total_neighbor_count_overflow"] = (
            self.edge_total_neighbor_count_overflow_loss(outputs, atom_types)
            if self.edge_total_neighbor_count_weight != 0
            else zero
        )
        losses["fragment_edge_consistency"] = (
            self.fragment_edge_consistency_loss(outputs, atom_types)
            if self.fragment_edge_consistency_weight != 0
            else zero
        )
        if self.smiles_weight != 0 and outputs.get("use_smiles_loss", False):
            if outputs.get("smiles_logits") is None or smiles_target_ids is None:
                raise ValueError(
                    "smiles_weight > 0 requires decoder logits and SMILES targets"
                )
            logits = outputs["smiles_logits"]
            if logits.shape[:2] != smiles_target_ids.shape:
                raise ValueError("SMILES logits and targets have incompatible shapes")
            losses["smiles"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                smiles_target_ids.reshape(-1),
                ignore_index=SMILES_PAD_INDEX,
            )
        else:
            losses["smiles"] = zero

        total = (
            self.heavy_fragment_weight * losses["heavy_fragment"]
            + self.heavy_fragment_presence_weight * losses["heavy_fragment_presence"]
            + self.heavy_neighbor_count_weight
            * losses["heavy_neighbor_count_overflow"]
            + self.h_parent_fragment_weight * losses["h_parent_fragment"]
            + self.h_parent_presence_weight * losses["h_parent_presence"]
            + self.h_parent_type_weight * losses["h_parent_type"]
            + self.h_attachment_weight * losses["h_attachment"]
            + self.h_count_weight * losses["h_count"]
            + self.h_entropy_weight * losses["h_entropy"]
            + self.edge_weight * losses["edge"]
            + self.edge_total_neighbor_count_weight
            * losses["edge_total_neighbor_count_overflow"]
            + self.fragment_edge_consistency_weight
            * losses["fragment_edge_consistency"]
            + self.smiles_weight * losses["smiles"]
        )
        losses["weighted"] = total
        return total, losses
