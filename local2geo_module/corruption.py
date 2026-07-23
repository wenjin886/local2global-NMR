from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .constants import NONE, NUM_BOND_TYPES


class SoftGraphCorruptor(nn.Module):
    """Corrupt heavy bonds and exchangeable H-to-heavy attachment logits."""

    def __init__(
        self,
        clean_margin_min: float = 1.5,
        clean_margin_max: float = 4.0,
        corruption_boost_min: float = 2.0,
        corruption_boost_max: float = 4.5,
        bond_type_confusion_probability: float = 0.12,
        false_positive_probability: float = 0.015,
        false_negative_probability: float = 0.08,
        valence_conflict_probability: float = 0.02,
        attachment_confusion_probability: float = 0.12,
        attachment_logit_noise_std: float = 0.35,
        logit_noise_std: float = 0.35,
    ):
        super().__init__()
        self.clean_margin_min = clean_margin_min
        self.clean_margin_max = clean_margin_max
        self.corruption_boost_min = corruption_boost_min
        self.corruption_boost_max = corruption_boost_max
        self.bond_type_confusion_probability = bond_type_confusion_probability
        self.false_positive_probability = false_positive_probability
        self.false_negative_probability = false_negative_probability
        self.valence_conflict_probability = valence_conflict_probability
        self.attachment_confusion_probability = attachment_confusion_probability
        self.attachment_logit_noise_std = attachment_logit_noise_std
        self.logit_noise_std = logit_noise_std

    @staticmethod
    def _symmetric_random(shape, device, dtype):
        values = torch.rand(shape, device=device, dtype=dtype)
        return 0.5 * (values + values.transpose(1, 2))

    def _heavy_logits(
        self,
        clean_bond_types: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms, _ = clean_bond_types.shape
        dtype = torch.get_default_dtype()
        safe_targets = clean_bond_types.clamp_min(0)
        clean_margin = self.clean_margin_min + (
            self.clean_margin_max - self.clean_margin_min
        ) * self._symmetric_random(
            (batch, atoms, atoms), clean_bond_types.device, dtype
        )
        logits = torch.zeros(
            (batch, atoms, atoms, NUM_BOND_TYPES),
            device=clean_bond_types.device,
            dtype=dtype,
        )
        logits.scatter_add_(
            -1, safe_targets.unsqueeze(-1), clean_margin.unsqueeze(-1)
        )
        upper = heavy_pair_mask & torch.triu(
            torch.ones_like(heavy_pair_mask, dtype=torch.bool), diagonal=1
        )
        true_edge = upper & safe_targets.ne(NONE)
        true_nonedge = upper & safe_targets.eq(NONE)

        random_class = torch.randint(
            1, NUM_BOND_TYPES, (batch, atoms, atoms), device=logits.device
        )
        random_class = torch.triu(random_class, diagonal=1)
        random_class = random_class + random_class.transpose(1, 2)
        type_class = 1 + safe_targets.remainder(NUM_BOND_TYPES - 1)
        event_random = self._symmetric_random(
            (batch, atoms, atoms), logits.device, dtype
        )
        type_event = true_edge & event_random.lt(
            self.bond_type_confusion_probability
        )
        alternative = torch.where(
            type_event, type_class, torch.zeros_like(safe_targets)
        )
        event_random = self._symmetric_random(
            (batch, atoms, atoms), logits.device, dtype
        )
        false_negative = true_edge & event_random.lt(
            self.false_negative_probability
        )
        alternative = torch.where(
            false_negative, torch.zeros_like(alternative), alternative
        )
        event_random = self._symmetric_random(
            (batch, atoms, atoms), logits.device, dtype
        )
        false_positive = true_nonedge & event_random.lt(
            self.false_positive_probability
        )
        alternative = torch.where(false_positive, random_class, alternative)

        degree = (safe_targets.ne(NONE) & heavy_pair_mask).sum(dim=-1)
        saturated = degree.ge(3)
        conflict_candidates = true_nonedge & (
            saturated[:, :, None] | saturated[:, None, :]
        )
        event_random = self._symmetric_random(
            (batch, atoms, atoms), logits.device, dtype
        )
        conflict = conflict_candidates & event_random.lt(
            self.valence_conflict_probability
        )
        alternative = torch.where(
            conflict, torch.ones_like(alternative), alternative
        )
        corrupted = type_event | false_negative | false_positive | conflict

        boost = self.corruption_boost_min + (
            self.corruption_boost_max - self.corruption_boost_min
        ) * self._symmetric_random(
            (batch, atoms, atoms), logits.device, dtype
        )
        update = torch.zeros_like(logits)
        update.scatter_add_(
            -1, alternative.unsqueeze(-1), boost.unsqueeze(-1)
        )
        logits = logits + update * corrupted.unsqueeze(-1)
        if self.logit_noise_std:
            noise = torch.randn_like(logits)
            noise = 0.5 * (noise + noise.transpose(1, 2))
            logits = logits + self.logit_noise_std * noise
        logits = 0.5 * (logits + logits.transpose(1, 2))
        logits = logits.masked_fill(~heavy_pair_mask.unsqueeze(-1), -20.0)
        logits[..., NONE] = torch.where(
            heavy_pair_mask,
            logits[..., NONE],
            torch.full_like(logits[..., NONE], 20.0),
        )
        return logits

    def _attachment_logits(
        self,
        h_attachment: torch.Tensor,
        attachment_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms = h_attachment.shape
        dtype = torch.get_default_dtype()
        logits = torch.full(
            (batch, atoms, atoms),
            -20.0,
            dtype=dtype,
            device=h_attachment.device,
        )
        valid_h = h_attachment.ge(0)
        margin = self.clean_margin_min + (
            self.clean_margin_max - self.clean_margin_min
        ) * torch.rand((batch, atoms), dtype=dtype, device=h_attachment.device)
        safe_parent = h_attachment.clamp_min(0)
        update = torch.zeros_like(logits)
        update.scatter_(-1, safe_parent.unsqueeze(-1), margin.unsqueeze(-1))
        logits = torch.where(attachment_mask, update, logits)
        if self.attachment_logit_noise_std:
            noise = self.attachment_logit_noise_std * torch.randn_like(logits)
            logits = torch.where(attachment_mask, logits + noise, logits)

        confusion = valid_h & torch.rand(
            (batch, atoms), device=logits.device
        ).lt(self.attachment_confusion_probability)
        random_scores = torch.rand_like(logits).masked_fill(
            ~attachment_mask, -1.0
        )
        wrong_parent = random_scores.argmax(dim=-1)
        boost = self.corruption_boost_min + (
            self.corruption_boost_max - self.corruption_boost_min
        ) * torch.rand((batch, atoms), dtype=dtype, device=logits.device)
        confused_update = torch.zeros_like(logits)
        confused_update.scatter_(
            -1, wrong_parent.unsqueeze(-1), boost.unsqueeze(-1)
        )
        logits = logits + confused_update * confusion.unsqueeze(-1)
        return logits.masked_fill(~attachment_mask, -20.0)

    def forward(
        self,
        clean_bond_types: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        h_attachment: torch.Tensor,
        attachment_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "heavy_edge_logits": self._heavy_logits(
                clean_bond_types, heavy_pair_mask
            ),
            "h_attachment_logits": self._attachment_logits(
                h_attachment, attachment_mask
            ),
        }
