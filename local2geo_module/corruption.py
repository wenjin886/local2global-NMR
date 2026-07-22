from __future__ import annotations

import torch
from torch import nn

from .constants import NONE, NUM_BOND_TYPES


class SoftGraphCorruptor(nn.Module):
    """Create symmetric noisy logits from a clean categorical heavy graph."""

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
        self.logit_noise_std = logit_noise_std

    @staticmethod
    def _symmetric_random(shape, device, dtype):
        values = torch.rand(shape, device=device, dtype=dtype)
        return 0.5 * (values + values.transpose(1, 2))

    def forward(
        self,
        clean_bond_types: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms, _ = clean_bond_types.shape
        dtype = torch.get_default_dtype()
        safe_targets = clean_bond_types.clamp_min(0)
        clean_margin = self.clean_margin_min + (
            self.clean_margin_max - self.clean_margin_min
        ) * self._symmetric_random((batch, atoms, atoms), clean_bond_types.device, dtype)
        logits = torch.zeros(
            (batch, atoms, atoms, NUM_BOND_TYPES),
            device=clean_bond_types.device,
            dtype=dtype,
        )
        logits.scatter_add_(-1, safe_targets.unsqueeze(-1), clean_margin.unsqueeze(-1))
        upper = pair_mask & torch.triu(
            torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
        )
        true_edge = upper & safe_targets.ne(NONE)
        true_nonedge = upper & safe_targets.eq(NONE)

        random_class = torch.randint(
            1, NUM_BOND_TYPES, (batch, atoms, atoms), device=logits.device
        )
        random_class = torch.triu(random_class, diagonal=1)
        random_class = random_class + random_class.transpose(1, 2)
        type_class = 1 + (safe_targets.remainder(NUM_BOND_TYPES - 1))
        event_random = self._symmetric_random((batch, atoms, atoms), logits.device, dtype)
        type_event = true_edge & event_random.lt(
            self.bond_type_confusion_probability
        )
        alternative = torch.zeros_like(safe_targets)
        alternative = torch.where(
            type_event,
            type_class,
            alternative,
        )
        event_random = self._symmetric_random((batch, atoms, atoms), logits.device, dtype)
        false_negative = true_edge & event_random.lt(
            self.false_negative_probability
        )
        alternative = torch.where(
            false_negative,
            torch.zeros_like(alternative),
            alternative,
        )
        event_random = self._symmetric_random((batch, atoms, atoms), logits.device, dtype)
        false_positive = true_nonedge & event_random.lt(self.false_positive_probability)
        alternative = torch.where(false_positive, random_class, alternative)

        degree = safe_targets.ne(NONE).sum(dim=-1)
        saturated = degree.ge(3)
        conflict_candidates = true_nonedge & (
            saturated[:, :, None] | saturated[:, None, :]
        )
        event_random = self._symmetric_random((batch, atoms, atoms), logits.device, dtype)
        conflict = conflict_candidates & event_random.lt(self.valence_conflict_probability)
        alternative = torch.where(conflict, torch.ones_like(alternative), alternative)
        corrupted = type_event | false_negative | false_positive | conflict

        boost = self.corruption_boost_min + (
            self.corruption_boost_max - self.corruption_boost_min
        ) * self._symmetric_random((batch, atoms, atoms), logits.device, dtype)
        update = torch.zeros_like(logits)
        update.scatter_add_(-1, alternative.unsqueeze(-1), boost.unsqueeze(-1))
        logits = logits + update * corrupted.unsqueeze(-1)

        if self.logit_noise_std:
            noise = torch.randn_like(logits)
            noise = 0.5 * (noise + noise.transpose(1, 2))
            logits = logits + self.logit_noise_std * noise
        logits = 0.5 * (logits + logits.transpose(1, 2))
        valid = pair_mask.unsqueeze(-1)
        logits = logits.masked_fill(~valid, -20.0)
        logits[..., NONE] = torch.where(
            pair_mask, logits[..., NONE], torch.full_like(logits[..., NONE], 20.0)
        )
        return logits
