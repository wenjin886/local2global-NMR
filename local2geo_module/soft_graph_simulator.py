from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .constants import NONE, NUM_BOND_TYPES


class SoftGraphSimulator(nn.Module):
    """Turn a clean SMILES graph into NMRToGraph-shaped soft logits.

    This module is only a test-input simulator. It is deliberately separate
    from the geometry solver and has no trainable parameters.
    """

    def __init__(
        self,
        clean_margin: float = 4.0,
        logit_noise_std: float = 0.10,
        bond_type_confusion_probability: float = 0.10,
        false_positive_probability: float = 0.01,
        false_negative_probability: float = 0.06,
        attachment_confusion_probability: float = 0.10,
        corruption_boost: float = 3.0,
    ):
        super().__init__()
        self.clean_margin = clean_margin
        self.logit_noise_std = logit_noise_std
        self.bond_type_confusion_probability = (
            bond_type_confusion_probability
        )
        self.false_positive_probability = false_positive_probability
        self.false_negative_probability = false_negative_probability
        self.attachment_confusion_probability = (
            attachment_confusion_probability
        )
        self.corruption_boost = corruption_boost

    @staticmethod
    def _symmetric_noise(shape, device, dtype) -> torch.Tensor:
        noise = torch.randn(shape, device=device, dtype=dtype)
        return 0.5 * (noise + noise.transpose(1, 2))

    def _heavy_logits(
        self,
        bond_types: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        corrupted: bool,
    ) -> torch.Tensor:
        targets = bond_types.clamp_min(NONE)
        dtype = torch.float32
        logits = torch.zeros(
            (*targets.shape, NUM_BOND_TYPES),
            dtype=dtype,
            device=targets.device,
        )
        logits.scatter_(
            -1,
            targets.unsqueeze(-1),
            torch.full(
                (*targets.shape, 1),
                self.clean_margin,
                dtype=dtype,
                device=targets.device,
            ),
        )
        if self.logit_noise_std:
            logits = logits + self.logit_noise_std * self._symmetric_noise(
                logits.shape, logits.device, logits.dtype
            )
        if corrupted:
            upper = heavy_pair_mask & torch.triu(
                torch.ones_like(heavy_pair_mask, dtype=torch.bool),
                diagonal=1,
            )
            random_values = torch.rand_like(targets, dtype=dtype)
            random_values = 0.5 * (
                random_values + random_values.transpose(1, 2)
            )
            true_edge = upper & targets.ne(NONE)
            nonedge = upper & targets.eq(NONE)
            type_event = true_edge & random_values.lt(
                self.bond_type_confusion_probability
            )
            false_negative = true_edge & random_values.gt(
                1.0 - self.false_negative_probability
            )
            false_positive = nonedge & random_values.lt(
                self.false_positive_probability
            )
            alternative = 1 + targets.remainder(NUM_BOND_TYPES - 1)
            random_class = torch.randint(
                1,
                NUM_BOND_TYPES,
                targets.shape,
                device=targets.device,
            )
            update_class = torch.where(
                type_event,
                alternative,
                torch.where(
                    false_negative,
                    torch.zeros_like(targets),
                    random_class,
                ),
            )
            event = type_event | false_negative | false_positive
            update = torch.zeros_like(logits)
            update.scatter_(
                -1,
                update_class.unsqueeze(-1),
                torch.full(
                    (*targets.shape, 1),
                    self.corruption_boost,
                    dtype=dtype,
                    device=targets.device,
                ),
            )
            update = update * event.unsqueeze(-1)
            update = update + update.transpose(1, 2)
            logits = logits + update

        logits = 0.5 * (logits + logits.transpose(1, 2))
        logits = logits.masked_fill(
            ~heavy_pair_mask.unsqueeze(-1), -20.0
        )
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
        corrupted: bool,
    ) -> torch.Tensor:
        dtype = torch.float32
        logits = torch.full(
            attachment_mask.shape,
            -20.0,
            dtype=dtype,
            device=h_attachment.device,
        )
        valid_h = h_attachment.ge(0)
        clean = torch.zeros_like(logits)
        clean.scatter_(
            -1,
            h_attachment.clamp_min(0).unsqueeze(-1),
            torch.full(
                (*h_attachment.shape, 1),
                self.clean_margin,
                dtype=dtype,
                device=h_attachment.device,
            ),
        )
        logits = torch.where(attachment_mask, clean, logits)
        if self.logit_noise_std:
            logits = torch.where(
                attachment_mask,
                logits + self.logit_noise_std * torch.randn_like(logits),
                logits,
            )
        if corrupted:
            confusion = valid_h & torch.rand(
                h_attachment.shape, device=logits.device
            ).lt(self.attachment_confusion_probability)
            random_scores = torch.rand_like(logits).masked_fill(
                ~attachment_mask, -1.0
            )
            wrong_parent = random_scores.argmax(dim=-1)
            update = torch.zeros_like(logits)
            update.scatter_(
                -1,
                wrong_parent.unsqueeze(-1),
                torch.full(
                    (*h_attachment.shape, 1),
                    self.corruption_boost,
                    dtype=dtype,
                    device=logits.device,
                ),
            )
            logits = logits + update * confusion.unsqueeze(-1)
        return logits.masked_fill(~attachment_mask, -20.0)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        corrupted: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if seed is None:
            heavy_edge_logits = self._heavy_logits(
                batch["bond_types"],
                batch["heavy_pair_mask"],
                corrupted,
            )
            h_attachment_logits = self._attachment_logits(
                batch["h_attachment"],
                batch["attachment_mask"],
                corrupted,
            )
            attachment_probabilities = torch.softmax(
                h_attachment_logits.masked_fill(
                    ~batch["attachment_mask"], -20.0
                ),
                dim=-1,
            ) * batch["attachment_mask"]
            attachment_probabilities = attachment_probabilities / (
                attachment_probabilities.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
            )
            return {
                "atom_types": batch["atomic_numbers"],
                "atom_mask": batch["atom_mask"],
                "heavy_mask": batch["heavy_mask"],
                "hydrogen_mask": batch["hydrogen_mask"],
                "heavy_edge_mask": batch["heavy_pair_mask"],
                "heavy_edge_logits": heavy_edge_logits,
                "h_attachment_logits": h_attachment_logits,
                "h_attachment_probabilities": attachment_probabilities,
                "assigned_h_count": attachment_probabilities.sum(dim=1),
            }
        devices = (
            [batch["bond_types"].device]
            if batch["bond_types"].is_cuda else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if batch["bond_types"].is_cuda:
                torch.cuda.manual_seed_all(seed)
            return self.forward(batch, corrupted=corrupted, seed=None)
