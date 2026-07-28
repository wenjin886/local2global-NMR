"""Lightning training logic for symmetry-aware multiset shift matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .model import SchNetShiftModel


class Shift3DModule(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_interactions: int = 6,
        num_rbf: int = 64,
        cutoff: float = 6.0,
        dropout: float = 0.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        h_loss_weight: float = 1.0,
        c_loss_weight: float = 1.0,
        equivalence_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SchNetShiftModel(
            hidden_dim=hidden_dim,
            num_interactions=num_interactions,
            num_rbf=num_rbf,
            cutoff=cutoff,
            dropout=dropout,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            batch["atomic_numbers"],
            batch["positions"],
            batch["atom_mask"],
        )

    @staticmethod
    def _class_means(
        values: torch.Tensor,
        classes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unique = torch.unique(classes, sorted=True)
        means = torch.stack([values[classes.eq(item)].mean() for item in unique])
        expanded = torch.empty_like(values)
        for item, mean in zip(unique, means):
            expanded[classes.eq(item)] = mean
        return means, expanded

    def _sample_losses(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        classes = batch["equivalence_classes"][index]
        h_mask = batch["h_prediction_mask"][index]
        h_raw = output["h_shifts"][index, h_mask]
        h_classes = classes[h_mask]
        _, h_aggregated = self._class_means(h_raw, h_classes)
        h_target = batch["h_targets"][index, batch["h_target_mask"][index]]
        if h_aggregated.numel() != h_target.numel():
            raise RuntimeError("Hydrogen prediction/target cardinality mismatch")
        h_sorted = h_aggregated.sort().values
        h_target_sorted = h_target.sort().values

        c_mask = batch["atomic_numbers"][index].eq(6) & batch["atom_mask"][index]
        c_raw = output["c_shifts"][index, c_mask]
        c_classes = classes[c_mask]
        c_aggregated, c_expanded = self._class_means(c_raw, c_classes)
        c_target = batch["c_targets"][index, batch["c_target_mask"][index]]
        if c_aggregated.numel() != c_target.numel():
            raise RuntimeError("Carbon prediction/target cardinality mismatch")
        c_sorted = c_aggregated.sort().values
        c_target_sorted = c_target.sort().values

        h_loss = F.smooth_l1_loss(h_sorted, h_target_sorted)
        c_loss = F.smooth_l1_loss(c_sorted, c_target_sorted)
        equivalence = 0.5 * (
            F.smooth_l1_loss(h_raw, h_aggregated)
            + F.smooth_l1_loss(c_raw, c_expanded)
        )
        return {
            "h_loss": h_loss,
            "c_loss": c_loss,
            "equivalence_loss": equivalence,
            "h_mae": (h_sorted - h_target_sorted).abs().mean(),
            "c_mae": (c_sorted - c_target_sorted).abs().mean(),
        }

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        output = self(batch)
        per_sample: List[Dict[str, torch.Tensor]] = [
            self._sample_losses(batch, output, index)
            for index in range(batch["atomic_numbers"].size(0))
        ]
        metrics = {
            key: torch.stack([sample[key] for sample in per_sample]).mean()
            for key in per_sample[0]
        }
        loss = (
            self.hparams.h_loss_weight * metrics["h_loss"]
            + self.hparams.c_loss_weight * metrics["c_loss"]
            + self.hparams.equivalence_loss_weight
            * metrics["equivalence_loss"]
        )
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["atomic_numbers"].size(0),
        )
        for key, value in metrics.items():
            self.log(
                f"{stage}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch["atomic_numbers"].size(0),
            )
        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
