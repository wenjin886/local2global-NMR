"""Lightning training module for the 2D-only hybrid topology prior."""

from __future__ import annotations

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .soft_graph_simulator import SoftGraphSimulator
from .topology_prior import SoftTopologyPrior


class HybridLocal2GeoModule(pl.LightningModule):
    """Pretrain soft graph correction and local bounds without 3D labels."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.05,
        residual_scale: float = 6.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        clean_margin: float = 4.0,
        logit_noise_std: float = 0.4,
        bond_type_confusion_probability: float = 0.15,
        false_positive_probability: float = 0.03,
        false_negative_probability: float = 0.12,
        attachment_confusion_probability: float = 0.15,
        corruption_boost: float = 6.0,
        edge_loss_weight: float = 1.0,
        attachment_loss_weight: float = 1.0,
        geometry_loss_weight: float = 0.3,
        one_three_loss_weight: float = 0.5,
        one_four_loss_weight: float = 0.5,
        distance_loss_weight: float = 0.5,
        ring_loss_weight: float = 0.2,
        conjugated_loss_weight: float = 0.2,
        torsion_loss_weight: float = 0.3,
        residual_loss_weight: float = 1e-3,
        positive_class_weight_cap: float = 20.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.prior = SoftTopologyPrior(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        self.simulator = SoftGraphSimulator(
            clean_margin=clean_margin,
            logit_noise_std=logit_noise_std,
            bond_type_confusion_probability=(
                bond_type_confusion_probability
            ),
            false_positive_probability=false_positive_probability,
            false_negative_probability=false_negative_probability,
            attachment_confusion_probability=(
                attachment_confusion_probability
            ),
            corruption_boost=corruption_boost,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        raw_graph: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.prior(
            atomic_numbers=batch["atomic_numbers"],
            formal_charges=batch["formal_charges"],
            atom_mask=batch["atom_mask"],
            heavy_mask=batch["heavy_mask"],
            hydrogen_mask=batch["hydrogen_mask"],
            pair_mask=batch["pair_mask"],
            heavy_pair_mask=batch["heavy_pair_mask"],
            attachment_mask=batch["attachment_mask"],
            raw_heavy_edge_logits=raw_graph["heavy_edge_logits"],
            raw_h_attachment_logits=raw_graph["h_attachment_logits"],
        )

    def correct_graph(
        self,
        batch: Dict[str, torch.Tensor],
        raw_graph: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Public import boundary for NMRToGraph and the XYZ evaluator."""
        return self(batch, raw_graph)

    def _balanced_bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        selected_targets = targets[mask].to(logits.dtype)
        selected_logits = logits[mask]
        if selected_logits.numel() == 0:
            return logits.sum() * 0.0
        positives = selected_targets.sum()
        negatives = selected_targets.numel() - positives
        positive_weight = (
            negatives / positives.clamp_min(1.0)
        ).clamp(1.0, self.hparams.positive_class_weight_cap)
        weights = torch.where(
            selected_targets.gt(0.5),
            positive_weight,
            torch.ones_like(selected_targets),
        )
        return (
            F.binary_cross_entropy_with_logits(
                selected_logits, selected_targets, reduction="none"
            ) * weights
        ).mean()

    @staticmethod
    def _masked_cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], targets[mask])

    @staticmethod
    def _masked_accuracy(
        prediction: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return prediction.sum() * 0.0
        return prediction[mask].eq(targets[mask]).float().mean()

    @staticmethod
    def _binary_f1(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = logits[mask].gt(0)
        truth = targets[mask].gt(0.5)
        true_positive = (prediction & truth).sum().float()
        false_positive = (prediction & ~truth).sum().float()
        false_negative = (~prediction & truth).sum().float()
        return 2.0 * true_positive / (
            2.0 * true_positive + false_positive + false_negative
        ).clamp_min(1.0)

    def _losses(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        atoms = batch["atom_mask"].size(1)
        upper = torch.triu(
            torch.ones(
                (atoms, atoms),
                device=self.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None]
        heavy_upper = batch["heavy_pair_mask"] & upper
        pair_upper = batch["pair_mask"] & upper

        edge = self._masked_cross_entropy(
            output["corrected_heavy_edge_logits"],
            batch["bond_types"],
            heavy_upper,
        )
        valid_h = batch["hydrogen_mask"] & batch["h_attachment"].ge(0)
        attachment = self._masked_cross_entropy(
            output["corrected_h_attachment_logits"],
            batch["h_attachment"],
            valid_h,
        )
        geometry = self._masked_cross_entropy(
            output["geometry_logits"],
            batch["geometry_classes"],
            batch["atom_mask"],
        )
        one_three = self._balanced_bce(
            output["one_three_logits"],
            batch["one_three_targets"],
            pair_upper,
        )
        one_four = self._balanced_bce(
            output["one_four_logits"],
            batch["one_four_targets"],
            pair_upper,
        )
        ring = self._balanced_bce(
            output["ring_logits"], batch["ring_bonds"], heavy_upper
        )
        conjugated = self._balanced_bce(
            output["conjugated_logits"],
            batch["conjugated_bonds"],
            heavy_upper,
        )
        torsion_mask = batch["torsion_classes"].ge(0) & upper
        torsion = (
            F.cross_entropy(
                output["torsion_logits"][torsion_mask],
                batch["torsion_classes"][torsion_mask],
            )
            if torsion_mask.any()
            else output["torsion_logits"].sum() * 0.0
        )
        ratio_13_mask = batch["one_three_targets"].gt(0.5) & pair_upper
        ratio_14_mask = batch["torsion_classes"].ge(0) & upper
        predicted_log_13 = output[
            "one_three_distance_ratio"
        ].clamp_min(1e-6).log()
        predicted_log_14 = output[
            "one_four_distance_ratio"
        ].clamp_min(1e-6).log()
        distance_13 = (
            F.smooth_l1_loss(
                predicted_log_13[ratio_13_mask],
                batch["one_three_log_ratio"][ratio_13_mask],
            )
            if ratio_13_mask.any()
            else predicted_log_13.sum() * 0.0
        )
        distance_14 = (
            F.smooth_l1_loss(
                predicted_log_14[ratio_14_mask],
                batch["one_four_log_ratio"][ratio_14_mask],
            )
            if ratio_14_mask.any()
            else predicted_log_14.sum() * 0.0
        )
        edge_residual = output["edge_residual"][
            batch["heavy_pair_mask"]
        ]
        attachment_residual = output["attachment_residual"][
            batch["attachment_mask"]
        ]
        residual = (
            edge_residual.square().mean()
            if edge_residual.numel()
            else output["edge_residual"].sum() * 0.0
        ) + (
            attachment_residual.square().mean()
            if attachment_residual.numel()
            else output["attachment_residual"].sum() * 0.0
        )
        distance = 0.5 * (distance_13 + distance_14)
        total = (
            self.hparams.edge_loss_weight * edge
            + self.hparams.attachment_loss_weight * attachment
            + self.hparams.geometry_loss_weight * geometry
            + self.hparams.one_three_loss_weight * one_three
            + self.hparams.one_four_loss_weight * one_four
            + self.hparams.distance_loss_weight * distance
            + self.hparams.ring_loss_weight * ring
            + self.hparams.conjugated_loss_weight * conjugated
            + self.hparams.torsion_loss_weight * torsion
            + self.hparams.residual_loss_weight * residual
        )
        return {
            "loss": total,
            "loss_edge": edge,
            "loss_attachment": attachment,
            "loss_geometry": geometry,
            "loss_one_three": one_three,
            "loss_one_four": one_four,
            "loss_distance_13": distance_13,
            "loss_distance_14": distance_14,
            "loss_ring": ring,
            "loss_conjugated": conjugated,
            "loss_torsion": torsion,
            "loss_residual": residual,
        }

    def _metrics(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        atoms = batch["atom_mask"].size(1)
        upper = torch.triu(
            torch.ones(
                (atoms, atoms),
                device=self.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )[None]
        heavy = batch["heavy_pair_mask"] & upper
        pair = batch["pair_mask"] & upper
        predicted_type = output[
            "corrected_heavy_edge_logits"
        ].argmax(dim=-1)
        true_type = batch["bond_types"]
        predicted_edge = predicted_type.ne(0)
        true_edge = true_type.ne(0)
        true_positive = (predicted_edge & true_edge & heavy).sum().float()
        false_positive = (
            predicted_edge & ~true_edge & heavy
        ).sum().float()
        false_negative = (
            ~predicted_edge & true_edge & heavy
        ).sum().float()
        edge_f1 = 2.0 * true_positive / (
            2.0 * true_positive + false_positive + false_negative
        ).clamp_min(1.0)
        bond_type_accuracy = self._masked_accuracy(
            predicted_type, true_type, heavy
        )
        valid_h = batch["hydrogen_mask"] & batch["h_attachment"].ge(0)
        attachment_accuracy = self._masked_accuracy(
            output["corrected_h_attachment_logits"].argmax(dim=-1),
            batch["h_attachment"],
            valid_h,
        )
        geometry_accuracy = self._masked_accuracy(
            output["geometry_logits"].argmax(dim=-1),
            batch["geometry_classes"],
            batch["atom_mask"],
        )
        return {
            "edge_f1": edge_f1,
            "bond_type_accuracy": bond_type_accuracy,
            "attachment_accuracy": attachment_accuracy,
            "geometry_accuracy": geometry_accuracy,
            "one_three_f1": self._binary_f1(
                output["one_three_logits"],
                batch["one_three_targets"],
                pair,
            ),
            "one_four_f1": self._binary_f1(
                output["one_four_logits"],
                batch["one_four_targets"],
                pair,
            ),
        }

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        batch_index: int,
    ) -> torch.Tensor:
        seed: Optional[int] = None
        if stage != "train":
            seed = 1729 + batch_index
        raw_graph = self.simulator(
            batch, corrupted=True, seed=seed
        )
        output = self(batch, raw_graph)
        losses = self._losses(batch, output)
        metrics = self._metrics(batch, output)
        values = {**losses, **metrics}
        self.log_dict(
            {f"{stage}/{key}": value for key, value in values.items()},
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=key_in_progress_bar(stage),
            batch_size=batch["atom_mask"].size(0),
        )
        return losses["loss"]

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "test", batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.trainer.max_epochs)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def key_in_progress_bar(stage: str) -> bool:
    return stage != "train"
