"""Lightning training with unequal-cardinality atom-to-peak set losses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .model import SchNetShiftModel


class Shift3DModule(pl.LightningModule):
    """Pretrain 3D2Shift from raw peak sets and graph environment labels.

    The primary loss compares every atom prediction directly with the raw peak
    set and therefore never requires matching cardinalities. Environment IDs
    are used only by the optional pretraining consistency regularizer.
    """

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
        h_huber_delta: float = 0.2,
        c_huber_delta: float = 2.0,
        h_outlier_cap: float = 2.0,
        c_outlier_cap: float = 20.0,
        h_softmin_temperature: float = 0.05,
        c_softmin_temperature: float = 0.5,
        stats_path: str | None = None,
        h_shift_mean: float | None = None,
        h_shift_std: float | None = None,
        c_shift_mean: float | None = None,
        c_shift_std: float | None = None,
    ) -> None:
        super().__init__()
        supplied_stats = (
            h_shift_mean,
            h_shift_std,
            c_shift_mean,
            c_shift_std,
        )
        if any(value is None for value in supplied_stats):
            if stats_path is None:
                h_shift_mean, h_shift_std = 0.0, 1.0
                c_shift_mean, c_shift_std = 0.0, 1.0
            else:
                statistics = json.loads(
                    Path(stats_path).read_text(encoding="utf-8")
                )
                h_shift_mean = float(statistics["hnmr_shift_mean"])
                h_shift_std = float(statistics["hnmr_shift_std"])
                c_shift_mean = float(statistics["cnmr_shift_mean"])
                c_shift_std = float(statistics["cnmr_shift_std"])
        assert h_shift_mean is not None and h_shift_std is not None
        assert c_shift_mean is not None and c_shift_std is not None
        if h_shift_std <= 0 or c_shift_std <= 0:
            raise ValueError("NMR shift standard deviations must be positive")
        self.save_hyperparameters()
        self.register_buffer(
            "shift_means",
            torch.tensor([h_shift_mean, c_shift_mean], dtype=torch.float32),
        )
        self.register_buffer(
            "shift_stds",
            torch.tensor([h_shift_std, c_shift_std], dtype=torch.float32),
        )
        self.model = SchNetShiftModel(
            hidden_dim=hidden_dim,
            num_interactions=num_interactions,
            num_rbf=num_rbf,
            cutoff=cutoff,
            dropout=dropout,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = self.model(
            batch["atomic_numbers"],
            batch["positions"],
            batch["atom_mask"],
        )
        return {
            "h_shifts_normalized": normalized["h_shifts"],
            "c_shifts_normalized": normalized["c_shifts"],
            "h_shifts": self._to_ppm(normalized["h_shifts"], nucleus=0),
            "c_shifts": self._to_ppm(normalized["c_shifts"], nucleus=1),
        }

    def _to_normalized(
        self, shifts_ppm: torch.Tensor, nucleus: int
    ) -> torch.Tensor:
        return (
            shifts_ppm - self.shift_means[nucleus]
        ) / self.shift_stds[nucleus]

    def _to_ppm(
        self, normalized_shifts: torch.Tensor, nucleus: int
    ) -> torch.Tensor:
        return (
            normalized_shifts * self.shift_stds[nucleus]
            + self.shift_means[nucleus]
        )

    def _normalized_scale(self, value_ppm: float, nucleus: int) -> float:
        std = (
            float(self.hparams.h_shift_std)
            if nucleus == 0
            else float(self.hparams.c_shift_std)
        )
        return value_ppm / std

    @staticmethod
    def _robust_pair_cost(
        predictions: torch.Tensor,
        peaks: torch.Tensor,
        delta: float,
        cap: float,
    ) -> torch.Tensor:
        difference = (
            predictions[:, None] - peaks[None, :]
        ).abs()
        huber = torch.where(
            difference < delta,
            0.5 * difference.square() / delta,
            difference - 0.5 * delta,
        )
        return huber.clamp_max(cap)

    @staticmethod
    def _soft_nearest(
        costs: torch.Tensor,
        dim: int,
        temperature: float,
    ) -> torch.Tensor:
        # Treat the soft assignment as the matching decision. Detaching it
        # prevents a bad candidate from receiving a gradient that pushes it
        # even farther away merely to reduce its assignment probability.
        weights = torch.softmax(-costs / temperature, dim=dim).detach()
        return (weights * costs).sum(dim=dim)

    @classmethod
    def _set_loss(
        cls,
        predictions: torch.Tensor,
        peaks: torch.Tensor,
        delta: float,
        cap: float,
        temperature: float,
    ) -> Dict[str, torch.Tensor]:
        if predictions.numel() == 0 or peaks.numel() == 0:
            zero = predictions.sum() * 0.0
            return {"loss": zero, "nearest_mae": zero}
        costs = cls._robust_pair_cost(predictions, peaks, delta, cap)
        atom_to_peak = cls._soft_nearest(
            costs, dim=1, temperature=temperature
        ).mean()
        peak_to_atom = cls._soft_nearest(
            costs, dim=0, temperature=temperature
        ).mean()
        absolute = (
            predictions[:, None] - peaks[None, :]
        ).abs()
        nearest_mae = 0.5 * (
            absolute.min(dim=1).values.mean()
            + absolute.min(dim=0).values.mean()
        )
        return {
            "loss": 0.5 * (atom_to_peak + peak_to_atom),
            "nearest_mae": nearest_mae,
        }

    @staticmethod
    def _environment_consistency(
        predictions: torch.Tensor,
        environment_ids: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        if predictions.numel() == 0:
            return predictions.sum() * 0.0
        _, inverse, counts = torch.unique(
            environment_ids,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        class_count = int(counts.numel())
        sums = predictions.new_zeros(class_count)
        sums.scatter_add_(0, inverse, predictions)
        means = sums / counts.to(predictions.dtype)
        atom_losses = F.smooth_l1_loss(
            predictions,
            means[inverse],
            beta=delta,
            reduction="none",
        )
        class_loss_sums = predictions.new_zeros(class_count)
        class_loss_sums.scatter_add_(0, inverse, atom_losses)
        class_losses = class_loss_sums / counts.to(predictions.dtype)
        # Every environment receives equal weight, independent of class size.
        repeated = counts.gt(1).to(predictions.dtype)
        return (
            class_losses * repeated
        ).sum() / repeated.sum().clamp_min(1.0)

    def _sample_losses(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        valid_atoms = batch["atom_mask"][index]
        atomic_numbers = batch["atomic_numbers"][index]
        h_atom_mask = valid_atoms & atomic_numbers.eq(1)
        c_atom_mask = valid_atoms & atomic_numbers.eq(6)
        h_predictions = output["h_shifts_normalized"][index, h_atom_mask]
        c_predictions = output["c_shifts_normalized"][index, c_atom_mask]
        h_peaks_ppm = batch["h_peak_shifts"][
            index, batch["h_peak_mask"][index]
        ]
        c_peaks_ppm = batch["c_peak_shifts"][
            index, batch["c_peak_mask"][index]
        ]
        h_peaks = self._to_normalized(h_peaks_ppm, nucleus=0)
        c_peaks = self._to_normalized(c_peaks_ppm, nucleus=1)

        h_set = self._set_loss(
            h_predictions,
            h_peaks,
            delta=self._normalized_scale(
                float(self.hparams.h_huber_delta), nucleus=0
            ),
            cap=self._normalized_scale(
                float(self.hparams.h_outlier_cap), nucleus=0
            ),
            temperature=self._normalized_scale(
                float(self.hparams.h_softmin_temperature), nucleus=0
            ),
        )
        c_set = self._set_loss(
            c_predictions,
            c_peaks,
            delta=self._normalized_scale(
                float(self.hparams.c_huber_delta), nucleus=1
            ),
            cap=self._normalized_scale(
                float(self.hparams.c_outlier_cap), nucleus=1
            ),
            temperature=self._normalized_scale(
                float(self.hparams.c_softmin_temperature), nucleus=1
            ),
        )

        equivalence = h_predictions.sum() * 0.0
        if (
            float(self.hparams.equivalence_loss_weight) > 0
            and "environment_ids" in batch
        ):
            environment_ids = batch["environment_ids"][index]
            h_equivalence = self._environment_consistency(
                h_predictions,
                environment_ids[h_atom_mask],
                delta=self._normalized_scale(
                    float(self.hparams.h_huber_delta), nucleus=0
                ),
            )
            c_equivalence = self._environment_consistency(
                c_predictions,
                environment_ids[c_atom_mask],
                delta=self._normalized_scale(
                    float(self.hparams.c_huber_delta), nucleus=1
                ),
            )
            equivalence = 0.5 * (h_equivalence + c_equivalence)

        h_predictions_ppm = output["h_shifts"][index, h_atom_mask]
        c_predictions_ppm = output["c_shifts"][index, c_atom_mask]
        h_nearest_mae_ppm = self._nearest_mae(
            h_predictions_ppm, h_peaks_ppm
        )
        c_nearest_mae_ppm = self._nearest_mae(
            c_predictions_ppm, c_peaks_ppm
        )
        return {
            "h_set_loss": h_set["loss"],
            "c_set_loss": c_set["loss"],
            "equivalence_loss": equivalence,
            "h_nearest_mae_ppm": h_nearest_mae_ppm,
            "c_nearest_mae_ppm": c_nearest_mae_ppm,
        }

    @staticmethod
    def _nearest_mae(
        predictions: torch.Tensor,
        peaks: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.numel() == 0 or peaks.numel() == 0:
            return predictions.sum() * 0.0
        absolute = (predictions[:, None] - peaks[None, :]).abs()
        return 0.5 * (
            absolute.min(dim=1).values.mean()
            + absolute.min(dim=0).values.mean()
        )

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        output = self(batch)
        per_sample = [
            self._sample_losses(batch, output, index)
            for index in range(batch["atomic_numbers"].size(0))
        ]
        metrics = {
            key: torch.stack([sample[key] for sample in per_sample]).mean()
            for key in per_sample[0]
        }
        loss = (
            self.hparams.h_loss_weight * metrics["h_set_loss"]
            + self.hparams.c_loss_weight * metrics["c_set_loss"]
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
