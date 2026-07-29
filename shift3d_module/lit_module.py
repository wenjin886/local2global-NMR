"""Lightning training with unequal-cardinality atom-to-peak set losses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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
        log_prediction_plots: bool = True,
        prediction_plot_samples: int = 9,
        h_plot_ppm_min: float = 0.0,
        h_plot_ppm_max: float = 10.0,
        c_plot_ppm_min: float = 0.0,
        c_plot_ppm_max: float = 230.0,
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
        self._validation_shift_examples: List[Dict[str, Any]] = []
        self._validation_shift_smiles: set[str] = set()
        self._collect_validation_shift_examples = False

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
        h_nearest = self._nearest_mae_components(
            h_predictions_ppm, h_peaks_ppm
        )
        c_nearest = self._nearest_mae_components(
            c_predictions_ppm, c_peaks_ppm
        )
        return {
            "h_set_loss": h_set["loss"],
            "c_set_loss": c_set["loss"],
            "equivalence_loss": equivalence,
            "h_nearest_mae_ppm": h_nearest["symmetric"],
            "h_atom_to_peak_mae_ppm": h_nearest["atom_to_peak"],
            "h_peak_to_atom_mae_ppm": h_nearest["peak_to_atom"],
            "c_nearest_mae_ppm": c_nearest["symmetric"],
            "c_atom_to_peak_mae_ppm": c_nearest["atom_to_peak"],
            "c_peak_to_atom_mae_ppm": c_nearest["peak_to_atom"],
        }

    @staticmethod
    def _nearest_mae_components(
        predictions: torch.Tensor,
        peaks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if predictions.numel() == 0 or peaks.numel() == 0:
            zero = predictions.sum() * 0.0
            return {
                "atom_to_peak": zero,
                "peak_to_atom": zero,
                "symmetric": zero,
            }
        absolute = (predictions[:, None] - peaks[None, :]).abs()
        atom_to_peak = absolute.min(dim=1).values.mean()
        peak_to_atom = absolute.min(dim=0).values.mean()
        return {
            "atom_to_peak": atom_to_peak,
            "peak_to_atom": peak_to_atom,
            "symmetric": 0.5 * (atom_to_peak + peak_to_atom),
        }

    @classmethod
    def _nearest_mae(
        cls,
        predictions: torch.Tensor,
        peaks: torch.Tensor,
    ) -> torch.Tensor:
        return cls._nearest_mae_components(
            predictions, peaks
        )["symmetric"]

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        output = self(batch)
        if stage == "val":
            self._collect_shift_examples(batch, output)
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
        if stage == "train":
            on_step, on_epoch = True, False
        else:
            on_step, on_epoch = True, True

        self.log(
            f"{stage}/loss",
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            batch_size=batch["atomic_numbers"].size(0),
        )
        for key, value in metrics.items():
            if stage == "train" and (
                "_atom_to_peak_mae_ppm" in key
                or "_peak_to_atom_mae_ppm" in key
            ):
                continue
            self.log(
                f"{stage}/{key}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch["atomic_numbers"].size(0),
            )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_shift_examples = []
        self._validation_shift_smiles = set()
        self._collect_validation_shift_examples = bool(
            self.hparams.log_prediction_plots
            and int(self.hparams.prediction_plot_samples) > 0
            and not self.trainer.sanity_checking
            and self.trainer.is_global_zero
        )

    def _collect_shift_examples(
        self,
        batch: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> None:
        if not self._collect_validation_shift_examples:
            return
        limit = int(self.hparams.prediction_plot_samples)
        for index, smiles in enumerate(batch["smiles"]):
            if len(self._validation_shift_examples) >= limit:
                return
            smiles = str(smiles)
            if smiles in self._validation_shift_smiles:
                continue
            valid_atoms = batch["atom_mask"][index]
            atomic_numbers = batch["atomic_numbers"][index]
            h_atoms = valid_atoms & atomic_numbers.eq(1)
            c_atoms = valid_atoms & atomic_numbers.eq(6)
            h_target = batch["h_peak_shifts"][
                index, batch["h_peak_mask"][index]
            ].detach().cpu()
            h_prediction = output["h_shifts"][
                index, h_atoms
            ].detach().cpu()
            c_target = batch["c_peak_shifts"][
                index, batch["c_peak_mask"][index]
            ].detach().cpu()
            c_prediction = output["c_shifts"][
                index, c_atoms
            ].detach().cpu()
            h_nearest = (
                self._nearest_mae_components(h_prediction, h_target)
                if h_prediction.numel() and h_target.numel()
                else None
            )
            c_nearest = (
                self._nearest_mae_components(c_prediction, c_target)
                if c_prediction.numel() and c_target.numel()
                else None
            )
            example = {
                "id": str(batch["id"][index]),
                "smiles": smiles,
                "h_target": h_target,
                "h_prediction": h_prediction,
                "h_nearest_mae_ppm": (
                    float(h_nearest["symmetric"])
                    if h_nearest is not None
                    else None
                ),
                "h_atom_to_peak_mae_ppm": (
                    float(h_nearest["atom_to_peak"])
                    if h_nearest is not None
                    else None
                ),
                "h_peak_to_atom_mae_ppm": (
                    float(h_nearest["peak_to_atom"])
                    if h_nearest is not None
                    else None
                ),
                "c_target": c_target,
                "c_prediction": c_prediction,
                "c_nearest_mae_ppm": (
                    float(c_nearest["symmetric"])
                    if c_nearest is not None
                    else None
                ),
                "c_atom_to_peak_mae_ppm": (
                    float(c_nearest["atom_to_peak"])
                    if c_nearest is not None
                    else None
                ),
                "c_peak_to_atom_mae_ppm": (
                    float(c_nearest["peak_to_atom"])
                    if c_nearest is not None
                    else None
                ),
            }
            self._validation_shift_examples.append(example)
            self._validation_shift_smiles.add(smiles)

    def on_validation_epoch_end(self) -> None:
        if (
            not self._collect_validation_shift_examples
            or not self._validation_shift_examples
            or self.trainer.sanity_checking
            or not self.trainer.is_global_zero
        ):
            return
        image = self._render_shift_stick_plot(
            self._validation_shift_examples,
            h_limits=(
                float(self.hparams.h_plot_ppm_min),
                float(self.hparams.h_plot_ppm_max),
            ),
            c_limits=(
                float(self.hparams.c_plot_ppm_min),
                float(self.hparams.c_plot_ppm_max),
            ),
        )
        for logger in self.trainer.loggers:
            if hasattr(logger, "log_image"):
                logger.log_image(
                    key="val/shift_target_vs_prediction",
                    images=[image],
                    step=self.global_step,
                )
            if hasattr(logger, "log_table"):
                logger.log_table(
                    key="val/shift_examples",
                    columns=[
                        "epoch",
                        "global_step",
                        "id",
                        "smiles",
                        "h_target_ppm",
                        "h_prediction_ppm",
                        "h_nearest_mae_ppm",
                        "h_atom_to_peak_mae_ppm",
                        "h_peak_to_atom_mae_ppm",
                        "c_target_ppm",
                        "c_prediction_ppm",
                        "c_nearest_mae_ppm",
                        "c_atom_to_peak_mae_ppm",
                        "c_peak_to_atom_mae_ppm",
                    ],
                    data=[
                        [
                            int(self.current_epoch),
                            int(self.global_step),
                            example["id"],
                            example["smiles"],
                            example["h_target"].tolist(),
                            example["h_prediction"].tolist(),
                            example["h_nearest_mae_ppm"],
                            example["h_atom_to_peak_mae_ppm"],
                            example["h_peak_to_atom_mae_ppm"],
                            example["c_target"].tolist(),
                            example["c_prediction"].tolist(),
                            example["c_nearest_mae_ppm"],
                            example["c_atom_to_peak_mae_ppm"],
                            example["c_peak_to_atom_mae_ppm"],
                        ]
                        for example in self._validation_shift_examples
                    ],
                    step=self.global_step,
                )
        self._collect_validation_shift_examples = False

    @staticmethod
    def _render_shift_stick_plot(
        examples: List[Dict[str, Any]],
        h_limits: tuple[float, float] = (0.0, 10.0),
        c_limits: tuple[float, float] = (0.0, 230.0),
    ):
        import numpy as np
        import textwrap
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D

        figure = Figure(
            figsize=(15.0, 16.0),
            dpi=120,
        )
        canvas = FigureCanvasAgg(figure)
        outer_grid = figure.add_gridspec(
            3,
            3,
            left=0.065,
            right=0.985,
            top=0.975,
            bottom=0.075,
            hspace=0.52,
            wspace=0.24,
        )
        target_color = "#4C78A8"
        prediction_color = "#E02020"

        for example_index, example in enumerate(examples[:9]):
            molecule_row, molecule_column = divmod(example_index, 3)
            molecule_grid = outer_grid[
                molecule_row, molecule_column
            ].subgridspec(2, 1, hspace=0.38)
            wrapped_smiles = "\n".join(
                textwrap.wrap(
                    str(example["smiles"]),
                    width=48,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
            for spectrum_row, (
                target_key,
                prediction_key,
                limits,
                nucleus,
                mae_key,
                precision,
            ) in enumerate(
                (
                    (
                        "h_target",
                        "h_prediction",
                        h_limits,
                        r"$^1$H NMR",
                        "h_nearest_mae_ppm",
                        3,
                    ),
                    (
                        "c_target",
                        "c_prediction",
                        c_limits,
                        r"$^{13}$C NMR",
                        "c_nearest_mae_ppm",
                        2,
                    ),
                )
            ):
                axis = figure.add_subplot(
                    molecule_grid[spectrum_row, 0]
                )
                targets = torch.as_tensor(
                    example[target_key]
                ).reshape(-1).numpy()
                predictions = torch.as_tensor(
                    example[prediction_key]
                ).reshape(-1).numpy()
                axis.vlines(
                    targets,
                    0.0,
                    1.0,
                    color=target_color,
                    linewidth=1.4,
                )
                axis.vlines(
                    predictions,
                    -1.0,
                    0.0,
                    color=prediction_color,
                    linewidth=1.2,
                )
                axis.axhline(0.0, color="black", linewidth=1.0)
                axis.set_xlim(*limits)
                axis.set_ylim(-1.08, 1.08)
                axis.set_yticks((-1.0, 0.0, 1.0))
                mae = example.get(mae_key)
                spectrum_title = nucleus
                if mae is not None:
                    spectrum_title += (
                        f" | symmetric MAE={mae:.{precision}f} ppm"
                    )
                axis.set_title(
                    (
                        f"{wrapped_smiles}\n{spectrum_title}"
                        if spectrum_row == 0
                        else spectrum_title
                    ),
                    fontsize=8.5,
                    pad=5.0,
                )
                axis.tick_params(labelsize=7.5)

        figure.supxlabel(
            "Chemical Shift (ppm)", fontsize=12, y=0.025
        )
        figure.supylabel(
            "NMR stick intensity", fontsize=12, x=0.015
        )
        figure.legend(
            handles=[
                Line2D([0], [0], color=target_color, label="target"),
                Line2D(
                    [0],
                    [0],
                    color=prediction_color,
                    label="model prediction",
                ),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()

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
