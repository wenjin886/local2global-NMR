from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch

from .corruption import SoftGraphCorruptor
from .loss import ProjectionGeometryLoss
from .metrics import local2geo_metrics
from .model import Local2GeoModel
from .visualization import (
    graph_image,
    sample_geometry_summary,
    write_sdf,
)


class LitLocal2Geo(pl.LightningModule):
    def __init__(
        self,
        model: Local2GeoModel,
        criterion: ProjectionGeometryLoss,
        train_corruptor: SoftGraphCorruptor,
        val_corruptor: SoftGraphCorruptor,
        lr: float = 2e-4,
        weight_decay: float = 1e-6,
        warm_up_steps: int = 500,
        val_corruption_seed: int = 1729,
        num_val_structures_to_log: int = 10,
        visualize_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_corruptor = train_corruptor
        self.val_corruptor = val_corruptor
        self.save_hyperparameters(ignore=[
            "model", "criterion", "train_corruptor", "val_corruptor"
        ])
        self._visualization_samples = []
        self._visualization_seen = 0
        self._visualization_rng = random.Random(val_corruption_seed)

    def _corrupt(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        corruptor = (
            self.train_corruptor if stage == "train" else self.val_corruptor
        )
        if stage == "train":
            return corruptor(
                batch["bond_types"],
                batch["heavy_pair_mask"],
                batch["h_attachment"],
                batch["attachment_mask"],
            )
        devices = []
        if batch["bond_types"].is_cuda:
            devices = [batch["bond_types"].device]
        with torch.random.fork_rng(devices=devices):
            seed = self.hparams.val_corruption_seed + batch_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            return corruptor(
                batch["bond_types"],
                batch["heavy_pair_mask"],
                batch["h_attachment"],
                batch["attachment_mask"],
            )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_graph: Dict[str, torch.Tensor],
        differentiable_relaxation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self.model(batch, noisy_graph, differentiable_relaxation)

    def _target_lengths(
        self,
        batch: Dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.model.relaxation.target_lengths(
            torch.nn.functional.one_hot(
                batch["bond_types"].clamp_min(0), num_classes=5
            ).to(dtype),
            batch["covalent_radii"],
        )

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        noisy = self._corrupt(batch, stage, batch_idx)
        outputs = self(
            batch,
            noisy,
            differentiable_relaxation=stage == "train",
        )
        clean_terms = self.model.clean_geometry_terms(
            batch, outputs["coordinates"]
        )
        loss, losses = self.criterion(outputs, batch, clean_terms)
        batch_size = batch["atom_mask"].size(0)
        self.log_dict(
            {f"{stage}/loss_{key}": value for key, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=False,
        )
        with torch.no_grad():
            target_lengths = self._target_lengths(
                batch, outputs["coordinates"].dtype
            )
            metrics = local2geo_metrics(
                outputs,
                batch,
                target_lengths,
                clash_distance_scale=self.model.relaxation.clash_distance_scale,
            )
        self.log_dict(
            {f"{stage}/{key}": value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=stage != "train",
        )
        if stage == "val":
            self._collect_visualization_samples(
                batch, outputs, target_lengths
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def on_validation_epoch_start(self) -> None:
        self._visualization_samples = []
        self._visualization_seen = 0
        self._visualization_rng = random.Random(
            int(self.hparams.val_corruption_seed) + int(self.current_epoch)
        )

    def _should_visualize(self) -> bool:
        frequency = int(self.hparams.visualize_every_n_epochs)
        return (
            int(self.hparams.num_val_structures_to_log) > 0
            and frequency > 0
            and int(self.current_epoch) % frequency == 0
            and not self.trainer.sanity_checking
        )

    def _collect_visualization_samples(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        target_lengths: torch.Tensor,
    ) -> None:
        if not self._should_visualize():
            return
        capacity = int(self.hparams.num_val_structures_to_log)
        for index in range(batch["atom_mask"].size(0)):
            self._visualization_seen += 1
            if len(self._visualization_samples) < capacity:
                slot = len(self._visualization_samples)
            else:
                slot = self._visualization_rng.randrange(
                    self._visualization_seen
                )
                if slot >= capacity:
                    continue
            size = int(batch["atom_mask"][index].sum())
            sample = {
                "smiles": batch["smiles"][index],
                "atom_mask": batch["atom_mask"][index, :size].detach().cpu(),
                "atomic_numbers": batch[
                    "atomic_numbers"
                ][index, :size].detach().cpu(),
                "formal_charges": batch[
                    "formal_charges"
                ][index, :size].detach().cpu(),
                "hydrogen_mask": batch[
                    "hydrogen_mask"
                ][index, :size].detach().cpu(),
                "vdw_radii": batch[
                    "vdw_radii"
                ][index, :size].detach().cpu(),
                "bond_types": batch[
                    "bond_types"
                ][index, :size, :size].detach().cpu().clamp_min(0),
                "predicted_bond_types": outputs[
                    "projected_edge_probabilities"
                ][index, :size, :size].detach().cpu().argmax(dim=-1),
                "seed_coordinates": outputs[
                    "seed_coordinates"
                ][index, :size].detach().cpu(),
                "coordinates": outputs[
                    "coordinates"
                ][index, :size].detach().cpu(),
                "target_lengths": target_lengths[
                    index, :size, :size
                ].detach().cpu(),
            }
            if slot == len(self._visualization_samples):
                self._visualization_samples.append(sample)
            else:
                self._visualization_samples[slot] = sample

    def on_validation_epoch_end(self) -> None:
        if (
            not self._visualization_samples
            or self.trainer.sanity_checking
            or not self.trainer.is_global_zero
        ):
            return
        try:
            import wandb
        except ImportError:
            return
        columns = [
            "epoch",
            "smiles",
            "num_atoms",
            "num_hydrogens",
            "clean_2d_graph",
            "projected_2d_graph",
            "seed_3d",
            "relaxed_3d",
            "bond_mae_angstrom",
            "min_nonbond_vdw_ratio",
        ]
        rows = []
        with tempfile.TemporaryDirectory(prefix="local2geo_wandb_") as directory:
            directory = Path(directory)
            for index, sample in enumerate(self._visualization_samples):
                stem = f"epoch_{int(self.current_epoch):04d}_{index:02d}"
                seed_path = directory / f"{stem}_seed.sdf"
                relaxed_path = directory / f"{stem}_relaxed.sdf"
                write_sdf(
                    seed_path,
                    sample["atomic_numbers"],
                    sample["formal_charges"],
                    sample["bond_types"],
                    sample["seed_coordinates"],
                    f"{sample['smiles']} seed",
                )
                write_sdf(
                    relaxed_path,
                    sample["atomic_numbers"],
                    sample["formal_charges"],
                    sample["bond_types"],
                    sample["coordinates"],
                    f"{sample['smiles']} relaxed",
                )
                summary = sample_geometry_summary(
                    sample,
                    sample["coordinates"],
                    sample["target_lengths"],
                )
                rows.append([
                    int(self.current_epoch),
                    sample["smiles"],
                    int(sample["atomic_numbers"].numel()),
                    int(sample["hydrogen_mask"].sum()),
                    wandb.Image(
                        graph_image(
                            sample["atomic_numbers"],
                            sample["formal_charges"],
                            sample["bond_types"],
                        ),
                        caption="clean all-atom graph",
                    ),
                    wandb.Image(
                        graph_image(
                            sample["atomic_numbers"],
                            sample["formal_charges"],
                            sample["predicted_bond_types"],
                        ),
                        caption="projected all-atom graph",
                    ),
                    wandb.Molecule(
                        str(seed_path.resolve()),
                        caption=f"{sample['smiles']} seed",
                    ),
                    wandb.Molecule(
                        str(relaxed_path.resolve()),
                        caption=f"{sample['smiles']} relaxed",
                    ),
                    summary["bond_mae_angstrom"],
                    summary["min_nonbond_vdw_ratio"],
                ])
            for logger in self.trainer.loggers:
                if hasattr(logger, "log_table"):
                    logger.log_table(
                        key="val/random_structures",
                        columns=columns,
                        data=rows,
                        step=self.global_step,
                    )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = max(int(self.hparams.warm_up_steps), 1)

        def schedule(step: int) -> float:
            return min((step + 1) / warmup, 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
