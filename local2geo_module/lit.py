from __future__ import annotations

from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from .corruption import SoftGraphCorruptor
from .loss import ProjectionGeometryLoss
from .metrics import local2geo_metrics
from .model import Local2GeoModel


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
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_corruptor = train_corruptor
        self.val_corruptor = val_corruptor
        self.save_hyperparameters(ignore=[
            "model", "criterion", "train_corruptor", "val_corruptor"
        ])

    def _corrupt(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        batch_idx: int,
    ) -> torch.Tensor:
        if stage == "train":
            return self.train_corruptor(batch["bond_types"], batch["pair_mask"])
        devices = []
        if batch["bond_types"].is_cuda:
            devices = [batch["bond_types"].device]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.hparams.val_corruption_seed + batch_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.hparams.val_corruption_seed + batch_idx)
            return self.val_corruptor(batch["bond_types"], batch["pair_mask"])

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_edge_logits: torch.Tensor,
        differentiable_relaxation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        return self.model(batch, noisy_edge_logits, differentiable_relaxation)

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
        clean_terms = self.model.clean_geometry_terms(batch, outputs["coordinates"])
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
            target_lengths = self.model.relaxation.target_lengths(
                torch.nn.functional.one_hot(
                    batch["bond_types"].clamp_min(0), num_classes=5
                ).to(outputs["coordinates"].dtype),
                batch["covalent_radii"],
            )
            metrics = local2geo_metrics(outputs, batch, target_lengths)
        self.log_dict(
            {f"{stage}/{key}": value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=stage != "train",
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

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
