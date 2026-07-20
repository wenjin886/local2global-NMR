from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from src.data.dataset import GraphBatch
from src.model.loss import NMRGraphLoss
from src.model.nmr_to_graph import NMRToGraph


class LitNMRToGraph(pl.LightningModule):
    def __init__(
            self,
            model: NMRToGraph,
            criterion: NMRGraphLoss,
            lr: float = 2e-4,
            weight_decay: float = 1e-12,
            warm_up_steps: int = 100,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, batch: GraphBatch) -> Dict[str, object]:
        return self.model(**batch.model_inputs())

    def basic_step(
            self,
            batch: GraphBatch,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
        outputs = self(batch)
        loss, losses = self.criterion(
            outputs=outputs,
            atom_types=batch.atom_types,
            bond_types=batch.bond_types,
            h_attachment=batch.h_attachment,
            heavy_fragment_labels=batch.heavy_fragment_labels,
            h_parent_fragment_labels=batch.h_parent_fragment_labels,
            h_parent_types=batch.h_parent_types,
            smiles_target_ids=batch.smiles_target_ids,
        )
        return loss, losses, outputs

    def _shared_step(self, batch: GraphBatch, stage: str) -> torch.Tensor:
        loss, losses, outputs = self.basic_step(batch)
        self.log_dict(
            {"%s/loss_%s" % (stage, key): value for key, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
        )
        metrics = self._batch_metrics(outputs, batch)
        self.log_dict(
            {"%s/%s" % (stage, key): value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
        )
        return loss

    def training_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    @staticmethod
    def _batch_metrics(
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> Dict[str, torch.Tensor]:
        edge_mask = outputs["heavy_edge_mask"] & torch.triu(
            torch.ones_like(outputs["heavy_edge_mask"], dtype=torch.bool),
            diagonal=1,
        )
        edge_mask = edge_mask & batch.bond_types.ge(0)
        if edge_mask.any():
            edge_prediction = outputs["heavy_edge_logits"].argmax(dim=-1)
            edge_accuracy = (
                edge_prediction[edge_mask] == batch.bond_types[edge_mask]
            ).float().mean()
        else:
            edge_accuracy = outputs["heavy_edge_logits"].sum() * 0.0

        predicted_counts = outputs["h_attachment_probabilities"].sum(dim=1)
        target_counts = torch.zeros_like(predicted_counts)
        for sample_index in range(batch.atom_types.size(0)):
            valid = outputs["hydrogen_mask"][sample_index] & batch.h_attachment[
                sample_index
            ].ge(0)
            targets = batch.h_attachment[sample_index, valid]
            if targets.numel() > 0:
                target_counts[sample_index].scatter_add_(
                    0,
                    targets,
                    torch.ones_like(targets, dtype=target_counts.dtype),
                )
        heavy_mask = outputs["heavy_mask"]
        h_count_mae = (
            (predicted_counts[heavy_mask] - target_counts[heavy_mask]).abs().mean()
            if heavy_mask.any()
            else predicted_counts.sum() * 0.0
        )
        fragment_targets = batch.heavy_fragment_labels
        fragment_valid = heavy_mask.unsqueeze(-1) & fragment_targets.ge(0)
        if fragment_valid.any():
            fragment_prediction = outputs["fragment_logits"].argmax(dim=-1)
            fragment_count_accuracy = (
                fragment_prediction[fragment_valid]
                == fragment_targets[fragment_valid]
            ).float().mean()
            fragment_presence_accuracy = (
                fragment_prediction[fragment_valid].gt(0)
                == fragment_targets[fragment_valid].gt(0)
            ).float().mean()
        else:
            fragment_count_accuracy = outputs["fragment_logits"].sum() * 0.0
            fragment_presence_accuracy = fragment_count_accuracy
        metrics = {
            "edge_accuracy": edge_accuracy,
            "fragment_count_accuracy": fragment_count_accuracy,
            "fragment_presence_accuracy": fragment_presence_accuracy,
            "h_count_mae": h_count_mae,
        }
        if outputs.get("smiles_token_ids") is not None:
            valid = batch.smiles_target_ids.ne(0)
            predictions = outputs["smiles_token_ids"]
            correct = predictions.eq(batch.smiles_target_ids) & valid
            metrics["smiles_token_accuracy"] = (
                correct.sum().float() / valid.sum().clamp_min(1)
            )
            metrics["smiles_exact_match"] = (
                (correct | ~valid).all(dim=1).float().mean()
            )
        return metrics

    def transfer_batch_to_device(
            self,
            batch: GraphBatch,
            device: torch.device,
            dataloader_idx: int,
    ) -> GraphBatch:
        return batch.to(device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.warm_up_steps <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,
            total_iters=self.hparams.warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
