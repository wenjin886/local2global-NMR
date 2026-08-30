"""Prior-only adaptation on frozen NMR-to-graph predictions."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from local2geo_module.topology_prior import SoftTopologyPrior
from src.data.dataset import GraphBatch
from src.lit.module import LitNMRToGraph
from src.model.loss import NMRGraphLoss
from src.model.nmr_to_graph import NMRToGraph

from .metrics import graph_exact_match_vectors


def _load_component_checkpoint(
    module: nn.Module,
    path: Optional[str],
    prefixes: Iterable[str],
    strict: bool,
) -> None:
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected:
            state = selected
            break
    incompatible = module.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            f"Loaded {path} non-strictly: "
            f"{len(incompatible.missing_keys)} missing and "
            f"{len(incompatible.unexpected_keys)} unexpected keys"
        )


class PriorOnlyNMRModule(pl.LightningModule):
    """Train only SoftTopologyPrior; never instantiate the 3D pipeline."""

    def __init__(
        self,
        nmr_to_graph: NMRToGraph,
        graph_criterion: NMRGraphLoss,
        topology_prior: SoftTopologyPrior,
        nmr_to_graph_checkpoint: Optional[str] = None,
        topology_prior_checkpoint: Optional[str] = None,
        checkpoint_strict: bool = True,
        corrected_edge_loss_weight: float = 1.0,
        corrected_attachment_loss_weight: float = 1.0,
        corrected_h_count_loss_weight: float = 0.25,
        corrected_neighbor_overflow_loss_weight: float = 1.0,
        corrected_carbon_valence_loss_weight: float = 0.25,
        topology_residual_loss_weight: float = 1e-3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        validation_examples: int = 9,
    ) -> None:
        super().__init__()
        self.nmr_to_graph = nmr_to_graph
        self.graph_criterion = graph_criterion
        if float(self.graph_criterion.smiles_weight) != 0.0:
            warnings.warn(
                "Prior-only validation is greedy; forcing "
                "graph_criterion.smiles_weight=0 for shape-safe graph loss.",
                stacklevel=2,
            )
            self.graph_criterion.smiles_weight = 0.0
        self.topology_prior = topology_prior
        self.save_hyperparameters(
            ignore=["nmr_to_graph", "graph_criterion", "topology_prior"]
        )
        _load_component_checkpoint(
            self.nmr_to_graph,
            nmr_to_graph_checkpoint,
            prefixes=("model.", "nmr_to_graph."),
            strict=checkpoint_strict,
        )
        _load_component_checkpoint(
            self.topology_prior,
            topology_prior_checkpoint,
            prefixes=("prior.", "topology_prior."),
            strict=checkpoint_strict,
        )
        for parameter in self.nmr_to_graph.parameters():
            parameter.requires_grad_(False)
        self.nmr_to_graph.eval()
        self._validation_examples: List[Dict[str, Any]] = []
        for name in ("raw_wrong", "raw_correct", "fixed", "broken"):
            self.register_buffer(
                f"_val_{name}", torch.zeros((), dtype=torch.float64),
                persistent=False,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.nmr_to_graph.eval()
        return self

    @staticmethod
    def _masks(
        atomic_numbers: torch.Tensor, atom_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        atoms = atom_mask.size(1)
        diagonal = torch.eye(
            atoms, device=atom_mask.device, dtype=torch.bool
        )[None]
        heavy_mask = atom_mask & atomic_numbers.ne(1)
        hydrogen_mask = atom_mask & atomic_numbers.eq(1)
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :] & ~diagonal
        return {
            "heavy_mask": heavy_mask,
            "hydrogen_mask": hydrogen_mask,
            "pair_mask": pair_mask,
            "heavy_pair_mask": (
                heavy_mask[:, :, None] & heavy_mask[:, None, :] & ~diagonal
            ),
            "attachment_mask": (
                hydrogen_mask[:, :, None] & heavy_mask[:, None, :]
            ),
        }

    @staticmethod
    def _attachment_probabilities(
        logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        probabilities = torch.softmax(
            logits.masked_fill(~mask, -20.0), dim=-1
        ) * mask.to(logits.dtype)
        return probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

    def forward(
        self, batch: GraphBatch, teacher_force_smiles: bool
    ) -> Dict[str, Any]:
        graph_inputs = batch.model_inputs()
        if not teacher_force_smiles:
            graph_inputs["smiles_input_ids"] = None
            graph_inputs["smiles_input_mask"] = None
        with torch.no_grad():
            raw = self.nmr_to_graph(
                **graph_inputs,
                teacher_force_smiles=teacher_force_smiles,
            )
        if raw.get("heavy_edge_logits") is None:
            raise ValueError("NMRToGraph must predict heavy edges")
        if raw.get("h_attachment_logits") is None:
            raise ValueError("NMRToGraph must predict H attachments")
        masks = self._masks(batch.atom_types, batch.atom_mask)
        corrected = self.topology_prior(
            atomic_numbers=batch.atom_types,
            # GraphBatch has no independent formal-charge observation.
            formal_charges=torch.zeros_like(batch.atom_types),
            atom_mask=batch.atom_mask,
            raw_heavy_edge_logits=raw["heavy_edge_logits"],
            raw_h_attachment_logits=raw["h_attachment_logits"],
            **masks,
        )
        corrected_attachment_probabilities = self._attachment_probabilities(
            corrected["corrected_h_attachment_logits"],
            masks["attachment_mask"],
        )
        corrected_outputs = {
            "heavy_edge_logits": corrected["corrected_heavy_edge_logits"],
            "heavy_edge_mask": masks["heavy_pair_mask"],
            "h_attachment_logits": corrected[
                "corrected_h_attachment_logits"
            ],
            "h_attachment_probabilities": (
                corrected_attachment_probabilities
            ),
            "heavy_mask": masks["heavy_mask"],
            "hydrogen_mask": masks["hydrogen_mask"],
        }
        return {
            "raw": raw,
            "corrected": corrected,
            "corrected_outputs": corrected_outputs,
            "masks": masks,
        }

    def _losses(
        self, batch: GraphBatch, output: Mapping[str, Any]
    ) -> Dict[str, torch.Tensor]:
        raw_loss, raw_parts = self.graph_criterion(
            outputs=output["raw"],
            atom_types=batch.atom_types,
            bond_types=batch.bond_types,
            h_attachment=batch.h_attachment,
            heavy_fragment_labels=batch.heavy_fragment_labels,
            h_parent_fragment_labels=batch.h_parent_fragment_labels,
            h_parent_types=batch.h_parent_types,
            smiles_target_ids=batch.smiles_target_ids,
        )
        corrected_outputs = output["corrected_outputs"]
        corrected_edge = self.graph_criterion.edge_loss(
            corrected_outputs, batch.bond_types
        )
        corrected_attachment = (
            self.graph_criterion._permutation_invariant_attachment_loss(
                corrected_outputs["h_attachment_probabilities"],
                corrected_outputs["hydrogen_mask"],
                batch.h_attachment,
            )
        )
        corrected_h_count = self.graph_criterion.hydrogen_count_loss(
            corrected_outputs["h_attachment_probabilities"],
            corrected_outputs["hydrogen_mask"],
            corrected_outputs["heavy_mask"],
            batch.h_attachment,
        )
        corrected_neighbor_overflow = (
            self.graph_criterion.edge_total_neighbor_count_overflow_loss(
                corrected_outputs, batch.atom_types
            )
        )
        corrected_carbon_valence = self.graph_criterion.carbon_valence_loss(
            corrected_outputs, batch.atom_types
        )
        masks = output["masks"]
        corrected = output["corrected"]
        residuals = []
        edge_residual = corrected["edge_residual"][masks["heavy_pair_mask"]]
        attachment_residual = corrected["attachment_residual"][
            masks["attachment_mask"]
        ]
        if edge_residual.numel():
            residuals.append(edge_residual.square().mean())
        if attachment_residual.numel():
            residuals.append(attachment_residual.square().mean())
        residual = (
            torch.stack(residuals).mean()
            if residuals
            else corrected["edge_residual"].sum() * 0.0
        )
        total = (
            float(self.hparams.corrected_edge_loss_weight) * corrected_edge
            + float(self.hparams.corrected_attachment_loss_weight)
            * corrected_attachment
            + float(self.hparams.corrected_h_count_loss_weight)
            * corrected_h_count
            + float(self.hparams.corrected_neighbor_overflow_loss_weight)
            * corrected_neighbor_overflow
            + float(self.hparams.corrected_carbon_valence_loss_weight)
            * corrected_carbon_valence
            + float(self.hparams.topology_residual_loss_weight) * residual
        )
        losses = {
            "loss": total,
            "loss_raw_graph": raw_loss.detach(),
            "loss_corrected_edge": corrected_edge,
            "loss_corrected_attachment": corrected_attachment,
            "loss_corrected_h_count": corrected_h_count,
            "loss_corrected_neighbor_overflow": corrected_neighbor_overflow,
            "loss_corrected_carbon_valence": corrected_carbon_valence,
            "loss_topology_residual": residual,
        }
        losses.update(
            {f"raw_{key}": value.detach() for key, value in raw_parts.items()}
        )
        return losses

    def _metrics(
        self, batch: GraphBatch, output: Mapping[str, Any]
    ) -> Dict[str, torch.Tensor]:
        raw = graph_exact_match_vectors(
            batch.atom_types,
            batch.atom_mask,
            batch.bond_types,
            batch.h_attachment,
            output["raw"]["heavy_edge_logits"],
            output["raw"]["h_attachment_probabilities"],
        )
        corrected = graph_exact_match_vectors(
            batch.atom_types,
            batch.atom_mask,
            batch.bond_types,
            batch.h_attachment,
            output["corrected_outputs"]["heavy_edge_logits"],
            output["corrected_outputs"]["h_attachment_probabilities"],
        )
        values = {
            "raw_graph_exact_match": raw["typed_exact"].mean(),
            "corrected_graph_exact_match": corrected["typed_exact"].mean(),
            "raw_connectivity_exact_match": raw[
                "connectivity_exact"
            ].mean(),
            "corrected_connectivity_exact_match": corrected[
                "connectivity_exact"
            ].mean(),
            "raw_h_count_exact_match": raw["h_count_exact"].mean(),
            "corrected_h_count_exact_match": corrected[
                "h_count_exact"
            ].mean(),
            "raw_edge_accuracy": raw["edge_accuracy"].mean(),
            "corrected_edge_accuracy": corrected["edge_accuracy"].mean(),
            "raw_connectivity_accuracy": raw[
                "connectivity_accuracy"
            ].mean(),
            "corrected_connectivity_accuracy": corrected[
                "connectivity_accuracy"
            ].mean(),
        }
        values["graph_exact_match_improvement"] = (
            corrected["typed_exact"] - raw["typed_exact"]
        ).mean()
        values["connectivity_exact_improvement"] = (
            corrected["connectivity_exact"]
            - raw["connectivity_exact"]
        ).mean()
        values["_raw_graph_exact_vector"] = raw["typed_exact"]
        values["_corrected_graph_exact_vector"] = corrected["typed_exact"]
        return values

    def _shared_step(
        self,
        batch: GraphBatch,
        stage: str,
        teacher_force_smiles: bool,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        output = self(batch, teacher_force_smiles=teacher_force_smiles)
        losses = self._losses(batch, output)
        metrics = self._metrics(batch, output)
        logged_metrics = {
            key: value for key, value in metrics.items() if not key.startswith("_")
        }
        if stage == "val":
            # These molecule-level exact metrics are reconstructed from global
            # counts in on_validation_epoch_end. This is robust to older
            # Lightning releases and makes the checkpoint monitor explicit.
            for key in (
                "raw_graph_exact_match",
                "corrected_graph_exact_match",
                "graph_exact_match_improvement",
            ):
                logged_metrics.pop(key)
        self.log_dict(
            {
                f"{stage}/{key}": value
                for key, value in {**losses, **logged_metrics}.items()
            },
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
            sync_dist=stage != "train",
            add_dataloader_idx=False,
        )
        return losses["loss"], {**output, **metrics}

    def training_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(
            batch, "train", teacher_force_smiles=True
        )
        return loss

    def validation_step(
        self,
        batch: GraphBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        # Deployment behavior: target SMILES is absent and decoding is greedy.
        if dataloader_idx == 0:
            loss, output = self._shared_step(
                batch, "val", teacher_force_smiles=False
            )
            self._update_fix_break_counts(output)
            if len(getattr(self.trainer, "val_dataloaders", [])) == 1:
                self._collect_validation_examples(batch, output)
            return loss
        output = self(batch, teacher_force_smiles=False)
        self._collect_validation_examples(batch, output)
        return None

    def on_validation_epoch_start(self) -> None:
        self._validation_examples = []
        for name in ("raw_wrong", "raw_correct", "fixed", "broken"):
            getattr(self, f"_val_{name}").zero_()

    def _update_fix_break_counts(self, output: Mapping[str, Any]) -> None:
        raw = output["_raw_graph_exact_vector"].detach().bool()
        corrected = output["_corrected_graph_exact_vector"].detach().bool()
        self._val_raw_wrong.add_((~raw).sum().double())
        self._val_raw_correct.add_(raw.sum().double())
        self._val_fixed.add_(((~raw) & corrected).sum().double())
        self._val_broken.add_((raw & (~corrected)).sum().double())

    def _collect_validation_examples(
        self, batch: GraphBatch, output: Mapping[str, Any]
    ) -> None:
        if self.trainer.sanity_checking:
            return
        remaining = int(self.hparams.validation_examples) - len(
            self._validation_examples
        )
        if remaining <= 0:
            return
        raw_edges = output["raw"]["heavy_edge_logits"].argmax(dim=-1)
        corrected_edges = output["corrected"][
            "corrected_heavy_edge_logits"
        ].argmax(dim=-1)
        raw_attachments = output["raw"]["h_attachment_probabilities"].argmax(
            dim=-1
        )
        corrected_attachments = output["corrected_outputs"][
            "h_attachment_probabilities"
        ].argmax(dim=-1)
        for index in range(min(remaining, batch.atom_types.size(0))):
            mask = batch.atom_mask[index]
            count = int(mask.sum())
            self._validation_examples.append(
                {
                    "atom_types": batch.atom_types[index, :count].detach().cpu(),
                    "target_edges": batch.bond_types[
                        index, :count, :count
                    ].detach().cpu(),
                    "raw_edges": raw_edges[index, :count, :count].detach().cpu(),
                    "corrected_edges": corrected_edges[
                        index, :count, :count
                    ].detach().cpu(),
                    "target_attachments": batch.h_attachment[
                        index, :count
                    ].detach().cpu(),
                    "raw_attachments": raw_attachments[
                        index, :count
                    ].detach().cpu(),
                    "corrected_attachments": corrected_attachments[
                        index, :count
                    ].detach().cpu(),
                }
            )

    @staticmethod
    def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result

    def on_validation_epoch_end(self) -> None:
        raw_wrong = self._distributed_sum(self._val_raw_wrong)
        raw_correct = self._distributed_sum(self._val_raw_correct)
        fixed = self._distributed_sum(self._val_fixed)
        broken = self._distributed_sum(self._val_broken)
        total = raw_wrong + raw_correct
        if total.item() <= 0:
            raise RuntimeError(
                "The full validation loader produced no samples; prior-only "
                "checkpoint metrics cannot be computed."
            )
        raw_exact = raw_correct / total
        corrected_exact = (raw_correct - broken + fixed) / total
        self.log(
            "val/raw_graph_exact_match", raw_exact.float(), sync_dist=False
        )
        self.log(
            "val/corrected_graph_exact_match",
            corrected_exact.float(),
            sync_dist=False,
        )
        self.log(
            "val/graph_exact_match_improvement",
            (corrected_exact - raw_exact).float(),
            sync_dist=False,
        )
        self.log(
            "val/prior_fix_rate",
            (fixed / raw_wrong.clamp_min(1.0)).float(),
            sync_dist=False,
        )
        self.log(
            "val/prior_break_rate",
            (broken / raw_correct.clamp_min(1.0)).float(),
            sync_dist=False,
        )
        self.log("val/raw_wrong_count", raw_wrong.float(), sync_dist=False)
        self.log("val/raw_correct_count", raw_correct.float(), sync_dist=False)
        if self.trainer.sanity_checking:
            return
        examples = self._gather_validation_examples()
        if not self.trainer.is_global_zero or not examples:
            return
        try:
            import wandb
        except ImportError:
            return
        rows = []
        for example in examples:
            atom_mask = torch.ones_like(example["atom_types"], dtype=torch.bool)
            graphs = [
                LitNMRToGraph._explicit_h_graph(
                    example["atom_types"],
                    atom_mask,
                    example[f"{name}_edges"],
                    example[f"{name}_attachments"],
                )
                for name in ("target", "raw", "corrected")
            ]
            positions = LitNMRToGraph._graph_layout(graphs[0])
            rows.append(
                [
                    wandb.Image(LitNMRToGraph._render_graph(graph, positions))
                    for graph in graphs
                ]
            )
        for logger in self.trainer.loggers:
            if hasattr(logger, "log_table"):
                logger.log_table(
                    key="val/prior_only_examples",
                    columns=[
                        "target_graph",
                        "predicted_graph_raw",
                        "predicted_graph_corrected",
                    ],
                    data=rows,
                    step=self.global_step,
                )

    def _gather_validation_examples(self) -> List[Dict[str, Any]]:
        local = self._validation_examples
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return local[: int(self.hparams.validation_examples)]
        gathered: List[Optional[List[Dict[str, Any]]]] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, local)
        merged = [
            example
            for rank_examples in gathered
            if rank_examples is not None
            for example in rank_examples
        ]
        return merged[: int(self.hparams.validation_examples)]

    def transfer_batch_to_device(
        self,
        batch: GraphBatch,
        device: torch.device,
        dataloader_idx: int,
    ) -> GraphBatch:
        return batch.to(device)

    def configure_optimizers(self):
        parameters = [
            parameter
            for parameter in self.topology_prior.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(self.trainer.max_epochs))
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
