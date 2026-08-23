from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.data.constants import (
    BOND_TYPE_CANDIDATES,
    SMILES_BOS_INDEX,
    SMILES_EOS_INDEX,
    SMILES_PAD_INDEX,
)
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
            num_val_examples_to_log: int = 10,
            inference_only_validation: bool = False,
            validation_stage: str = "graph",
            check_bond_order: bool = False,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_val_examples_to_log = num_val_examples_to_log
        self.inference_only_validation = inference_only_validation
        if validation_stage not in {"smiles", "fragment", "graph"}:
            raise ValueError(
                "validation_stage must be 'smiles', 'fragment', or 'graph'"
            )
        self.validation_stage = validation_stage
        self._validation_examples = []
        self._graph_validation_examples = []
        # Deprecated no-op accepted only so an existing experiment config can
        # still instantiate this module while resuming its checkpoint.
        _ = check_bond_order
        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(
            self,
            batch: GraphBatch,
            teacher_force_smiles: Optional[bool] = None,
    ) -> Dict[str, object]:
        return self.model(
            **batch.model_inputs(),
            teacher_force_smiles=teacher_force_smiles,
        )

    def basic_step(
            self,
            batch: GraphBatch,
            teacher_force_smiles: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]:
        outputs = self(batch, teacher_force_smiles=teacher_force_smiles)
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

    def _shared_step(
            self,
            batch: GraphBatch,
            stage: str,
            teacher_force_smiles: Optional[bool] = None,
    ) -> torch.Tensor:
        loss, losses, outputs = self.basic_step(
            batch, teacher_force_smiles=teacher_force_smiles
        )
        self.log_dict(
            {"%s/loss_%s" % (stage, key): value for key, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
            add_dataloader_idx=False,
        )
        if stage != "train":
            metrics = self._batch_metrics(
                outputs,
                batch,
                smiles_mode="teacher" if teacher_force_smiles else "greedy",
            )
            self.log_dict(
                {"%s/%s" % (stage, key): value for key, value in metrics.items()},
                on_step=False,
                on_epoch=True,
                batch_size=batch.atom_types.size(0),
                add_dataloader_idx=False,
            )
            if self.validation_stage in {"fragment", "graph"}:
                self._log_fragment_carbon_valence_metrics(
                    stage, outputs, batch
                )
        return loss

    def training_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", teacher_force_smiles=True)

    def validation_step(
            self,
            batch: GraphBatch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        if self.inference_only_validation:
            if dataloader_idx == 0:
                loss, losses, outputs = self.basic_step(
                    batch, teacher_force_smiles=True
                )
                self.log_dict(
                    {
                        "val/loss_%s" % key: value
                        for key, value in losses.items()
                    },
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch.atom_types.size(0),
                    add_dataloader_idx=False,
                )
                if self.validation_stage in {"fragment", "graph"}:
                    self._log_fragment_carbon_valence_metrics(
                        "val", outputs, batch
                    )
                return loss

            outputs = self(batch, teacher_force_smiles=False)
            metrics = self._inference_metrics(outputs, batch)
            self.log_dict(
                {
                    "val_inference/%s" % key: value
                    for key, value in metrics.items()
                },
                on_step=False,
                on_epoch=True,
                batch_size=batch.atom_types.size(0),
                add_dataloader_idx=False,
            )
            if self.validation_stage in {"fragment", "graph"}:
                self._log_fragment_carbon_valence_metrics(
                    "val_inference", outputs, batch
                )
            self._collect_validation_examples(outputs, batch)
            self._collect_graph_validation_examples(outputs, batch)
            return None
        if dataloader_idx == 0:
            return self._shared_step(
                batch, "val", teacher_force_smiles=True
            )
        outputs = self(batch, teacher_force_smiles=False)
        metrics = self._smiles_metrics(outputs, batch, smiles_mode="greedy")
        self.log_dict(
            {"val_generation/%s" % key: value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
            add_dataloader_idx=False,
        )
        self._collect_validation_examples(outputs, batch)
        self._collect_graph_validation_examples(outputs, batch)

    def _inference_metrics(
            self,
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> Dict[str, torch.Tensor]:
        """Small, stage-aware metric set from the exact inference path."""
        metrics = {}
        predictions = outputs.get("smiles_token_ids")
        if predictions is not None:
            valid = batch.smiles_target_ids.ne(SMILES_PAD_INDEX)
            correct = predictions.eq(batch.smiles_target_ids)
            metrics["smiles_exact_accuracy"] = (
                (correct | ~valid).all(dim=1).float().mean()
            )
        if self.validation_stage in {"fragment", "graph"}:
            metrics["heavy_fragment_score"] = self._heavy_fragment_score(
                outputs, batch
            )
        if self.validation_stage == "graph":
            graph_metrics = self._graph_metrics(outputs, batch)
            for key in (
                "graph_score",
                "edge_precision",
                "edge_recall",
                "predicted_to_target_edge_count_ratio",
            ):
                metrics[key] = graph_metrics[key]
        return metrics

    @staticmethod
    def _fragment_carbon_valence_metrics(
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> Dict[str, Tuple[torch.Tensor, int]]:
        fragment_counts = outputs["fragment_logits"].argmax(dim=-1)
        valences = NMRGraphLoss.fragment_carbon_valences_from_counts(
            fragment_counts
        )
        carbon_mask = outputs["heavy_mask"] & batch.atom_types.eq(6)
        valid_carbon = carbon_mask & valences.eq(4)
        dtype = outputs["fragment_logits"].dtype

        num_carbons = int(carbon_mask.sum().item())
        num_valid_carbons = valid_carbon.sum().to(dtype=dtype)
        carbons_per_molecule = carbon_mask.sum(dim=-1)
        valid_per_molecule = valid_carbon.sum(dim=-1)
        carbon_containing = carbons_per_molecule.gt(0)
        num_carbon_molecules = int(carbon_containing.sum().item())
        num_all_valid_molecules = (
            valid_per_molecule.eq(carbons_per_molecule) & carbon_containing
        ).sum().to(dtype=dtype)
        invalid_carbons = (
            carbons_per_molecule - valid_per_molecule
        ).sum().to(dtype=dtype)
        num_molecules = batch.atom_types.size(0)

        return {
            "fragment_carbon_valence_accuracy": (
                num_valid_carbons / max(num_carbons, 1), num_carbons
            ),
            "fragment_molecule_all_carbon_valid_rate": (
                num_all_valid_molecules / max(num_carbon_molecules, 1),
                num_carbon_molecules,
            ),
            "fragment_average_invalid_carbons_per_molecule": (
                invalid_carbons / max(num_molecules, 1), num_molecules
            ),
        }

    def _log_fragment_carbon_valence_metrics(
            self,
            stage: str,
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> None:
        metrics = self._fragment_carbon_valence_metrics(outputs, batch)
        for name, (value, denominator) in metrics.items():
            if denominator == 0:
                continue
            # Weight epoch reduction by the metric's true denominator. This
            # gives exact carbon-level and molecule-level rates across batches.
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=denominator,
                add_dataloader_idx=False,
            )

    @staticmethod
    def _heavy_fragment_score(
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> torch.Tensor:
        predictions = outputs["fragment_logits"].argmax(dim=-1)
        scores = []
        for sample_index in range(predictions.size(0)):
            heavy = outputs["heavy_mask"][sample_index]
            target = batch.heavy_fragment_labels[sample_index, heavy]
            predicted = predictions[sample_index, heavy]
            valid = target.ge(0)
            if target.numel() == 0:
                continue
            target_presence = target.gt(0)
            predicted_presence = predicted.gt(0)
            tp = (
                valid & target_presence & predicted_presence
            ).sum(dim=0).float()
            fp = (
                valid & ~target_presence & predicted_presence
            ).sum(dim=0).float()
            fn = (
                valid & target_presence & ~predicted_presence
            ).sum(dim=0).float()
            denominator = 2 * tp + fp + fn
            supported = denominator.gt(0)
            presence_f1 = (
                (2 * tp[supported] / denominator[supported]).mean()
                if supported.any()
                else predictions.sum() * 0.0 + 1.0
            )
            positive = valid & target.gt(0)
            positive_accuracy = (
                predicted[positive].eq(target[positive]).float().mean()
                if positive.any()
                else predictions.sum() * 0.0 + 1.0
            )
            scores.append(
                torch.sqrt(
                    (presence_f1 * positive_accuracy).clamp_min(0.0)
                )
            )
        if scores:
            return torch.stack(scores).mean()
        return predictions.sum() * 0.0

    def test_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test", teacher_force_smiles=False)

    def _batch_metrics(
            self,
            outputs: Dict[str, object],
            batch: GraphBatch,
            smiles_mode: str,
    ) -> Dict[str, torch.Tensor]:
        zero = outputs["fragment_logits"].sum() * 0.0
        graph_metrics = (
            self._graph_metrics(outputs, batch)
            if outputs.get("heavy_edge_logits") is not None
            else {}
        )

        fragment_prediction = outputs["fragment_logits"].argmax(dim=-1)
        fragment_scores = []
        presence_macro_f1s = []
        positive_count_accuracies = []
        h_parent_type_accuracies = []
        h_parent_fragment_accuracies = []
        h_attachment_accuracies = []
        h_count_maes = []
        heavy_neighbor_count_violation_rates = []
        for sample_index in range(batch.atom_types.size(0)):
            heavy = outputs["heavy_mask"][sample_index]
            target_fragment = batch.heavy_fragment_labels[sample_index, heavy]
            predicted_fragment = fragment_prediction[sample_index, heavy]
            valid_fragment = target_fragment.ge(0)
            if target_fragment.numel() > 0:
                target_presence = target_fragment.gt(0)
                predicted_presence = predicted_fragment.gt(0)
                tp = (
                    valid_fragment & target_presence & predicted_presence
                ).sum(dim=0).float()
                fp = (
                    valid_fragment & ~target_presence & predicted_presence
                ).sum(dim=0).float()
                fn = (
                    valid_fragment & target_presence & ~predicted_presence
                ).sum(dim=0).float()
                denominator = 2 * tp + fp + fn
                supported = denominator.gt(0)
                presence_f1 = (
                    (2 * tp[supported] / denominator[supported]).mean()
                    if supported.any() else zero + 1.0
                )
                positive = valid_fragment & target_fragment.gt(0)
                positive_accuracy = (
                    predicted_fragment[positive].eq(target_fragment[positive]).float().mean()
                    if positive.any() else zero + 1.0
                )
                presence_macro_f1s.append(presence_f1)
                positive_count_accuracies.append(positive_accuracy)
                fragment_scores.append(
                    torch.sqrt((presence_f1 * positive_accuracy).clamp_min(0.0))
                )
                neighbor_count_caps = (
                    self.criterion.heavy_neighbor_count_caps(
                        batch.atom_types[sample_index]
                    )[heavy]
                )
                predicted_neighbor_counts = predicted_fragment.sum(dim=-1)
                heavy_neighbor_count_violation_rates.append(
                    predicted_neighbor_counts.gt(neighbor_count_caps).float().mean()
                )

            hydrogen = outputs["hydrogen_mask"][sample_index]
            target_hydrogen = hydrogen & batch.h_parent_types[sample_index].ge(0)
            predicted_rows = hydrogen.nonzero(as_tuple=False).flatten()
            target_rows = target_hydrogen.nonzero(as_tuple=False).flatten()
            if predicted_rows.numel() and target_rows.numel():
                predicted_type_index = outputs["h_parent_type_logits"][
                    sample_index, predicted_rows
                ].argmax(dim=-1)
                predicted_types = outputs["parent_atom_types"][predicted_type_index]
                target_types = batch.h_parent_types[sample_index, target_rows]
                predicted_parent_fragments = outputs["h_parent_fragment_logits"][
                    sample_index, predicted_rows
                ].argmax(dim=-1)
                target_parent_fragments = batch.h_parent_fragment_labels[
                    sample_index, target_rows
                ]
                type_cost = predicted_types[:, None].ne(target_types[None, :]).float()
                fragment_cost = predicted_parent_fragments[:, None].ne(
                    target_parent_fragments[None, :]
                ).float().mean(dim=-1)
                row, column = linear_sum_assignment(
                    (type_cost + fragment_cost).detach().cpu().numpy()
                )
                row = torch.as_tensor(row, device=predicted_types.device)
                column = torch.as_tensor(column, device=predicted_types.device)
                type_correct = predicted_types[row].eq(target_types[column])
                fragment_correct = predicted_parent_fragments[row].eq(
                    target_parent_fragments[column]
                ).all(dim=-1)
                h_parent_type_accuracies.append(type_correct.float().mean())
                h_parent_fragment_accuracies.append(fragment_correct.float().mean())

                if outputs.get("h_attachment_logits") is not None:
                    predicted_attachment = outputs["h_attachment_logits"][
                        sample_index, predicted_rows
                    ].argmax(dim=-1)
                    target_attachment = batch.h_attachment[sample_index, target_rows]
                    num_atoms = batch.atom_types.size(1)
                    predicted_histogram = torch.bincount(
                        predicted_attachment, minlength=num_atoms
                    )
                    target_histogram = torch.bincount(
                        target_attachment, minlength=num_atoms
                    )
                    h_attachment_accuracies.append(
                        torch.minimum(
                            predicted_histogram, target_histogram
                        ).sum().float() / target_attachment.numel()
                    )

                    soft_counts = outputs["h_attachment_probabilities"][
                        sample_index
                    ].sum(dim=0)
                    h_count_maes.append(
                        (soft_counts[heavy] - target_histogram[heavy]).abs().mean()
                    )

        def mean(values):
            return torch.stack(values).mean() if values else zero

        metrics = {
            "heavy_fragment_score": mean(fragment_scores),
            "heavy_fragment_presence_macro_f1": mean(presence_macro_f1s),
            "heavy_fragment_positive_count_accuracy": mean(
                positive_count_accuracies
            ),
            "h_parent_type_accuracy": mean(h_parent_type_accuracies),
            "h_parent_fragment_exact_accuracy": mean(
                h_parent_fragment_accuracies
            ),
            "h_attachment_multiset_accuracy": mean(h_attachment_accuracies),
            "h_count_mae": mean(h_count_maes),
            "heavy_neighbor_count_violation_rate": mean(
                heavy_neighbor_count_violation_rates
            ),
        }
        metrics.update(graph_metrics)
        if outputs.get("h_attachment_logits") is None:
            metrics.pop("h_attachment_multiset_accuracy")
            metrics.pop("h_count_mae")
        metrics.update(self._smiles_metrics(outputs, batch, smiles_mode))
        return metrics

    @staticmethod
    def _graph_metrics(
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> Dict[str, torch.Tensor]:
        """Molecule-macro heavy-edge metrics without none-class domination."""
        predictions = outputs["heavy_edge_logits"].argmax(dim=-1)
        graph_scores = []
        existence_f1s = []
        existence_precisions = []
        existence_recalls = []
        predicted_to_target_ratios = []
        typed_bond_recalls = []
        for sample_index in range(predictions.size(0)):
            valid_pairs = outputs["heavy_edge_mask"][sample_index] & torch.triu(
                torch.ones_like(
                    outputs["heavy_edge_mask"][sample_index],
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            valid_pairs &= batch.bond_types[sample_index].ge(0)
            predicted = predictions[sample_index, valid_pairs]
            target = batch.bond_types[sample_index, valid_pairs]
            if target.numel() == 0:
                continue

            predicted_exists = predicted.ne(0)
            target_exists = target.ne(0)
            true_positive = (predicted_exists & target_exists).sum().float()
            predicted_positive = predicted_exists.sum().float()
            target_positive = target_exists.sum().float()

            precision = torch.where(
                predicted_positive.gt(0),
                true_positive / predicted_positive.clamp_min(1.0),
                target_positive.eq(0).to(dtype=true_positive.dtype),
            )
            recall = torch.where(
                target_positive.gt(0),
                true_positive / target_positive.clamp_min(1.0),
                torch.ones_like(true_positive),
            )
            existence_f1 = torch.where(
                (precision + recall).gt(0),
                2 * precision * recall / (precision + recall),
                torch.zeros_like(precision),
            )
            typed_correct = (
                predicted.eq(target) & target_exists
            ).sum().float()
            typed_bond_recall = torch.where(
                target_positive.gt(0),
                typed_correct / target_positive.clamp_min(1.0),
                predicted_positive.eq(0).to(dtype=typed_correct.dtype),
            )
            graph_score = torch.sqrt(
                (existence_f1 * typed_bond_recall).clamp_min(0.0)
            )
            existence_f1s.append(existence_f1)
            existence_precisions.append(precision)
            existence_recalls.append(recall)
            predicted_to_target_ratios.append(
                predicted_positive / target_positive.clamp_min(1.0)
            )
            typed_bond_recalls.append(typed_bond_recall)
            graph_scores.append(graph_score)

        zero = outputs["heavy_edge_logits"].sum() * 0.0

        def mean(values):
            return torch.stack(values).mean() if values else zero

        return {
            "graph_score": mean(graph_scores),
            "bond_existence_f1": mean(existence_f1s),
            "edge_precision": mean(existence_precisions),
            "edge_recall": mean(existence_recalls),
            "predicted_to_target_edge_count_ratio": mean(
                predicted_to_target_ratios
            ),
            "typed_bond_recall": mean(typed_bond_recalls),
        }

    def _smiles_metrics(
            self,
            outputs: Dict[str, object],
            batch: GraphBatch,
            smiles_mode: str,
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        if outputs.get("smiles_token_ids") is not None:
            valid = batch.smiles_target_ids.ne(SMILES_PAD_INDEX)
            predictions = outputs["smiles_token_ids"]
            correct = predictions.eq(batch.smiles_target_ids) & valid
            logits = outputs.get("smiles_logits")
            if smiles_mode == "teacher" and logits is not None:
                metrics["smiles_teacher_token_accuracy"] = (
                    correct.sum().float() / valid.sum().clamp_min(1)
                )
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch.smiles_target_ids.reshape(-1),
                    ignore_index=SMILES_PAD_INDEX,
                )
                metrics["smiles_teacher_perplexity"] = token_loss.clamp_max(20).exp()
            if smiles_mode == "greedy" and self.model.smiles_vocab is not None:
                metrics["smiles_greedy_exact_match"] = (
                    (correct | ~valid).all(dim=1).float().mean()
                )
                (
                    validity,
                    stereo_agnostic,
                    composition_exact,
                ) = self._greedy_smiles_metrics(
                    predictions,
                    batch.smiles_target_ids,
                    batch.atom_types,
                    batch.atom_mask,
                )
                metrics["smiles_greedy_validity"] = validity
                metrics["smiles_greedy_stereo_agnostic_exact_match"] = (
                    stereo_agnostic
                )
                metrics["smiles_greedy_element_composition_exact"] = (
                    composition_exact
                )
        return metrics

    def on_validation_epoch_start(self) -> None:
        self._validation_examples = []
        self._graph_validation_examples = []

    def _format_fragments(
            self,
            atom_types: torch.Tensor,
            fragment_counts: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> str:
        rows = []
        neighbor_count_caps = self.criterion.heavy_neighbor_count_caps(atom_types)
        for atom_index in heavy_mask.nonzero(as_tuple=False).flatten().tolist():
            counts = fragment_counts[atom_index]
            neighbor_count = int(counts.clamp_min(0).sum())
            max_neighbors = int(neighbor_count_caps[atom_index])
            ports = [
                f"{candidate}x{int(count)}"
                for candidate, count in zip(BOND_TYPE_CANDIDATES, counts.tolist())
                if count > 0
            ]
            rows.append(
                f"{atom_index}:Z{int(atom_types[atom_index])}"
                f"(neighbors={neighbor_count}/{max_neighbors})="
                + (",".join(ports) if ports else "none")
            )
        return "; ".join(rows)

    def _collect_validation_examples(
            self,
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> None:
        if (
            self.num_val_examples_to_log <= 0
            or self.trainer.sanity_checking
            or outputs.get("smiles_token_ids") is None
            or self.model.smiles_vocab is None
        ):
            return
        remaining = self.num_val_examples_to_log - len(self._validation_examples)
        if remaining <= 0:
            return
        fragment_predictions = outputs["fragment_logits"].argmax(dim=-1)
        for sample_index in range(min(remaining, batch.atom_types.size(0))):
            target_smiles = self._decode_smiles(batch.smiles_target_ids[sample_index])
            predicted_smiles = self._decode_smiles(
                outputs["smiles_token_ids"][sample_index]
            )
            from rdkit import Chem, rdBase
            with rdBase.BlockLogs():
                predicted_molecule = Chem.MolFromSmiles(predicted_smiles)
                target_molecule = Chem.MolFromSmiles(target_smiles)
            predicted_valid = predicted_molecule is not None
            non_stereo_exact = False
            target_composition = self._composition_from_atomic_numbers(
                batch.atom_types[
                    sample_index, batch.atom_mask[sample_index]
                ].tolist()
            )
            predicted_composition = None
            if predicted_molecule is not None and target_molecule is not None:
                predicted_non_stereo = Chem.MolToSmiles(
                    predicted_molecule, canonical=True, isomericSmiles=False
                )
                target_non_stereo = Chem.MolToSmiles(
                    target_molecule, canonical=True, isomericSmiles=False
                )
                non_stereo_exact = predicted_non_stereo == target_non_stereo
            if predicted_molecule is not None:
                predicted_composition = self._molecule_composition(
                    predicted_molecule
                )
            target_fragments = ""
            predicted_fragments = ""
            if self.validation_stage != "smiles":
                heavy_mask = outputs["heavy_mask"][sample_index]
                target_fragments = self._format_fragments(
                    batch.atom_types[sample_index],
                    batch.heavy_fragment_labels[sample_index],
                    heavy_mask,
                )
                predicted_fragments = self._format_fragments(
                    batch.atom_types[sample_index],
                    fragment_predictions[sample_index],
                    heavy_mask,
                )
            self._validation_examples.append([
                int(self.current_epoch),
                target_smiles,
                predicted_smiles,
                predicted_smiles == target_smiles,
                predicted_valid,
                non_stereo_exact,
                self._format_composition(target_composition),
                self._format_composition(predicted_composition),
                predicted_composition == target_composition,
                target_fragments,
                predicted_fragments,
            ])

    def _collect_graph_validation_examples(
            self,
            outputs: Dict[str, object],
            batch: GraphBatch,
    ) -> None:
        if (
            self.num_val_examples_to_log <= 0
            or self.trainer.sanity_checking
            or outputs.get("heavy_edge_logits") is None
            or outputs.get("h_attachment_logits") is None
        ):
            return
        remaining = (
            self.num_val_examples_to_log
            - len(self._graph_validation_examples)
        )
        if remaining <= 0:
            return
        predicted_edge_types = outputs["heavy_edge_logits"].argmax(dim=-1)
        predicted_h_attachments = outputs["h_attachment_logits"].argmax(dim=-1)
        for sample_index in range(min(remaining, batch.atom_types.size(0))):
            self._graph_validation_examples.append({
                "atom_types": batch.atom_types[sample_index].detach().cpu(),
                "atom_mask": batch.atom_mask[sample_index].detach().cpu(),
                "target_edge_types": (
                    batch.bond_types[sample_index].detach().cpu()
                ),
                "predicted_edge_types": (
                    predicted_edge_types[sample_index].detach().cpu()
                ),
                "target_h_attachments": (
                    batch.h_attachment[sample_index].detach().cpu()
                ),
                "predicted_h_attachments": (
                    predicted_h_attachments[sample_index].detach().cpu()
                ),
            })

    def on_validation_epoch_end(self) -> None:
        if (
            self.trainer.sanity_checking
            or not self.trainer.is_global_zero
        ):
            return
        for logger in self.trainer.loggers:
            if hasattr(logger, "log_table") and self._validation_examples:
                logger.log_table(
                    key="val/examples",
                    columns=[
                        "epoch",
                        "target_smiles",
                        "predicted_smiles",
                        "smiles_exact",
                        "smiles_valid",
                        "smiles_non_stereo_exact",
                        "target_composition",
                        "predicted_composition",
                        "composition_exact",
                        "target_fragments",
                        "predicted_fragments",
                    ],
                    data=self._validation_examples,
                    step=self.global_step,
                )
        if not self._graph_validation_examples:
            return

        import wandb

        graph_rows = []
        for example in self._graph_validation_examples:
            target_graph = self._explicit_h_graph(
                atom_types=example["atom_types"],
                atom_mask=example["atom_mask"],
                edge_types=example["target_edge_types"],
                h_attachments=example["target_h_attachments"],
            )
            predicted_graph = self._explicit_h_graph(
                atom_types=example["atom_types"],
                atom_mask=example["atom_mask"],
                edge_types=example["predicted_edge_types"],
                h_attachments=example["predicted_h_attachments"],
            )
            heavy_positions = self._graph_layout(target_graph)
            graph_rows.append([
                wandb.Image(
                    self._render_graph(target_graph, heavy_positions)
                ),
                wandb.Image(
                    self._render_graph(predicted_graph, heavy_positions)
                ),
            ])

        for logger in self.trainer.loggers:
            if hasattr(logger, "log_table"):
                logger.log_table(
                    key="val/graph_examples",
                    columns=["target_graph", "predicted_graph_raw"],
                    data=graph_rows,
                    step=self.global_step,
                )

    @staticmethod
    def _explicit_h_graph(
            atom_types: torch.Tensor,
            atom_mask: torch.Tensor,
            edge_types: torch.Tensor,
            h_attachments: torch.Tensor,
    ):
        import networkx as nx

        graph = nx.Graph()
        valid_nodes = atom_mask.bool().nonzero(as_tuple=False).flatten().tolist()
        for node_index in valid_nodes:
            graph.add_node(
                node_index,
                atomic_number=int(atom_types[node_index]),
            )

        heavy_nodes = [
            node_index
            for node_index in valid_nodes
            if int(atom_types[node_index]) != 1
        ]
        for left_position, left_node in enumerate(heavy_nodes):
            for right_node in heavy_nodes[left_position + 1:]:
                bond_type = int(edge_types[left_node, right_node])
                if bond_type > 0:
                    graph.add_edge(
                        left_node,
                        right_node,
                        bond_type=bond_type,
                    )

        valid_node_set = set(valid_nodes)
        for hydrogen_node in valid_nodes:
            if int(atom_types[hydrogen_node]) != 1:
                continue
            parent_node = int(h_attachments[hydrogen_node])
            if (
                    parent_node in valid_node_set
                    and int(atom_types[parent_node]) != 1
            ):
                graph.add_edge(
                    hydrogen_node,
                    parent_node,
                    bond_type=1,
                )
        return graph

    @staticmethod
    def _graph_layout(graph):
        import networkx as nx

        heavy_nodes = [
            node
            for node, attributes in graph.nodes(data=True)
            if int(attributes["atomic_number"]) != 1
        ]
        if not heavy_nodes:
            return {}
        if len(heavy_nodes) == 1:
            node = heavy_nodes[0]
            return {node: (0.0, 0.0)}
        heavy_graph = graph.subgraph(heavy_nodes)
        return nx.spring_layout(heavy_graph, seed=0)

    @staticmethod
    def _place_hydrogens(graph, heavy_positions):
        """Place H around its own parent without sharing H coordinates."""
        import math
        import numpy as np

        positions = {
            node: np.asarray(position, dtype=float).copy()
            for node, position in heavy_positions.items()
        }
        if heavy_positions:
            center = np.stack(list(positions.values())).mean(axis=0)
        else:
            center = np.zeros(2, dtype=float)
        radius = 0.45 if len(heavy_positions) <= 1 else 0.18

        for parent_node, parent_position in heavy_positions.items():
            hydrogen_nodes = sorted(
                neighbor
                for neighbor in graph.neighbors(parent_node)
                if int(graph.nodes[neighbor]["atomic_number"]) == 1
            )
            if not hydrogen_nodes:
                continue
            outward = np.asarray(parent_position) - center
            if np.linalg.norm(outward) > 1e-8:
                base_angle = math.atan2(outward[1], outward[0])
            else:
                base_angle = parent_node * 2.399963229728653
            angular_step = min(0.75, math.pi / max(len(hydrogen_nodes), 1))
            offset_center = (len(hydrogen_nodes) - 1) / 2
            for offset, hydrogen_node in enumerate(hydrogen_nodes):
                angle = base_angle + (offset - offset_center) * angular_step
                positions[hydrogen_node] = (
                    np.asarray(parent_position)
                    + radius * np.array([math.cos(angle), math.sin(angle)])
                )

        unplaced_hydrogens = [
            node
            for node, attributes in graph.nodes(data=True)
            if (
                int(attributes["atomic_number"]) == 1
                and node not in positions
            )
        ]
        for offset, hydrogen_node in enumerate(sorted(unplaced_hydrogens)):
            angle = 2 * math.pi * offset / max(len(unplaced_hydrogens), 1)
            positions[hydrogen_node] = center + 1.2 * np.array([
                math.cos(angle),
                math.sin(angle),
            ])
        return positions

    @staticmethod
    def _render_graph(graph, heavy_positions):
        import networkx as nx
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        element_symbols = {
            1: "H",
            6: "C",
            7: "N",
            8: "O",
            9: "F",
            14: "Si",
            15: "P",
            16: "S",
            17: "Cl",
            35: "Br",
            53: "I",
        }
        element_colors = {
            1: "#F2F2F2",
            6: "#4A4A4A",
            7: "#4C78A8",
            8: "#E45756",
            9: "#72B7B2",
            14: "#B8A48A",
            15: "#F2A541",
            16: "#EAC435",
            17: "#54A24B",
            35: "#A65F5F",
            53: "#7A5195",
        }
        figure_size = min(10.0, max(5.0, graph.number_of_nodes() ** 0.5 * 1.4))
        figure = Figure(figsize=(figure_size, figure_size), dpi=120)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        axis.set_axis_off()
        positions = LitNMRToGraph._place_hydrogens(
            graph, heavy_positions
        )

        nodes = list(graph.nodes)
        atomic_numbers = [
            int(graph.nodes[node]["atomic_number"]) for node in nodes
        ]
        labels = {
            node: (
                "H"
                if atomic_number == 1
                else (
                    f"{element_symbols.get(atomic_number, f'Z{atomic_number}')}"
                    f"{node}"
                )
            )
            for node, atomic_number in zip(nodes, atomic_numbers)
        }
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=nodes,
            node_color=[
                element_colors.get(atomic_number, "#BDBDBD")
                for atomic_number in atomic_numbers
            ],
            node_size=[
                260 if atomic_number == 1 else 520
                for atomic_number in atomic_numbers
            ],
            edgecolors="#222222",
            linewidths=0.8,
            ax=axis,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=7,
            font_color="black",
            ax=axis,
        )

        bond_styles = {
            1: (1.5, "solid"),
            2: (3.0, "solid"),
            3: (4.5, "solid"),
            4: (2.0, "dashed"),
        }
        for bond_type, (width, style) in bond_styles.items():
            edges = [
                (left, right)
                for left, right, attributes in graph.edges(data=True)
                if int(attributes["bond_type"]) == bond_type
            ]
            if edges:
                nx.draw_networkx_edges(
                    graph,
                    positions,
                    edgelist=edges,
                    width=width,
                    style=style,
                    edge_color="#333333",
                    ax=axis,
                )
        edge_labels = {
            (left, right): {2: "=", 3: "≡", 4: "ar"}[int(attributes["bond_type"])]
            for left, right, attributes in graph.edges(data=True)
            if int(attributes["bond_type"]) in {2, 3, 4}
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                positions,
                edge_labels=edge_labels,
                font_size=7,
                rotate=False,
                ax=axis,
            )
        figure.tight_layout(pad=0.2)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        figure.clear()
        return image

    def _decode_smiles(self, token_ids: torch.Tensor) -> str:
        tokens = []
        for token_id in token_ids.detach().cpu().tolist():
            if token_id == SMILES_EOS_INDEX:
                break
            if token_id in (SMILES_PAD_INDEX, SMILES_BOS_INDEX):
                continue
            if token_id >= len(self.model.smiles_vocab):
                return "<unk>"
            token = self.model.smiles_vocab[token_id]
            if token == "<unk>":
                return "<unk>"
            tokens.append(token)
        return "".join(tokens)

    @staticmethod
    def _composition_from_atomic_numbers(atomic_numbers):
        counts = {}
        for atomic_number in atomic_numbers:
            atomic_number = int(atomic_number)
            counts[atomic_number] = counts.get(atomic_number, 0) + 1
        return tuple(sorted(counts.items()))

    @classmethod
    def _molecule_composition(cls, molecule):
        from rdkit import Chem

        molecule_with_h = Chem.AddHs(molecule)
        return cls._composition_from_atomic_numbers(
            atom.GetAtomicNum() for atom in molecule_with_h.GetAtoms()
        )

    @staticmethod
    def _format_composition(composition) -> str:
        if composition is None:
            return "invalid"
        return ",".join(
            f"Z{atomic_number}x{count}"
            for atomic_number, count in composition
        )

    def _greedy_smiles_metrics(
            self,
            predictions,
            targets,
            atom_types,
            atom_mask,
    ):
        from rdkit import Chem, rdBase

        valid_values = []
        stereo_agnostic_values = []
        composition_exact_values = []
        device = predictions.device
        for sample_index, (prediction, target) in enumerate(
                zip(predictions, targets)
        ):
            predicted_smiles = self._decode_smiles(prediction)
            target_smiles = self._decode_smiles(target)
            with rdBase.BlockLogs():
                predicted_molecule = Chem.MolFromSmiles(predicted_smiles)
                target_molecule = Chem.MolFromSmiles(target_smiles)
            valid_values.append(float(predicted_molecule is not None))
            target_atomic_numbers = atom_types[
                sample_index, atom_mask[sample_index]
            ].detach().cpu().tolist()
            target_composition = self._composition_from_atomic_numbers(
                target_atomic_numbers
            )
            if predicted_molecule is None or target_molecule is None:
                stereo_agnostic_values.append(0.0)
            else:
                predicted_canonical = Chem.MolToSmiles(
                    predicted_molecule, canonical=True, isomericSmiles=False
                )
                target_canonical = Chem.MolToSmiles(
                    target_molecule, canonical=True, isomericSmiles=False
                )
                stereo_agnostic_values.append(
                    float(predicted_canonical == target_canonical)
                )
            if predicted_molecule is None:
                composition_exact_values.append(0.0)
                continue
            predicted_composition = self._molecule_composition(
                predicted_molecule
            )
            composition_exact_values.append(
                float(predicted_composition == target_composition)
            )
        return (
            torch.tensor(valid_values, device=device).mean(),
            torch.tensor(stereo_agnostic_values, device=device).mean(),
            torch.tensor(composition_exact_values, device=device).mean(),
        )

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
