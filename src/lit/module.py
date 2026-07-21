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
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_val_examples_to_log = num_val_examples_to_log
        self._validation_examples = []
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
        return loss

    def training_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", teacher_force_smiles=True)

    def validation_step(
            self,
            batch: GraphBatch,
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        if dataloader_idx == 0:
            return self._shared_step(
                batch, "val", teacher_force_smiles=True
            )
        outputs = self(batch, teacher_force_smiles=False)
        metrics = self._batch_metrics(outputs, batch, smiles_mode="greedy")
        self.log_dict(
            {"val_generation/%s" % key: value for key, value in metrics.items()},
            on_step=False,
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
            add_dataloader_idx=False,
        )
        self._collect_validation_examples(outputs, batch)

    def test_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test", teacher_force_smiles=False)

    def _batch_metrics(
            self,
            outputs: Dict[str, object],
            batch: GraphBatch,
            smiles_mode: str,
    ) -> Dict[str, torch.Tensor]:
        zero = outputs["fragment_logits"].sum() * 0.0

        edge_accuracy = None
        if outputs.get("heavy_edge_logits") is not None:
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
                edge_accuracy = zero

        fragment_prediction = outputs["fragment_logits"].argmax(dim=-1)
        fragment_scores = []
        presence_macro_f1s = []
        positive_count_accuracies = []
        atom_exact_accuracies = []
        h_parent_type_accuracies = []
        h_parent_fragment_accuracies = []
        h_parent_joint_accuracies = []
        h_attachment_accuracies = []
        h_count_maes = []
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
                atom_exact = (
                    predicted_fragment.eq(target_fragment) | ~valid_fragment
                ).all(dim=-1).float().mean()
                presence_macro_f1s.append(presence_f1)
                positive_count_accuracies.append(positive_accuracy)
                atom_exact_accuracies.append(atom_exact)
                fragment_scores.append(
                    torch.sqrt((presence_f1 * positive_accuracy).clamp_min(0.0))
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
                h_parent_joint_accuracies.append(
                    (type_correct & fragment_correct).float().mean()
                )

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
            "heavy_fragment_atom_exact_accuracy": mean(atom_exact_accuracies),
            "h_parent_type_accuracy": mean(h_parent_type_accuracies),
            "h_parent_fragment_exact_accuracy": mean(
                h_parent_fragment_accuracies
            ),
            "h_parent_joint_accuracy": mean(h_parent_joint_accuracies),
            "h_attachment_multiset_accuracy": mean(h_attachment_accuracies),
            "h_count_mae": mean(h_count_maes),
        }
        if edge_accuracy is not None:
            metrics["edge_accuracy"] = edge_accuracy
        if outputs.get("h_attachment_logits") is None:
            metrics.pop("h_attachment_multiset_accuracy")
            metrics.pop("h_count_mae")
        if outputs.get("smiles_token_ids") is not None:
            valid = batch.smiles_target_ids.ne(SMILES_PAD_INDEX)
            predictions = outputs["smiles_token_ids"]
            correct = predictions.eq(batch.smiles_target_ids) & valid
            metrics[f"smiles_{smiles_mode}_token_accuracy"] = (
                correct.sum().float() / valid.sum().clamp_min(1)
            )
            metrics[f"smiles_{smiles_mode}_exact_match"] = (
                (correct | ~valid).all(dim=1).float().mean()
            )
            logits = outputs.get("smiles_logits")
            if smiles_mode == "teacher" and logits is not None:
                token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch.smiles_target_ids.reshape(-1),
                    ignore_index=SMILES_PAD_INDEX,
                )
                metrics["smiles_teacher_perplexity"] = token_loss.clamp_max(20).exp()
            if smiles_mode == "greedy" and self.model.smiles_vocab is not None:
                validity, stereo_agnostic = self._greedy_smiles_metrics(
                    predictions, batch.smiles_target_ids
                )
                metrics["smiles_greedy_validity"] = validity
                metrics["smiles_greedy_stereo_agnostic_exact_match"] = (
                    stereo_agnostic
                )
        return metrics

    def on_validation_epoch_start(self) -> None:
        self._validation_examples = []

    @staticmethod
    def _format_fragments(
            atom_types: torch.Tensor,
            fragment_counts: torch.Tensor,
            heavy_mask: torch.Tensor,
    ) -> str:
        rows = []
        for atom_index in heavy_mask.nonzero(as_tuple=False).flatten().tolist():
            counts = fragment_counts[atom_index]
            ports = [
                f"{candidate}x{int(count)}"
                for candidate, count in zip(BOND_TYPE_CANDIDATES, counts.tolist())
                if count > 0
            ]
            rows.append(
                f"{atom_index}:Z{int(atom_types[atom_index])}="
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
            if predicted_molecule is not None and target_molecule is not None:
                predicted_non_stereo = Chem.MolToSmiles(
                    predicted_molecule, canonical=True, isomericSmiles=False
                )
                target_non_stereo = Chem.MolToSmiles(
                    target_molecule, canonical=True, isomericSmiles=False
                )
                non_stereo_exact = predicted_non_stereo == target_non_stereo
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
                target_fragments,
                predicted_fragments,
            ])

    def on_validation_epoch_end(self) -> None:
        if (
            not self._validation_examples
            or self.trainer.sanity_checking
            or not self.trainer.is_global_zero
        ):
            return
        columns = [
            "epoch",
            "target_smiles",
            "predicted_smiles",
            "smiles_exact",
            "smiles_valid",
            "smiles_non_stereo_exact",
            "target_fragments",
            "predicted_fragments",
        ]
        for logger in self.trainer.loggers:
            if hasattr(logger, "log_table"):
                logger.log_table(
                    key="val/examples",
                    columns=columns,
                    data=self._validation_examples,
                    step=self.global_step,
                )

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

    def _greedy_smiles_metrics(self, predictions, targets):
        from rdkit import Chem, rdBase

        valid_values = []
        stereo_agnostic_values = []
        device = predictions.device
        for prediction, target in zip(predictions, targets):
            predicted_smiles = self._decode_smiles(prediction)
            target_smiles = self._decode_smiles(target)
            with rdBase.BlockLogs():
                predicted_molecule = Chem.MolFromSmiles(predicted_smiles)
                target_molecule = Chem.MolFromSmiles(target_smiles)
            valid_values.append(float(predicted_molecule is not None))
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
        return (
            torch.tensor(valid_values, device=device).mean(),
            torch.tensor(stereo_agnostic_values, device=device).mean(),
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
