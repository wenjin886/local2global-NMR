"""Lightning pipeline connecting NMR, soft topology, 3D, and shift models."""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from local2geo_module.geometry_solver import DifferentiableGeometrySolver
from local2geo_module.topology_prior import SoftTopologyPrior
from shift3d_module.lit_module import Shift3DModule
from src.data.constants import (
    SMILES_BOS_INDEX,
    SMILES_EOS_INDEX,
    SMILES_PAD_INDEX,
)
from src.data.dataset import GraphBatch
from src.model.loss import NMRGraphLoss
from src.model.nmr_to_graph import NMRToGraph

from .refiner import SpectrumConditionedEGNNRefiner
from .metrics import (
    geometry_quality,
    graph_to_canonical_smiles,
    rdkit_graph_quality,
    write_xyz,
)


def _load_component_checkpoint(
    module: nn.Module,
    path: Optional[str],
    prefixes: Iterable[str] = (),
    strict: bool = True,
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
            f"{len(incompatible.missing_keys)} missing, "
            f"{len(incompatible.unexpected_keys)} unexpected keys"
        )


class EndToEndNMRModule(pl.LightningModule):
    """Train generated coordinates with graph, chemistry, and NMR losses.

    Dataset coordinates are neither accepted nor read. Coordinates are always
    generated from NMRToGraph logits through SoftTopologyPrior and the
    differentiable geometry solver before residual refinement.
    """

    def __init__(
        self,
        nmr_to_graph: NMRToGraph,
        graph_criterion: NMRGraphLoss,
        topology_prior: SoftTopologyPrior,
        geometry_solver: DifferentiableGeometrySolver,
        coordinate_refiner: SpectrumConditionedEGNNRefiner,
        shift_model: Shift3DModule,
        nmr_to_graph_checkpoint: Optional[str] = None,
        topology_prior_checkpoint: Optional[str] = None,
        shift_model_checkpoint: Optional[str] = None,
        checkpoint_strict: bool = True,
        freeze_nmr_to_graph: bool = True,
        freeze_topology_prior: bool = True,
        freeze_shift_model: bool = True,
        input_shifts_are_normalized: bool = True,
        graph_loss_weight: float = 1.0,
        nmr_loss_weight: float = 0.01,
        chemistry_loss_weight: float = 0.1,
        displacement_loss_weight: float = 0.01,
        smiles_loss_weight: float = 1.0,
        corrected_edge_loss_weight: float = 1.0,
        corrected_attachment_loss_weight: float = 1.0,
        topology_residual_loss_weight: float = 1e-3,
        greedy_probability_start: float = 0.0,
        greedy_probability_end: float = 0.0,
        teacher_only_steps: int = 1000,
        greedy_transition_steps: int = 9000,
        greedy_schedule: str = "cosine",
        greedy_sampling_seed: int = 42,
        learning_rate: float = 1e-4,
        topology_learning_rate: Optional[float] = None,
        refiner_learning_rate: Optional[float] = None,
        weight_decay: float = 1e-5,
        validation_examples: int = 9,
        validation_xyz_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not freeze_shift_model:
            raise ValueError(
                "The first end-to-end stage requires freeze_shift_model=true"
            )
        if not 0.0 <= greedy_probability_start <= 1.0:
            raise ValueError("greedy_probability_start must be in [0, 1]")
        if not 0.0 <= greedy_probability_end <= 1.0:
            raise ValueError("greedy_probability_end must be in [0, 1]")
        if teacher_only_steps < 0 or greedy_transition_steps < 0:
            raise ValueError("curriculum step counts must be non-negative")
        if greedy_schedule not in {"linear", "cosine"}:
            raise ValueError("greedy_schedule must be 'linear' or 'cosine'")
        self.nmr_to_graph = nmr_to_graph
        self.graph_criterion = graph_criterion
        if float(self.graph_criterion.smiles_weight) != 0.0:
            warnings.warn(
                "EndToEndNMRModule owns the teacher-only SMILES CE; "
                "forcing graph_criterion.smiles_weight=0 to keep greedy "
                "validation shape-safe.",
                stacklevel=2,
            )
            self.graph_criterion.smiles_weight = 0.0
        self.topology_prior = topology_prior
        self.geometry_solver = geometry_solver
        self.coordinate_refiner = coordinate_refiner
        self.shift_model = shift_model
        self.save_hyperparameters(
            ignore=[
                "nmr_to_graph",
                "graph_criterion",
                "topology_prior",
                "geometry_solver",
                "coordinate_refiner",
                "shift_model",
            ]
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
        _load_component_checkpoint(
            self.shift_model,
            shift_model_checkpoint,
            prefixes=("shift_model.",),
            strict=checkpoint_strict,
        )
        self._set_frozen(self.shift_model, True)
        self._set_frozen(self.nmr_to_graph, freeze_nmr_to_graph)
        self._set_frozen(self.topology_prior, freeze_topology_prior)
        self._validation_examples: List[Dict[str, Any]] = []
        self._curriculum_origin_step = 0

    @staticmethod
    def _set_frozen(module: nn.Module, frozen: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Lightning recursively toggles descendants. Keep frozen evaluators
        # deterministic while retaining autograd with respect to coordinates.
        self.shift_model.eval()
        if bool(self.hparams.freeze_nmr_to_graph):
            self.nmr_to_graph.eval()
        if bool(self.hparams.freeze_topology_prior):
            self.topology_prior.eval()
        return self

    @staticmethod
    def _prior_masks(
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

    def forward(
        self,
        batch: GraphBatch,
        teacher_force_smiles: bool = False,
    ) -> Dict[str, Any]:
        graph_inputs = batch.model_inputs()
        if not teacher_force_smiles:
            # Do not expose target tokens or even target sequence length to
            # greedy generation.
            graph_inputs["smiles_input_ids"] = None
            graph_inputs["smiles_input_mask"] = None
        if bool(self.hparams.freeze_nmr_to_graph):
            with torch.no_grad():
                graph_output = self.nmr_to_graph(
                    **graph_inputs,
                    teacher_force_smiles=teacher_force_smiles,
                )
        else:
            graph_output = self.nmr_to_graph(
                **graph_inputs,
                teacher_force_smiles=teacher_force_smiles,
            )
        if (
            graph_output.get("heavy_edge_logits") is None
            or graph_output.get("h_attachment_logits") is None
        ):
            raise ValueError("NMRToGraph must predict edges and H attachments")
        masks = self._prior_masks(batch.atom_types, batch.atom_mask)
        # Formal charge is not an independently observed input in GraphBatch.
        # Neutral zeros avoid leaking it from target SMILES/graph labels.
        formal_charges = torch.zeros_like(batch.atom_types)
        freeze_entire_upstream = bool(
            self.hparams.freeze_nmr_to_graph
            and self.hparams.freeze_topology_prior
        )
        if freeze_entire_upstream:
            with torch.no_grad():
                prior_output = self.topology_prior(
                    atomic_numbers=batch.atom_types,
                    formal_charges=formal_charges,
                    atom_mask=batch.atom_mask,
                    raw_heavy_edge_logits=graph_output["heavy_edge_logits"],
                    raw_h_attachment_logits=graph_output[
                        "h_attachment_logits"
                    ],
                    **masks,
                )
        else:
            prior_output = self.topology_prior(
                atomic_numbers=batch.atom_types,
                formal_charges=formal_charges,
                atom_mask=batch.atom_mask,
                raw_heavy_edge_logits=graph_output["heavy_edge_logits"],
                raw_h_attachment_logits=graph_output[
                    "h_attachment_logits"
                ],
                **masks,
            )
        geometry = self.geometry_solver(
            atomic_numbers=batch.atom_types,
            atom_mask=batch.atom_mask,
            heavy_mask=masks["heavy_mask"],
            hydrogen_mask=masks["hydrogen_mask"],
            heavy_edge_logits=prior_output["corrected_heavy_edge_logits"],
            h_attachment_logits=prior_output[
                "corrected_h_attachment_logits"
            ],
            # Freezing pretrained modules must not silently change the
            # geometry algorithm or its public differentiable semantics.
            differentiable=True,
            geometry_probabilities_override=torch.softmax(
                prior_output["geometry_logits"], dim=-1
            ),
            local_geometry_priors=prior_output,
        )
        # Expose the shared masks alongside solver outputs so losses and
        # metrics use exactly the same graph support as coordinate generation.
        geometry.update(masks)
        refined = self.coordinate_refiner(
            coordinates=geometry["coordinates"],
            graph_atom_features=graph_output["graph_atom_features"],
            h_peak_features=graph_output["h_peak_features_clean"],
            h_peak_mask=batch.h_nmr_mask,
            c_peak_features=graph_output["c_peak_features_clean"],
            c_peak_mask=batch.c_nmr_mask,
            edge_probabilities=geometry["edge_probabilities"],
            atom_mask=batch.atom_mask,
        )
        shift_batch = self._shift_batch(batch, refined["coordinates"])
        shift_output = self.shift_model(shift_batch)
        return {
            "graph": graph_output,
            "topology": prior_output,
            "geometry": geometry,
            "refined": refined,
            "shift_batch": shift_batch,
            "shift": shift_output,
            "teacher_force_smiles": teacher_force_smiles,
        }

    def _shift_batch(
        self, batch: GraphBatch, positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if bool(self.hparams.input_shifts_are_normalized):
            h_targets = self.shift_model._to_ppm(batch.h_nmr, nucleus=0)
            c_targets = self.shift_model._to_ppm(batch.c_nmr, nucleus=1)
        else:
            h_targets, c_targets = batch.h_nmr, batch.c_nmr
        return {
            "atomic_numbers": batch.atom_types,
            "positions": positions,
            "atom_mask": batch.atom_mask,
            "h_peak_shifts": h_targets,
            "h_peak_mask": batch.h_nmr_mask,
            "c_peak_shifts": c_targets,
            "c_peak_mask": batch.c_nmr_mask,
        }

    def _losses(
        self,
        batch: GraphBatch,
        output: Mapping[str, Any],
        include_smiles_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        graph_loss, graph_parts = self.graph_criterion(
            outputs=output["graph"],
            atom_types=batch.atom_types,
            bond_types=batch.bond_types,
            h_attachment=batch.h_attachment,
            heavy_fragment_labels=batch.heavy_fragment_labels,
            h_parent_fragment_labels=batch.h_parent_fragment_labels,
            h_parent_types=batch.h_parent_types,
            smiles_target_ids=batch.smiles_target_ids,
        )
        sample_shift_losses = [
            self.shift_model._sample_losses(
                output["shift_batch"], output["shift"], index
            )
            for index in range(batch.atom_types.size(0))
        ]
        shift_metrics = {
            key: torch.stack([sample[key] for sample in sample_shift_losses]).mean()
            for key in sample_shift_losses[0]
        }
        nmr_loss = (
            float(self.shift_model.hparams.h_loss_weight)
            * shift_metrics["h_set_loss"]
            + float(self.shift_model.hparams.c_loss_weight)
            * shift_metrics["c_set_loss"]
        )
        smiles_loss = nmr_loss.sum() * 0.0
        if include_smiles_loss and float(self.hparams.smiles_loss_weight) > 0:
            smiles_logits = output["graph"].get("smiles_logits")
            if smiles_logits is None:
                raise ValueError(
                    "Teacher-forced SMILES loss requires an enabled decoder"
                )
            if smiles_logits.shape[:2] != batch.smiles_target_ids.shape:
                raise ValueError(
                    "Teacher-forced SMILES logits and targets have "
                    "incompatible shapes"
                )
            smiles_loss = F.cross_entropy(
                smiles_logits.reshape(-1, smiles_logits.size(-1)),
                batch.smiles_target_ids.reshape(-1),
                ignore_index=SMILES_PAD_INDEX,
            )
        geometry = output["geometry"]
        refined_terms = self.geometry_solver.terms(
            positions=output["refined"]["coordinates"].float(),
            probabilities=geometry["edge_probabilities"],
            geometry_probabilities=geometry["geometry_probabilities"],
            atom_mask=batch.atom_mask,
            pair_mask=geometry["pair_mask"],
            covalent_radii=geometry["covalent_radii"],
            vdw_radii=geometry["vdw_radii"],
            reduction="mean",
            local_geometry_priors=output["topology"],
        )
        chemistry_loss = self.geometry_solver.total(refined_terms)
        displacement_loss = output["refined"]["displacement"].square().sum(
            dim=-1
        )[batch.atom_mask].mean()
        corrected = output["topology"]
        heavy_pair_mask = output["geometry"]["heavy_pair_mask"]
        upper = torch.triu(
            torch.ones_like(heavy_pair_mask, dtype=torch.bool), diagonal=1
        )
        edge_mask = heavy_pair_mask & upper & batch.bond_types.ge(0)
        corrected_edge_logits = corrected["corrected_heavy_edge_logits"]
        if edge_mask.any():
            class_weights = corrected_edge_logits.new_full(
                (corrected_edge_logits.size(-1),),
                float(self.graph_criterion.edge_bond_class_weight),
            )
            class_weights[0] = float(
                self.graph_criterion.edge_none_class_weight
            )
            corrected_edge_loss = F.cross_entropy(
                corrected_edge_logits[edge_mask],
                batch.bond_types[edge_mask].long(),
                weight=class_weights,
            )
            edge_probabilities = torch.softmax(
                corrected_edge_logits[edge_mask], dim=-1
            )
            corrected_edge_entropy = -(
                edge_probabilities
                * edge_probabilities.clamp_min(1e-12).log()
            ).sum(-1).mean()
            corrected_edge_confidence = edge_probabilities.max(-1).values.mean()
        else:
            corrected_edge_loss = corrected_edge_logits.sum() * 0.0
            corrected_edge_entropy = corrected_edge_loss
            corrected_edge_confidence = corrected_edge_loss
        corrected_attachment_probabilities = output["geometry"][
            "h_attachment_probabilities"
        ]
        corrected_attachment_loss = (
            self.graph_criterion._permutation_invariant_attachment_loss(
                corrected_attachment_probabilities,
                output["geometry"]["hydrogen_mask"],
                batch.h_attachment,
            )
        )
        attachment_mask = output["geometry"]["hydrogen_mask"]
        attachment_entropy = -(
            corrected_attachment_probabilities.clamp_min(1e-12).log()
            * corrected_attachment_probabilities
        ).sum(-1)
        corrected_attachment_entropy = (
            attachment_entropy[attachment_mask].mean()
            if attachment_mask.any()
            else corrected_attachment_loss * 0.0
        )
        edge_residual = corrected["edge_residual"][heavy_pair_mask]
        attachment_residual = corrected["attachment_residual"][
            output["geometry"]["attachment_mask"]
        ]
        residual_terms = []
        if edge_residual.numel():
            residual_terms.append(edge_residual.square().mean())
        if attachment_residual.numel():
            residual_terms.append(attachment_residual.square().mean())
        topology_residual_loss = (
            torch.stack(residual_terms).mean()
            if residual_terms
            else corrected_edge_logits.sum() * 0.0
        )
        total = (
            float(self.hparams.graph_loss_weight) * graph_loss
            + float(self.hparams.nmr_loss_weight) * nmr_loss
            + float(self.hparams.smiles_loss_weight) * smiles_loss
            + float(self.hparams.chemistry_loss_weight) * chemistry_loss
            + float(self.hparams.displacement_loss_weight) * displacement_loss
            + float(self.hparams.corrected_edge_loss_weight)
            * corrected_edge_loss
            + float(self.hparams.corrected_attachment_loss_weight)
            * corrected_attachment_loss
            + float(self.hparams.topology_residual_loss_weight)
            * topology_residual_loss
        )
        values = {
            "loss": total,
            "loss_graph": graph_loss,
            "loss_nmr": nmr_loss,
            "loss_smiles": smiles_loss,
            "loss_chemistry": chemistry_loss,
            "loss_displacement": displacement_loss,
            "loss_corrected_edge": corrected_edge_loss,
            "loss_corrected_attachment": corrected_attachment_loss,
            "loss_topology_residual": topology_residual_loss,
            "corrected_edge_entropy": corrected_edge_entropy,
            "corrected_edge_confidence": corrected_edge_confidence,
            "corrected_attachment_entropy": corrected_attachment_entropy,
            "h_nearest_mae_ppm": shift_metrics["h_nearest_mae_ppm"],
            "c_nearest_mae_ppm": shift_metrics["c_nearest_mae_ppm"],
        }
        values.update(
            {f"graph_{key}": value for key, value in graph_parts.items()}
        )
        values.update(
            {f"geometry_{key}": value for key, value in refined_terms.items()}
        )
        values.update(self._corrected_graph_metrics(batch, output))
        return values

    @staticmethod
    def _corrected_graph_metrics(
        batch: GraphBatch, output: Mapping[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Permutation-invariant molecule exact match for explicit H graphs."""
        predicted_edges = output["topology"][
            "corrected_heavy_edge_logits"
        ].argmax(dim=-1)
        target_edges = batch.bond_types
        heavy_mask = output["geometry"]["heavy_mask"]
        hydrogen_mask = output["geometry"]["hydrogen_mask"]
        pair = heavy_mask[:, :, None] & heavy_mask[:, None, :]
        upper = torch.triu(torch.ones_like(pair), diagonal=1)
        valid_pair = pair & upper & target_edges.ge(0)
        attachment_probabilities = output["geometry"][
            "h_attachment_probabilities"
        ]
        target_counts = torch.zeros_like(attachment_probabilities.sum(dim=1))
        for sample_index in range(target_counts.size(0)):
            valid_h = hydrogen_mask[sample_index] & batch.h_attachment[
                sample_index
            ].ge(0)
            parents = batch.h_attachment[sample_index, valid_h].long()
            if parents.numel():
                target_counts[sample_index].scatter_add_(
                    0, parents, torch.ones_like(parents, dtype=target_counts.dtype)
                )
        predicted_counts = torch.zeros_like(target_counts)
        predicted_parents = attachment_probabilities.argmax(dim=-1)
        for sample_index in range(target_counts.size(0)):
            parents = predicted_parents[sample_index, hydrogen_mask[sample_index]]
            if parents.numel():
                predicted_counts[sample_index].scatter_add_(
                    0, parents, torch.ones_like(parents, dtype=target_counts.dtype)
                )
        exact = []
        edge_accuracies = []
        for sample_index in range(target_counts.size(0)):
            sample_pairs = valid_pair[sample_index]
            edge_equal = predicted_edges[sample_index, sample_pairs].eq(
                target_edges[sample_index, sample_pairs]
            )
            edges_exact = edge_equal.all() if edge_equal.numel() else torch.tensor(
                True, device=target_counts.device
            )
            counts_exact = predicted_counts[sample_index, heavy_mask[sample_index]].eq(
                target_counts[sample_index, heavy_mask[sample_index]]
            ).all()
            exact.append((edges_exact & counts_exact).float())
            edge_accuracies.append(
                edge_equal.float().mean()
                if edge_equal.numel()
                else target_counts.new_tensor(1.0)
            )
        zero = predicted_edges.sum() * 0.0
        return {
            "corrected_graph_exact_match": torch.stack(exact).mean() if exact else zero,
            "corrected_edge_accuracy": (
                torch.stack(edge_accuracies).mean() if edge_accuracies else zero
            ),
        }

    def _shared_step(
        self,
        batch: GraphBatch,
        stage: str,
        teacher_force_smiles: bool = False,
    ) -> torch.Tensor:
        output = self(batch, teacher_force_smiles=teacher_force_smiles)
        losses = self._losses(
            batch,
            output,
            include_smiles_loss=teacher_force_smiles,
        )
        self.log_dict(
            {f"{stage}/{key}": value for key, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.atom_types.size(0),
            add_dataloader_idx=False,
        )
        return losses["loss"]

    def training_step(self, batch: GraphBatch, batch_idx: int) -> torch.Tensor:
        greedy_probability = self._greedy_probability()
        use_greedy = self._sample_greedy_batch(greedy_probability)
        self.log(
            "train/greedy_probability",
            greedy_probability,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/is_greedy_batch",
            float(use_greedy),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/realized_greedy_ratio",
            float(use_greedy),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/realized_teacher_ratio",
            float(not use_greedy),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return self._shared_step(
            batch,
            "train",
            teacher_force_smiles=not use_greedy,
        )

    def on_train_start(self) -> None:
        # Deliberately reset on every fit/resume. A resumed phase receives a
        # fresh teacher-only warm-up before its configured transition.
        self._curriculum_origin_step = int(self.global_step)

    def _greedy_probability_for_elapsed(self, elapsed_steps: int) -> float:
        start = float(self.hparams.greedy_probability_start)
        end = float(self.hparams.greedy_probability_end)
        teacher_steps = int(self.hparams.teacher_only_steps)
        transition_steps = int(self.hparams.greedy_transition_steps)
        if elapsed_steps < teacher_steps:
            return start
        if transition_steps == 0:
            return end
        progress = min(
            1.0,
            max(0.0, (elapsed_steps - teacher_steps) / transition_steps),
        )
        if str(self.hparams.greedy_schedule) == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return start + (end - start) * progress

    def _greedy_probability(self) -> float:
        elapsed = max(0, int(self.global_step) - self._curriculum_origin_step)
        return self._greedy_probability_for_elapsed(elapsed)

    def _sample_greedy_batch(self, probability: float) -> bool:
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        # global_step and seed are identical on every DDP rank, so every rank
        # follows the same teacher/greedy computation branch.
        generator = random.Random(
            int(self.hparams.greedy_sampling_seed) + int(self.global_step)
        )
        return generator.random() < probability

    def validation_step(
        self,
        batch: GraphBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        # Validation always follows the deployment path, independently of the
        # current training curriculum.
        output = self(batch, teacher_force_smiles=False)
        if dataloader_idx == 0:
            losses = self._losses(batch, output)
            self.log_dict(
                {f"val/{key}": value for key, value in losses.items()},
                on_step=False,
                on_epoch=True,
                batch_size=batch.atom_types.size(0),
                add_dataloader_idx=False,
                sync_dist=True,
            )
            if len(getattr(self.trainer, "val_dataloaders", [])) == 1:
                self._collect_validation_examples(batch, output)
            return losses["loss"]
        self._collect_validation_examples(batch, output)
        return None

    def on_validation_epoch_start(self) -> None:
        self._validation_examples = []

    def _decode_smiles(self, token_ids: torch.Tensor) -> str:
        vocabulary = self.nmr_to_graph.smiles_vocab
        if vocabulary is None:
            return "<decoder-disabled>"
        tokens = []
        for token_id in token_ids.detach().cpu().tolist():
            if token_id == SMILES_EOS_INDEX:
                break
            if token_id in (SMILES_PAD_INDEX, SMILES_BOS_INDEX):
                continue
            if token_id < 0 or token_id >= len(vocabulary):
                return "<unk>"
            token = vocabulary[token_id]
            if token == "<unk>":
                return "<unk>"
            tokens.append(token)
        return "".join(tokens)

    def _collect_validation_examples(
        self, batch: GraphBatch, output: Mapping[str, Any]
    ) -> None:
        limit = int(self.hparams.validation_examples)
        if limit <= 0 or self.trainer.sanity_checking:
            return
        remaining = limit - len(self._validation_examples)
        if remaining <= 0:
            return
        raw_edges = output["graph"]["heavy_edge_logits"].argmax(dim=-1)
        corrected_edges = output["topology"][
            "corrected_heavy_edge_logits"
        ].argmax(dim=-1)
        attachments = output["topology"][
            "corrected_h_attachment_logits"
        ].argmax(dim=-1)
        predicted_tokens = output["graph"].get("smiles_token_ids")
        shift_batch = output["shift_batch"]
        for index in range(min(remaining, batch.atom_types.size(0))):
            mask = batch.atom_mask[index]
            atom_count = int(mask.sum().item())
            h_atoms = mask & batch.atom_types[index].eq(1)
            c_atoms = mask & batch.atom_types[index].eq(6)
            self._validation_examples.append(
                {
                    "atom_types": batch.atom_types[index, mask].detach().cpu(),
                    "coordinates": output["refined"]["coordinates"][
                        index, mask
                    ].detach().cpu(),
                    "covalent_radii": output["geometry"]["covalent_radii"][
                        index, mask
                    ].detach().cpu(),
                    "vdw_radii": output["geometry"]["vdw_radii"][
                        index, mask
                    ].detach().cpu(),
                    "target_edges": batch.bond_types[
                        index, :atom_count, :atom_count
                    ].detach().cpu(),
                    "raw_edges": raw_edges[
                        index, :atom_count, :atom_count
                    ].detach().cpu(),
                    "corrected_edges": corrected_edges[
                        index, :atom_count, :atom_count
                    ].detach().cpu(),
                    "target_attachments": batch.h_attachment[
                        index, :atom_count
                    ].detach().cpu(),
                    "predicted_attachments": attachments[
                        index, :atom_count
                    ].detach().cpu(),
                    "target_smiles": self._decode_smiles(
                        batch.smiles_target_ids[index]
                    ),
                    "predicted_smiles": (
                        self._decode_smiles(predicted_tokens[index])
                        if predicted_tokens is not None
                        else "<decoder-disabled>"
                    ),
                    "h_target": shift_batch["h_peak_shifts"][
                        index, batch.h_nmr_mask[index]
                    ].detach().cpu(),
                    "h_prediction": output["shift"]["h_shifts"][
                        index, h_atoms
                    ].detach().cpu(),
                    "c_target": shift_batch["c_peak_shifts"][
                        index, batch.c_nmr_mask[index]
                    ].detach().cpu(),
                    "c_prediction": output["shift"]["c_shifts"][
                        index, c_atoms
                    ].detach().cpu(),
                }
            )

    @staticmethod
    def _all_atom_bonds(
        atom_types: torch.Tensor,
        heavy_edges: torch.Tensor,
        attachments: torch.Tensor,
    ) -> torch.Tensor:
        bonds = heavy_edges.clone()
        for hydrogen in atom_types.eq(1).nonzero(as_tuple=False).flatten():
            parent = int(attachments[hydrogen])
            if 0 <= parent < atom_types.numel() and int(atom_types[parent]) != 1:
                bonds[hydrogen, parent] = 1
                bonds[parent, hydrogen] = 1
        return bonds

    @staticmethod
    def _render_structure(
        atom_types: torch.Tensor,
        coordinates: torch.Tensor,
        bonds: torch.Tensor,
    ):
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        colors = {
            1: "#DDDDDD",
            6: "#333333",
            7: "#3B6FE8",
            8: "#E33A3A",
            9: "#66C96A",
            15: "#E39A27",
            16: "#E2C62E",
            17: "#55B85A",
        }
        figure = Figure(figsize=(5.0, 4.2), dpi=120)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111, projection="3d")
        xyz = coordinates.numpy()
        for left in range(atom_types.numel()):
            for right in range(left + 1, atom_types.numel()):
                if int(bonds[left, right]) > 0:
                    axis.plot(*xyz[[left, right]].T, color="#777777", linewidth=1.5)
        axis.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=[colors.get(int(z), "#A45CC5") for z in atom_types],
            s=[28 if int(z) == 1 else 70 for z in atom_types],
            depthshade=True,
        )
        axis.set_axis_off()
        axis.view_init(elev=22, azim=35)
        figure.tight_layout(pad=0.1)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        figure.clear()
        return image

    @staticmethod
    def _render_graph(atom_types: torch.Tensor, bonds: torch.Tensor):
        import networkx as nx
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        graph = nx.Graph()
        for index, atomic_number in enumerate(atom_types.tolist()):
            graph.add_node(index, label=str(atomic_number))
        for left in range(atom_types.numel()):
            for right in range(left + 1, atom_types.numel()):
                if int(bonds[left, right]) > 0:
                    graph.add_edge(left, right, order=int(bonds[left, right]))
        positions = nx.spring_layout(graph, seed=17)
        figure = Figure(figsize=(4.2, 3.6), dpi=110)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        nx.draw_networkx(
            graph,
            positions,
            labels=nx.get_node_attributes(graph, "label"),
            node_size=350,
            font_size=7,
            width=[graph.edges[e]["order"] for e in graph.edges],
            ax=axis,
        )
        axis.set_axis_off()
        figure.tight_layout(pad=0.1)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        figure.clear()
        return image

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        examples = self._gather_validation_examples()
        # Every rank must participate in the collective above. Rendering and
        # W&B writes remain global-zero-only after the gather completes.
        if not self.trainer.is_global_zero or not examples:
            return
        try:
            import wandb
        except ImportError:
            return
        rows = []
        quality_rows = []
        xyz_root = None
        if self.hparams.validation_xyz_dir:
            xyz_root = (
                Path(str(self.hparams.validation_xyz_dir))
                / f"epoch_{int(self.current_epoch):03d}_step_{int(self.global_step)}"
            )
        for sample_index, example in enumerate(examples):
            target_bonds = self._all_atom_bonds(
                example["atom_types"],
                example["target_edges"],
                example["target_attachments"],
            )
            predicted_bonds = self._all_atom_bonds(
                example["atom_types"],
                example["corrected_edges"],
                example["predicted_attachments"],
            )
            raw_bonds = self._all_atom_bonds(
                example["atom_types"],
                example["raw_edges"],
                example["predicted_attachments"],
            )
            graph_smiles = graph_to_canonical_smiles(
                example["atom_types"], predicted_bonds
            )
            graph_quality = rdkit_graph_quality(
                example["atom_types"], predicted_bonds
            )
            geometric_quality = geometry_quality(
                example["atom_types"],
                example["coordinates"],
                predicted_bonds,
                example["covalent_radii"],
                example["vdw_radii"],
            )
            quality = {**graph_quality, **geometric_quality}
            quality_rows.append(quality)
            graph_exact = self._example_graph_exact(example)
            xyz_path = "<disabled>"
            if xyz_root is not None:
                path = xyz_root / f"sample_{sample_index:02d}.xyz"
                write_xyz(
                    path,
                    example["atom_types"],
                    example["coordinates"],
                    comment=(
                        f"target_smiles={example['target_smiles']} "
                        f"predicted_smiles={example['predicted_smiles']} "
                        f"graph_smiles={graph_smiles or '<invalid>'}"
                    ),
                )
                xyz_path = str(path)
            rows.append(
                [
                    int(self.current_epoch),
                    example["target_smiles"],
                    example["predicted_smiles"],
                    graph_smiles or "<invalid-graph>",
                    graph_exact,
                    quality["validity"],
                    quality["connected"],
                    quality["atom_stability"],
                    quality["molecule_stability"],
                    quality["clash_free"],
                    quality["bond_length_mae_angstrom"],
                    quality["min_nonbond_vdw_ratio"],
                    xyz_path,
                    wandb.Image(
                        self._render_structure(
                            example["atom_types"],
                            example["coordinates"],
                            predicted_bonds,
                        )
                    ),
                    wandb.Image(
                        self._render_graph(example["atom_types"], target_bonds)
                    ),
                    wandb.Image(
                        self._render_graph(example["atom_types"], raw_bonds)
                    ),
                    wandb.Image(
                        self._render_graph(example["atom_types"], predicted_bonds)
                    ),
                    example["h_target"].tolist(),
                    example["h_prediction"].tolist(),
                    example["c_target"].tolist(),
                    example["c_prediction"].tolist(),
                ]
            )
        for logger in self.trainer.loggers:
            if hasattr(logger, "log_table"):
                logger.log_table(
                    key="val/end_to_end_examples",
                    columns=[
                        "epoch",
                        "target_smiles",
                        "predicted_smiles",
                        "corrected_graph_smiles",
                        "corrected_graph_exact_match",
                        "3d_validity",
                        "connected",
                        "atom_stability",
                        "molecule_stability",
                        "clash_free",
                        "bond_length_mae_angstrom",
                        "min_nonbond_vdw_ratio",
                        "xyz_path",
                        "generated_3d",
                        "target_graph",
                        "predicted_graph_raw",
                        "predicted_graph_corrected",
                        "h_target_ppm",
                        "h_prediction_ppm",
                        "c_target_ppm",
                        "c_prediction_ppm",
                    ],
                    data=rows,
                    step=self.global_step,
                )
            experiment = getattr(logger, "experiment", None)
            if experiment is not None and quality_rows:
                panel_metrics = {}
                for key in quality_rows[0]:
                    finite_values = [
                        row[key]
                        for row in quality_rows
                        if math.isfinite(float(row[key]))
                    ]
                    if finite_values:
                        panel_metrics[f"val_3d/{key}"] = sum(finite_values) / len(
                            finite_values
                        )
                experiment.log(panel_metrics, step=self.global_step)
                if xyz_root is not None:
                    for path in sorted(xyz_root.glob("*.xyz")):
                        experiment.save(
                            str(path), base_path=str(xyz_root), policy="now"
                        )

    @staticmethod
    def _example_graph_exact(example: Mapping[str, Any]) -> bool:
        atom_types = example["atom_types"]
        heavy = atom_types.ne(1)
        upper = torch.triu(torch.ones_like(example["target_edges"]), diagonal=1).bool()
        pairs = heavy[:, None] & heavy[None, :] & upper
        if not torch.equal(
            example["corrected_edges"][pairs], example["target_edges"][pairs]
        ):
            return False
        heavy_indices = heavy.nonzero(as_tuple=False).flatten()
        for parent in heavy_indices.tolist():
            target_count = int(
                (example["target_attachments"][atom_types.eq(1)] == parent).sum()
            )
            predicted_count = int(
                (example["predicted_attachments"][atom_types.eq(1)] == parent).sum()
            )
            if target_count != predicted_count:
                return False
        return True

    def _gather_validation_examples(self) -> List[Dict[str, Any]]:
        """Collect the deterministic validation panel from every DDP rank."""
        local_examples = self._validation_examples
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return local_examples[: int(self.hparams.validation_examples)]
        gathered: List[Optional[List[Dict[str, Any]]]] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, local_examples)
        merged = [
            example
            for rank_examples in gathered
            if rank_examples is not None
            for example in rank_examples
        ]
        return merged[: int(self.hparams.validation_examples)]

    def configure_optimizers(self):
        groups = []
        assigned = set()
        module_groups = (
            (
                "topology_prior",
                self.topology_prior,
                self.hparams.topology_learning_rate,
            ),
            (
                "coordinate_refiner",
                self.coordinate_refiner,
                self.hparams.refiner_learning_rate,
            ),
        )
        for name, module, configured_lr in module_groups:
            parameters = [
                parameter for parameter in module.parameters()
                if parameter.requires_grad
            ]
            if parameters:
                assigned.update(id(parameter) for parameter in parameters)
                groups.append(
                    {
                        "params": parameters,
                        "lr": float(
                            self.hparams.learning_rate
                            if configured_lr is None else configured_lr
                        ),
                        "name": name,
                    }
                )
        remaining = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in assigned
        ]
        if remaining:
            groups.append(
                {
                    "params": remaining,
                    "lr": float(self.hparams.learning_rate),
                    "name": "other",
                }
            )
        if not groups:
            raise ValueError("No trainable parameters in end-to-end pipeline")
        optimizer = torch.optim.AdamW(
            groups,
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
