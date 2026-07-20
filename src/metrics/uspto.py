from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.data.constants import (
    MULTIPLICITY_VOCAB,
    SMILES_SPECIAL_TOKENS,
    normalize_multiplicity_label,
)
from src.data.smiles import tokenize_smiles_tokens
from .base import Metrics, read_json


def _values(tensor, mask=None):
    array = tensor.detach().cpu().numpy()
    if mask is not None:
        array = array[mask.detach().cpu().numpy().astype(bool)]
    return np.asarray(array).reshape(-1).tolist()


def _continuous_summary(values, prefix):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 1.0,
            f"{prefix}_median": 0.0,
        }
    return {
        f"{prefix}_count": int(array.size),
        f"{prefix}_min": round(float(array.min()), 6),
        f"{prefix}_max": round(float(array.max()), 6),
        f"{prefix}_mean": round(float(array.mean()), 6),
        f"{prefix}_std": round(float(array.std()), 6) or 1.0,
        f"{prefix}_median": round(float(np.median(array)), 6),
    }


class USPTOPreprocessMetrics(Metrics):
    """Accumulate training-set statistics for NMR preprocessing.

    Multiplicity is categorical, so its vocabulary and histogram are recorded;
    z-score statistics are recorded only for continuous quantities.
    """

    def __init__(
            self,
            json_path: Optional[Union[str, Path]] = None,
            summarize_hidden: bool = False,
            hidden_prefix: str = "_",
    ):
        dataset_infos = read_json(json_path) if json_path else {}
        self.ref_smiles = set(dataset_infos.get("smiles", []))
        self.ref_atom_hist = dataset_infos.get("atom_hist")
        self.prefix = hidden_prefix if summarize_hidden else ""
        self.reset()

    def update(self, data):
        if not isinstance(data, (list, tuple)):
            data = [data]
        for item in data:
            num_atoms = int(item.h.shape[0])
            self.max_num_atoms = max(self.max_num_atoms, num_atoms)
            self.max_num_heavy_atoms = max(
                self.max_num_heavy_atoms, int(item.h.ne(1).sum().item())
            )

            fragment_labels = getattr(
                item, "heavy_fragment_labels",
                getattr(item, "heavy_atom_local_labels", None),
            )
            if fragment_labels is not None and fragment_labels.numel():
                valid = fragment_labels[fragment_labels.ge(0)]
                if valid.numel():
                    self.max_neighbor_type_count = max(
                        self.max_neighbor_type_count, int(valid.max().item())
                    )

            self.hnmr_shifts.extend(_values(item.h_nmr))
            self.cnmr_shifts.extend(_values(item.c_nmr))
            self.n_hnmr_shifts_per_mol.append(int(item.h_nmr.numel()))
            self.n_cnmr_shifts_per_mol.append(int(item.c_nmr.numel()))

            if hasattr(item, "h_nmr_integration"):
                mask = getattr(item, "h_nmr_integration_mask", None)
                self.hnmr_integrations.extend(_values(item.h_nmr_integration, mask))
            if hasattr(item, "h_nmr_j"):
                mask = getattr(item, "h_nmr_j_mask", None)
                self.hnmr_j_values.extend(_values(item.h_nmr_j, mask))
                if mask is not None:
                    self.n_j_values_per_peak.extend(
                        mask.sum(dim=-1).detach().cpu().tolist()
                    )
            if hasattr(item, "h_nmr_multiplicity"):
                for value in item.h_nmr_multiplicity:
                    self.multiplicity_counts[normalize_multiplicity_label(value)] += 1
            smiles = getattr(item, "isomeric_smiles", getattr(item, "smiles", ""))
            smiles_tokens = tokenize_smiles_tokens(smiles)
            self.smiles_token_counts.update(smiles_tokens)
            self.smiles_token_lengths.append(len(smiles_tokens) + 1)  # EOS

    def summarize(self) -> dict:
        p = self.prefix
        summary = {
            "max_num_atoms": self.max_num_atoms,
            "max_num_heavy_atoms": self.max_num_heavy_atoms,
            "max_neighbor_type_count": self.max_neighbor_type_count,
        }
        summary.update(_continuous_summary(self.hnmr_shifts, f"{p}hnmr_shift"))
        summary.update(_continuous_summary(self.cnmr_shifts, f"{p}cnmr_shift"))
        summary.update(_continuous_summary(
            self.hnmr_integrations, f"{p}hnmr_integration"
        ))
        summary.update(_continuous_summary(self.hnmr_j_values, f"{p}hnmr_j"))
        summary.update(_continuous_summary(
            self.n_j_values_per_peak, f"{p}hnmr_j_count"
        ))
        summary.update(_continuous_summary(
            self.n_hnmr_shifts_per_mol, f"{p}n_hnmr_shifts_per_mol"
        ))
        summary.update(_continuous_summary(
            self.n_cnmr_shifts_per_mol, f"{p}n_cnmr_shifts_per_mol"
        ))
        observed_labels = sorted(
            label for label in self.multiplicity_counts
            if label not in MULTIPLICITY_VOCAB
        )
        labels = MULTIPLICITY_VOCAB + observed_labels
        summary["multiplicity_labels"] = labels
        summary["multiplicity_counts"] = {
            label: int(self.multiplicity_counts.get(label, 0))
            for label in labels
        }
        summary["num_multiplicity_classes"] = len(labels)
        observed_smiles_tokens = sorted(
            token for token in self.smiles_token_counts
            if token not in SMILES_SPECIAL_TOKENS
        )
        smiles_vocab = SMILES_SPECIAL_TOKENS + observed_smiles_tokens
        summary["smiles_vocab"] = smiles_vocab
        summary["smiles_token_counts"] = {
            token: int(self.smiles_token_counts.get(token, 0))
            for token in smiles_vocab
        }
        summary["smiles_vocab_size"] = len(smiles_vocab)
        summary["max_smiles_tokens"] = max(self.smiles_token_lengths, default=1)
        return summary

    def reset(self):
        self.max_num_atoms = 0
        self.max_num_heavy_atoms = 0
        self.max_neighbor_type_count = 0
        self.hnmr_shifts = []
        self.cnmr_shifts = []
        self.hnmr_integrations = []
        self.hnmr_j_values = []
        self.n_j_values_per_peak = []
        self.n_hnmr_shifts_per_mol = []
        self.n_cnmr_shifts_per_mol = []
        self.multiplicity_counts = Counter()
        self.smiles_token_counts = Counter()
        self.smiles_token_lengths = []
