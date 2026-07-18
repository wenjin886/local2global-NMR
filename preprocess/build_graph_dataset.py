"""Add explicit-H graph targets to preprocessed NMR samples."""

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

from src.data.constants import MAX_J_VALUES, MULTIPLICITY_MISSING_INDEX
from src.data.dataset import graph_targets_from_smiles


def get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def build_graph_dataset(input_path: str, output_path: str) -> None:
    items = torch.load(input_path)
    processed = []
    for item in items:
        smiles = get_value(item, "smiles")
        targets = graph_targets_from_smiles(smiles)
        h_nmr = torch.as_tensor(get_value(item, "h_nmr"), dtype=torch.float)
        integration = get_value(item, "h_nmr_integration")
        integration_is_available = integration is not None
        if integration is None:
            integration = torch.zeros_like(h_nmr)
        integration_mask = get_value(item, "h_nmr_integration_mask")
        if integration_mask is None:
            integration_mask = torch.full_like(
                h_nmr, integration_is_available, dtype=torch.bool
            )
        multiplicity = get_value(item, "h_nmr_multiplicity")
        multiplicity_is_available = multiplicity is not None
        if multiplicity is None:
            multiplicity = torch.full_like(
                h_nmr, MULTIPLICITY_MISSING_INDEX, dtype=torch.long
            )
        multiplicity_mask = get_value(item, "h_nmr_multiplicity_mask")
        if multiplicity_mask is None:
            multiplicity_mask = torch.full_like(
                h_nmr, multiplicity_is_available, dtype=torch.bool
            )
        j_values = get_value(item, "h_nmr_j")
        if j_values is None:
            j_values = torch.zeros((h_nmr.numel(), MAX_J_VALUES))
        j_mask = get_value(item, "h_nmr_j_mask")
        if j_mask is None:
            j_mask = torch.as_tensor(j_values).ne(0)
        processed.append({
            "smiles": smiles,
            "h": targets["h"],
            "h_nmr": h_nmr,
            "h_nmr_integration": torch.as_tensor(integration, dtype=torch.float),
            "h_nmr_integration_mask": torch.as_tensor(
                integration_mask, dtype=torch.bool
            ),
            "h_nmr_multiplicity": torch.as_tensor(multiplicity, dtype=torch.long),
            "h_nmr_multiplicity_mask": torch.as_tensor(
                multiplicity_mask, dtype=torch.bool
            ),
            "h_nmr_j": torch.as_tensor(j_values, dtype=torch.float),
            "h_nmr_j_mask": torch.as_tensor(j_mask, dtype=torch.bool),
            "c_nmr": torch.as_tensor(get_value(item, "c_nmr"), dtype=torch.float),
            "bond_types": targets["bond_types"],
            "h_attachment": targets["h_attachment"],
            "heavy_fragment_labels": targets["heavy_fragment_labels"],
            "h_parent_fragment_labels": targets["h_parent_fragment_labels"],
            "h_parent_types": targets["h_parent_types"],
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    build_graph_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
