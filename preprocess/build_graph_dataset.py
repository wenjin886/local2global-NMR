"""Add explicit-H graph targets to preprocessed NMR samples."""

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

from src.data.dataset import graph_targets_from_smiles, load_local_vocab


def get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def build_graph_dataset(input_path: str, output_path: str, local_vocab_path: str) -> None:
    items = torch.load(input_path)
    local_vocab = load_local_vocab(local_vocab_path)
    processed = []
    for item in items:
        smiles = get_value(item, "smiles")
        targets = graph_targets_from_smiles(smiles, local_vocab)
        h_nmr = torch.as_tensor(get_value(item, "h_nmr"), dtype=torch.float)
        integration = get_value(item, "h_nmr_integration")
        if integration is None:
            integration = torch.ones_like(h_nmr)
        processed.append({
            "smiles": smiles,
            "h": targets["h"],
            "h_nmr": h_nmr,
            "h_nmr_integration": torch.as_tensor(integration, dtype=torch.float),
            "c_nmr": torch.as_tensor(get_value(item, "c_nmr"), dtype=torch.float),
            "bond_types": targets["bond_types"],
            "h_attachment": targets["h_attachment"],
            "local_labels": targets["local_labels"],
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--local_vocab_path",
        default="data/uspto/label_vocab.json",
    )
    args = parser.parse_args()
    build_graph_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        local_vocab_path=args.local_vocab_path,
    )


if __name__ == "__main__":
    main()
