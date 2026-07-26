"""Scan materialized splits for heavy-graph connectivity and pair imbalance.

Only ``data.h`` and ``data.bond_types`` are used. Statistics therefore require
neither RDKit nor SMILES parsing and exactly match the targets consumed by the
heavy-edge head.
"""

import argparse
import gc
import json
import os.path as osp
from collections import Counter
from typing import Any, Dict, Optional, Sequence

import torch
from tqdm import tqdm


def _get_value(sample: Any, key: str):
    if isinstance(sample, dict):
        return sample[key]
    return getattr(sample, key)


def _num_connected_components(adjacency: torch.Tensor) -> int:
    """Return the number of components in a square boolean adjacency matrix."""
    num_nodes = adjacency.size(0)
    if num_nodes == 0:
        return 0
    unseen = set(range(num_nodes))
    num_components = 0
    while unseen:
        num_components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = adjacency[node].nonzero(
                as_tuple=False
            ).flatten().tolist()
            for neighbor in neighbors:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return num_components


class HeavyGraphConnectivityCounter:
    """Accumulate molecule-, atom-, and unordered heavy-pair statistics."""

    def __init__(self):
        self.num_molecules = 0
        self.num_molecules_multiple_heavy_components = 0
        self.num_molecules_single_heavy_atom = 0
        self.num_heavy_atoms = 0
        self.num_isolated_heavy_atoms = 0
        self.isolated_heavy_atom_types = Counter()
        self.num_bonded_heavy_pairs = 0
        self.num_nonbonded_heavy_pairs = 0
        self.heavy_component_count_histogram = Counter()

    def update(self, sample: Any) -> None:
        atom_types = torch.as_tensor(_get_value(sample, "h")).reshape(-1)
        bond_types = torch.as_tensor(_get_value(sample, "bond_types"))
        if bond_types.shape != (atom_types.numel(), atom_types.numel()):
            raise ValueError(
                "bond_types must be [num_atoms, num_atoms], got "
                f"{tuple(bond_types.shape)} for {atom_types.numel()} atoms"
            )

        heavy_indices = atom_types.ne(1).nonzero(
            as_tuple=False
        ).flatten()
        num_heavy = heavy_indices.numel()
        heavy_bond_types = bond_types.index_select(
            0, heavy_indices
        ).index_select(1, heavy_indices)
        if heavy_bond_types.numel() and heavy_bond_types.min().item() < 0:
            raise ValueError("Materialized per-sample bond_types cannot be negative")
        adjacency = heavy_bond_types.gt(0)
        adjacency.fill_diagonal_(False)

        num_components = _num_connected_components(adjacency)
        heavy_degrees = adjacency.sum(dim=-1)
        isolated = heavy_degrees.eq(0)
        isolated_types = atom_types[heavy_indices[isolated]]

        upper_triangle = torch.triu(
            torch.ones(
                (num_heavy, num_heavy),
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        bonded_pairs = (adjacency & upper_triangle).sum().item()
        num_pairs = num_heavy * (num_heavy - 1) // 2

        self.num_molecules += 1
        self.num_molecules_multiple_heavy_components += int(
            num_components > 1
        )
        self.num_molecules_single_heavy_atom += int(num_heavy == 1)
        self.num_heavy_atoms += num_heavy
        self.num_isolated_heavy_atoms += isolated.sum().item()
        self.isolated_heavy_atom_types.update(
            int(atomic_number) for atomic_number in isolated_types.tolist()
        )
        self.num_bonded_heavy_pairs += bonded_pairs
        self.num_nonbonded_heavy_pairs += num_pairs - bonded_pairs
        self.heavy_component_count_histogram[num_components] += 1

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    def summarize(self) -> Dict[str, Any]:
        return {
            "num_molecules": self.num_molecules,
            "num_molecules_multiple_heavy_components": (
                self.num_molecules_multiple_heavy_components
            ),
            "multiple_heavy_components_ratio": self._ratio(
                self.num_molecules_multiple_heavy_components,
                self.num_molecules,
            ),
            "heavy_component_count_histogram": {
                str(count): frequency
                for count, frequency
                in sorted(self.heavy_component_count_histogram.items())
            },
            "num_heavy_atoms": self.num_heavy_atoms,
            "num_isolated_heavy_atoms": self.num_isolated_heavy_atoms,
            "isolated_heavy_atom_ratio": self._ratio(
                self.num_isolated_heavy_atoms,
                self.num_heavy_atoms,
            ),
            "isolated_heavy_atom_types": {
                str(atomic_number): count
                for atomic_number, count
                in sorted(self.isolated_heavy_atom_types.items())
            },
            "num_molecules_single_heavy_atom": (
                self.num_molecules_single_heavy_atom
            ),
            "single_heavy_atom_molecule_ratio": self._ratio(
                self.num_molecules_single_heavy_atom,
                self.num_molecules,
            ),
            "num_bonded_heavy_pairs": self.num_bonded_heavy_pairs,
            "num_nonbonded_heavy_pairs": self.num_nonbonded_heavy_pairs,
            "bonded_to_nonbonded_pair_ratio": self._ratio(
                self.num_bonded_heavy_pairs,
                self.num_nonbonded_heavy_pairs,
            ),
        }


def _save_json(results: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)


def scan_splits(
        data_dir: str,
        splits: Sequence[str],
        output_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    if output_path is None:
        output_path = osp.join(data_dir, "graph_connectivity_stats.json")
    results = {}
    overall = HeavyGraphConnectivityCounter()
    for split in splits:
        split_path = osp.join(data_dir, f"{split}.pt")
        if not osp.isfile(split_path):
            raise FileNotFoundError(split_path)
        print(f"Loading {split_path}")
        try:
            samples = torch.load(split_path, map_location="cpu")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "The split was serialized with a package unavailable in this "
                "environment. Install the original dataset dependencies "
                "(commonly torch_geometric) before scanning."
            ) from error
        split_counter = HeavyGraphConnectivityCounter()
        for sample in tqdm(samples, desc=f"Scanning {split}"):
            split_counter.update(sample)
            overall.update(sample)
        results[split] = split_counter.summarize()
        results["overall"] = overall.summarize()
        _save_json(results, output_path)
        print(
            f"Finished {split}: "
            f"{split_counter.num_molecules:,} molecules"
        )
        del samples
        gc.collect()
    print(f"Saved statistics to {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count heavy-graph components, isolated atoms, and bonded/nonbonded "
            "unordered heavy pairs in materialized .pt splits."
        )
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train.pt, val.pt, and test.pt",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test", "train"],
        help="Split names to scan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path; defaults to "
            "<data_dir>/graph_connectivity_stats.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scan_splits(
        data_dir=args.data_dir,
        splits=args.splits,
        output_path=args.output,
    )
