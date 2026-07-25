from types import SimpleNamespace

import torch

from preprocess.count_graph_connectivity import HeavyGraphConnectivityCounter


def sample(atom_types, edges):
    atom_types = torch.tensor(atom_types)
    bond_types = torch.zeros(
        (atom_types.numel(), atom_types.numel()),
        dtype=torch.uint8,
    )
    for left, right, bond_type in edges:
        bond_types[left, right] = bond_types[right, left] = bond_type
    return SimpleNamespace(h=atom_types, bond_types=bond_types)


def test_heavy_graph_connectivity_counter():
    counter = HeavyGraphConnectivityCounter()
    # Two explicit H plus a connected C-O heavy graph.
    counter.update(sample(
        [1, 1, 6, 8],
        [(2, 3, 1)],
    ))
    # C-N are connected while Cl is an isolated heavy component.
    counter.update(sample(
        [6, 7, 17],
        [(0, 1, 2)],
    ))
    # One-heavy-atom molecule has one component and one isolated heavy atom.
    counter.update(sample(
        [1, 1, 8],
        [],
    ))

    summary = counter.summarize()
    assert summary["num_molecules"] == 3
    assert summary["num_molecules_multiple_heavy_components"] == 1
    assert summary["multiple_heavy_components_ratio"] == 1 / 3
    assert summary["heavy_component_count_histogram"] == {"1": 2, "2": 1}
    assert summary["num_heavy_atoms"] == 6
    assert summary["num_isolated_heavy_atoms"] == 2
    assert summary["isolated_heavy_atom_types"] == {"8": 1, "17": 1}
    assert summary["num_molecules_single_heavy_atom"] == 1
    assert summary["num_bonded_heavy_pairs"] == 2
    assert summary["num_nonbonded_heavy_pairs"] == 2
    assert summary["bonded_to_nonbonded_pair_ratio"] == 1.0
