import torch

from preprocess.audit_fragment_carbon_valence import (
    FragmentCarbonValenceCounter,
    fragment_carbon_valences,
)
from src.data.constants import BOND_TYPE_CANDIDATES


def _fragment(**counts):
    prediction = torch.zeros(len(BOND_TYPE_CANDIDATES), dtype=torch.long)
    for candidate, count in counts.items():
        prediction[BOND_TYPE_CANDIDATES.index(candidate.replace("_", "-"))] = count
    return prediction


def test_fragment_carbon_valence_metrics_include_fused_aromatic_carbon():
    predictions = torch.stack([
        torch.stack([
            _fragment(**{"1_1": 1, "6_4": 2}),
            _fragment(**{"1_1": 3, "6_1": 1}),
            _fragment(),
        ]),
        torch.stack([
            _fragment(**{"6_4": 3}),
            _fragment(**{"1_1": 2, "6_1": 1}),
            _fragment(**{"6_2": 2}),
        ]),
    ])
    atom_types = torch.tensor([[6, 6, 8], [6, 6, 6]])
    atom_mask = torch.ones_like(atom_types, dtype=torch.bool)

    valences = fragment_carbon_valences(predictions)
    assert valences.tolist() == [[4, 4, 0], [4, 3, 4]]

    counter = FragmentCarbonValenceCounter()
    counter.update(predictions, atom_types, atom_mask)
    results = counter.summarize()

    assert results["fragment_carbon_valence_accuracy"] == 0.8
    assert results["fragment_molecule_all_carbon_valid_rate"] == 0.5
    assert results["average_invalid_carbons_per_molecule"] == 0.5
    assert (
        results["average_invalid_carbons_per_carbon_containing_molecule"]
        == 0.5
    )
