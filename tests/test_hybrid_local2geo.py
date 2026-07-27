from pathlib import Path

import pytest
import torch

pytest.importorskip("pytorch_lightning")
pytest.importorskip("pyarrow")

from local2geo_module.data import (  # noqa: E402
    ParquetSmilesDataset,
    collate_local2geo,
    graph_from_smiles,
)
from local2geo_module.geometry_solver import (  # noqa: E402
    DifferentiableGeometrySolver,
)
from local2geo_module.lit_module import HybridLocal2GeoModule  # noqa: E402
from local2geo_module.soft_graph_simulator import (  # noqa: E402
    SoftGraphSimulator,
)


EXAMPLE_PARQUETS = [
    Path("data/uspto/exp_data/example_data_1.parquet"),
    Path("data/uspto/exp_data/example_data_2.parquet"),
]


def test_parquet_splits_are_nonempty_and_disjoint():
    datasets = {
        split: ParquetSmilesDataset(EXAMPLE_PARQUETS, split)
        for split in ("train", "val", "test")
    }
    values = {name: set(dataset.smiles) for name, dataset in datasets.items()}
    assert values["train"]
    assert values["val"]
    assert values["test"]
    assert values["train"].isdisjoint(values["val"])
    assert values["train"].isdisjoint(values["test"])
    assert values["val"].isdisjoint(values["test"])


def test_2d_only_targets_mark_butane_heavy_torsion_as_anti():
    graph = graph_from_smiles("CCCC")
    heavy = graph["atomic_numbers"].ne(1)
    torsion = graph["torsion_classes"][heavy][:, heavy]
    assert torsion.eq(2).sum() == 2
    assert torch.isfinite(graph["one_four_log_ratio"]).all()


def test_coordinate_loss_reaches_corrupted_raw_graph_logits():
    batch = collate_local2geo([graph_from_smiles("CCCC")])
    raw = SoftGraphSimulator(corruption_boost=6.0)(
        batch, corrupted=True, seed=7
    )
    raw["heavy_edge_logits"].requires_grad_(True)
    raw["h_attachment_logits"].requires_grad_(True)
    module = HybridLocal2GeoModule(hidden_dim=32, num_layers=1)
    learned = module.correct_graph(batch, raw)
    solver = DifferentiableGeometrySolver(
        num_steps=1,
        seed_mode="differentiable",
        one_three_distance_weight=2.0,
        one_four_distance_weight=2.0,
    )
    output = solver(
        atomic_numbers=batch["atomic_numbers"],
        atom_mask=batch["atom_mask"],
        heavy_mask=batch["heavy_mask"],
        hydrogen_mask=batch["hydrogen_mask"],
        heavy_edge_logits=learned["corrected_heavy_edge_logits"],
        h_attachment_logits=learned["corrected_h_attachment_logits"],
        differentiable=True,
        geometry_probabilities_override=torch.softmax(
            learned["geometry_logits"], dim=-1
        ),
        local_geometry_priors={
            key: learned[key]
            for key in (
                "one_three_probability",
                "one_four_probability",
                "one_three_distance_ratio",
                "one_four_distance_ratio",
            )
        },
    )
    gradients = torch.autograd.grad(
        output["coordinates"].square().sum(),
        (
            raw["heavy_edge_logits"],
            raw["h_attachment_logits"],
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
