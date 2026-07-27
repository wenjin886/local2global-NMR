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
from local2geo_module.eval_hybrid import build_parser  # noqa: E402
from local2geo_module.lit_module import HybridLocal2GeoModule  # noqa: E402
from local2geo_module.seed_generator import (  # noqa: E402
    SoftDistanceStressSeed,
)
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
        seed_mode="soft_stress",
        soft_stress_steps=3,
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
        coordinate_seed=3,
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


def _oracle_local_priors(batch):
    return {
        "one_three_probability": batch["one_three_targets"],
        "one_four_probability": batch["one_four_targets"],
        "one_three_distance_ratio": batch[
            "one_three_log_ratio"
        ].exp(),
        "one_four_distance_ratio": batch[
            "one_four_log_ratio"
        ].exp(),
        "one_four_validity": (
            batch["heavy_mask"][:, :, None]
            & batch["heavy_mask"][:, None, :]
        ).float(),
    }


def test_soft_stress_seed_is_parameter_free_and_differentiable():
    batch = collate_local2geo([graph_from_smiles("CCCC")])
    raw = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=3)
    raw["heavy_edge_logits"].requires_grad_(True)
    raw["h_attachment_logits"].requires_grad_(True)
    graph_builder = DifferentiableGeometrySolver(num_steps=0)
    graph = graph_builder.soft_graph(
        raw["heavy_edge_logits"],
        raw["h_attachment_logits"],
        batch["pair_mask"],
        batch["heavy_pair_mask"],
        batch["attachment_mask"],
    )
    seed_module = SoftDistanceStressSeed(num_steps=3)
    seed = seed_module(
        atom_mask=batch["atom_mask"],
        heavy_mask=batch["heavy_mask"],
        probabilities=graph["edge_probabilities"],
        covalent_radii=batch["covalent_radii"],
        vdw_radii=batch["vdw_radii"],
        bond_length_scales=graph_builder.bond_length_scales,
        local_geometry_priors=_oracle_local_priors(batch),
        differentiable=True,
        generator=torch.Generator().manual_seed(3),
    )
    assert sum(
        parameter.numel()
        for parameter in seed_module.parameters()
    ) == 0
    gradients = torch.autograd.grad(
        seed.square().sum(),
        (
            raw["heavy_edge_logits"],
            raw["h_attachment_logits"],
        ),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_soft_stress_seed_extends_a_long_chain():
    batch = collate_local2geo([
        graph_from_smiles("CCCCCCCCCCCCC")
    ])
    raw = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=3)
    geometry = torch.nn.functional.one_hot(
        batch["geometry_classes"].clamp_min(0), 7
    ).float()

    def coordinates(mode):
        solver = DifferentiableGeometrySolver(
            num_steps=0,
            seed_mode=mode,
            soft_stress_steps=96,
        )
        return solver(
            batch["atomic_numbers"],
            batch["atom_mask"],
            batch["heavy_mask"],
            batch["hydrogen_mask"],
            raw["heavy_edge_logits"],
            raw["h_attachment_logits"],
            differentiable=False,
            geometry_probabilities_override=geometry,
            local_geometry_priors=(
                _oracle_local_priors(batch)
                if mode == "soft_stress" else None
            ),
            coordinate_seed=3,
        )["seed_coordinates"][0]

    legacy = coordinates("differentiable")
    stress = coordinates("soft_stress")
    heavy = batch["heavy_mask"][0]
    legacy_diameter = torch.cdist(
        legacy[heavy], legacy[heavy]
    ).max()
    stress_diameter = torch.cdist(
        stress[heavy], stress[heavy]
    ).max()
    assert stress_diameter > 1.5 * legacy_diameter


def test_hybrid_eval_defaults_to_soft_stress():
    args = build_parser().parse_args([
        "--checkpoint", "model.ckpt", "--smiles", "CC"
    ])
    assert args.seed_mode == "soft_stress"
