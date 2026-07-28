import json

import torch

from shift3d_module.data import collate_shift_samples
from shift3d_module.lit_module import Shift3DModule
from shift3d_module.model import SchNetShiftModel


def _sample():
    return {
        "id": "ethane",
        "smiles": "CC",
        "atomic_numbers": torch.tensor([6, 6, 1, 1, 1, 1, 1, 1]),
        "positions": torch.randn(8, 3),
        "environment_ids": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
        # Deliberately unequal cardinalities: one proton peak for six H atoms,
        # and two carbon lines for two C atoms in one graph environment.
        "h_peak_shifts": torch.tensor([1.0]),
        "h_peak_integrations": torch.tensor([6.0]),
        "h_peak_integration_mask": torch.tensor([True]),
        "c_peak_shifts": torch.tensor([9.8, 10.2]),
    }


def test_schnet_outputs_are_invariant_to_rotation_and_translation():
    torch.manual_seed(0)
    model = SchNetShiftModel(
        hidden_dim=32, num_interactions=2, num_rbf=16, cutoff=5.0
    ).eval()
    sample = _sample()
    batch = collate_shift_samples([sample])
    first = model(
        batch["atomic_numbers"], batch["positions"], batch["atom_mask"]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transformed = batch["positions"] @ rotation.T + 3.5
    second = model(batch["atomic_numbers"], transformed, batch["atom_mask"])
    assert torch.allclose(first["h_shifts"], second["h_shifts"], atol=1e-5)
    assert torch.allclose(first["c_shifts"], second["c_shifts"], atol=1e-5)


def test_lightning_loss_supports_unequal_atom_and_peak_cardinalities():
    module = Shift3DModule(
        hidden_dim=32, num_interactions=2, num_rbf=16, cutoff=5.0
    )
    batch = collate_shift_samples([_sample()])
    output = module(batch)
    losses = module._sample_losses(batch, output, 0)
    assert set(losses) == {
        "h_set_loss",
        "c_set_loss",
        "equivalence_loss",
        "h_nearest_mae_ppm",
        "c_nearest_mae_ppm",
    }
    total = (
        losses["h_set_loss"]
        + losses["c_set_loss"]
        + losses["equivalence_loss"]
    )
    assert torch.isfinite(total)
    total.backward()
    assert any(parameter.grad is not None for parameter in module.parameters())


def test_bidirectional_set_loss_is_differentiable_with_extra_predictions():
    predictions = torch.tensor([1.0, 2.0, 8.0], requires_grad=True)
    targets = torch.tensor([1.1, 7.9])
    result = Shift3DModule._set_loss(
        predictions,
        targets,
        delta=0.2,
        cap=2.0,
        temperature=0.05,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["nearest_mae"])
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_environment_consistency_penalizes_only_within_class_spread():
    environment_ids = torch.tensor([0, 0, 1])
    equal = Shift3DModule._environment_consistency(
        torch.tensor([1.0, 1.0, 5.0]),
        environment_ids,
        delta=0.2,
    )
    spread = Shift3DModule._environment_consistency(
        torch.tensor([1.0, 2.0, 5.0]),
        environment_ids,
        delta=0.2,
    )
    assert equal.eq(0)
    assert spread > 0


def test_shift_normalization_uses_training_statistics_and_returns_ppm(tmp_path):
    stats_path = tmp_path / "dataset_infos_train.json"
    stats_path.write_text(
        json.dumps(
            {
                "hnmr_shift_mean": 5.0,
                "hnmr_shift_std": 2.0,
                "cnmr_shift_mean": 100.0,
                "cnmr_shift_std": 50.0,
            }
        ),
        encoding="utf-8",
    )
    module = Shift3DModule(
        hidden_dim=16,
        num_interactions=1,
        num_rbf=8,
        stats_path=str(stats_path),
    )
    h_ppm = torch.tensor([3.0, 5.0, 7.0])
    c_ppm = torch.tensor([50.0, 100.0, 150.0])
    assert torch.equal(
        module._to_normalized(h_ppm, nucleus=0),
        torch.tensor([-1.0, 0.0, 1.0]),
    )
    assert torch.equal(
        module._to_normalized(c_ppm, nucleus=1),
        torch.tensor([-1.0, 0.0, 1.0]),
    )
    assert torch.equal(
        module._to_ppm(torch.tensor([-1.0, 0.0, 1.0]), nucleus=0),
        h_ppm,
    )
    assert torch.equal(
        module._to_ppm(torch.tensor([-1.0, 0.0, 1.0]), nucleus=1),
        c_ppm,
    )
