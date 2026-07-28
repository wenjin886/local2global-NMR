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
        "equivalence_classes": torch.tensor([0, 0, 1, 1, 1, 1, 1, 1]),
        "h_prediction_mask": torch.tensor(
            [False, False, True, True, True, True, True, True]
        ),
        "h_shifts": torch.full((6,), 1.0),
        "c_shifts": torch.tensor([10.0]),
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


def test_lightning_loss_matches_equal_cardinality_symmetry_multisets():
    module = Shift3DModule(
        hidden_dim=32, num_interactions=2, num_rbf=16, cutoff=5.0
    )
    batch = collate_shift_samples([_sample()])
    output = module(batch)
    losses = module._sample_losses(batch, output, 0)
    assert set(losses) == {
        "h_loss",
        "c_loss",
        "equivalence_loss",
        "h_mae",
        "c_mae",
    }
    total = sum(losses.values())
    total.backward()
    assert any(parameter.grad is not None for parameter in module.parameters())
