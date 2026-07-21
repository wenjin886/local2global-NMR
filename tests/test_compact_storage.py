from types import SimpleNamespace

import torch

from src.data.storage import compact_sample_storage


def test_compact_storage_removes_legacy_fields_and_narrows_dtypes():
    sample = SimpleNamespace(
        h=torch.tensor([1, 6, 8]),
        bond_types=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        h_attachment=torch.tensor([1, -100, -100]),
        heavy_fragment_labels=torch.tensor([
            [-100] * 22, [0] * 22, [0] * 22
        ]),
        h_parent_fragment_labels=torch.tensor([
            [0] * 22, [-100] * 22, [-100] * 22
        ]),
        h_parent_types=torch.tensor([6, -100, -100]),
        h_nmr_multiplicity=torch.tensor([3]),
        smiles_token_ids=torch.tensor([4, 5]),
        smiles="CO",
        canonical_smiles="CO",
        isomeric_smiles="CO",
        canno_h=torch.tensor([6, 8, 1]),
        hydrogen_neighbors=torch.tensor([6]),
        is_aromatic_heavy_atoms=torch.tensor([0, 0]),
        heavy_atom_local_labels=torch.zeros((2, 22), dtype=torch.int32),
    )

    compact_sample_storage(sample)

    assert sample.h.dtype == torch.uint8
    assert sample.bond_types.dtype == torch.uint8
    assert sample.heavy_fragment_labels.dtype == torch.int8
    assert sample.h_parent_fragment_labels.dtype == torch.int8
    assert sample.h_parent_types.dtype == torch.int8
    assert sample.h_attachment.dtype == torch.int16
    assert sample.h_nmr_multiplicity.dtype == torch.int16
    assert sample.smiles_token_ids.dtype == torch.int16
    assert sample.isomeric_smiles == "CO"
    for key in (
        "smiles",
        "canonical_smiles",
        "canno_h",
        "hydrogen_neighbors",
        "is_aromatic_heavy_atoms",
        "heavy_atom_local_labels",
    ):
        assert not hasattr(sample, key)


def test_compact_storage_rejects_lossy_integer_cast():
    sample = SimpleNamespace(
        h=torch.tensor([300]),
        bond_types=torch.zeros((1, 1), dtype=torch.long),
        isomeric_smiles="[300H]",
    )
    try:
        compact_sample_storage(sample)
    except ValueError as error:
        assert "do not fit" in str(error)
    else:
        raise AssertionError("Expected a checked dtype overflow")
