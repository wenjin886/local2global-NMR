from types import SimpleNamespace

import h5py
import torch

from preprocess.uspto_3d_nmr import (
    build_3d2shift_dataset,
    load_nmr_split_index,
    load_or_build_index_cache,
)


def test_nmr_pt_files_are_the_authoritative_split_source(tmp_path):
    torch.save([], tmp_path / "train.pt")
    torch.save(
        [SimpleNamespace(isomeric_smiles="CC")],
        tmp_path / "val.pt",
    )
    torch.save(
        [SimpleNamespace(isomeric_smiles="CO")],
        tmp_path / "test.pt",
    )
    split_for_smiles, indices, audit = load_nmr_split_index(tmp_path)
    assert split_for_smiles["CC"] == "val"
    assert split_for_smiles["CO"] == "test"
    assert "CC" not in indices["train"]
    assert audit["nmr_records/val"] == 1


def test_coordinate_and_nmr_indices_are_reloaded_from_cache(tmp_path):
    coordinates = tmp_path / "coordinates"
    nmr = tmp_path / "nmr"
    coordinates.mkdir()
    nmr.mkdir()
    for split in ("train", "val", "test"):
        torch.save(
            [SimpleNamespace(isomeric_smiles="CC")] if split == "train" else [],
            nmr / f"{split}.pt",
        )
    with h5py.File(coordinates / "train_molecules.h5", "w") as handle:
        group = handle.create_group("0000001")
        group.attrs["smiles"] = "CC"
    cache = tmp_path / "index_cache.pt"
    first = load_or_build_index_cache(coordinates, nmr, cache)
    second = load_or_build_index_cache(coordinates, nmr, cache)
    assert cache.exists()
    assert dict(first[0]) == dict(second[0])
    assert first[2] == second[2] == {"CC": "train"}


def test_builder_uses_pt_split_and_keeps_raw_unequal_peak_sets(tmp_path):
    coordinates = tmp_path / "coordinates"
    nmr = tmp_path / "nmr"
    output = tmp_path / "output"
    coordinates.mkdir()
    nmr.mkdir()
    record = SimpleNamespace(
        isomeric_smiles="CC",
        h_nmr=torch.tensor([1.1]),
        h_nmr_integration=torch.tensor([6.0]),
        h_nmr_integration_mask=torch.tensor([True]),
        c_nmr=torch.tensor([9.8, 10.0, 10.2]),
    )
    torch.save([], nmr / "train.pt")
    torch.save([record], nmr / "val.pt")
    torch.save([], nmr / "test.pt")

    # The coordinate source split is intentionally train while the NMR split
    # is val. The output must follow the authoritative NMR .pt split.
    with h5py.File(coordinates / "train_molecules.h5", "w") as handle:
        group = handle.create_group("0000001")
        group.attrs["smiles"] = "CC"
        features = group.create_group("atom_features")
        features.create_dataset(
            "atom_charges", data=[6, 6, 1, 1, 1, 1, 1, 1]
        )
        features.create_dataset("atom_mask", data=[1] * 8)
        features.create_dataset(
            "atom_coords", data=torch.randn(3, 8, 3).numpy()
        )

    report = build_3d2shift_dataset(
        nmr_dir=nmr,
        coords_dir=coordinates,
        output_dir=output,
        shard_size=1,
    )
    assert report["counts"]["accepted/val"] == 1
    assert report["splits"]["train"]["count"] == 0
    assert report["splits"]["val"]["count"] == 1
    samples = torch.load(
        output / "val" / "shard_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    sample = samples[0]
    assert sample["coordinate_source_split"] == "train"
    assert sample["positions"].shape == (3, 8, 3)
    assert sample["environment_ids"].shape == (8,)
    assert torch.unique(sample["environment_ids"][:2]).numel() == 1
    assert torch.unique(sample["environment_ids"][2:]).numel() == 1
    assert sample["h_peak_shifts"].numel() == 1
    assert sample["c_peak_shifts"].numel() == 3
