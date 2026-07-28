from types import SimpleNamespace

import h5py
import torch

from preprocess.uspto_3d_nmr import (
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
