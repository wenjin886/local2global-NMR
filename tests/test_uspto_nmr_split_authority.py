from types import SimpleNamespace

import torch

from preprocess.uspto_3d_nmr import load_nmr_split_index


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
