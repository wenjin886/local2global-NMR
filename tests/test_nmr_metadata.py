import json

import torch

from src.data.transforms import NormalizeNMR


class Sample:
    pass


def test_normalize_nmr_respects_metadata_masks(tmp_path):
    stats = {
        "hnmr_shift_mean": 5.0,
        "hnmr_shift_std": 2.0,
        "cnmr_shift_mean": 100.0,
        "cnmr_shift_std": 50.0,
        "hnmr_integration_mean": 2.0,
        "hnmr_integration_std": 1.0,
        "hnmr_j_mean": 7.0,
        "hnmr_j_std": 2.0,
    }
    path = tmp_path / "dataset_infos.json"
    path.write_text(json.dumps(stats))
    sample = Sample()
    sample.h_nmr = torch.tensor([[3.0, 7.0, 0.0]])
    sample.h_nmr_mask = torch.tensor([[True, True, False]])
    sample.c_nmr = torch.tensor([[50.0, 150.0, 0.0]])
    sample.c_nmr_mask = torch.tensor([[True, True, False]])
    sample.h_nmr_integration = torch.tensor([3.0, 0.0])
    sample.h_nmr_integration_mask = torch.tensor([True, False])
    sample.h_nmr_j = torch.tensor([[9.0, 0.0], [0.0, 0.0]])
    sample.h_nmr_j_mask = torch.tensor([[True, False], [False, False]])

    output = NormalizeNMR(str(path), encode_smiles=False)(sample)

    assert torch.equal(output.h_nmr, torch.tensor([[-1.0, 1.0, 0.0]]))
    assert torch.equal(output.c_nmr, torch.tensor([[-1.0, 1.0, 0.0]]))
    assert torch.equal(output.h_nmr_integration, torch.tensor([1.0, 0.0]))
    assert torch.equal(output.h_nmr_j, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_multiplicity_vocab_keeps_every_training_label(tmp_path):
    stats = {
        "hnmr_shift_mean": 0.0,
        "hnmr_shift_std": 1.0,
        "cnmr_shift_mean": 0.0,
        "cnmr_shift_std": 1.0,
        "hnmr_integration_mean": 0.0,
        "hnmr_integration_std": 1.0,
        "hnmr_j_mean": 0.0,
        "hnmr_j_std": 1.0,
        "multiplicity_labels": [
            "<pad>", "<missing>", "<unk>", "d", "ddddd", "dtdd",
        ],
    }
    path = tmp_path / "dataset_infos.json"
    path.write_text(json.dumps(stats))
    sample = Sample()
    sample.h_nmr = torch.zeros(4)
    sample.c_nmr = torch.zeros(1)
    sample.h_nmr_multiplicity = ["d", "ddddd", "dtdd", "new-label"]

    output = NormalizeNMR(str(path), encode_smiles=False)(sample)

    assert output.h_nmr_multiplicity.tolist() == [3, 4, 5, 2]
    assert output.h_nmr_multiplicity_mask.tolist() == [True, True, True, True]
