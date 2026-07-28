import torch

from shift3d_module.data import ShiftDataset


def test_split_is_loaded_once_from_one_pt_file(tmp_path):
    samples = [
        {
            "id": f"molecule:conf{index}",
            "positions": torch.randn(8, 3),
            "conformer_index": index,
        }
        for index in range(3)
    ]
    torch.save(
        {"version": 4, "split": "train", "samples": samples},
        tmp_path / "train.pt",
    )
    dataset = ShiftDataset(tmp_path, "train")
    assert len(dataset) == 3
    assert [dataset[index]["conformer_index"] for index in range(3)] == [
        0,
        1,
        2,
    ]
