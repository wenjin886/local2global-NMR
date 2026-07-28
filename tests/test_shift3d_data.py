import json

from shift3d_module.data import ShardedShiftDataset, ShardShuffleSampler


def test_shard_sampler_shuffles_without_interleaving_shard_reads(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    counts = [3, 2, 4]
    (split / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "split": "train",
                "count": sum(counts),
                "shards": [
                    {"path": f"shard_{index:05d}.pt", "count": count}
                    for index, count in enumerate(counts)
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = ShardedShiftDataset(tmp_path, "train")
    sampler = ShardShuffleSampler(dataset, seed=7)

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert sorted(first) == list(range(sum(counts)))
    assert sorted(second) == list(range(sum(counts)))
    assert first != second

    def shard_for_index(index):
        if index < 3:
            return 0
        if index < 5:
            return 1
        return 2

    shard_sequence = [shard_for_index(index) for index in first]
    runs = [
        shard
        for index, shard in enumerate(shard_sequence)
        if index == 0 or shard != shard_sequence[index - 1]
    ]
    assert sorted(runs) == [0, 1, 2]
