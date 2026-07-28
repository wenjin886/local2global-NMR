# 3D2Shift

Standalone SchNet pretraining from atom types and offline-expanded RDKit
conformers to per-atom 1H/13C chemical shifts.

The encoder follows the continuous-filter distance message passing used by
[SchNet](https://doi.org/10.1063/1.5019779) and the original
[CASCADE](https://doi.org/10.1039/D1SC03343C) model. It deliberately does not
perform conformer pooling: the builder writes every stored conformer as an
independent `.pt` sample, and all conformers are visited in every epoch.

Build the dataset:

```bash
python -m preprocess.uspto_3d_nmr build \
  --nmr-dir data/uspto/preprocessed \
  --coords-dir /path/to/preprocessed_coordinates \
  --output-dir data/uspto/3d2shift
```

The final train/validation/test assignment is read exclusively from
`--nmr-dir/train.pt`, `val.pt`, and `test.pt`, as produced by
`uspto_nmr_preprocess.py`. The split names on the coordinate HDF5 files are
used only to locate coordinates.

The builder does not read the source parquet files. It preserves the raw
`h_nmr` and `c_nmr` peak sets from the three `.pt` files, without expanding
hydrogen integrations, collapsing carbon lines, or rejecting samples because
the number of peaks differs from the number of atoms. Hydrogen integration is
retained as optional metadata but is not used by the current set loss.

For each explicit atom, `environment_ids` is generated from the SMILES graph
with chirality-aware RDKit canonical symmetry ranks. The HDF5 atomic-number
sequence is checked against the explicit-H RDKit molecule before these labels
are accepted, so the environment IDs, atom types, and every stored conformer
share one atom order.

Dataset version 3 expands conformers offline: every saved sample contains one
`positions: [N, 3]` tensor plus its `conformer_index`. Training never selects or
generates conformers. A shard-aware sampler shuffles shard order and sample
order within each shard every epoch, while keeping reads localized so a random
sample does not trigger repeated loading of entire `.pt` shards.

The expensive coordinate and NMR split indices are cached at
`OUTPUT_DIR/index_cache.pt`. Subsequent dataset rebuilds reuse this file.
The cache is invalidated automatically when any source HDF5/PT size or
modification time changes; `--rebuild-index-cache` forces regeneration.

The primary objective is a robust bidirectional set loss:

`L_set = 1/2 mean_atom softmin_peak cost + 1/2 mean_peak softmin_atom cost`.

It operates directly on per-atom predictions and supports arbitrary atom/peak
cardinalities. A capped Huber pair cost reduces the influence of extra or
spurious peaks. During standalone pretraining, `L_equiv` additionally
penalizes prediction spread among atoms with the same `environment_id`.
For end-to-end NMR -> 3D -> NMR training, set
`model.equivalence_loss_weight=0`; graph environment labels are then not
required by the downstream loss.

The SchNet heads predict normalized shifts. H and C peak targets are
standardized independently with the training-only statistics in
`dataset_infos_train.json`; validation and test statistics are never used.
Set/equivalence losses operate in normalized space, while predictions and
nearest-set MAE metrics are converted back to ppm. The ppm-valued Huber,
outlier-cap, and soft-matching settings in the config are converted internally,
so they retain their physical interpretation. The resolved mean/std values are
stored in the Lightning checkpoint, allowing evaluation without re-reading the
JSON file.

Train and evaluate:

```bash
python -m shift3d_module.train
python -m shift3d_module.eval \
  --checkpoint outputs/shift3d/checkpoints/last.ckpt \
  --data-dir data/uspto/3d2shift
```

Hydra overrides work normally, for example
`logger.mode=offline data.batch_size=64`.
