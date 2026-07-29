# 3D2Shift

Standalone SchNet pretraining from atom types and offline-expanded RDKit
conformers to per-atom 1H/13C chemical shifts.

The encoder follows the continuous-filter distance message passing used by
[SchNet](https://doi.org/10.1063/1.5019779) and the original
[CASCADE](https://doi.org/10.1039/D1SC03343C) model. It deliberately does not
perform conformer pooling: the builder writes every stored conformer as an
independent entry in one `.pt` file per split, and all conformers are visited
in every epoch.

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

Dataset version 4 expands conformers offline: every saved sample contains one
`positions: [N, 3]` tensor plus its `conformer_index`. Training never selects or
generates conformers. The builder produces `train.pt`, `val.pt`, and `test.pt`.
Each complete split is loaded into CPU memory once during DataModule setup;
epoch shuffling is then purely an in-memory index permutation with no shard
I/O. The default uses `num_workers=0` so the loaded Python object graph is not
replicated into worker processes.

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

When W&B logging is enabled, every validation run records
`val/shift_target_vs_prediction`. The two-row stick plot shows raw target peaks
upward in blue and de-normalized per-atom model predictions downward in red,
with separate ppm ranges for 1H and 13C. By default each plot uses a 3-by-3
molecule layout containing nine different SMILES. Each molecule card stacks
its 1H and 13C plots, and long SMILES wrap across lines instead of being
truncated. The three conformers of one molecule therefore do not fill the
plot. Sample count and ppm limits are controlled by
`model.prediction_plot_samples`, `model.h_plot_ppm_*`, and
`model.c_plot_ppm_*`. Each subplot title includes its per-sample symmetric
nearest MAE. The companion `val/shift_examples` W&B table records the SMILES,
raw target/prediction ppm lists, and separate 1H/13C symmetric,
atom-to-peak, and peak-to-atom nearest MAE values. Validation metrics also log
`h/c_atom_to_peak_mae_ppm` and `h/c_peak_to_atom_mae_ppm` separately.
