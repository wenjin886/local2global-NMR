# 3D2Shift

Standalone SchNet pretraining from atom types and one randomly selected RDKit
conformer to per-atom 1H/13C chemical shifts.

The encoder follows the continuous-filter distance message passing used by
[SchNet](https://doi.org/10.1063/1.5019779) and the original
[CASCADE](https://doi.org/10.1039/D1SC03343C) model. It deliberately does not
perform conformer pooling: each stored conformer is an independent geometry
augmentation view during training.

Build a conservative dataset:

```bash
python -m preprocess.uspto_3d_nmr build \
  --parquet data/uspto/exp_data/*.parquet \
  --nmr-dir data/uspto/preprocessed \
  --coords-dir /path/to/preprocessed_coordinates \
  --output-dir data/uspto/3d2shift
```

The final train/validation/test assignment is read exclusively from
`--nmr-dir/train.pt`, `val.pt`, and `test.pt`, as produced by
`uspto_nmr_preprocess.py`. The split names on the coordinate HDF5 files are
used only to locate coordinates.

The default keeps only samples whose integrated hydrogen multiset exactly
matches the explicit hydrogen count and whose carbon line count exactly
matches the number of chirality-aware carbon symmetry classes. Run the same
command with `--audit-only` first on the full source collection and inspect
`audit.json`. `--hydrogen-policy exact_or_carbon_bound` and
`--carbon-policy collapse` are explicit, less conservative alternatives.

`h_peak_counts` preserves the source integration for every proton peak.
Carbon integrals are not treated as atom counts; `c_equivalence_class_sizes`
stores the class multiplicities derived from the SMILES graph, while the
class-to-shift assignment remains latent and is solved by multiset matching.

Train and evaluate:

```bash
python -m shift3d_module.train
python -m shift3d_module.eval \
  --checkpoint outputs/shift3d/checkpoints/last.ckpt \
  --data-dir data/uspto/3d2shift
```

Hydra overrides work normally, for example
`logger.mode=offline data.batch_size=64`.
