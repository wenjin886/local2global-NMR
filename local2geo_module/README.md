# local2geo_module

This is a standalone pretraining project for a sharp-but-soft heavy-atom graph
projector and a differentiable local-geometry initializer. It uses only SMILES
and clean 2D graph supervision. No conformer coordinates, RDKit 3D embedding,
or force-field targets are read.

## Data flow

```text
SMILES from data/uspto/preprocessed/{train,val,test}.pt
  -> clean element-sorted heavy graph and 2D-derived local geometry class
  -> synthetic noisy five-class edge logits
  -> residual SoftGraphProjector
  -> sharp-but-soft edge probabilities
  -> learned canonical coordinate seed (trained only by 2D-derived priors)
  -> batched differentiable local relaxation
  -> locally constrained heavy-atom coordinates
```

The heavy atom ordering matches the element-grouped/canonical-rank convention
used by `src.data.dataset.graph_targets_from_smiles`. Explicit hydrogens are not
placed yet; their clean counts condition expected valence and local geometry.

The relaxation never selects a hard neighbor list. Bond forces are weighted by
soft edge probabilities, while the all-neighbor angle loss is evaluated from
weighted direction moments in `O(B*N^2)` rather than materializing all
`O(B*N^3)` triples. Its functional gradient steps use `create_graph=True`
during training, so local-geometry losses propagate through coordinates and
projected probabilities to the noisy input logits.

## Local run

Use an environment with the root project's dependencies:

```bash
conda activate spec2struc
WANDB_MODE=offline python -m local2geo_module.train
```

Small smoke run:

```bash
WANDB_MODE=offline python -m local2geo_module.train \
  datamodule.train_limit=32 \
  datamodule.val_limit=8 \
  datamodule.train_batch_size=2 \
  lit_module.model.relaxation.num_steps=2 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1
```

Useful Hydra overrides:

```bash
python -m local2geo_module.train \
  data_path=/path/to/full/preprocessed \
  log_path=/scratch/$USER/local2geo_logs \
  datamodule.train_batch_size=16 \
  lit_module.model.relaxation.num_steps=8
```

`DATA_PATH` and `LOG_PATH` environment variables provide the same overrides.

## Logged objectives and metrics

The main graph losses are balanced projected edge CE, supervised logit margin,
presence/type entropy, expected degree and valence, geometry-class CE, and a
small projection-residual penalty. Coordinates are evaluated only against
2D-derived covalent-radius bond lengths, local angle classes, planarity, and
nonbond clash priors.

Validation logs include edge precision/recall/F1, conditional bond-type
accuracy, graph exact match, entropy, geometry-class accuracy, bond MAE, angle
MAE, clash rate, a local geometry score, and a combined `val/score` used for
checkpointing.

Run the self-contained unit tests with:

```bash
python -m unittest discover -s local2geo_module/tests -v
```

## Export XYZ from a checkpoint

For a checkpoint left in its Hydra run directory, the evaluator automatically
finds the saved `.hydra/config.yaml`:

```bash
python -m local2geo_module.eval \
  --checkpoint logs/local2geo/runs/RUN/checkpoints/last.ckpt \
  --smiles "CCO" \
  --output ethanol.xyz
```

The default `clean` input mode constructs deterministic sharp logits from the
SMILES 2D graph. To inspect the projector under a reproducible validation-style
corruption, add `--input-mode val-corrupted --seed 1729`.

Multiple SMILES are evaluated as one batch and written to `local2geo_outputs/`
unless another directory is selected:

```bash
python -m local2geo_module.eval \
  --checkpoint /path/to/model.ckpt \
  --config /path/to/training/config.yaml \
  --smiles "CCO" "c1ccccc1" \
  --output-dir xyz_results
```

The current pretrained module is heavy-atom only, so these XYZ files do not yet
contain explicit hydrogens.

## Integration contract

At integration time, replace synthetic `noisy_edge_logits` with
`NMRToGraph`'s `heavy_edge_logits`. The default projector uses atom types,
masks, predicted heavy-edge logits, and H counts. Formal charge is parsed for
future experiments but disabled by default because the current NMR path does
not provide it. NMR information then enters through the raw edge
logits, and later NMR-conditioned SE(3) blocks can additionally consume
`graph_atom_features` and `peak_features`.

The module deliberately outputs probabilities and never applies argmax in the
training path.
