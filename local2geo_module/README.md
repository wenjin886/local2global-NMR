# local2geo_module

This is a standalone pretraining project for a sharp-but-soft all-atom graph
projector and a differentiable local-geometry initializer. It uses only SMILES
and clean 2D graph supervision. No conformer coordinates, RDKit 3D embedding,
or force-field targets are read.

## Data flow

```text
SMILES from data/uspto/preprocessed/{train,val,test}.pt
  -> explicit-H graph: H slots first, element-sorted heavy slots second
  -> synthetic noisy heavy-edge logits and H-to-heavy attachment logits
  -> residual heavy-edge and attachment projection
  -> unified sharp-but-soft all-atom edge probabilities
  -> learned canonical coordinate seed (trained only by 2D-derived priors)
  -> topology-aware differentiable local relaxation
  -> locally constrained explicit-H coordinates
```

The complete atom ordering matches `src.data.dataset.graph_targets_from_smiles`:
exchangeable hydrogen slots first, followed by heavy atoms grouped by element
and canonical rank. H attachment rows use a softmax over heavy parents. Their
column sums are supervised by per-heavy H counts, and a small entropy objective
encourages sharp rows without assigning physical meaning to equivalent H IDs.

The relaxation never selects a hard neighbor list. Bond forces are weighted by
soft edge probabilities, while the all-neighbor angle loss is evaluated from
weighted direction moments in `O(B*N^2)` rather than materializing all
`O(B*N^3)` triples. Coordinate forces use interaction sums rather than a
batch-wide mean, so an update does not shrink with batch size or molecule size.
Soft `Q @ Q` path mass distinguishes 1-3 contacts from more distant nonbonded
pairs, and a per-atom p-norm prevents severe clashes from being diluted by all
other pairs. Functional gradient steps use `create_graph=True` during training,
so local losses reach both heavy-edge and H-attachment input logits.

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
  datamodule.max_total_atoms=64 \
  lit_module.model.relaxation.num_steps=2 \
  lit_module.num_val_structures_to_log=0 \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=2 \
  +trainer.limit_val_batches=1
```

Useful Hydra overrides:

```bash
python -m local2geo_module.train \
  data_path=/path/to/full/preprocessed \
  log_path=/scratch/$USER/local2geo_logs \
  datamodule.train_batch_size=16 \
  lit_module.model.relaxation.num_steps=16
```

`DATA_PATH` and `LOG_PATH` environment variables provide the same overrides.

## Logged objectives and metrics

The main graph losses are balanced heavy-edge CE, supervised logit margin,
presence/type entropy, expected degree and valence, H-count and H-attachment
entropy objectives, geometry-class CE, and small projection-residual penalties.
Coordinates are evaluated only against 2D-derived covalent-radius bond lengths,
local angle classes, planarity, and topology-aware nonbond clash priors.

Validation logs include edge precision/recall/F1, conditional bond-type
accuracy, graph exact match, H-attachment multiset accuracy/count MAE,
all-atom and heavy-only bond/angle/clash metrics, geometry scores, and a
combined `val/score` used for checkpointing.

Every `lit_module.visualize_every_n_epochs` validation epochs, reservoir
sampling selects `lit_module.num_val_structures_to_log` random validation
molecules. W&B logs a table containing their clean and projected 2D graphs,
seed and relaxed 3D SDF structures, bond MAE, and minimum nonbonded vdW ratio.
The sampling changes by epoch and is not restricted to the first validation
batch.

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

New all-atom checkpoints export explicit hydrogens. Checkpoints trained with the
old heavy-only architecture are not compatible with the all-atom model.

## Integration contract

At integration time, replace the synthetic `heavy_edge_logits` and
`h_attachment_logits` with the identically named `NMRToGraph` outputs. The
initializer assembles these into a unified all-atom soft adjacency and returns
coordinates in the same H-first atom-slot order. Formal charge is parsed for
future experiments but disabled by default because the current NMR path does
not provide it. Later NMR-conditioned SE(3) blocks can additionally consume
`graph_atom_features` and `peak_features`.

The module deliberately outputs probabilities and never applies argmax in the
training path.
