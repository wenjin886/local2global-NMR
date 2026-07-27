# local2geo_module

`local2geo_module` contains both a parameter-free differentiable all-atom
geometry solver and an optional learned soft-topology prior. Together they
consume the soft heavy-edge and H-attachment logits produced by `NMRToGraph`
and return explicit-H coordinates in the same atom-slot order.

Neither path uses conformers, 3D labels, RDKit embedding, force fields, or a
learned coordinate MLP. The learned prior is trained only from clean 2D graphs
and synthetic corruptions of their logits.

## Components

The soft-graph test-input simulator and production geometry solver are separate:

```python
from local2geo_module import (
    DifferentiableGeometrySolver,
    SoftGraphSimulator,
)
```

- `SoftGraphSimulator` converts a clean SMILES graph into finite,
  NMRToGraph-shaped `heavy_edge_logits` and `h_attachment_logits`. It supports
  clean-soft and corrupted-soft modes and is used only for tests and demos.
- `DifferentiableGeometrySolver` consumes atom types, masks, and those two
  logits. It has zero trainable parameters and always performs geometry
  operations in FP32 with autocast disabled.

The solver uses fixed chemistry priors:

- bond targets from covalent-radius sums and fixed bond-type scale factors;
- soft VSEPR-like geometry probabilities derived from atom type, soft degree,
  and soft single/double/triple/aromatic statistics without `argmax`;
- angle constraints from fixed ideal geometry cosines;
- trigonal-planar constraints;
- a soft excluded-volume lower bound for every unbonded atom pair, including
  one-three pairs such as H...H within a methyl group.

The production solver embeds fixed radius entries for H, B, C, N, O, F, Si,
P, S, Cl, Br, and I, covering the current dataset. It raises on an unsupported
element so extending the table is explicit rather than silently using an
incorrect fallback.

Coordinates are obtained by unrolled stress minimization. Three seed modes are
kept for explicit ablation:

- `soft_stress`: the hybrid default. It first opens the heavy-atom skeleton,
  then attaches H through soft parent probabilities and local outward
  directions, and finally runs a short all-atom reconciliation. Every stage
  uses soft bond, 1--3/1--4, uncertainty-aware path-distance, and
  excluded-volume stress. Its differentiable path has no `argmax`, hard graph
  search, `eigh`, or detach.
- `differentiable`: the legacy graph-smoothed spherical seed.
- `mds`: the detached hard shortest-path/MDS proposal used only as an
  evaluation comparison.

Each `SoftDistanceStressSeed` step is now a weighted SMACOF update over
predicted D12/D13/D14 targets. 2-hop and
3-hop membership is derived differentiably from the corrected graph rather
than the less accurate auxiliary membership heads. A small linear confidence
term preserves gradients to low-confidence edges, while weak graph/vdW terms
act only as lower bounds. The same objective is reused by the following
relaxation, so increasing `--num-steps` no longer switches to a compacting
moment-angle objective.

## Atom order and integration

The SMILES data helper and `NMRToGraph` use the same order:

```text
[all exchangeable H slots] + [element-grouped canonical heavy slots]
```

`NMRToGraph` updates features but does not permute atom slots. Production use:

```python
solver = DifferentiableGeometrySolver(num_steps=32)

geometry = solver(
    atomic_numbers=atom_types,
    atom_mask=atom_mask,
    heavy_mask=nmr_graph_outputs["heavy_mask"],
    hydrogen_mask=nmr_graph_outputs["hydrogen_mask"],
    heavy_edge_logits=nmr_graph_outputs["heavy_edge_logits"],
    h_attachment_logits=nmr_graph_outputs["h_attachment_logits"],
    differentiable=True,
)

coordinates = geometry["coordinates"]  # FP32, [B, N, 3]
```

`graph_atom_features` and NMR peak features bypass this parameter-free solver
and condition the later SE(3) refiner directly.

The outer NMR model may use `bf16-mixed` or `16-mixed`. The solver casts its
logits to FP32 inside an autocast-disabled region. The cast remains
differentiable. Coordinates and later SE(3) distance/direction calculations
should preferably remain FP32, while scalar feature MLPs may use mixed
precision.

## Hybrid soft-topology pretraining (no 3D labels)

The hybrid path replaces discrete neighbour/path projection with a dense
message-passing model. It predicts:

- residual corrections to heavy-bond and H-attachment logits;
- soft 1--3 and 1--4 membership;
- local distance as an endpoint-covalent-radius log-ratio;
- geometry class, ring, conjugation, and torsion-class auxiliaries.

Corrected logits use `corrected = raw + residual`, leaving an exact identity
gradient path from later coordinate/NMR losses back to the original
`NMRToGraph` logits. There is no `topk`, threshold, or `argmax` in this learned
training path. MDS remains an explicit detached evaluation-only ablation.

The default config reads the two example parquet files directly:

```bash
python -m local2geo_module.train
```

The default W&B mode is `offline`; use online logging on HPC with:

```bash
python -m local2geo_module.train \
  logger.mode=online \
  data.parquet_paths='[/path/to/parquet_directory]'
```

Useful smoke-test overrides are:

```bash
python -m local2geo_module.train \
  data.train_limit=32 data.val_limit=8 \
  data.num_workers=0 trainer.max_epochs=1 \
  logger.enabled=false
```

Canonical non-stereochemical SMILES are deduplicated and deterministically
split 80/10/10, preventing the same 2D graph from leaking across splits.
Supervision is generated entirely from SMILES connectivity:

- bond and H-parent labels from the clean graph;
- graph-distance-2/3 membership;
- ring/conjugation and geometry classes;
- VSEPR/covalent-radius 1--3 targets;
- planar, ring-gauche, or acyclic-heavy-chain anti 1--4 targets.

After training, load a Lightning checkpoint and write explicit-H XYZ:

```bash
python -m local2geo_module.eval_hybrid \
  --checkpoint outputs/local2geo_hybrid/.../checkpoints/last.ckpt \
  --smiles "CCCC" "c1ccccc1" \
  --input-mode clean-soft \
  --seed-mode soft_stress \
  --soft-stress-steps 96 \
  --soft-stress-heavy-fraction 0.65 \
  --soft-stress-hydrogen-fraction 0.20 \
  --num-steps 256 \
  --output-dir hybrid_local2geo_outputs \
  --write-sdf
```

Use `--input-mode corrupted-soft` to measure graph-error recovery. For later
end-to-end NMR training, import `HybridLocal2GeoModule.correct_graph` (or the
contained `SoftTopologyPrior`) and run the geometry solver with
`seed_mode="soft_stress"` and `differentiable=True`.

## SMILES demo

No checkpoint is required:

```bash
python -m local2geo_module.eval \
  --smiles "CCO" "c1ccccc1" \
  --input-mode clean-soft \
  --seed-mode mds \
  --mds-inflation 1.15 \
  --mds-stress-steps 384 \
  --num-steps 256 \
  --unbonded-distance-scale 0.80 \
  --unbonded-weight 2.0 \
  --output-dir local2geo_outputs \
  --write-sdf
```

For a direct seed ablation with otherwise identical relaxation:

```bash
python -m local2geo_module.eval \
  --smiles "CCCCCCCCCCCCC" \
  --seed-mode differentiable \
  --output-dir local2geo_outputs/soft_seed
```

To evaluate the fully-soft prior initializer directly from SMILES:

```bash
python -m local2geo_module.eval_prior \
  --smiles "CCO" "c1ccccc1" \
  --input-mode clean-soft \
  --num-steps 400 \
  --output-dir prior_initializer_outputs \
  --write-sdf
```

This path writes explicit-hydrogen XYZ coordinates. It uses the SMILES only to
simulate NMRToGraph-shaped soft probabilities; the prior initializer receives
the resulting edge and geometry probabilities rather than RDKit coordinates.
Use `--input-mode corrupted-soft` to test robustness to graph errors.

`--unbonded-distance-scale` multiplies the sum of the two vdW radii and
therefore controls the soft lower bound for every unbonded pair. Values around
`0.75`, `0.80`, and `0.85` are useful initial comparisons; overly large values
can expand strained rings or other legitimate close contacts.

To test robustness to NMRToGraph-like mistakes:

```bash
python -m local2geo_module.eval \
  --smiles "CCO" \
  --input-mode corrupted-soft \
  --seed 1729 \
  --output ethanol.xyz
```

The XYZ contains explicit H. The optional SDF uses clean connectivity so a
viewer does not infer false bonds from distance alone.

## Tests

The regression suite includes three complex molecules taken from the dataset:

```text
COC(=O)Cc1c(C)oc2cc(N)ccc2c1=O
C=C(CSCCCSc1ccc(C(=O)C(C)(C)N2CCOCC2)cc1)C(=O)OC
O=C(CC(F)(F)F)NC[C@H]1CN(c2ccc3c(c2)CCCc2cn[nH]c2-3)C(=O)O1
```

Run:

```bash
python -m unittest discover -s local2geo_module/tests -v
```

Tests cover atom ordering, explicit H counts, clean/corrupted soft-graph
simulation, zero trainable parameters, FP32 behavior inside BF16 autocast,
finite nonzero gradients to both input logits, local-energy reduction, batch
independence, and XYZ/SDF export.

## W&B visualization

`visualization.py` retains the 2D graph and 3D SDF utilities. The hybrid
standalone trainer logs all scalar recovery/local-prior metrics through its
Lightning logger; 3D structure logging can be added after a checkpoint is
meaningful because coordinate relaxation is deliberately not part of the
2D-only pretraining loss.
