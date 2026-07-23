# local2geo_module

`local2geo_module` is a parameter-free, differentiable all-atom geometry
initializer. It consumes the soft heavy-edge and H-attachment logits produced
by `NMRToGraph` and returns explicit-H coordinates in the same atom-slot order.

It does not use conformers, 3D labels, RDKit embedding, force fields, a learned
coordinate MLP, or a graph projection network.

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

Coordinates are obtained by unrolled stress minimization from a deterministic
graph-smoothed seed. With `differentiable=True`, every update uses
`create_graph=True`, so downstream coordinate losses reach both heavy-edge and
H-attachment logits.

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

## SMILES demo

No checkpoint is required:

```bash
python -m local2geo_module.eval \
  --smiles "CCO" "c1ccccc1" \
  --input-mode clean-soft \
  --num-steps 256 \
  --unbonded-distance-scale 0.80 \
  --unbonded-weight 2.0 \
  --output-dir local2geo_outputs \
  --write-sdf
```

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

`visualization.py` retains the 2D graph and 3D SDF utilities. Because this
initializer has no standalone training loop, random validation structure
logging belongs in the main NMR Lightning module after integration rather than
in a separate local2geo trainer.
