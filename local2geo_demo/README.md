# local2geo demo

This standalone demo converts a SMILES string into an XYZ file without using
3D labels, RDKit conformer embedding, or a trained model.

The pipeline is:

1. RDKit parses the SMILES and optionally adds explicit hydrogens.
2. The exact 2D bonds are converted into symmetric five-class soft edge
   probabilities (`none/single/double/triple/aromatic`). Confidence and logit
   noise are controllable so predicted soft graphs can be simulated.
3. A thresholded projection defines the working local neighborhoods.
4. Covalent-radius bond lengths and hybridization/VSEPR-like local angle
   templates produce a deterministic 3D tree assembly.
5. A short PyTorch optimization enforces bond, angle, planarity, ring-closure,
   and non-local clash priors.
6. Coordinates and geometry diagnostics are returned; no RMSD or conformer
   target is used.

Run from the repository root with an environment containing this project's
dependencies:

```bash
python -m local2geo_demo 'CCO' -o /tmp/ethanol.xyz
```

Simulate a less confident graph:

```bash
python -m local2geo_demo 'c1ccccc1' -o /tmp/benzene.xyz \
  --edge-confidence 0.85 --logit-noise 0.35 --seed 7
```

Useful options:

- `--no-hydrogens`: generate only atoms explicitly present in the SMILES.
- `--edge-threshold`: hard-projection threshold applied to bonded probability.
- `--steps`: number of unrolled prior-relaxation steps.

## Intended future interface

`SoftMolecularGraph.edge_probabilities` has the same final categorical axis as
the current NMR graph model. A future integration can replace
`simulate_soft_graph` with `softmax(heavy_edge_logits)` while retaining the
geometry initializer. Learned atom, edge, global, and peak-memory features are
deliberately absent here: they should condition bounded geometry corrections
and the later SE(3) refinement, not the fixed chemical-prior baseline.

## Scope

This is a local-geometry initializer, not a conformer generator or force
field. Torsions are deterministic initial choices, long-range interactions are
not modeled, stereochemistry is not yet enforced, and unusual hypervalent
centers use a generic spherical fallback. The XYZ should be judged by local
bond/angle/planarity/clash diagnostics rather than agreement with a particular
conformer.
