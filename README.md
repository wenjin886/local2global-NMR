# local2global-NMR

The current implementation covers the supervised `NMR -> graph` stage:

```text
explicit atom slots (data.h) + unassigned 1H/13C peaks
    -> shared atom-spectrum cross-attention
    -> bidirectional H/heavy-atom interaction
    -> heavy-heavy bond logits + H-to-heavy attachment probabilities
```

All atoms use the same embedding and NMR cross-attention. The representation is
split into H and heavy atoms only after NMR conditioning. Heavy-heavy bonds use
five classes (`none`, `single`, `double`, `triple`, `aromatic`); every H row is
normalized over heavy atoms and therefore predicts one attachment.

## Data contract

`NMRGraphDataset` accepts serialized samples containing `smiles`, `h_nmr`, and
`c_nmr`. If graph targets are absent, they are built from an explicit-H RDKit
molecule so that `data.h` and all targets share one atom ordering. For faster
training, materialize them once:

```bash
python preprocess/build_graph_dataset.py \
  --input_path data/uspto/preprocessed/train.pt \
  --output_path data/uspto/preprocessed/graph/train.pt
```

Repeat for validation and test splits and point `DATA_PATH` at the graph folder.

## Training

The project follows the Hydra/Lightning organization of
`frcnt/equivariant-neural-diffusion`:

```bash
pip install -e '.[test]'
DATA_PATH=data/uspto/preprocessed/graph \
python src/train.py -cn train_uspto_graph
```

The graph objective contains heavy-edge classification, permutation-invariant H
attachment, per-heavy-atom H-count consistency, and element-specific local
environment classification. It intentionally contains no valence loss.

## Tests

```bash
python -m pytest -q
```
