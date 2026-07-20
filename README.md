# local2global-NMR

This repository currently implements the supervised `NMR -> molecular graph`
stage of the planned `NMR -> 3D -> NMR` pipeline. The graph model follows a
local-to-global curriculum:

```text
element-sorted explicit atom slots + separately embedded 1H/13C peak sets
    (1H shift + optional integration/multiplicity/J metadata)
    -> shared atom-spectrum cross-attention
    -> element-grouped ordered heavy-atom queries
    -> factorized local-fragment prediction
    -> molecule-local H-to-heavy retrieval
    -> H-context aggregation into heavy queries
    -> symmetric heavy-heavy edge prediction
```

No molecular formula is used. `data.h` supplies the exact explicit atom
inventory, including H. The current objective intentionally contains no valence
loss.

## Factorized fragment representation

The previous element-specific local-label vocabulary was replaced because its
carbon vocabulary exceeded 500 classes and had a strong long tail. Each heavy
atom now predicts a categorical count for every neighbor-element/bond-type port:

```python
BOND_TYPE_CANDIDATES = [
    "1-1",
    "6-1", "6-2", "6-3", "6-4",
    "7-1", "7-2", "7-3", "7-4",
    "8-1", "8-2", "8-4",
    "9-1",
    "15-1", "15-2", "15-4",
    "16-1", "16-2", "16-4",
    "17-1", "35-1", "53-1",
]
```

For example, a methoxy carbon has `1-1: 3` and `8-1: 1`. Every port predicts
one of the count classes `0..4`. A separate presence loss reduces domination by
zero-count targets.

Every H slot also predicts the element and complete factorized fragment of its
parent heavy atom. These H targets are matched permutation-invariantly with the
Hungarian algorithm, because explicit H slots have no intrinsic ordering.

## Ordered heavy queries

Input atoms are sorted by element. Heavy targets use the same element groups and
RDKit canonical ranks to break ties within each group. The model adds learned
within-heavy query embeddings and decodes them against all NMR-conditioned atom
features. The resulting indices form one stable interface for fragment targets,
H-parent classes, and heavy-edge targets.

This order is an internal training coordinate system. A final molecular graph
can still be canonicalized after connectivity is predicted.

## H-to-heavy retrieval

H attachment is a dynamic classifier whose class count is the number of heavy
atoms in that molecule. H and heavy features are projected into a shared space:

```text
score(H_h, A_i) = cosine(project_H(H_h), project_A(A_i)) / temperature
```

Invalid and padded atoms are masked before the softmax. Attachment supervision
is permutation-invariant over H slots. The soft attachment matrix is then used
to aggregate proton context into every heavy query before edge prediction.

## Heavy-heavy graph prediction

The edge head produces symmetric logits for five classes:

```text
none, single, double, triple, aromatic
```

Predicted edges and H attachments are converted back to realized fragment
counts. A consistency loss compares those counts with the local fragment head:

```text
predicted local ports <-> realized global edges
```

This provides the differentiable local-to-global interface needed by the later
conformer model.

## Data contract

Each materialized sample contains:

```text
h                           element-sorted explicit atom slots
h_nmr, c_nmr                unassigned peak lists
h_nmr_integration           optional proton integrations
h_nmr_integration_mask      which integrations are observed
h_nmr_multiplicity          raw labels on disk; categorical IDs after transform
h_nmr_multiplicity_mask     which multiplicities are observed
h_nmr_j                     padded per-peak J-value sets
h_nmr_j_mask                valid entries in each J-value set
heavy_fragment_labels       [N, num_fragment_types]
h_parent_fragment_labels    [N, num_fragment_types]
h_parent_types              [N]
h_attachment                H row -> ordered heavy query index
bond_types                  [N, N] heavy-heavy targets
```

Each dataset-specific preprocess is responsible for materializing these graph
targets together with its spectrum fields. `NMRGraphDataset` can still
construct missing targets lazily from `smiles`, but materialization avoids
running RDKit in every training epoch.

## Leakage-safe USPTO split

USPTO preprocessing removes atom and bond stereochemistry and converts every
SMILES to a canonical non-isomeric identity before splitting. Consequently,
variants containing `@`, `/`, or `\\` cannot occur in different splits.
The default split is deterministic with seed 0:

```text
train : validation : test = 0.85 : 0.05 : 0.10
```

By default, repeated records for the same non-stereochemical molecule are
globally deduplicated. The command writes `train.pt`, `val.pt`, `test.pt`, and
an auditable `split_manifest.json`:

```bash
python preprocess/uspto_nmr_preprocess.py \
  --parquet_dir data/uspto/exp_data \
  --save_dir data/uspto/preprocessed \
  --seed 0
```

To retain repeated experimental spectra without leakage, pass
`--keep_duplicate_records`. All records belonging to one canonical molecule
will still be assigned to the same split. `dataset_infos.json` is always
computed from the training split only.

## Spectrum metadata and normalization

USPTO preprocessing keeps every proton peak as one aligned record: sorting by
shift also reorders its `nH`, `category`, and `j_values`. `dataset_infos.json`
records training-corpus mean/std for continuous H shift, C shift, integration,
and individual J values. Multiplicity is categorical, so the file records every
training label, including rare compound labels such as `ddddd` and `dtdd`, plus
an aligned histogram rather than an arbitrary mean/std over category IDs. The
raw strings remain in the preprocessed samples and are converted to IDs only by
the dataset transform. Only labels absent from the training vocabulary map to
`<unk>` at validation or inference time.

`NormalizeNMR` reads these statistics and z-score normalizes continuous values
in the dataset transform. Missing integration and J entries remain zero under
their availability masks. The model then uses independent H and C shift
embeddings. The H branch adds optional integration, multiplicity, and
permutation-invariant J-set embeddings before the two nuclei are concatenated.

The metadata interfaces can be disabled independently for datasets that do not
provide them:

```yaml
lit_module.model.use_h_integration: false
lit_module.model.use_h_multiplicity: false
lit_module.model.use_h_j: false
datamodule.transform.normalize_h_integration: false
datamodule.transform.normalize_h_j: false
```

Compute normalization statistics from the training split only and reuse that
same `dataset_infos.json` for validation and test data.

## Training curriculum

### Stage 1: fragments only

```bash
DATA_PATH=data/uspto/preprocessed \
python src/train.py -cn train_uspto_fragment
```

Optimized losses:

```text
heavy fragment count + presence
H parent fragment count + presence
H parent element
```

### Stage 2: add H-parent retrieval

Start from the Stage-1 checkpoint and enable attachment/count losses while
keeping edge losses disabled:

```bash
python src/train.py -cn train_uspto_graph \
  ckpt_path=/path/to/fragment.ckpt \
  lit_module.criterion.edge_weight=0 \
  lit_module.criterion.fragment_edge_consistency_weight=0
```

### Stage 3: full graph

```bash
python src/train.py -cn train_uspto_graph \
  ckpt_path=/path/to/attachment.ckpt
```

The full loss is:

```text
fragment supervision
+ H parent-environment supervision
+ H-to-heavy retrieval
+ per-heavy H-count consistency
+ heavy-edge classification
+ fragment-edge consistency
```

## Representation analysis

The forward output exposes:

```text
atom_features_pre_ca
atom_features
heavy_query_features
hydrogen_attachment_features
heavy_attachment_features
graph_atom_features
attention
```

These tensors can be used for pre/post-CA t-SNE, fragment linear probes,
H-parent retrieval analysis, and atom-spectrum attention visualization.

## Installation and tests

```bash
pip install -e '.[test]'
python -m pytest -q
```
