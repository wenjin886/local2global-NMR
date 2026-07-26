# local2global-NMR

This repository currently implements the supervised `NMR -> molecular graph`
stage of the planned `NMR -> 3D -> NMR` pipeline. The graph model follows a
local-to-global curriculum:

```text
element-sorted explicit atom slots + separately embedded 1H/13C peak sets
    (1H shift + optional integration/multiplicity/J metadata)
    -> concatenated joint atom/1H/13C self-attention
    -> element-grouped ordered heavy-atom queries
    -> H/heavy atom interaction and refined atom features
    -> optional canonical-SMILES auxiliary decoding
    -> factorized local-fragment prediction
    -> molecule-local H-to-heavy retrieval
    -> H-context aggregation into heavy queries
    -> symmetric heavy-heavy edge prediction
```

No molecular formula is used. `data.h` supplies the exact explicit atom
inventory, including H. The objective contains a coarse heavy-atom
neighbor-count constraint, but intentionally does not impose a formal-valence or
bond-order-weighted valence loss.

Matrix parameters inside the joint encoder and SMILES decoder use Xavier
uniform initialization. Other graph heads retain their module-specific PyTorch
defaults, and SMILES/atom padding embedding rows remain zero.

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
within-heavy query embeddings to the jointly encoded atom features, then decodes
the resulting seeds against the complete joint atom/NMR memory. The resulting
indices form one stable interface for fragment targets, H-parent classes, and
heavy-edge targets.

This order is an internal training coordinate system. A final molecular graph
can still be canonicalized after connectivity is predicted.

## Optional SMILES auxiliary task

A causal canonical-SMILES decoder can read either the original joint memory or
the refined atom features together with the encoded NMR peaks. Three independent
switches control its use:

```yaml
lit_module.model.use_smiles_loss: true
lit_module.criterion.smiles_weight: 1.0
lit_module.model.use_smiles_conditioning: false
lit_module.model.smiles_memory: refined_atom_nmr
```

`use_smiles_loss` enables the auxiliary sequence objective. With the default
`use_smiles_conditioning: false`, the decoder is an independent training head:
its hidden states and logits never enter the graph branch. `smiles_memory: joint`
lets its loss update the joint encoder only. `smiles_memory: refined_atom_nmr`
instead supplies the post-heavy-query/post-interaction atom features followed by
the jointly encoded NMR peak features. The SMILES loss can then help distinguish
ordered atom features and atom inventory while still leaving the graph forward
path independent of generated SMILES.

`use_smiles_conditioning` retains the earlier experimental interface that lets
atom features read decoder hidden states. This creates the opposite dependency,
so it is supported only with `smiles_memory: joint`; combining it with
`refined_atom_nmr` would introduce a cycle and raises an explicit configuration
error.

Training and the full validation loader use teacher forcing. A separate fixed
validation subset uses greedy self-conditioned generation, making the exposure
gap visible without autoregressively decoding the entire validation set.
For an explicit teacher-forced inference path, set:

```yaml
lit_module.model.teacher_force_smiles_during_eval: true
```

To run the proposed upper-bound conditioning experiment:

```bash
python src/train.py -cn train_uspto_graph \
  lit_module.model.use_smiles_conditioning=true
```

The generation target is canonical isomeric SMILES, so atom chirality and bond
stereochemistry are preserved. `rxn.chemutils.tokenization.tokenize_smiles`
produces chemical tokens such as `[C@H]`, `Cl`, `/`, and `\\`. The vocabulary
is built from the training split during preprocessing and stored in
`dataset_infos_train.json` together with BOS/EOS/PAD/UNK control tokens. Validation
and test tokens absent from the training vocabulary map to `<unk>`.

## Validation metrics

Checkpoint selection is independent of all loss weights and monitors:

```text
val/heavy_fragment_score
    = sqrt(
        heavy_fragment_presence_macro_f1
        * heavy_fragment_positive_count_accuracy
      )
```

Fragment metrics are computed per molecule and then averaged. Validation also
reports atom-level fragment exact accuracy, permutation-invariant H-parent type
and environment accuracy, H-attachment multiset accuracy, and H-count MAE. The
fragment argmax predictions additionally report the fraction of heavy atoms
whose predicted number of directly bonded neighbors exceeds the
dataset-observed element limit.

When heavy-edge prediction is enabled, checkpoint selection uses the
molecule-macro graph score:

```text
graph_score = sqrt(bond_existence_f1 * typed_bond_recall)
```

`bond_existence_f1` ignores bond type and evaluates whether each heavy-atom pair
is connected. `typed_bond_recall` counts a target bond only when both its atom
pair and bond class are correct. This avoids the large no-bond class dominating
the monitor.

The datamodule deterministically samples 1024 validation molecules for greedy
SMILES generation:

```yaml
datamodule.val_generation_size: 1024
datamodule.val_generation_seed: 0
```

The full loader reports teacher-forced token accuracy and perplexity. The fixed
subset reports greedy exact match, RDKit validity, stereo-agnostic exact match,
and full element-composition exactness under the
`val_generation/` namespace. The first 10 molecules of this deterministic
subset are also logged once per epoch as a W&B table containing
target/predicted SMILES, validity, exactness, element compositions, and readable
target/predicted heavy-atom fragment counts with `neighbors=current/maximum`
summaries.
This reuses the existing greedy outputs and does not run an additional
generation pass. A
`LearningRateMonitor` records the optimizer learning rate at every step,
including warmup.

For full-graph training, the same deterministic subset supplies the first 10
graph examples without an additional forward pass. `val/graph_examples` has
exactly two W&B columns, `target_graph` and `predicted_graph_raw`. Both are
NetworkX renderings of the complete explicit-H graph. The predicted rendering
contains only bonds actually selected by the heavy-edge and H-attachment
argmaxes; it performs no RDKit sanitization, graph repair, or error
highlighting. Only target heavy-atom coordinates are shared between the two
panels. Hydrogens are independently arranged around their own target or
predicted parent, so permutation-invariant H slots do not create misleading
long bonds across the molecule.

## Heavy-fragment neighbor-count constraint

For each heavy atom, the model converts every fragment count distribution into
an expected count and sums over the 22 neighbor-element/bond-type candidates:

```text
expected_neighbor_count = sum_candidate sum_count count * p(count)
neighbor_count_overflow_loss
    = mean((relu(expected_neighbor_count - element_cap) / element_cap)^2)
```

This counts neighbors, so single, double, triple, and aromatic bonds each add
one; it never computes a bond-order sum or formal valence. A complete scan of
the current train/validation/test splits gives H 1, C 4, N 4, O 2,
F/Cl/Br/I 1, P 4, and S 4 in every split. These values are stored in
`DEFAULT_MAX_NEIGHBOR_COUNTS` and can be overridden with
`criterion.max_heavy_neighbor_counts`.

`train_uspto_fragment.yaml` keeps
`heavy_neighbor_count_weight: 0.0` and `smiles_memory: joint`, so checkpoints
trained before this optional constraint and refined SMILES memory were added can
continue training without changing their objective or parameter graph. The
full-graph config enables the constraint with weight `0.01`. The lookup is a
non-persistent buffer, and deprecated `heavy_degree_*` configuration names are
accepted as aliases for checkpoint/config compatibility. H-parent fragment
supervision is retained and is not subject to this overflow constraint.

Full-graph training also constrains the graph actually realized by the edge and
H-attachment heads. For every heavy atom:

```text
expected_heavy_neighbors = sum_j (1 - p(edge_ij = none))
expected_H_neighbors = sum_h p(h -> i)
expected_total_neighbors
    = expected_heavy_neighbors + expected_H_neighbors
```

`edge_total_neighbor_count_overflow` applies the same normalized squared
overflow against the dataset-observed element cap. It is fully differentiable,
counts every non-none heavy bond as one neighbor irrespective of bond order,
and uses soft H assignments rather than supplementing hydrogens from valence.
`train_uspto_graph.yaml` enables it with weight `0.1`.

Heavy-edge cross entropy separately weights non-bonds and bonds. Edge class
zero (`none`) uses `criterion.edge_none_class_weight`; every non-zero bond
class uses the shared `criterion.edge_bond_class_weight`. The graph-training
configuration uses `0.2` and `1.0`, respectively, to reduce domination by the
roughly ten-times-more-common non-bonded heavy-atom pairs. Both values default
to `1.0`, so older configurations retain unweighted cross entropy.

Immediately before heavy-edge readout, the optional graph joint encoder
concatenates all fragment/H-context-refined atom tokens (including explicit H)
with the jointly encoded H-NMR and C-NMR peak tokens. Masked self-attention then
lets updated atoms and spectra interact globally, and the atom-token portion of
its output is used for pairwise edge classification:

```text
[refined atom tokens; H-NMR tokens; C-NMR tokens]
                         |
             graph joint self-attention
                         |
              refined heavy-edge features
```

`use_graph_joint_encoder` controls this stage and
`num_graph_joint_layers` controls its depth. The graph configuration enables
one layer; the default remains disabled for architectural compatibility. Since
enabling it introduces new trainable parameters, the graph configuration loads
an older checkpoint as weights with `load_weights_strict: false` and starts a
new optimizer/scheduler state rather than attempting a full training-state
resume.

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
h_nmr_multiplicity          int16 categorical IDs from the train vocabulary
h_nmr_multiplicity_mask     which multiplicities are observed
h_nmr_j                     padded per-peak J-value sets
h_nmr_j_mask                valid entries in each J-value set
heavy_fragment_labels       int8 [N, num_fragment_types]
h_parent_fragment_labels    int8 [N, num_fragment_types]
h_parent_types              int8 [N]
h_attachment                int16 H row -> ordered heavy query index
bond_types                  uint8 [N, N] heavy-heavy targets
isomeric_smiles             canonical stereochemistry-preserving target
smiles_token_ids            int16 token IDs for isomeric_smiles
```

Atomic numbers are stored as `uint8`. Superseded `canno_h`,
`hydrogen_neighbors`, `heavy_atom_local_labels`, and separately stored aromatic
flags are removed. Aromaticity remains represented by bond type 4 and can be
derived from fragment or edge targets. Original and non-stereochemical SMILES
are used during preprocessing/splitting but are not duplicated in every saved
sample.

Each dataset-specific preprocess is responsible for materializing these graph
targets together with its spectrum fields. `NMRGraphDataset` can still
construct missing targets lazily from `isomeric_smiles`, but materialization
avoids running RDKit in every training epoch.

## Leakage-safe USPTO split

USPTO preprocessing removes atom and bond stereochemistry and converts every
SMILES to a canonical non-isomeric identity before splitting. Consequently,
variants containing `@`, `/`, or `\\` cannot occur in different splits.
This non-isomeric identity is used only for grouping and deduplication; it does
not replace the stereochemistry-preserving SMILES generation target.
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
will still be assigned to the same split. `dataset_infos_train.json` is always
computed from the training split only.

## Spectrum metadata and normalization

USPTO preprocessing keeps every proton peak as one aligned record: sorting by
shift also reorders its `nH`, `category`, and `j_values`.
`dataset_infos_{train,val,test}.json` records the corresponding split's
descriptive statistics for continuous H shift, C shift, integration, and J
values. Only the train file is used for runtime normalization and categorical
vocabularies. Multiplicity is categorical, so it records every observed label,
including rare compound labels such as `ddddd` and `dtdd`, plus an aligned
histogram rather than an arbitrary mean/std over category IDs. The
raw strings are scanned before saving, then converted once during preprocessing
with the training vocabulary. Only labels absent from the training vocabulary
map to `<unk>` in validation or test data. Runtime transforms therefore only
normalize continuous values and do not repeat SMILES tokenization or categorical
mapping every epoch.
The info files carry categorical-mapping and compact-storage versions, so
rerunning the preprocessor can detect fully materialized splits without loading
the large `.pt` files again. Existing split files are upgraded one at a time and
atomically replaced only after the mapped, compact replacement is written
successfully.

`NormalizeNMR` reads these statistics and z-score normalizes continuous values
after variable-size samples have been padded into a batch. This replaces many
per-sample tensor operations with one vectorized operation per field. Padding
and missing integration/J entries remain zero under their masks. The model then
uses independent H and C shift embeddings. The H branch adds optional
integration, multiplicity, and permutation-invariant J-set embeddings before
the two nuclei are concatenated.

The metadata interfaces can be disabled independently for datasets that do not
provide them:

```yaml
lit_module.model.use_h_integration: false
lit_module.model.use_h_multiplicity: false
lit_module.model.use_h_j: false
datamodule.batch_transform.normalize_h_integration: false
datamodule.batch_transform.normalize_h_j: false
```

Compute normalization statistics from the training split only and reuse that
same `dataset_infos_train.json` for validation and test normalization/mapping.

To audit whether a non-isolation constraint is valid and measure heavy-pair
class imbalance, scan the materialized targets directly:

```bash
python preprocess/count_graph_connectivity.py \
  --data_dir data/uspto/preprocessed
```

The script processes one split at a time and writes
`graph_connectivity_stats.json`. It reports the fraction of molecules with
multiple heavy connected components, isolated-heavy counts and element types,
single-heavy-atom molecules, and the bonded/nonbonded unordered heavy-pair
ratio. It uses only `h` and `bond_types`; RDKit and SMILES parsing are not
needed, although the Python package used to serialize each `.pt` sample must be
installed to load it.

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

The fragment configuration sets `predict_attachments: false` and
`predict_edges: false`. Attachment, H-context aggregation, and the dense
`[B,N,N,3D]` heavy-edge readout are skipped entirely instead of being computed
with zero loss weights. Training steps log losses only; molecule-wise Hungarian
classification metrics are evaluated during validation.

### Stage 2: add H-parent retrieval

Start from the Stage-1 checkpoint and enable attachment/count losses while
keeping edge losses disabled:

```bash
python src/train.py -cn train_uspto_graph \
  ckpt_path=/path/to/fragment.ckpt \
  lit_module.model.predict_edges=false \
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
+ heavy-fragment neighbor-count overflow
+ H parent-environment supervision
+ H-to-heavy retrieval
+ per-heavy H-count consistency
+ heavy-edge classification
+ fragment-edge consistency
```

## Representation analysis

The forward output exposes:

```text
atom_features_pre_joint
atom_features
heavy_query_features
joint_features
hydrogen_attachment_features
heavy_attachment_features
graph_atom_features
attention
```

These tensors can be used for pre/post-joint-encoder t-SNE, fragment linear
probes, H-parent retrieval analysis, and joint atom-spectrum attention
visualization. SMILES generation reads the joint memory as an auxiliary task;
with `smiles_memory: refined_atom_nmr`, it instead reads the refined atom/NMR
memory so its loss also supervises ordered atom refinement. Its decoder states
still do not condition the graph pipeline by default.

## Installation and tests

```bash
pip install -e '.[test]'
python -m pytest -q
```
