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

## SMILES decoder and optional BiXT fusion

A causal canonical-SMILES decoder can read either the original joint memory or
the refined atom/NMR memory. The staged configurations use:

```yaml
lit_module.model.use_smiles_loss: true
lit_module.model.use_smiles_conditioning: false
lit_module.model.use_smiles_joint_bixt: true
lit_module.model.smiles_memory: joint
lit_module.model.num_smiles_layers: 5
```

`use_smiles_joint_bixt` adds exactly one terminal bidirectional cross-attention
block after the causal decoder. Following
[BiXT](https://arxiv.org/abs/2402.12138), one shared atom/NMR-to-SMILES
similarity matrix is normalized in both directions to update the SMILES hidden
states and complete joint atom/H-NMR/C-NMR memory simultaneously. SMILES logits
come from the updated hidden states, and downstream atom queries read the
updated joint memory. During greedy generation, every next token is selected
from the BiXT-updated logits.

The BiXT depth is intentionally fixed at one. Both directions are computed from
pre-update inputs, so SMILES logits only read causal decoder states and the
original joint memory. Feeding the SMILES-updated joint memory into a second
BiXT block would leak later teacher-forced tokens back into earlier logits.
With the implemented two-sided FFNs, five decoder layers plus terminal BiXT
contains slightly more SMILES-path parameters than six decoder layers, but
fewer than seven; it is therefore the intended near-six-layer comparison.

`use_smiles_conditioning` retains the previous `atom_smiles_layer` for
ablation, but it and `use_smiles_joint_bixt` are mutually exclusive. BiXT
requires `smiles_memory: joint`; the refined-memory path remains available when
BiXT is disabled.

The generation target is canonical isomeric SMILES, so atom chirality and bond
stereochemistry are preserved. `rxn.chemutils.tokenization.tokenize_smiles`
produces chemical tokens such as `[C@H]`, `Cl`, `/`, and `\\`. The vocabulary
is built from the training split during preprocessing and stored in
`dataset_infos_train.json` together with BOS/EOS/PAD/UNK control tokens. Validation
and test tokens absent from the training vocabulary map to `<unk>`.

## Validation losses and inference metrics

Every stage uses two validation paths:

```yaml
inference_only_validation: true
datamodule.val_generation_size: 1024
datamodule.val_generation_seed: 0
```

The first dataloader traverses the complete validation set with teacher forcing
and logs all standard `val/loss_*` terms, including `val/loss_weighted`, so
train/val loss divergence remains visible. These losses never determine
checkpoint selection.

The second dataloader is the deterministic 1024-molecule subset. It performs
greedy generation, feeds generated SMILES hidden states through BiXT, and
executes the same fragment/graph path used at inference. No teacher-forced
representation contributes to `val_inference/*`. The compact metric set is
stage-aware:

```text
SMILES:
  val_inference/smiles_exact_accuracy

fragment:
  val_inference/smiles_exact_accuracy
  val_inference/heavy_fragment_score

graph:
  val_inference/smiles_exact_accuracy
  val_inference/heavy_fragment_score
  val_inference/graph_score
  val_inference/edge_precision
  val_inference/edge_recall
  val_inference/predicted_to_target_edge_count_ratio
```

The fragment score is the geometric mean of fragment-presence macro-F1 and
positive-count accuracy. The graph score is the geometric mean of
bond-existence F1 and typed-bond recall. Checkpoint selection uses the main
score for its stage and is independent of tunable loss weights.

The first 10 molecules are still logged once per epoch as a W&B table containing
target/predicted SMILES, validity, exactness, element compositions, and readable
target/predicted heavy-atom fragment counts. This reuses the greedy outputs and
does not run an additional generation pass. A `LearningRateMonitor` records the
optimizer learning rate at every step.

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

The SMILES-only stage disables this objective. Fragment and graph stages enable
the fragment-side neighbor-count constraint with weight `0.1`. The lookup is a
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

Full-graph training additionally constrains the expected bond-order valence of
every carbon atom to four. Single, double, and triple heavy bonds contribute
`1`, `2`, and `3`, while aromatic bonds each contribute one sigma bond plus one
soft pi contribution per carbon with any incident aromatic bond. This keeps a
three-aromatic-bond fused carbon at valence four. Every soft H attachment
contributes `1`. `carbon_valence_weight` controls the normalized squared error
and `train_uspto_graph.yaml` enables it with weight `0.1`.

Fragment checkpoint carbon valence can be audited without new dataset labels:

```bash
python -m preprocess.audit_fragment_carbon_valence \
  --config configs/train_uspto_fragment.yaml \
  --checkpoint /path/to/fragment.ckpt \
  --data-dir /path/to/materialized/uspto \
  --split val \
  --output fragment_carbon_valence_val.json
```

The report includes carbon-level valence accuracy, the fraction of
carbon-containing molecules in which every predicted carbon has valence four,
and the average number of invalid carbons per molecule. By default the model
uses its inference-time, non-teacher-forced SMILES path; pass
`--teacher-force-smiles` only for a teacher-forced diagnostic.

Fragment training applies the same chemistry directly to the soft count
distributions with `fragment_carbon_valence_weight`. Expected single, double,
triple, and aromatic port counts contribute `1`, `2`, `3`, and `1`; the soft
probability that any aromatic port is present adds one further pi contribution.
`train_uspto_fragment.yaml` enables the normalized squared error with weight
`0.1`. Validation logs argmax carbon-valence accuracy, the all-carbon-valid
molecule rate, and the average invalid-carbon count per molecule on both the
full teacher-forced path and the inference subset.

Heavy-edge cross entropy separately weights non-bonds and bonds. Edge class
zero (`none`) uses `criterion.edge_none_class_weight`; every non-zero bond
class uses the shared `criterion.edge_bond_class_weight`. The graph-training
configuration uses `0.4` and `1.0`, respectively, to reduce domination by the
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

### Stage 0: joint encoder and SMILES

```bash
DATA_PATH=data/uspto/preprocessed \
python src/train.py -cn train_uspto_smiles
```

Only SMILES cross-entropy has non-zero weight. This pretrains the atom/NMR
embeddings, joint encoder, five-layer causal decoder, and the SMILES-output side
of terminal BiXT. Checkpoint selection uses greedy SMILES exact accuracy.

### Stage 1: add fragments

```bash
DATA_PATH=data/uspto/preprocessed \
python src/train.py -cn train_uspto_fragment \
  ckpt_path=/path/to/smiles.ckpt
```

The fragment stage adds heavy-fragment and H-parent-environment supervision
while retaining the SMILES objective. Attachment and edge prediction remain
disabled. Non-strict weight loading initializes newly introduced or
architecture-changed parameters while starting a new optimizer/scheduler.

### Stage 2: full graph

```bash
DATA_PATH=data/uspto/preprocessed \
python src/train.py -cn train_uspto_graph \
  ckpt_path=/path/to/fragment.ckpt
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
visualization. With `use_smiles_joint_bixt: true`, `joint_features` contains the
SMILES-updated atom/NMR memory and `attention.smiles_joint_bixt` exposes both
directions of the shared attention matrix. `graph_joint_features` contains the
later fragment/H-context-aware atom/NMR refinement used by heavy-edge
prediction.

## Installation and tests

```bash
pip install -e '.[test]'
python -m pytest -q
```

# End-to-end NMR cycle training

The first end-to-end stage is implemented in `end2end_module` and configured
by `configs/train_end2end.yaml`:

```text
NMRToGraph
  -> pretrained SoftTopologyPrior
  -> differentiable geometry solver
  -> pooled-spectrum residual EGNN coordinate refiner
  -> frozen Shift3DModule
  -> NMR set loss
```

This path consumes `GraphBatch`, which has no coordinate field. Dataset XYZ is
therefore neither loaded nor available to the model. The topology prior is the
only graph-correction module; the EGNN changes coordinates only. Its minimal
inputs are generated coordinates, `graph_atom_features`, separately
masked-mean-pooled H/C peak features, and corrected all-atom soft bond
probabilities. Dense pair communication keeps a low-confidence or missed bond
from making two fragments invisible to the refiner.

The Shift3D checkpoint remains in evaluation mode with all parameters frozen,
but its forward pass is deliberately not wrapped in `torch.no_grad()`. NMR loss
gradients therefore flow through predicted shifts to refined coordinates and
the trainable upstream modules. Normalized graph-dataset peak shifts are
converted back to ppm with the statistics embedded in `Shift3DModule` before
the set loss is evaluated.

Set the component checkpoints and launch with:

```bash
export DATA_PATH=/path/to/uspto/preprocessed
export NMR2GRAPH_CKPT=/path/to/nmr2graph.ckpt
export TOPOLOGY_PRIOR_CKPT=/path/to/local2geo_prior.ckpt
export SHIFT3D_CKPT=/path/to/shift3d.ckpt
python -m end2end_module.train
```

The validation datamodule deterministically selects nine molecules. Every
validation run writes a W&B table containing generated 3D structures, raw and
SoftTopologyPrior-corrected predicted graphs, target graphs, predicted/target
SMILES, and predicted/target H/C NMR values. There is intentionally no target
3D structure in the table.
