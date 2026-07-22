from typing import List, Tuple


BOND_TYPE_CANDIDATES = [
    "1-1",
    "6-1", "6-2", "6-3", "6-4",
    "7-1", "7-2", "7-3", "7-4",
    "8-1", "8-2", "8-4",
    "9-1",
    "15-1", "15-2", "15-4",
    "16-1", "16-2", "16-4",
    "17-1",
    "35-1",
    "53-1",
]

HEAVY_ATOM_TYPES = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
# Broad maximum coordination numbers. This constrains neighbor count rather
# than bond-order-weighted valence and is intentionally permissive for charged
# N/O and hypervalent P/S chemistry.
DEFAULT_MAX_HEAVY_DEGREES = {
    6: 4,
    7: 4,
    8: 3,
    9: 1,
    14: 4,
    15: 5,
    16: 6,
    17: 1,
    35: 1,
    53: 1,
}
NUM_BOND_TYPES = 5  # none, single, double, triple, aromatic

# Actual labels are collected without frequency filtering by the training-set
# metrics. Only the control tokens are fixed globally.
MULTIPLICITY_VOCAB = ["<pad>", "<missing>", "<unk>"]
MULTIPLICITY_TO_INDEX = {
    value: index for index, value in enumerate(MULTIPLICITY_VOCAB)
}
MULTIPLICITY_PAD_INDEX = MULTIPLICITY_TO_INDEX["<pad>"]
MULTIPLICITY_MISSING_INDEX = MULTIPLICITY_TO_INDEX["<missing>"]
MULTIPLICITY_UNKNOWN_INDEX = MULTIPLICITY_TO_INDEX["<unk>"]
MAX_J_VALUES = 6

SMILES_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
SMILES_PAD_INDEX = 0
SMILES_BOS_INDEX = 1
SMILES_EOS_INDEX = 2
SMILES_UNKNOWN_INDEX = 3


def normalize_multiplicity_label(value) -> str:
    if value is None:
        return "<missing>"
    value = str(value).strip().lower()
    if not value or value in {"nan", "none", "null"}:
        return "<missing>"
    return value


def parse_bond_type_candidates(
        candidates: List[str] = BOND_TYPE_CANDIDATES,
) -> List[Tuple[int, int]]:
    return [tuple(int(value) for value in candidate.split("-")) for candidate in candidates]
