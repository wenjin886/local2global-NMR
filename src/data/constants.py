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
NUM_BOND_TYPES = 5  # none, single, double, triple, aromatic

# USPTO stores proton multiplicity as a categorical string.  Keep padding,
# genuinely missing metadata, and out-of-vocabulary values distinct.
MULTIPLICITY_VOCAB = [
    "<pad>", "<missing>", "<unk>",
    "s", "d", "t", "q", "p", "hept", "m", "h",
    "dd", "dt", "td", "ddd", "dq", "qd", "tt", "qt", "tq",
    "dddd", "ddt", "dtd", "dtt", "dqt", "dp", "pd", "pt",
    "ddtd", "ddq", "dddt", "ddddt", "dttt", "heptd", "dh",
    "dqd"
]
MULTIPLICITY_TO_INDEX = {
    value: index for index, value in enumerate(MULTIPLICITY_VOCAB)
}
MULTIPLICITY_PAD_INDEX = MULTIPLICITY_TO_INDEX["<pad>"]
MULTIPLICITY_MISSING_INDEX = MULTIPLICITY_TO_INDEX["<missing>"]
MULTIPLICITY_UNKNOWN_INDEX = MULTIPLICITY_TO_INDEX["<unk>"]
MAX_J_VALUES = 6


def multiplicity_to_index(value) -> int:
    if value is None:
        return MULTIPLICITY_MISSING_INDEX
    value = str(value).strip().lower()
    if not value or value in {"nan", "none", "null"}:
        return MULTIPLICITY_MISSING_INDEX
    return MULTIPLICITY_TO_INDEX.get(value, MULTIPLICITY_UNKNOWN_INDEX)


def parse_bond_type_candidates(
        candidates: List[str] = BOND_TYPE_CANDIDATES,
) -> List[Tuple[int, int]]:
    return [tuple(int(value) for value in candidate.split("-")) for candidate in candidates]
