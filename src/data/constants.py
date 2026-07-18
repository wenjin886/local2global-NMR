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


def parse_bond_type_candidates(
        candidates: List[str] = BOND_TYPE_CANDIDATES,
) -> List[Tuple[int, int]]:
    return [tuple(int(value) for value in candidate.split("-")) for candidate in candidates]

