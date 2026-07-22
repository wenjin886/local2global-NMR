import math


NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC = range(5)
NUM_BOND_TYPES = 5

BOND_ORDERS = [0.0, 1.0, 2.0, 3.0, 1.5]
BOND_LENGTH_SCALES = [1.0, 1.0, 0.90, 0.85, 0.93]

GEOMETRY_NAMES = (
    "terminal",
    "linear",
    "bent",
    "trigonal_planar",
    "trigonal_pyramidal",
    "tetrahedral",
    "other",
)
GEOMETRY_TO_INDEX = {name: index for index, name in enumerate(GEOMETRY_NAMES)}
GEOMETRY_COSINES = (
    0.0,
    -1.0,
    math.cos(math.radians(104.5)),
    math.cos(math.radians(120.0)),
    math.cos(math.radians(107.0)),
    -1.0 / 3.0,
    0.0,
)
PLANAR_GEOMETRY_INDEX = GEOMETRY_TO_INDEX["trigonal_planar"]
