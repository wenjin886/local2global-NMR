import argparse
import json
from pathlib import Path

from .geometry import generate_geometry, write_xyz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a locally reasonable XYZ structure from SMILES using "
            "simulated soft edges and fixed chemical priors."
        )
    )
    parser.add_argument("smiles", help="Input SMILES string")
    parser.add_argument("-o", "--output", required=True, help="Output .xyz path")
    parser.add_argument(
        "--edge-confidence",
        type=float,
        default=0.97,
        help="Probability assigned to the correct class for true bonds",
    )
    parser.add_argument(
        "--nonedge-bond-prob",
        type=float,
        default=0.002,
        help="Total bonded probability assigned to a non-edge",
    )
    parser.add_argument(
        "--logit-noise",
        type=float,
        default=0.0,
        help="Symmetric Gaussian noise added to simulated edge logits",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.5,
        help="Minimum bonded probability retained by hard projection",
    )
    parser.add_argument(
        "--steps", type=int, default=600, help="Prior-relaxation Adam steps"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-hydrogens",
        action="store_true",
        help="Do not add explicit hydrogens before geometry generation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_geometry(
        smiles=args.smiles,
        add_hydrogens=not args.no_hydrogens,
        edge_confidence=args.edge_confidence,
        nonedge_bond_probability=args.nonedge_bond_prob,
        logit_noise=args.logit_noise,
        edge_threshold=args.edge_threshold,
        relaxation_steps=args.steps,
        seed=args.seed,
    )
    output = Path(args.output)
    write_xyz(output, result)
    print(json.dumps({
        "output": str(output.resolve()),
        "num_atoms": len(result.symbols),
        **result.diagnostics,
    }, indent=2, sort_keys=True))
