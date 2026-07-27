from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .data import collate_local2geo, graph_from_smiles
from .eval import _resolve_device, _safe_stem, write_xyz
from .geometry_solver import DifferentiableGeometrySolver
from .prior_initializer import PriorGeometryInitializer
from .soft_graph_simulator import SoftGraphSimulator
from .visualization import write_sdf


def _move_batch(
    batch: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_prior_smiles(
    smiles: List[str],
    initializer: PriorGeometryInitializer,
    simulator: SoftGraphSimulator,
    output: Optional[Path],
    output_dir: Path,
    corrupted: bool,
    seed: int,
    device: torch.device,
    write_sdf_files: bool,
) -> List[Path]:
    """Convert SMILES to explicit-H XYZ using the dense-soft initializer."""
    samples = [graph_from_smiles(value) for value in smiles]
    cpu_batch = collate_local2geo(samples)
    batch = _move_batch(cpu_batch, device)
    simulated = simulator(batch, corrupted=corrupted, seed=seed)

    # Reuse only the fixed soft-graph and VSEPR rules. No coordinate
    # relaxation from DifferentiableGeometrySolver is executed.
    graph_builder = DifferentiableGeometrySolver(num_steps=0).to(device)
    graph = graph_builder.soft_graph(
        simulated["heavy_edge_logits"].float(),
        simulated["h_attachment_logits"].float(),
        batch["pair_mask"],
        batch["heavy_pair_mask"],
        batch["attachment_mask"],
    )
    geometry_probabilities = graph_builder.soft_geometry_probabilities(
        batch["atomic_numbers"],
        batch["atom_mask"],
        graph["edge_probabilities"],
    )
    result = initializer(
        atomic_numbers=batch["atomic_numbers"],
        atom_mask=batch["atom_mask"],
        edge_probabilities=graph["edge_probabilities"],
        geometry_probabilities=geometry_probabilities,
        differentiable=False,
        seed=seed,
    )
    coordinates = result["coordinates"].cpu()
    stress = result["stress"].cpu()

    paths: List[Path] = []
    mode = "corrupted-soft" if corrupted else "clean-soft"
    for index, sample in enumerate(samples):
        atom_count = int(sample["atomic_numbers"].numel())
        output_path = (
            output
            if output is not None
            else output_dir / f"{_safe_stem(sample['smiles'], index)}.xyz"
        )
        comment = (
            f"SMILES={sample['smiles']} prior_initializer input={mode}; "
            f"stress={float(stress[index]):.6f}; explicit hydrogens"
        )
        write_xyz(
            output_path,
            sample["atomic_numbers"],
            coordinates[index, :atom_count],
            comment,
        )
        paths.append(output_path.resolve())
        if write_sdf_files:
            write_sdf(
                output_path.with_suffix(".sdf"),
                sample["atomic_numbers"],
                sample["formal_charges"],
                sample["bond_types"],
                coordinates[index, :atom_count],
                sample["smiles"],
            )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate explicit-H XYZ coordinates from SMILES with the "
            "dense-soft PriorGeometryInitializer."
        )
    )
    parser.add_argument(
        "--smiles",
        nargs="+",
        required=True,
        help="One or more quoted SMILES strings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XYZ path; valid only with one SMILES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prior_initializer_outputs"),
    )
    parser.add_argument(
        "--input-mode",
        choices=("clean-soft", "corrupted-soft"),
        default="clean-soft",
    )
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--num-restarts", type=int, default=1)
    parser.add_argument("--init-scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--write-sdf",
        action="store_true",
        help="Also write an SDF using the clean SMILES connectivity.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is not None and len(args.smiles) != 1:
        raise ValueError("--output can only be used with one SMILES")
    if (
        args.num_steps < 0
        or args.num_restarts < 1
        or args.init_scale <= 0
        or args.margin <= 0
        or args.noise_std < 0
    ):
        raise ValueError(
            "num-steps and noise-std must be non-negative; num-restarts, "
            "init-scale, and margin must be positive"
        )

    device = _resolve_device(args.device)
    simulator = SoftGraphSimulator(
        clean_margin=args.margin,
        logit_noise_std=args.noise_std,
    ).to(device)
    initializer = PriorGeometryInitializer(
        num_steps=args.num_steps,
        init_scale=args.init_scale,
        num_restarts=args.num_restarts,
    ).to(device)
    paths = evaluate_prior_smiles(
        smiles=args.smiles,
        initializer=initializer,
        simulator=simulator,
        output=args.output,
        output_dir=args.output_dir,
        corrupted=args.input_mode == "corrupted-soft",
        seed=args.seed,
        device=device,
        write_sdf_files=args.write_sdf,
    )
    print(f"Device:       {device}")
    print(f"Parameters:   {sum(p.numel() for p in initializer.parameters())}")
    print(f"Input mode:   {args.input_mode}")
    for path in paths:
        print(f"Wrote:        {path}")


if __name__ == "__main__":
    main()
