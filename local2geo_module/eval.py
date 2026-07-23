from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from rdkit import Chem

from .data import collate_local2geo, graph_from_smiles
from .geometry_solver import DifferentiableGeometrySolver
from .soft_graph_simulator import SoftGraphSimulator
from .visualization import write_sdf


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch(
    batch: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _element_symbols(atomic_numbers: Iterable[int]) -> List[str]:
    table = Chem.GetPeriodicTable()
    return [table.GetElementSymbol(int(number)) for number in atomic_numbers]


def write_xyz(
    output_path: Path,
    atomic_numbers: torch.Tensor,
    coordinates: torch.Tensor,
    comment: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = _element_symbols(atomic_numbers.tolist())
    lines = [str(len(symbols)), comment]
    for symbol, (x, y, z) in zip(symbols, coordinates.tolist()):
        lines.append(f"{symbol:<2s} {x: .8f} {y: .8f} {z: .8f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_stem(smiles: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", smiles).strip("._")
    return f"{index:04d}_{stem[:60] or 'molecule'}"


def evaluate_smiles(
    smiles: List[str],
    solver: DifferentiableGeometrySolver,
    simulator: SoftGraphSimulator,
    output: Optional[Path],
    output_dir: Path,
    corrupted: bool,
    seed: int,
    device: torch.device,
    write_sdf_files: bool,
) -> List[Path]:
    samples = [graph_from_smiles(value) for value in smiles]
    cpu_batch = collate_local2geo(samples)
    batch = _move_batch(cpu_batch, device)
    soft_graph = simulator(batch, corrupted=corrupted, seed=seed)
    outputs = solver(
        atomic_numbers=batch["atomic_numbers"],
        atom_mask=batch["atom_mask"],
        heavy_mask=batch["heavy_mask"],
        hydrogen_mask=batch["hydrogen_mask"],
        heavy_edge_logits=soft_graph["heavy_edge_logits"],
        h_attachment_logits=soft_graph["h_attachment_logits"],
        differentiable=False,
    )
    coordinates = outputs["coordinates"].cpu()

    paths = []
    for index, sample in enumerate(samples):
        atom_count = int(sample["atomic_numbers"].numel())
        output_path = (
            output
            if output is not None
            else output_dir / f"{_safe_stem(sample['smiles'], index)}.xyz"
        )
        mode = "corrupted-soft" if corrupted else "clean-soft"
        sample_mask = batch["atom_mask"][index:index + 1, :atom_count]
        sample_pair_mask = outputs[
            "pair_mask"
        ][index:index + 1, :atom_count, :atom_count]
        terms = solver.terms(
            coordinates[index:index + 1, :atom_count],
            outputs["edge_probabilities"][
                index:index + 1, :atom_count, :atom_count
            ],
            outputs["geometry_probabilities"][
                index:index + 1, :atom_count
            ],
            sample_mask,
            sample_pair_mask,
            outputs["covalent_radii"][
                index:index + 1, :atom_count
            ],
            outputs["vdw_radii"][index:index + 1, :atom_count],
        )
        comment = (
            f"SMILES={sample['smiles']} local2geo input={mode}; "
            f"bond={float(terms['bond']):.6f} "
            f"angle={float(terms['angle']):.6f} "
            f"clash={float(terms['clash']):.6f}; explicit hydrogens"
        )
        write_xyz(
            output_path,
            sample["atomic_numbers"],
            coordinates[index, :atom_count],
            comment,
        )
        paths.append(output_path.resolve())
        if write_sdf_files:
            sdf_path = output_path.with_suffix(".sdf")
            write_sdf(
                sdf_path,
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
            "Run the parameter-free FP32 geometry solver on SMILES-derived "
            "NMRToGraph-shaped soft logits."
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
        help="Output .xyz path; valid only for a single SMILES.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local2geo_outputs"),
    )
    parser.add_argument(
        "--input-mode",
        choices=("clean-soft", "corrupted-soft"),
        default="clean-soft",
    )
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--num-steps",
        "--relax-steps",
        dest="num_steps",
        type=int,
        default=256,
    )
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--write-sdf",
        action="store_true",
        help="Also write an SDF with explicit clean connectivity.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is not None and len(args.smiles) != 1:
        raise ValueError("--output can only be used with one SMILES")
    if args.margin <= 0 or args.num_steps < 0 or args.step_size <= 0:
        raise ValueError("margin, num-steps, and step-size must be positive")
    # device = _resolve_device(args.device)
    device = torch.device("cpu")
    print(f"Evaluating {len(args.smiles)} SMILES on {device}...")
    simulator = SoftGraphSimulator(
        clean_margin=args.margin,
        logit_noise_std=args.noise_std,
    ).to(device)
    solver = DifferentiableGeometrySolver(
        num_steps=args.num_steps,
        step_size=args.step_size,
    ).to(device)
    paths = evaluate_smiles(
        smiles=args.smiles,
        solver=solver,
        simulator=simulator,
        output=args.output,
        output_dir=args.output_dir,
        corrupted=args.input_mode == "corrupted-soft",
        seed=args.seed,
        device=device,
        write_sdf_files=args.write_sdf,
    )
    print(f"Device:       {device}")
    print(f"Solver dtype: {torch.float32}")
    print(f"Parameters:   {sum(p.numel() for p in solver.parameters())}")
    for path in paths:
        print(f"Wrote:        {path}")


if __name__ == "__main__":
    main()
