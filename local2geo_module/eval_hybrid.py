"""Load a trained hybrid prior and write explicit-H XYZ coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .data import collate_local2geo, graph_from_smiles
from .eval import _resolve_device, _safe_stem, write_xyz
from .geometry_solver import DifferentiableGeometrySolver
from .lit_module import HybridLocal2GeoModule
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


def _simulator_from_checkpoint(
    module: HybridLocal2GeoModule,
    noise_std: float,
) -> SoftGraphSimulator:
    h = module.hparams
    return SoftGraphSimulator(
        clean_margin=float(h.clean_margin),
        logit_noise_std=noise_std,
        bond_type_confusion_probability=float(
            h.bond_type_confusion_probability
        ),
        false_positive_probability=float(
            h.false_positive_probability
        ),
        false_negative_probability=float(
            h.false_negative_probability
        ),
        attachment_confusion_probability=float(
            h.attachment_confusion_probability
        ),
        corruption_boost=float(h.corruption_boost),
    )


def evaluate_smiles(
    smiles: List[str],
    checkpoint: Path,
    output: Optional[Path],
    output_dir: Path,
    input_mode: str,
    noise_std: float,
    seed: int,
    device: torch.device,
    num_steps: int,
    step_size: float,
    seed_mode: str,
    one_three_weight: float,
    one_four_weight: float,
    bond_probability_power: float,
    angle_probability_power: float,
    soft_stress_steps: int,
    soft_stress_step_size: float,
    soft_stress_init_scale: float,
    write_sdf_files: bool,
) -> List[Path]:
    module = HybridLocal2GeoModule.load_from_checkpoint(
        str(checkpoint), map_location=device
    ).to(device)
    module.eval()
    simulator = _simulator_from_checkpoint(module, noise_std).to(device)
    solver = DifferentiableGeometrySolver(
        num_steps=num_steps,
        step_size=step_size,
        seed_mode=seed_mode,
        one_three_distance_weight=one_three_weight,
        one_four_distance_weight=one_four_weight,
        bond_probability_power=bond_probability_power,
        angle_probability_power=angle_probability_power,
        soft_stress_steps=soft_stress_steps,
        soft_stress_step_size=soft_stress_step_size,
        soft_stress_init_scale=soft_stress_init_scale,
    ).to(device)

    samples = [graph_from_smiles(value) for value in smiles]
    cpu_batch = collate_local2geo(samples)
    batch = _move_batch(cpu_batch, device)
    with torch.no_grad():
        raw_graph = simulator(
            batch,
            corrupted=input_mode == "corrupted-soft",
            seed=seed,
        )
        learned = module.correct_graph(batch, raw_graph)
        learned_geometry = torch.softmax(
            learned["geometry_logits"], dim=-1
        )
        local_priors = {
            "one_three_probability": learned[
                "one_three_probability"
            ],
            "one_four_probability": learned[
                "one_four_probability"
            ],
            "one_three_distance_ratio": learned[
                "one_three_distance_ratio"
            ],
            "one_four_distance_ratio": learned[
                "one_four_distance_ratio"
            ],
            "one_four_validity": (
                batch["heavy_mask"][:, :, None]
                & batch["heavy_mask"][:, None, :]
            ).to(learned["one_four_probability"].dtype),
        }
    result = solver(
        atomic_numbers=batch["atomic_numbers"],
        atom_mask=batch["atom_mask"],
        heavy_mask=batch["heavy_mask"],
        hydrogen_mask=batch["hydrogen_mask"],
        heavy_edge_logits=learned["corrected_heavy_edge_logits"],
        h_attachment_logits=learned[
            "corrected_h_attachment_logits"
        ],
        differentiable=False,
        geometry_probabilities_override=learned_geometry,
        local_geometry_priors=local_priors,
        coordinate_seed=seed,
    )
    coordinates = result["coordinates"].cpu()

    paths = []
    for index, sample in enumerate(samples):
        atom_count = sample["atomic_numbers"].numel()
        output_path = (
            output
            if output is not None
            else output_dir / f"{_safe_stem(sample['smiles'], index)}.xyz"
        )
        terms = result["geometry_terms"]
        comment = (
            f"SMILES={sample['smiles']} hybrid_local2geo "
            f"input={input_mode} seed={seed_mode}; "
            f"bond={float(terms['bond']):.6f} "
            f"angle={float(terms['angle']):.6f} "
            f"clash={float(terms['clash']):.6f} "
            f"one3={float(terms['one_three_distance']):.6f} "
            f"one4={float(terms['one_four_distance']):.6f}; "
            "explicit hydrogens"
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
            "Load a 2D-only hybrid topology checkpoint and convert SMILES "
            "to explicit-hydrogen XYZ coordinates."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smiles", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hybrid_local2geo_outputs"),
    )
    parser.add_argument(
        "--input-mode",
        choices=("clean-soft", "corrupted-soft"),
        default="clean-soft",
    )
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument(
        "--seed-mode",
        choices=("soft_stress", "differentiable", "mds"),
        default="soft_stress",
    )
    parser.add_argument("--soft-stress-steps", type=int, default=96)
    parser.add_argument(
        "--soft-stress-step-size", type=float, default=0.06
    )
    parser.add_argument(
        "--soft-stress-init-scale", type=float, default=1.5
    )
    parser.add_argument("--one-three-weight", type=float, default=2.0)
    parser.add_argument("--one-four-weight", type=float, default=2.0)
    parser.add_argument(
        "--bond-probability-power", type=float, default=1.0
    )
    parser.add_argument(
        "--angle-probability-power", type=float, default=1.0
    )
    parser.add_argument("--write-sdf", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.output is not None and len(args.smiles) != 1:
        raise ValueError("--output requires exactly one SMILES")
    if (
        args.num_steps < 0
        or args.step_size <= 0
        or args.noise_std < 0
        or args.one_three_weight < 0
        or args.one_four_weight < 0
        or args.bond_probability_power <= 0
        or args.angle_probability_power <= 0
        or args.soft_stress_steps < 0
        or args.soft_stress_step_size <= 0
        or args.soft_stress_init_scale <= 0
    ):
        raise ValueError(
            "num-steps/noise/weights must be non-negative; step-size and "
            "probability powers must be positive"
        )
    device = _resolve_device(args.device)
    paths = evaluate_smiles(
        smiles=args.smiles,
        checkpoint=args.checkpoint,
        output=args.output,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        noise_std=args.noise_std,
        seed=args.seed,
        device=device,
        num_steps=args.num_steps,
        step_size=args.step_size,
        seed_mode=args.seed_mode,
        one_three_weight=args.one_three_weight,
        one_four_weight=args.one_four_weight,
        bond_probability_power=args.bond_probability_power,
        angle_probability_power=args.angle_probability_power,
        soft_stress_steps=args.soft_stress_steps,
        soft_stress_step_size=args.soft_stress_step_size,
        soft_stress_init_scale=args.soft_stress_init_scale,
        write_sdf_files=args.write_sdf,
    )
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    for path in paths:
        print(f"Wrote:      {path}")


if __name__ == "__main__":
    main()
