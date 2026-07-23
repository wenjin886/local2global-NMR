from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from .constants import NONE, NUM_BOND_TYPES
from .data import collate_local2geo, graph_from_smiles


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_training_config(checkpoint_path: Path) -> Optional[Path]:
    """Find the Hydra config saved beside a training checkpoint."""
    for directory in (checkpoint_path.parent, *checkpoint_path.parents):
        candidate = directory / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


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


def clean_edge_logits(
    bond_types: torch.Tensor,
    pair_mask: torch.Tensor,
    margin: float = 4.0,
) -> torch.Tensor:
    """Convert a categorical 2D graph to deterministic sharp, finite logits."""
    targets = bond_types.clamp_min(NONE)
    logits = torch.zeros(
        (*targets.shape, NUM_BOND_TYPES),
        device=targets.device,
        dtype=torch.get_default_dtype(),
    )
    logits.scatter_(
        -1,
        targets.unsqueeze(-1),
        torch.full((*targets.shape, 1), margin, device=targets.device),
    )
    logits = logits.masked_fill(~pair_mask.unsqueeze(-1), -20.0)
    logits[..., NONE] = torch.where(
        pair_mask,
        logits[..., NONE],
        torch.full_like(logits[..., NONE], 20.0),
    )
    return logits


def load_module(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    relax_steps: int = -1
):
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if config_path is None:
        config_path = _find_training_config(checkpoint_path)
    if config_path is None:
        raise FileNotFoundError(
            "Could not find .hydra/config.yaml above the checkpoint. "
            "Pass the training config explicitly with --config."
        )
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    cfg = OmegaConf.load(config_path)
    module = hydra.utils.instantiate(cfg.lit_module)
    checkpoint = _load_torch(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = module.load_state_dict(state_dict, strict=False)
    tolerated_unexpected = {
        key for key in incompatible.unexpected_keys
        if key.endswith(".bond_embedding")
    }
    unsupported_unexpected = (
        set(incompatible.unexpected_keys) - tolerated_unexpected
    )
    if incompatible.missing_keys or unsupported_unexpected:
        details = []
        if incompatible.missing_keys:
            details.append(f"missing keys: {sorted(incompatible.missing_keys)}")
        if unsupported_unexpected:
            details.append(
                f"unexpected keys: {sorted(unsupported_unexpected)}"
            )
        raise RuntimeError(
            "Checkpoint is incompatible with the configured model ("
            + "; ".join(details)
            + ")"
        )
    module.eval()
    module.to(device)
    if relax_steps >= 0:
        module.model.relaxation.num_steps = relax_steps
        print(f"Set relaxation.num_steps = {relax_steps} for evaluation")
    return module, config_path


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
    module,
    smiles: List[str],
    output: Optional[Path],
    output_dir: Path,
    input_mode: str,
    clean_margin: float,
    seed: int,
    device: torch.device,
) -> List[Path]:
    samples = [graph_from_smiles(value) for value in smiles]
    cpu_batch = collate_local2geo(samples)
    batch = _move_batch(cpu_batch, device)

    if input_mode == "clean":
        edge_logits = clean_edge_logits(
            batch["bond_types"], batch["pair_mask"], margin=clean_margin
        )
    else:
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            edge_logits = module.val_corruptor(
                batch["bond_types"], batch["pair_mask"]
            )

    # Do not use torch.inference_mode(): the coordinate relaxation computes
    # forces with autograd even though inference does not retain the graph.
    outputs = module(
        batch,
        edge_logits,
        differentiable_relaxation=False,
    )
    coordinates = outputs["coordinates"].detach().cpu()
    probabilities = outputs["projected_edge_probabilities"].detach().cpu()

    paths = []
    for index, sample in enumerate(samples):
        atom_count = int(sample["atomic_numbers"].numel())
        if output is not None:
            output_path = output
        else:
            output_path = output_dir / f"{_safe_stem(sample['smiles'], index)}.xyz"
        pair_mask = cpu_batch["pair_mask"][index, :atom_count, :atom_count]
        upper = torch.triu(pair_mask, diagonal=1)
        confidence = probabilities[index, :atom_count, :atom_count].amax(dim=-1)
        mean_confidence = float(confidence[upper].mean()) if upper.any() else 1.0
        comment = (
            f"SMILES={sample['smiles']} local2geo input={input_mode} "
            f"mean_edge_confidence={mean_confidence:.6f}; heavy atoms only"
        )
        write_xyz(
            output_path,
            sample["atomic_numbers"],
            coordinates[index, :atom_count],
            comment,
        )
        paths.append(output_path.resolve())
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained local2geo checkpoint and export heavy-atom XYZ "
            "coordinates for one or more SMILES."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Lightning .ckpt file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Training config.yaml. By default, search for .hydra/config.yaml "
            "above the checkpoint."
        ),
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
        help="Directory used when --output is omitted.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("clean", "val-corrupted"),
        default="clean",
        help=(
            "Use deterministic logits from the SMILES graph, or a seeded "
            "validation-style corruption."
        ),
    )
    parser.add_argument(
        "--clean-margin",
        type=float,
        default=4.0,
        help="Correct-class logit margin used by clean input mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Random seed used by val-corrupted input mode.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--relax-steps",
        default=-1,
        type=int,)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is not None and len(args.smiles) != 1:
        raise ValueError("--output can only be used with one SMILES")
    if args.clean_margin <= 0:
        raise ValueError("--clean-margin must be positive")

    device = _resolve_device(args.device)
    module, config_path = load_module(
        args.checkpoint,
        config_path=args.config,
        device=device,
        relax_steps=args.relax_steps
    )
    paths = evaluate_smiles(
        module=module,
        smiles=args.smiles,
        output=args.output,
        output_dir=args.output_dir,
        input_mode=args.input_mode,
        clean_margin=args.clean_margin,
        seed=args.seed,
        device=device,
    )
    print(f"Loaded checkpoint: {args.checkpoint.expanduser().resolve()}")
    print(f"Loaded config:     {config_path}")
    print(f"Device:            {device}")
    for path in paths:
        print(f"Wrote:             {path}")


if __name__ == "__main__":
    main()
