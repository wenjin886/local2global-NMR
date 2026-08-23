"""Audit carbon valence in fragment-head predictions from a checkpoint."""

import argparse
import json
import os.path as osp
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from src.data.constants import BOND_TYPE_CANDIDATES
from src.data.dataset import NMRGraphDataset, TransformingCollator
from src.model.loss import NMRGraphLoss


def fragment_carbon_valences(fragment_predictions: torch.Tensor) -> torch.Tensor:
    """Return integer carbon valences implied by argmax fragment counts."""
    if fragment_predictions.size(-1) != len(BOND_TYPE_CANDIDATES):
        raise ValueError(
            "fragment_predictions last dimension must match "
            f"BOND_TYPE_CANDIDATES ({len(BOND_TYPE_CANDIDATES)})"
        )
    return NMRGraphLoss.fragment_carbon_valences_from_counts(
        fragment_predictions
    )


@dataclass
class FragmentCarbonValenceCounter:
    num_molecules: int = 0
    num_carbon_containing_molecules: int = 0
    num_all_carbon_valid_molecules: int = 0
    num_carbons: int = 0
    num_valid_carbons: int = 0

    def update(
            self,
            fragment_predictions: torch.Tensor,
            atom_types: torch.Tensor,
            atom_mask: torch.Tensor,
    ) -> None:
        valences = fragment_carbon_valences(fragment_predictions)
        carbon_mask = atom_mask.bool() & atom_types.eq(6)
        valid = valences.eq(4) & carbon_mask
        for sample_index in range(atom_types.size(0)):
            sample_carbon = carbon_mask[sample_index]
            num_carbons = int(sample_carbon.sum().item())
            num_valid = int(valid[sample_index].sum().item())
            self.num_molecules += 1
            self.num_carbons += num_carbons
            self.num_valid_carbons += num_valid
            if num_carbons:
                self.num_carbon_containing_molecules += 1
                self.num_all_carbon_valid_molecules += int(
                    num_valid == num_carbons
                )

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    def summarize(self) -> Dict[str, Any]:
        invalid_carbons = self.num_carbons - self.num_valid_carbons
        return {
            "num_molecules": self.num_molecules,
            "num_carbon_containing_molecules": (
                self.num_carbon_containing_molecules
            ),
            "num_carbons": self.num_carbons,
            "num_valid_carbons": self.num_valid_carbons,
            "num_invalid_carbons": invalid_carbons,
            "num_all_carbon_valid_molecules": (
                self.num_all_carbon_valid_molecules
            ),
            "fragment_carbon_valence_accuracy": self._ratio(
                self.num_valid_carbons, self.num_carbons
            ),
            "fragment_molecule_all_carbon_valid_rate": self._ratio(
                self.num_all_carbon_valid_molecules,
                self.num_carbon_containing_molecules,
            ),
            "average_invalid_carbons_per_molecule": self._ratio(
                invalid_carbons, self.num_molecules
            ),
            "average_invalid_carbons_per_carbon_containing_molecule": (
                self._ratio(
                    invalid_carbons,
                    self.num_carbon_containing_molecules,
                )
            ),
        }


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_fragment_model(
        config_path: str,
        checkpoint_path: str,
        data_dir: Optional[str] = None,
):
    # Imported lazily so the pure metric helpers remain usable without Hydra.
    import hydra
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    if data_dir is not None:
        # Override before model instantiation because the SMILES vocabulary
        # path is interpolated from data_path.
        config.data_path = data_dir
    model = hydra.utils.instantiate(config.lit_module.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=True)
    return config, model


def audit_checkpoint(
        config_path: str,
        checkpoint_path: str,
        split: str = "val",
        data_dir: Optional[str] = None,
        split_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        device: str = "auto",
        teacher_force_smiles: bool = False,
        max_molecules: Optional[int] = None,
) -> Dict[str, Any]:
    import hydra
    from tqdm import tqdm

    config, model = _load_fragment_model(
        config_path, checkpoint_path, data_dir=data_dir
    )
    if split_path is None:
        path_key = f"{split}_path"
        if path_key not in config.datamodule:
            raise ValueError(f"Config datamodule has no {path_key}")
        split_path = str(config.datamodule[path_key])
    if not osp.isfile(split_path):
        raise FileNotFoundError(split_path)

    transform = hydra.utils.instantiate(config.datamodule.batch_transform)
    dataset = NMRGraphDataset(split_path)
    if max_molecules is not None:
        if max_molecules <= 0:
            raise ValueError("max_molecules must be positive")
        dataset = Subset(dataset, range(min(max_molecules, len(dataset))))
    if batch_size is None:
        batch_size = int(config.datamodule.val_batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=TransformingCollator(transform),
    )

    selected_device = _device(device)
    model = model.eval().to(selected_device)
    counter = FragmentCarbonValenceCounter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Auditing {split} fragments"):
            batch = batch.to(selected_device)
            outputs = model(
                **batch.model_inputs(),
                teacher_force_smiles=teacher_force_smiles,
            )
            counter.update(
                outputs["fragment_logits"].argmax(dim=-1).cpu(),
                batch.atom_types.cpu(),
                batch.atom_mask.cpu(),
            )
    results = counter.summarize()
    results.update({
        "checkpoint": checkpoint_path,
        "config": config_path,
        "split": split,
        "split_path": split_path,
        "teacher_force_smiles": teacher_force_smiles,
    })
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="configs/train_uspto_fragment.yaml",
        help="Model config matching the checkpoint",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-force-smiles", action="store_true")
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results = audit_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        data_dir=args.data_dir,
        split_path=args.split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        teacher_force_smiles=args.teacher_force_smiles,
        max_molecules=args.max_molecules,
    )
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
