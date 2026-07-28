"""Build a compact, sharded USPTO 3D -> NMR pretraining dataset.

The source spectra are MestreNova simulations. Hydrogen peak integrations are
expanded into an atom-sized multiset. Carbon labels are matched to
chirality-aware RDKit symmetry classes; raw carbon *line* counts are not used
as atom counts because heteronuclear coupling can split one resonance into
many lines.

The output is deliberately independent from the fragment/NMR-to-graph
datasets. Each molecule is stored once with all available RDKit conformers;
the training dataloader chooses one conformer as geometry augmentation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem
from tqdm import tqdm


URL_USPTO = "https://zenodo.org/records/17766755/files/uspto.tar.gz?download=1"
SPLITS = ("train", "val", "test")
DATASET_VERSION = 1


def read_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _format_uspto_mol_idx(mol_idx: Any) -> str:
    if isinstance(mol_idx, bytes):
        mol_idx = mol_idx.decode("utf-8")
    if isinstance(mol_idx, str):
        return mol_idx if mol_idx.isdigit() else f"{int(mol_idx):07d}"
    return f"{int(mol_idx):07d}"


def _copy_uspto_molecule_without_spectra(
    src_file: h5py.File,
    dst_file: h5py.File,
    mol_idx: Any,
) -> bool:
    mol_idx = _format_uspto_mol_idx(mol_idx)
    if mol_idx not in src_file:
        return False
    source = src_file[mol_idx]
    target = dst_file.create_group(mol_idx)
    target.attrs["mol_idx"] = mol_idx
    if "smiles" in source.attrs:
        target.attrs["smiles"] = source.attrs["smiles"]
    if "atom_features" not in source:
        return False
    source.copy("atom_features", target)
    return True


def _save_h5_item_to_npz_dict(
    item: h5py.Dataset | h5py.Group,
    save_key: str,
    npz_data: Dict[str, np.ndarray],
) -> None:
    for attr_key, attr_value in item.attrs.items():
        npz_data[f"{save_key}_attr_{attr_key}"] = np.asarray(attr_value)
    if isinstance(item, h5py.Dataset):
        npz_data[save_key] = item[()]
        return
    for subkey in item.keys():
        _save_h5_item_to_npz_dict(item[subkey], f"{save_key}_{subkey}", npz_data)


def read_uspto_h5(
    file_path: str,
    save_dir: str | Path | None = None,
    split_name: str = "split_indices_dedup",
) -> None:
    """Split the large ChefNMR HDF5 file while omitting dense spectra."""
    save_dir = osp.dirname(file_path) if save_dir is None else str(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    remaining = [
        split
        for split in SPLITS
        if not osp.exists(osp.join(save_dir, f"{split}_molecules.h5"))
    ]
    if not remaining:
        print("All coordinate splits are already present.")
        return

    indices: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    with h5py.File(file_path, "r", swmr=True) as source:
        if split_name not in source:
            raise KeyError(f"Cannot find split group {split_name!r}")
        for key in (split_name, "valid_indices_h", "valid_indices_c"):
            if key in source:
                _save_h5_item_to_npz_dict(source[key], key, indices)
        split_group = source[split_name]
        for split in remaining:
            output_path = osp.join(save_dir, f"{split}_molecules.h5")
            copied = 0
            with h5py.File(output_path, "w") as target:
                target.attrs.update(
                    source_file=file_path, split_name=split_name, split=split
                )
                for mol_idx in tqdm(split_group[split][()], desc=f"Saving {split}"):
                    copied += int(
                        _copy_uspto_molecule_without_spectra(
                            source, target, mol_idx
                        )
                    )
            counts[split] = copied
    for split, count in counts.items():
        indices[f"exported_{split}_count"] = np.asarray(count, dtype=np.int64)
    np.savez(osp.join(save_dir, "indices.npz"), **indices)


def download_and_split_uspto(target_dir: str) -> None:
    download_dir = osp.join(target_dir, "download")
    os.makedirs(download_dir, exist_ok=True)
    archive = osp.join(download_dir, "uspto.tar.gz")
    if not osp.exists(archive):
        urllib.request.urlretrieve(URL_USPTO, archive)
    data_dir = osp.join(download_dir, "data")
    if not osp.exists(data_dir):
        with tarfile.open(archive, "r") as handle:
            handle.extractall(download_dir)
    read_uspto_h5(
        osp.join(data_dir, "uspto", "molecules.h5"),
        save_dir=osp.join(download_dir, "preprocessed"),
    )


def canonical_smiles(smiles: str, isomeric: bool = True) -> Optional[str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=isomeric)


def _as_peak_list(value: Any) -> List[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return list(value)


def _integer_integration(peak: Mapping[str, Any]) -> Optional[int]:
    try:
        value = float(peak.get("nH"))
    except (TypeError, ValueError):
        return None
    # MestreNova occasionally emits zero-integration artefact peaks. They do
    # not contribute an atom to the expanded multiset and should be ignored,
    # rather than invalidating the entire molecule.
    if not np.isfinite(value) or abs(value - round(value)) > 1e-4 or value < 0:
        return None
    return int(round(value))


def expand_hydrogen_shifts(
    peaks: Sequence[Mapping[str, Any]],
) -> Optional[torch.Tensor]:
    expanded: List[float] = []
    for peak in peaks:
        count = _integer_integration(peak)
        try:
            shift = float(peak["delta"])
        except (KeyError, TypeError, ValueError):
            return None
        if count is None or not np.isfinite(shift):
            return None
        expanded.extend([shift] * count)
    return torch.tensor(sorted(expanded), dtype=torch.float32)


def hydrogen_peak_tensors(
    peaks: Sequence[Mapping[str, Any]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    values: List[Tuple[float, int]] = []
    for peak in peaks:
        count = _integer_integration(peak)
        try:
            shift = float(peak["delta"])
        except (KeyError, TypeError, ValueError):
            return None
        if count is None or not np.isfinite(shift):
            return None
        values.append((shift, count))
    values.sort()
    return (
        torch.tensor([value[0] for value in values], dtype=torch.float32),
        torch.tensor([value[1] for value in values], dtype=torch.long),
    )


def _carbon_line_values(
    peaks: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    values: List[float] = []
    weights: List[float] = []
    for peak in peaks:
        try:
            shift = float(peak["delta (ppm)"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(shift):
            continue
        weight = 1.0
        for key in ("integral", "intensity"):
            try:
                candidate = abs(float(peak.get(key)))
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate) and candidate > 0:
                weight = candidate
                break
        values.append(shift)
        weights.append(weight)
    order = np.argsort(values)
    return np.asarray(values)[order], np.asarray(weights)[order]


def collapse_carbon_lines(
    peaks: Sequence[Mapping[str, Any]],
    class_count: int,
    max_cluster_span: float = 15.0,
) -> Optional[torch.Tensor]:
    """Collapse sorted multiplet lines into a known number of resonances.

    This exact one-dimensional dynamic program minimizes weighted within-group
    variance over contiguous groups. It is intentionally opt-in: raw lines can
    include pathological simulated multiplets, so ``carbon_policy=exact`` is
    the conservative default.
    """
    values, weights = _carbon_line_values(peaks)
    line_count = len(values)
    if class_count <= 0 or line_count < class_count:
        return None
    if line_count == class_count:
        return torch.tensor(values, dtype=torch.float32)

    sw = np.concatenate([[0.0], np.cumsum(weights)])
    swx = np.concatenate([[0.0], np.cumsum(weights * values)])
    swx2 = np.concatenate([[0.0], np.cumsum(weights * values * values)])

    def interval_cost(left: int, right: int) -> float:
        total = sw[right] - sw[left]
        first = swx[right] - swx[left]
        second = swx2[right] - swx2[left]
        return max(0.0, second - first * first / max(total, 1e-12))

    inf = float("inf")
    costs = np.full((class_count + 1, line_count + 1), inf)
    previous = np.full((class_count + 1, line_count + 1), -1, dtype=np.int32)
    costs[0, 0] = 0.0
    for groups in range(1, class_count + 1):
        for right in range(groups, line_count + 1):
            best_cost = inf
            best_left = -1
            for left in range(groups - 1, right):
                candidate = costs[groups - 1, left] + interval_cost(left, right)
                if candidate < best_cost:
                    best_cost, best_left = candidate, left
            costs[groups, right] = best_cost
            previous[groups, right] = best_left

    intervals: List[Tuple[int, int]] = []
    right = line_count
    for groups in range(class_count, 0, -1):
        left = int(previous[groups, right])
        if left < 0:
            return None
        intervals.append((left, right))
        right = left
    intervals.reverse()
    if any(values[right - 1] - values[left] > max_cluster_span for left, right in intervals):
        return None
    centres = [
        float(np.average(values[left:right], weights=weights[left:right]))
        for left, right in intervals
    ]
    return torch.tensor(sorted(centres), dtype=torch.float32)


def _explicit_molecule(smiles: str) -> Optional[Chem.Mol]:
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.AddHs(molecule)


def symmetry_classes(molecule: Chem.Mol) -> torch.Tensor:
    ranks = Chem.CanonicalRankAtoms(
        molecule, breakTies=False, includeChirality=True
    )
    mapping: Dict[int, int] = {}
    compact: List[int] = []
    for rank in ranks:
        if int(rank) not in mapping:
            mapping[int(rank)] = len(mapping)
        compact.append(mapping[int(rank)])
    return torch.tensor(compact, dtype=torch.long)


def _normalize_coordinates(
    atom_coords: np.ndarray,
    atom_mask: np.ndarray,
) -> np.ndarray:
    """Return conformers as ``[C, N, 3]`` after applying the atom mask."""
    coords = np.asarray(atom_coords)
    mask = np.asarray(atom_mask).astype(bool).reshape(-1)
    if coords.ndim == 2:
        coords = coords[None, ...]
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Unexpected atom_coords shape: {coords.shape}")
    if coords.shape[1] != len(mask) and coords.shape[0] == len(mask):
        coords = np.transpose(coords, (1, 0, 2))
    if coords.shape[1] != len(mask):
        raise ValueError(
            f"Coordinate/mask mismatch: {coords.shape} versus {mask.shape}"
        )
    return np.asarray(coords[:, mask, :], dtype=np.float32)


def molecule_from_h5(
    group: h5py.Group,
    smiles: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    molecule = _explicit_molecule(smiles)
    if molecule is None:
        return None, "invalid_smiles"
    features = group.get("atom_features")
    if features is None:
        return None, "missing_atom_features"
    mask = np.asarray(features["atom_mask"][()]).astype(bool).reshape(-1)
    charges = np.asarray(features["atom_charges"][()]).reshape(-1)[mask].astype(int)
    expected = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    if not np.array_equal(charges, expected):
        return None, "atom_order_mismatch"
    try:
        positions = _normalize_coordinates(features["atom_coords"][()], mask)
    except ValueError:
        return None, "coordinate_shape_mismatch"
    return {
        "atomic_numbers": torch.tensor(charges, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.float32),
        "equivalence_classes": symmetry_classes(molecule),
    }, "ok"


def convert_to_data(mol_from_h5: h5py.Group, smiles: str):
    """Compatibility converter retained for the original ``*_with_nmr.pt`` flow."""
    try:
        from torch_geometric.data import Data
    except ImportError as error:
        raise ImportError(
            "The legacy PyG output path requires torch_geometric. "
            "The sharded 3D2Shift builder does not."
        ) from error

    data = Data(smiles=smiles)
    atom_features = mol_from_h5["atom_features"]
    atom_coords = atom_features["atom_coords"][()]
    atom_charges = atom_features["atom_charges"][()]
    atom_mask = atom_features["atom_mask"][()]
    mask = np.asarray(atom_mask).astype(bool).reshape(-1)
    data.num_nodes = int(mask.sum())
    data.h = torch.tensor(
        np.asarray(atom_charges).reshape(-1)[mask], dtype=torch.long
    )
    # Keep all conformers. This fixes the old [:num_atoms] slicing, which
    # sliced the conformer axis when atom_coords had shape [C, N, 3].
    data.pos = torch.tensor(
        _normalize_coordinates(atom_coords, mask), dtype=torch.float32
    )
    return data


def compact_nmr_data(data: Any) -> Any:
    """Retain shift counts needed by 3D2Shift in the legacy compact files."""
    keep = {
        "h_nmr",
        "h_nmr_integration",
        "h_nmr_integration_mask",
        "c_nmr",
        "smiles",
        "canonical_smiles",
        "isomeric_smiles",
    }
    for key in list(data.keys()):
        if key not in keep:
            del data[key]
    return data


def add_nmr_to_coord_data(
    coord_data: Any,
    nmr_data_split: Sequence[Any],
    nmr_smiles_idx: Mapping[str, int],
) -> Any:
    """Attach the NMR record selected by the pre-existing NMR split."""
    nmr_data = nmr_data_split[nmr_smiles_idx[coord_data.smiles]]
    coord_data.h_nmr = nmr_data.h_nmr
    coord_data.c_nmr = nmr_data.c_nmr
    if hasattr(nmr_data, "h_nmr_integration"):
        coord_data.h_nmr_integration = nmr_data.h_nmr_integration
    if hasattr(nmr_data, "h_nmr_integration_mask"):
        coord_data.h_nmr_integration_mask = nmr_data.h_nmr_integration_mask
    return coord_data


def _load_torch(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_nmr_split_index(
    nmr_dir: str | Path,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], Counter]:
    """Load the authoritative split assignment from train/val/test.pt."""
    split_for_smiles: Dict[str, str] = {}
    indices: Dict[str, Dict[str, int]] = {}
    audit: Counter = Counter()
    for split in SPLITS:
        path = Path(nmr_dir) / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing authoritative NMR split file: {path}"
            )
        records = _load_torch(path)
        split_index: Dict[str, int] = {}
        for index, record in enumerate(records):
            smiles = getattr(record, "isomeric_smiles", None)
            if smiles is None:
                smiles = getattr(record, "smiles", None)
            key = canonical_smiles(str(smiles), isomeric=True)
            if key is None:
                audit[f"invalid_nmr_smiles/{split}"] += 1
                continue
            previous = split_for_smiles.get(key)
            if previous is not None and previous != split:
                raise ValueError(
                    f"NMR split leakage for {key!r}: {previous} and {split}"
                )
            split_for_smiles[key] = split
            split_index[key] = index
            audit[f"nmr_records/{split}"] += 1
        indices[split] = split_index
    return split_for_smiles, indices, audit


def preprocess_uspto_only_nmr_3d_coords(
    nmr_dir: str,
    target_dir: str,
    coords_dir: str,
) -> None:
    """Reproduce the original PyG files using NMR .pt splits as authority.

    Coordinate HDF5 split names are deliberately ignored for final assignment:
    every coordinate SMILES is looked up in the already-created NMR
    train/val/test files.
    """
    os.makedirs(target_dir, exist_ok=True)
    _, split_indices, _ = load_nmr_split_index(nmr_dir)
    compact_records: Dict[str, Sequence[Any]] = {}
    size_info = {
        "only_nmr": {split: 0 for split in SPLITS},
        "coords_with_nmr": {split: 0 for split in SPLITS},
    }
    for split in SPLITS:
        compact_path = Path(target_dir) / f"{split}_only_nmr.pt"
        records = None
        if compact_path.exists():
            candidate = _load_torch(compact_path)
            if not candidate or hasattr(candidate[0], "h_nmr_integration"):
                records = candidate
            else:
                print(
                    f"Rebuilding {compact_path}: the existing compact file "
                    "does not contain h_nmr_integration."
                )
        if records is None:
            records = [
                compact_nmr_data(item)
                for item in tqdm(
                    _load_torch(Path(nmr_dir) / f"{split}.pt"),
                    desc=f"Compacting {split} NMR",
                )
            ]
            torch.save(records, compact_path)
        compact_records[split] = records
        size_info["only_nmr"][split] = len(records)

    coordinate_data: Dict[str, List[Any]] = {split: [] for split in SPLITS}
    for coordinate_split in SPLITS:
        path = Path(coords_dir) / f"{coordinate_split}_molecules.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as handle:
            for mol_idx in tqdm(
                handle.keys(), desc=f"Matching {coordinate_split} coordinates"
            ):
                group = handle[mol_idx]
                smiles = read_str(group.attrs["smiles"])
                key = canonical_smiles(smiles, isomeric=True)
                if key is None:
                    continue
                for nmr_split in SPLITS:
                    if key not in split_indices[nmr_split]:
                        continue
                    data = convert_to_data(group, key)
                    data = add_nmr_to_coord_data(
                        data,
                        compact_records[nmr_split],
                        split_indices[nmr_split],
                    )
                    coordinate_data[nmr_split].append(data)

    for split in SPLITS:
        output_path = Path(target_dir) / f"{split}_with_nmr.pt"
        torch.save(coordinate_data[split], output_path)
        size_info["coords_with_nmr"][split] = len(coordinate_data[split])
    (Path(target_dir) / "data_size_info.json").write_text(
        json.dumps(size_info, indent=2), encoding="utf-8"
    )


def _carbon_class_count(
    atomic_numbers: torch.Tensor,
    classes: torch.Tensor,
) -> int:
    return int(torch.unique(classes[atomic_numbers.eq(6)]).numel())


def _hydrogen_training_mask(
    molecule: Chem.Mol,
    policy: str,
    target_count: int,
    max_missing_hydrogens: int,
) -> Optional[torch.Tensor]:
    atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    )
    all_h = atomic_numbers.eq(1)
    carbon_h = torch.tensor(
        [
            atom.GetAtomicNum() == 1
            and atom.GetNeighbors()[0].GetAtomicNum() == 6
            for atom in molecule.GetAtoms()
        ],
        dtype=torch.bool,
    )
    if target_count == int(all_h.sum()):
        return all_h
    if policy == "exact_or_carbon_bound" and target_count == int(carbon_h.sum()):
        return carbon_h
    if (
        policy == "partial_missing"
        and 0 < int(all_h.sum()) - target_count <= max_missing_hydrogens
    ):
        return all_h
    return None


def targets_from_row(
    row: Mapping[str, Any],
    structure: Dict[str, Any],
    structure_smiles: str,
    hydrogen_policy: str,
    carbon_policy: str,
    max_carbon_cluster_span: float,
    max_missing_hydrogens: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    molecule = _explicit_molecule(structure_smiles)
    if molecule is None:
        return None, "invalid_smiles"
    hydrogen_peaks = _as_peak_list(row.get("h_nmr_peaks"))
    peak_tensors = hydrogen_peak_tensors(hydrogen_peaks)
    h_targets = expand_hydrogen_shifts(hydrogen_peaks)
    if h_targets is None or h_targets.numel() == 0:
        return None, "invalid_hydrogen_integration"
    h_mask = _hydrogen_training_mask(
        molecule,
        hydrogen_policy,
        int(h_targets.numel()),
        max_missing_hydrogens=max_missing_hydrogens,
    )
    if h_mask is None:
        return None, "hydrogen_count_mismatch"

    class_count = _carbon_class_count(
        structure["atomic_numbers"], structure["equivalence_classes"]
    )
    if class_count == 0:
        return None, "no_carbon_targets"
    carbon_peaks = _as_peak_list(row.get("c_nmr_peaks"))
    if carbon_policy == "exact":
        values, _ = _carbon_line_values(carbon_peaks)
        c_targets = (
            torch.tensor(values, dtype=torch.float32)
            if len(values) == class_count
            else None
        )
    elif carbon_policy == "collapse":
        c_targets = collapse_carbon_lines(
            carbon_peaks, class_count, max_carbon_cluster_span
        )
    else:
        raise ValueError(f"Unknown carbon policy: {carbon_policy}")
    if c_targets is None:
        return None, "carbon_count_mismatch"
    carbon_classes = structure["equivalence_classes"][
        structure["atomic_numbers"].eq(6)
    ]
    class_sizes = torch.stack(
        [
            carbon_classes.eq(item).sum()
            for item in torch.unique(carbon_classes, sorted=True)
        ]
    )
    assert peak_tensors is not None
    return {
        "h_shifts": h_targets,
        "h_peak_shifts": peak_tensors[0],
        "h_peak_counts": peak_tensors[1],
        "h_prediction_mask": h_mask,
        "c_shifts": c_targets,
        # These sizes belong to the ordered symmetry classes, not directly to
        # the sorted c_shifts. Their shift assignment is established by the
        # per-batch multiset matching loss.
        "c_equivalence_class_sizes": class_sizes,
    }, "ok"


def iter_parquet_rows(
    paths: Sequence[str | Path],
    batch_size: int = 4096,
) -> Iterator[Dict[str, Any]]:
    columns = ["smiles", "h_nmr_peaks", "c_nmr_peaks"]
    for path in paths:
        parquet = pq.ParquetFile(str(path))
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def index_coordinates(
    coords_dir: str | Path,
) -> Tuple[Dict[str, List[Tuple[str, str]]], Counter]:
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    audit: Counter = Counter()
    for split in SPLITS:
        path = osp.join(str(coords_dir), f"{split}_molecules.h5")
        if not osp.exists(path):
            audit[f"missing_coordinate_split/{split}"] += 1
            continue
        with h5py.File(path, "r") as handle:
            for mol_idx in tqdm(handle.keys(), desc=f"Indexing {split} coordinates"):
                smiles = read_str(handle[mol_idx].attrs.get("smiles", ""))
                key = canonical_smiles(smiles, isomeric=True)
                if key is None:
                    audit["invalid_coordinate_smiles"] += 1
                    continue
                index[key].append((split, str(mol_idx)))
                audit[f"coordinates/{split}"] += 1
    return index, audit


def _source_fingerprint(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    fingerprint = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists():
            fingerprint.append({"path": str(resolved), "missing": True})
            continue
        stat = resolved.stat()
        fingerprint.append(
            {
                "path": str(resolved),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return fingerprint


def load_or_build_index_cache(
    coords_dir: str | Path,
    nmr_dir: str | Path,
    cache_path: str | Path,
    rebuild: bool = False,
) -> Tuple[
    Dict[str, List[Tuple[str, str]]],
    Counter,
    Dict[str, str],
    Counter,
]:
    """Cache the expensive coordinate and authoritative NMR split indices."""
    coordinate_paths = [
        Path(coords_dir) / f"{split}_molecules.h5" for split in SPLITS
    ]
    nmr_paths = [Path(nmr_dir) / f"{split}.pt" for split in SPLITS]
    metadata = {
        "version": DATASET_VERSION,
        "coordinate_sources": _source_fingerprint(coordinate_paths),
        "nmr_sources": _source_fingerprint(nmr_paths),
    }
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        cached = _load_torch(cache_path)
        if cached.get("metadata") == metadata:
            print(f"Loading reusable indices from {cache_path}")
            return (
                cached["coordinate_index"],
                Counter(cached["coordinate_audit"]),
                cached["split_for_smiles"],
                Counter(cached["nmr_audit"]),
            )
        print(f"Ignoring stale index cache at {cache_path}")

    coordinate_index, coordinate_audit = index_coordinates(coords_dir)
    split_for_smiles, _, nmr_audit = load_nmr_split_index(nmr_dir)
    payload = {
        "metadata": metadata,
        "coordinate_index": coordinate_index,
        "coordinate_audit": dict(coordinate_audit),
        "split_for_smiles": split_for_smiles,
        "nmr_audit": dict(nmr_audit),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    print(f"Saved reusable indices to {cache_path}")
    return coordinate_index, coordinate_audit, split_for_smiles, nmr_audit


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_size: int) -> None:
        self.directory = root / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.shard_size = shard_size
        self.buffer: List[Dict[str, Any]] = []
        self.shards: List[Dict[str, Any]] = []

    def add(self, sample: Dict[str, Any]) -> None:
        self.buffer.append(sample)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        name = f"shard_{len(self.shards):05d}.pt"
        path = self.directory / name
        temporary = path.with_suffix(".pt.tmp")
        torch.save(self.buffer, temporary)
        os.replace(temporary, path)
        self.shards.append({"path": name, "count": len(self.buffer)})
        self.buffer = []

    def finish(self) -> Dict[str, Any]:
        self.flush()
        manifest = {
            "version": DATASET_VERSION,
            "split": self.split,
            "count": sum(item["count"] for item in self.shards),
            "shards": self.shards,
        }
        path = self.directory / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def build_3d2shift_dataset(
    parquet_paths: Sequence[str | Path],
    nmr_dir: str | Path,
    coords_dir: str | Path,
    output_dir: str | Path,
    hydrogen_policy: str = "exact",
    carbon_policy: str = "exact",
    max_carbon_cluster_span: float = 15.0,
    max_missing_hydrogens: int = 2,
    shard_size: int = 4096,
    audit_only: bool = False,
    index_cache: str | Path | None = None,
    rebuild_index_cache: bool = False,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = (
        output / "index_cache.pt"
        if index_cache is None
        else Path(index_cache)
    )
    coordinate_index, audit, split_for_smiles, nmr_audit = (
        load_or_build_index_cache(
            coords_dir=coords_dir,
            nmr_dir=nmr_dir,
            cache_path=cache_path,
            rebuild=rebuild_index_cache,
        )
    )
    audit.update(nmr_audit)
    writers = (
        {}
        if audit_only
        else {
            split: ShardWriter(output, split, shard_size) for split in SPLITS
        }
    )
    handles = {
        split: h5py.File(
            osp.join(str(coords_dir), f"{split}_molecules.h5"), "r"
        )
        for split in SPLITS
        if osp.exists(osp.join(str(coords_dir), f"{split}_molecules.h5"))
    }
    seen_coordinate_records: set[Tuple[str, str, str]] = set()
    try:
        for row in tqdm(iter_parquet_rows(parquet_paths), desc="Matching spectra"):
            smiles = str(row.get("smiles", ""))
            key = canonical_smiles(smiles, isomeric=True)
            if key is None:
                audit["invalid_parquet_smiles"] += 1
                continue
            output_split = split_for_smiles.get(key)
            if output_split is None:
                audit["spectrum_not_in_nmr_splits"] += 1
                continue
            references = coordinate_index.get(key, [])
            if not references:
                audit["spectrum_without_coordinates"] += 1
                continue
            for coordinate_split, mol_idx in references:
                record_key = (coordinate_split, mol_idx, key)
                if record_key in seen_coordinate_records:
                    audit["duplicate_spectrum_for_coordinate"] += 1
                    continue
                group = handles[coordinate_split][mol_idx]
                coordinate_smiles = read_str(group.attrs["smiles"])
                structure, reason = molecule_from_h5(group, coordinate_smiles)
                if structure is None:
                    audit[f"rejected/{reason}"] += 1
                    continue
                h_preview = expand_hydrogen_shifts(
                    _as_peak_list(row.get("h_nmr_peaks"))
                )
                if h_preview is not None:
                    total_h = int(structure["atomic_numbers"].eq(1).sum())
                    audit[
                        f"hydrogen_total_minus_integrated/"
                        f"{total_h - int(h_preview.numel())}"
                    ] += 1
                carbon_classes = _carbon_class_count(
                    structure["atomic_numbers"],
                    structure["equivalence_classes"],
                )
                carbon_lines, _ = _carbon_line_values(
                    _as_peak_list(row.get("c_nmr_peaks"))
                )
                audit[
                    f"carbon_lines_minus_classes/"
                    f"{len(carbon_lines) - carbon_classes}"
                ] += 1
                targets, reason = targets_from_row(
                    row,
                    structure,
                    structure_smiles=coordinate_smiles,
                    hydrogen_policy=hydrogen_policy,
                    carbon_policy=carbon_policy,
                    max_carbon_cluster_span=max_carbon_cluster_span,
                    max_missing_hydrogens=max_missing_hydrogens,
                )
                if targets is None:
                    audit[f"rejected/{reason}"] += 1
                    continue
                seen_coordinate_records.add(record_key)
                audit[f"accepted/{output_split}"] += 1
                if audit_only:
                    continue
                sample = {
                    "id": f"{output_split}:{coordinate_split}:{mol_idx}",
                    "smiles": coordinate_smiles,
                    "coordinate_source_split": coordinate_split,
                    **structure,
                    **targets,
                }
                writers[output_split].add(sample)
    finally:
        for handle in handles.values():
            handle.close()

    manifests = {
        split: writer.finish() for split, writer in writers.items()
    }
    report = {
        "version": DATASET_VERSION,
        "parquet_paths": [str(Path(path)) for path in parquet_paths],
        "nmr_dir": str(Path(nmr_dir)),
        "coords_dir": str(Path(coords_dir)),
        "hydrogen_policy": hydrogen_policy,
        "carbon_policy": carbon_policy,
        "max_carbon_cluster_span": max_carbon_cluster_span,
        "max_missing_hydrogens": max_missing_hydrogens,
        "index_cache": str(cache_path),
        "audit_only": audit_only,
        "counts": dict(sorted(audit.items())),
        "splits": manifests,
    }
    (output / "audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split-h5")
    split.add_argument("--h5-path", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--split-name", default="split_indices_dedup")

    legacy = subparsers.add_parser(
        "legacy-pt",
        help="Build the original *_with_nmr.pt files using NMR .pt splits.",
    )
    legacy.add_argument("--nmr-dir", required=True)
    legacy.add_argument("--coords-dir", required=True)
    legacy.add_argument("--output-dir", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--parquet", nargs="+", required=True)
    build.add_argument(
        "--nmr-dir",
        required=True,
        help="Directory containing authoritative train.pt/val.pt/test.pt.",
    )
    build.add_argument("--coords-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument(
        "--hydrogen-policy",
        choices=("exact", "exact_or_carbon_bound", "partial_missing"),
        default="exact",
    )
    build.add_argument(
        "--carbon-policy", choices=("exact", "collapse"), default="exact"
    )
    build.add_argument("--max-carbon-cluster-span", type=float, default=15.0)
    build.add_argument("--max-missing-hydrogens", type=int, default=2)
    build.add_argument("--shard-size", type=int, default=4096)
    build.add_argument("--audit-only", action="store_true")
    build.add_argument(
        "--index-cache",
        default=None,
        help="Reusable index cache; defaults to OUTPUT_DIR/index_cache.pt.",
    )
    build.add_argument("--rebuild-index-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "split-h5":
        read_uspto_h5(
            args.h5_path, args.output_dir, split_name=args.split_name
        )
        return
    if args.command == "legacy-pt":
        preprocess_uspto_only_nmr_3d_coords(
            nmr_dir=args.nmr_dir,
            target_dir=args.output_dir,
            coords_dir=args.coords_dir,
        )
        return
    report = build_3d2shift_dataset(
        parquet_paths=args.parquet,
        nmr_dir=args.nmr_dir,
        coords_dir=args.coords_dir,
        output_dir=args.output_dir,
        hydrogen_policy=args.hydrogen_policy,
        carbon_policy=args.carbon_policy,
        max_carbon_cluster_span=args.max_carbon_cluster_span,
        max_missing_hydrogens=args.max_missing_hydrogens,
        shard_size=args.shard_size,
        audit_only=args.audit_only,
        index_cache=args.index_cache,
        rebuild_index_cache=args.rebuild_index_cache,
    )
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
