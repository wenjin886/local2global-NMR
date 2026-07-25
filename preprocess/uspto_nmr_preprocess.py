import os
import os.path as osp
import argparse
import gc

import urllib.request
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data

from rdkit import Chem
import tarfile
import pandas as pd

import time
from typing import List

import json
# import logging
# import lmdb
import pickle
from functools import lru_cache
# log = utils.get_pylogger(__name__)

# import h5py
from src.metrics.uspto import USPTOPreprocessMetrics
from src.metrics.base import read_json, save_json
from src.data.constants import (
    MAX_J_VALUES,
    MULTIPLICITY_MISSING_INDEX,
    MULTIPLICITY_UNKNOWN_INDEX,
    SMILES_UNKNOWN_INDEX,
    normalize_multiplicity_label,
)
from src.data.smiles import tokenize_smiles_tokens
from src.data.storage import COMPACT_STORAGE_VERSION, compact_sample_storage
SEED = 0
SPLIT_NAMES = ("train", "val", "test")
CATEGORICAL_MAPPING_VERSION = 1



def process_atoms(
    atom_idx: np.ndarray, 
    atom_pos: List[np.ndarray], 
    db_id: str, smiles: str, original_smiles: str,
    nmr_data: List[Data],
    split: str,
    ): 
    data_list = []
    for data in nmr_data:
        data.num_atoms = len(atom_idx)
        data.h = torch.from_numpy(atom_idx)
        
        data.original_smiles = original_smiles
        
        
        if len(atom_pos) > 1 and split == 'train':
            for i, pos in enumerate(atom_pos):
                data.id = f"{db_id}_{i}"
                data_ = data.clone()
                data_.pos = torch.from_numpy(pos)
                # print(data_)
                data_list.append(data_)
            # raise ValueError("Stop here | len(atom_pos) > 1 and split == 'train'")
        else:
            data.id = f"{db_id}_0"
            data.pos = torch.from_numpy(atom_pos[0])
            data_list.append(data)
    
    return data_list


def _parse_j_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        values = value.strip("_").split("_") if value.strip("_") else []
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = value
    else:
        values = [value]
    parsed = []
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            parsed.append(number)
    return parsed[:MAX_J_VALUES]


def _optional_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    return (value, True) if np.isfinite(value) else (0.0, False)


def preprocess_parquet(df: pd.DataFrame):
    """
    columns:
      ['smiles', 'hsqc_nmr_peaks', 'hsqc_nmr_spectrum', 'h_nmr_peaks',
       'h_nmr_spectra', 'molecular_formula', 'c_nmr_peaks', 'ir_spectra',
       'msms_cfmid_positive_10ev', 'msms_cfmid_positive_20ev',
       'msms_cfmid_positive_40ev', 'msms_cfmid_fragments_positive',
       'msms_cfmid_negative_10ev', 'msms_cfmid_negative_20ev',
       'msms_cfmid_negative_40ev', 'msms_cfmid_fragments_negative',
       'c_nmr_spectra', 'msms_iceberg_positive',
       'msms_iceberg_fragments_positive', 'msms_scarf_positive',
       'msms_scarf_fragments_positive']
    """
    data_list =[]
    from src.data.dataset import (
        canonicalize_smiles_with_stereo,
        canonicalize_smiles_without_stereo,
        graph_targets_from_smiles,
    )

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing parquet files", unit="file"):
        h_peaks = []
        for peak in row['h_nmr_peaks']:
            shift = float(peak['delta'])
            integration, integration_available = _optional_float(peak.get('nH'))
            multiplicity = normalize_multiplicity_label(peak.get('category'))

            j_values = _parse_j_values(peak.get('j_values'))
            h_peaks.append((
                shift,
                integration,
                integration_available,
                multiplicity,
                j_values,
            ))
        h_peaks.sort(key=lambda values: values[0])

        c_nmr_peaks = []
        for peak in row['c_nmr_peaks']:
            c_nmr_peaks.append(peak['delta (ppm)'])

        num_h_peaks = len(h_peaks)
        h_nmr_j = torch.zeros((num_h_peaks, MAX_J_VALUES), dtype=torch.float)
        h_nmr_j_mask = torch.zeros((num_h_peaks, MAX_J_VALUES), dtype=torch.bool)
        for peak_index, values in enumerate(h_peaks):
            j_values = values[-1]
            if j_values:
                h_nmr_j[peak_index, :len(j_values)] = torch.tensor(j_values)
                h_nmr_j_mask[peak_index, :len(j_values)] = True
        
        smiles = row['smiles']
        canonical_smiles = canonicalize_smiles_without_stereo(smiles)
        isomeric_smiles = canonicalize_smiles_with_stereo(smiles)
        graph_targets = graph_targets_from_smiles(smiles)
        data = Data(
            smiles=smiles,
            canonical_smiles=canonical_smiles,
            isomeric_smiles=isomeric_smiles,
            h_nmr=torch.tensor([values[0] for values in h_peaks], dtype=torch.float),
            c_nmr=torch.tensor(sorted(c_nmr_peaks), dtype=torch.float),
            h_nmr_integration=torch.tensor(
                [values[1] for values in h_peaks], dtype=torch.float
            ),
            h_nmr_integration_mask=torch.tensor(
                [values[2] for values in h_peaks], dtype=torch.bool
            ),
            h_nmr_multiplicity=[values[3] for values in h_peaks],
            h_nmr_j=h_nmr_j,
            h_nmr_j_mask=h_nmr_j_mask,
            h = graph_targets["h"],
            bond_types=graph_targets["bond_types"],
            h_attachment=graph_targets["h_attachment"],
            h_parent_types=graph_targets["h_parent_types"],
            h_parent_fragment_labels=graph_targets["h_parent_fragment_labels"],
            heavy_fragment_labels=graph_targets["heavy_fragment_labels"],
        )
        data_list.append(data)
    return data_list

def split_by_canonical_smiles(
        data_list,
        ratios=(0.85, 0.05, 0.10),
        seed=SEED,
        deduplicate=False,
):
    """Split molecule groups without non-stereochemical SMILES leakage."""
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("ratios must contain three non-negative values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1")

    from src.data.dataset import canonicalize_smiles_without_stereo

    groups = {}
    for data in data_list:
        key = getattr(data, "canonical_smiles", None)
        if key is None:
            key = canonicalize_smiles_without_stereo(data.smiles)
            data.canonical_smiles = key
        groups.setdefault(key, []).append(data)

    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    num_groups = len(keys)
    num_train = int(ratios[0] * num_groups)
    num_val = int(ratios[1] * num_groups)
    key_splits = {
        "train": keys[:num_train],
        "val": keys[num_train:num_train + num_val],
        "test": keys[num_train + num_val:],
    }

    splits = {}
    for split, split_keys in key_splits.items():
        if deduplicate:
            splits[split] = [groups[key][0] for key in split_keys]
        else:
            splits[split] = [
                item for key in split_keys for item in groups[key]
            ]

    key_sets = {name: set(values) for name, values in key_splits.items()}
    assert key_sets["train"].isdisjoint(key_sets["val"])
    assert key_sets["train"].isdisjoint(key_sets["test"])
    assert key_sets["val"].isdisjoint(key_sets["test"])
    return splits, key_splits, groups


def _split_paths(save_dir):
    return {
        split: {
            "data": osp.join(save_dir, f"{split}.pt"),
            "info": osp.join(save_dir, f"dataset_infos_{split}.json"),
        }
        for split in SPLIT_NAMES
    }


def _load_torch(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path)


def _atomic_torch_save(data, path):
    temporary_path = f"{path}.mapping.tmp"
    try:
        torch.save(data, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if osp.exists(temporary_path):
            os.remove(temporary_path)


def _save_split_metrics(splits, paths):
    """Scan raw labels before any categorical mapping is applied."""
    for split in SPLIT_NAMES:
        metrics = USPTOPreprocessMetrics()
        metrics.update(splits[split])
        save_json(metrics.summarize(), paths[split]["info"])
        print(f"Saved normalization statistics to {paths[split]['info']}")


def _encode_categorical_inputs(data_list, train_infos, split):
    multiplicity_mapping = {
        label: index
        for index, label in enumerate(train_infos["multiplicity_labels"])
    }
    smiles_mapping = {
        token: index for index, token in enumerate(train_infos["smiles_vocab"])
    }
    num_unknown_multiplicity = 0
    num_unknown_smiles = 0
    for data in tqdm(
            data_list,
            desc=f"Encoding {split} multiplicity/SMILES",
            unit="molecule",
    ):
        multiplicity = getattr(data, "h_nmr_multiplicity", None)
        if multiplicity is None:
            labels = ["<missing>"] * int(data.h_nmr.numel())
        elif torch.is_tensor(multiplicity):
            # Already encoded files are idempotent. Raw labels cannot be
            # reconstructed from IDs, so only validate them against train vocab.
            if multiplicity.numel() and (
                multiplicity.min().item() < 0
                or multiplicity.max().item() >= len(multiplicity_mapping)
            ):
                raise ValueError(
                    f"{split} contains multiplicity IDs outside the train vocabulary"
                )
            data.h_nmr_multiplicity = multiplicity.to(torch.int16)
            if not hasattr(data, "h_nmr_multiplicity_mask"):
                data.h_nmr_multiplicity_mask = multiplicity.ne(
                    MULTIPLICITY_MISSING_INDEX
                )
            labels = None
        else:
            labels = [normalize_multiplicity_label(value) for value in multiplicity]

        if labels is not None:
            encoded = []
            for label in labels:
                index = multiplicity_mapping.get(
                    label, MULTIPLICITY_UNKNOWN_INDEX
                )
                num_unknown_multiplicity += int(index == MULTIPLICITY_UNKNOWN_INDEX)
                encoded.append(index)
            data.h_nmr_multiplicity = torch.tensor(encoded, dtype=torch.int16)
            data.h_nmr_multiplicity_mask = torch.tensor([
                label != "<missing>" for label in labels
            ], dtype=torch.bool)

        smiles_ids = getattr(data, "smiles_token_ids", None)
        if smiles_ids is not None:
            smiles_ids = torch.as_tensor(smiles_ids)
            if smiles_ids.numel() and (
                smiles_ids.min().item() < 0
                or smiles_ids.max().item() >= len(smiles_mapping)
            ):
                raise ValueError(
                    f"{split} contains SMILES IDs outside the train vocabulary"
                )
            data.smiles_token_ids = smiles_ids.to(torch.int16)
        else:
            smiles = getattr(data, "isomeric_smiles", None)
            if smiles is None:
                smiles = getattr(data, "smiles", "")
            encoded = []
            for token in tokenize_smiles_tokens(smiles):
                index = smiles_mapping.get(token, SMILES_UNKNOWN_INDEX)
                num_unknown_smiles += int(index == SMILES_UNKNOWN_INDEX)
                encoded.append(index)
            data.smiles_token_ids = torch.tensor(encoded, dtype=torch.int16)

    print(
        f"{split}: mapped {len(data_list)} molecules; "
        f"unknown multiplicities={num_unknown_multiplicity}, "
        f"unknown SMILES tokens={num_unknown_smiles}"
    )
    return data_list


def _map_and_save_split(data_list, path, info_path, train_infos, split):
    data_list = _encode_categorical_inputs(data_list, train_infos, split)
    for data in tqdm(
            data_list,
            desc=f"Compacting {split} storage",
            unit="molecule",
    ):
        compact_sample_storage(data)
    _atomic_torch_save(data_list, path)
    split_infos = read_json(info_path)
    split_infos["categorical_mapping_version"] = CATEGORICAL_MAPPING_VERSION
    split_infos["categorical_vocab_source"] = "dataset_infos_train.json"
    split_infos["compact_storage_version"] = COMPACT_STORAGE_VERSION
    save_json(split_infos, info_path)
    print(f"Saved {len(data_list)} mapped, compact samples to {path}")


def preprocess_uspto(
        parquet_dir: str,
        save_dir: str = None,
        split_ratios=(0.85, 0.05, 0.10),
        seed=SEED,
        deduplicate=True,
):
    if save_dir is None:
        save_dir = osp.dirname(parquet_dir)
    os.makedirs(save_dir, exist_ok=True)
    paths = _split_paths(save_dir)
    data_files_exist = all(
        osp.isfile(paths[split]["data"]) for split in SPLIT_NAMES
    )
    info_files_exist = all(
        osp.isfile(paths[split]["info"]) for split in SPLIT_NAMES
    )
    already_mapped = info_files_exist and all(
        read_json(paths[split]["info"]).get("categorical_mapping_version")
        == CATEGORICAL_MAPPING_VERSION
        for split in SPLIT_NAMES
    )
    already_compact = info_files_exist and all(
        read_json(paths[split]["info"]).get("compact_storage_version")
        == COMPACT_STORAGE_VERSION
        for split in SPLIT_NAMES
    )

    if data_files_exist and already_mapped and already_compact:
        print(
            "Found complete, already-mapped compact split artifacts; "
            "nothing to do"
        )
        return

    if data_files_exist:
        status = "complete" if info_files_exist else "missing some metrics files"
        print(
            f"Found existing train/val/test.pt ({status}); "
            "skipping parquet preprocessing"
        )
        for split in SPLIT_NAMES:
            info_path = paths[split]["info"]
            if osp.isfile(info_path):
                continue
            print(f"Start loading {paths[split]['data']} for dataset info...")
            start_time = time.time()
            data_list = _load_torch(paths[split]["data"])
            print(f"Done loading. Time taken: {time.time() - start_time:.2f} seconds.")
            first_multiplicity = (
                getattr(data_list[0], "h_nmr_multiplicity", None)
                if data_list else None
            )
            if split == "train" and torch.is_tensor(first_multiplicity):
                raise RuntimeError(
                    "dataset_infos_train.json is missing, but train.pt is already "
                    "mapped; the original categorical labels cannot be recovered"
                )
            metrics = USPTOPreprocessMetrics()
            metrics.update(data_list)
            save_json(metrics.summarize(), info_path)
            print(f"Saved normalization statistics to {info_path}")
            del data_list, metrics
            gc.collect()

        train_infos = read_json(paths["train"]["info"])
        for split in SPLIT_NAMES:
            print(f"Start loading {paths[split]['data']} for mapping/compaction...")
            start_time = time.time()
            data_list = _load_torch(paths[split]["data"])
            print(f"Done loading. Time taken: {time.time() - start_time:.2f} seconds.")
            _map_and_save_split(
                data_list,
                paths[split]["data"],
                paths[split]["info"],
                train_infos,
                split,
            )
            del data_list
            gc.collect()
        return

    file_list = sorted(
        (file for file in os.listdir(parquet_dir) if file.endswith(".parquet")),
        key=lambda value: int(value.split('.')[0].split('_')[-1]),
    )
    total_data_list = []
    for file in tqdm(file_list, total=len(file_list), desc="Reading parquet files", unit="file"):
        df = pd.read_parquet(osp.join(parquet_dir, file))
        data_list = preprocess_parquet(df)
        total_data_list.extend(data_list)

    splits, key_splits, groups = split_by_canonical_smiles(
        total_data_list,
        ratios=split_ratios,
        seed=seed,
        deduplicate=deduplicate,
    )
    num_input_records = len(total_data_list)
    split_counts = {name: len(values) for name, values in splits.items()}
    # Metrics must see raw categorical labels. All three splits are encoded
    # afterwards with the train vocabulary to keep token IDs consistent.
    _save_split_metrics(splits, paths)
    train_infos = read_json(paths["train"]["info"])
    del total_data_list
    gc.collect()
    for split in SPLIT_NAMES:
        data_list = splits.pop(split)
        _map_and_save_split(
            data_list,
            paths[split]["data"],
            paths[split]["info"],
            train_infos,
            split,
        )
        del data_list
        gc.collect()

    manifest = {
        "seed": int(seed),
        "requested_ratios": list(split_ratios),
        "deduplicated": bool(deduplicate),
        "num_input_records": num_input_records,
        "num_unique_non_stereo_molecules": len(groups),
        "num_removed_duplicate_records": (
            num_input_records - len(groups) if deduplicate else 0
        ),
        "num_records": split_counts,
        "num_molecules": {
            name: len(values) for name, values in key_splits.items()
        },
        "canonical_smiles": key_splits,
    }
    save_json(manifest, osp.join(save_dir, "split_manifest.json"))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", default="./data/uspto/exp_data/")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--keep_duplicate_records",
        action="store_true",
        help="Keep repeated spectra, while grouping each molecule into one split.",
    )
    args = parser.parse_args()
    preprocess_uspto(
        parquet_dir=args.parquet_dir,
        save_dir=args.save_dir,
        seed=args.seed,
        deduplicate=not args.keep_duplicate_records,
    )
