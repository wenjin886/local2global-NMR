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
import h5py
from src.data.dataset import canonicalize_smiles_without_stereo
URL_USPTO = "https://zenodo.org/records/17766755/files/uspto.tar.gz?download=1"
def _save_h5_item_to_npz_dict(
    item: h5py.Dataset | h5py.Group,
    save_key: str,
    npz_data: dict,
    ):
    for attr_key, attr_value in item.attrs.items():
        npz_data[f"{save_key}_attr_{attr_key}"] = np.asarray(attr_value)

    if isinstance(item, h5py.Dataset):
        npz_data[save_key] = item[()]
        return

    for subkey in item.keys():
        _save_h5_item_to_npz_dict(item[subkey], f"{save_key}_{subkey}", npz_data)



    return True


def read_uspto_h5(
    file_path: str,
    save_dir: str | Path | None = None,
    split_name: str = "split_indices_dedup",
    ):
    """
    Split a large USPTO molecules.h5 by split_indices_dedup.

    Molecular data are saved as one HDF5 file per split. Each molecule group
    only contains mol_idx, smiles and atom_features; spectra are not copied.
    The selected split group and valid_indices* are flattened and saved to
    indices.npz.
    """
    save_dir = osp.dirname(file_path) if save_dir is None else str(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    file_list = os.listdir(save_dir)
    split_list = ["train", "val", "test"]
    for file in file_list:
        if file.endswith(".h5"):
            if "train" in file: 
                split_list.remove("train")
            elif "val" in file:
                split_list.remove("val")
            elif "test" in file:
                split_list.remove("test")
    if len(split_list) == 0:
        print("All splits are already processed")
        return 

    indices = {}
    split_counts = {}
    with h5py.File(file_path, 'r', swmr=True) as f:
        if split_name not in f:
            raise KeyError(f"Cannot find split group '{split_name}' in {file_path}")

        for key in [split_name, "valid_indices_h", "valid_indices_c"]:
            if key in f:
                _save_h5_item_to_npz_dict(f[key], key, indices)

        split_group = f[split_name]
        for split in split_list:
            if split not in split_group:
                raise KeyError(f"Cannot find '{split}' in split group '{split_name}'")

            split_indices = split_group[split][()]
            out_path = osp.join(save_dir, f"{split}_molecules.h5")
            copied = 0
            with h5py.File(out_path, "w") as out_f:
                out_f.attrs["source_file"] = file_path
                out_f.attrs["split_name"] = split_name
                out_f.attrs["split"] = split
                for mol_idx in tqdm(split_indices, desc=f"Saving {split} molecules", unit="mol"):
                    copied += int(_copy_uspto_molecule_without_spectra(f, out_f, mol_idx))

            split_counts[split] = copied

    indices["exported_train_count"] = np.asarray(split_counts.get("train", 0), dtype=np.int64)
    indices["exported_val_count"] = np.asarray(split_counts.get("val", 0), dtype=np.int64)
    indices["exported_test_count"] = np.asarray(split_counts.get("test", 0), dtype=np.int64)
    np.savez(osp.join(save_dir, "indices.npz"), **indices)


def read_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value

def preprocess_uspto_with_3d_coords(target_dir: str):

    """
    item from lmdb:
        atoms: list, atom symbols
        coordinates: np.ndarray, (n_atoms, 3)
        atoms_target: np.ndarray, (n_atoms, 1), nmr shifts
        atoms_target_mask: np.ndarray, (n_atoms, 1), nmr shifts mask
        smiles: str
        db_id: str
        mol: Chem.Mol
        inchikey: str
    """

    # Download
    download_dir = os.path.join(target_dir, "download")
    os.makedirs(download_dir, exist_ok=True)
    fname = osp.join(download_dir, "uspto.tar.gz")
    if not osp.exists(fname):
        print(f"The downloaded files will be placed in '{download_dir}'.")
        print(f"Downloading '{URL_USPTO}'...")
        urllib.request.urlretrieve(URL_USPTO, fname)
        print(f"Done downloading '{URL_USPTO}'...")

    data_dir = osp.join(download_dir, "data")
    if not osp.exists(data_dir):
        # with zipfile.ZipFile(fname, 'r') as zip_ref:
            # zip_ref.extractall(download_dir)
        with tarfile.open(fname, 'r') as tar_ref:
            tar_ref.extractall(download_dir)
        print(f"Done extracting '{fname}'...")
    
    print(f"Preprocessing USPTO...")
    h5_path = osp.join(data_dir, "uspto/molecules.h5")
    preprocessed_dir = osp.join(download_dir, "preprocessed")
    
    read_uspto_h5(h5_path, save_dir=preprocessed_dir, split_name="split_indices_dedup")
    print(f"Done preprocessing USPTO. Saved files in '{preprocessed_dir}'.")


def _format_uspto_mol_idx(mol_idx) -> str:
    if isinstance(mol_idx, bytes):
        mol_idx = mol_idx.decode("utf-8")
    if isinstance(mol_idx, str):
        return mol_idx if mol_idx.isdigit() else f"{int(mol_idx):07d}"
    return f"{int(mol_idx):07d}"

def _copy_uspto_molecule_without_spectra(
    src_file: h5py.File,
    dst_file: h5py.File,
    mol_idx,
    ):
    mol_idx = _format_uspto_mol_idx(mol_idx)
    if mol_idx not in src_file:
        print(f"Missing molecule {mol_idx}; skipped.")
        return False

    src_group = src_file[mol_idx]
    dst_group = dst_file.create_group(mol_idx)
    dst_group.attrs["mol_idx"] = mol_idx
    if "smiles" in src_group.attrs:
        dst_group.attrs["smiles"] = src_group.attrs["smiles"]

    if "atom_features" in src_group:
        src_group.copy("atom_features", dst_group)
    else:
        print(f"Molecule {mol_idx} has no atom_features; saved attrs only.")

def convert_to_data(mol_from_h5, smiles):
    data = Data(smiles=smiles)
    atom_features = mol_from_h5["atom_features"]

    atom_coords = atom_features["atom_coords"][()]
    atom_charges = atom_features["atom_charges"][()]
    atom_mask = atom_features["atom_mask"][()]

    num_atoms = int(atom_mask.sum())
    data.num_nodes = num_atoms
    data.h = torch.from_numpy(atom_charges[:num_atoms])
    data.pos = torch.from_numpy(atom_coords[:num_atoms])

    return data

def compact_nmr_data(data):
    for key in data.keys():
        if key not in ["h_nmr", "c_nmr", "isomeric_smiles"]:
            del data[key]
    return data

def add_nmr_to_coord_data(coord_data, nmr_data_split, nmr_smiles_idx):
    smiles = coord_data.smiles
    idx = nmr_smiles_idx[smiles]
    nmr_data = nmr_data_split[idx]
    coord_data.h_nmr = nmr_data.h_nmr
    coord_data.c_nmr = nmr_data.c_nmr
    return coord_data

def preprocess_uspto_only_nmr_3d_coords(
    nmr_dir: str, target_dir: str,
    coords_dir: str, 
    ):
    """
    Preprocess USPTO dataset with 3D coordinates.

    Args:
        coords_dir (str): Directory containing the 3D coordinates. (src file is molecules.h5 and processed through read_uspto_h5)
        target_dir (str): Directory to save the preprocessed data.
    """
    nmr_smiles_idx = {
        "val": {},
        "test": {},
        "train": {},
    }
    # compact nmr data
    for split in nmr_smiles_idx.keys():
        print(f"Step 1: Processing NMR data for {split}...")
        nmr_data_list = torch.load(osp.join(nmr_dir, f"{split}.pt"))
        nmr_data_list = [compact_nmr_data(datai) for datai in nmr_data_list]
        nmr_smiles_idx[split] = {datai.isomeric_smiles: i for i,datai in enumerate(nmr_data_list)}
        torch.save(nmr_data_list, osp.join(target_dir, f"{split}_only_nmr.pt"))
        print(f"Step 1: Done processing NMR data for {split}...")
        del nmr_data_list
    
    coord_data = {
        "train": [],
        "test": [],
        "val": [],
    }
    for coord_split in coord_data.keys():
        f_name = osp.join(coords_dir, f"{coord_split}_molecules.h5")
        with h5py.File(f_name, 'r') as f:
            for mol_idx in tqdm(f.keys(), desc=f"Processing {coord_split} file", total=len(f.keys())):
                mol = f[mol_idx]
                smiles = read_str(mol.attrs["smiles"])

                for nmr_split in nmr_smiles_idx.keys():
                    if smiles in nmr_smiles_idx[nmr_split]:
                        datai = convert_to_data(mol, smiles)
                        coord_data[coord_split].append(datai)
                        break
    
    for coord_split in coord_data.keys():
        nmr_data_split = torch.load(osp.join(target_dir, f"{coord_split}_only_nmr.pt"))
        coord_data[coord_split] = [
            add_nmr_to_coord_data(
                datai, nmr_data_split, nmr_smiles_idx[coord_split]
                ) for datai in coord_data[coord_split]]
        torch.save(coord_data[coord_split], osp.join(target_dir, f"{coord_split}_with_nmr.pt"))
        del coord_data[coord_split]
        del nmr_data_split
            
    
        
    
    # for split in split_list:
    #     torch.save(new_split_data[split], )
    #     print()
    


    pass

if __name__ == "__main__":
    # coords_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/END_NMR/data/uspto/download/preprocessed"
    # tgt_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/preprocessed/3d_to_nmr"
    # split_file = osp.join(osp.dirname(tgt_dir), "split_manifest.json")
    nmr_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/local2global/data/uspto/preprocessed"
    preprocess_uspto_only_nmr_3d_coords(nmr_dir)
