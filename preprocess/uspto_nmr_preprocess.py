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
SEED = 0
SPLIT_NAMES = ("train", "val", "test")
CATEGORICAL_MAPPING_VERSION = 1




# URL_USPTO = "https://zenodo.org/records/17766755/files/uspto.tar.gz?download=1"

# target_dir = "../data/uspto/"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     handlers=[
#         logging.StreamHandler(),  # 输出到 terminal
#         logging.FileHandler(osp.join(target_dir, "preprocess.log"), mode="a"),  # 保存到文件
#     ],
# )

# log = logging.getLogger(__name__)

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

# def get_xyz_txt(atoms, pos, des):
#     xyz_i = f"{len(atoms)}\n"
#     xyz_i += f"{des}\n"

#     # deal with atoms within the molecule
#     for atom,p in zip(atoms, pos):
#         xyz_i += f"{atom} {p[0]} {p[1]} {p[2]} \n"
#     return xyz_i

# def get_clean_smiles(mol: Chem.Mol | str):
#     if type(mol) == str:
#         try:
#             mol = Chem.MolFromSmiles(mol)
#         except Exception as e:
#             print(f"Error parsing SMILES: {e}")
#             return None

#     mol = Chem.RemoveAllHs(mol)
#     Chem.RemoveStereochemistry(mol)
#     return Chem.MolToSmiles(mol)

# def check_I_valence(mol: Chem.Mol | str):
#     has_I, is_I_3_bonds, is_I_5_bonds = False,False, False
#     if type(mol) == str:
#         mol = Chem.MolFromSmiles(mol)
#     atoms = mol.GetAtoms()
#     for atom in atoms:
#         h = atom.GetAtomicNum()

#         if h == 53:
#             has_I = True

#             explicit_valence = atom.GetExplicitValence()
#             implicit_valence = atom.GetImplicitValence()
#             total_valence = explicit_valence + implicit_valence

#             if total_valence == 3:
#                 is_I_3_bonds = True
#                 break
#             elif total_valence == 5:
#                 is_I_5_bonds = True
#                 break
#     return has_I, is_I_3_bonds, is_I_5_bonds
    

# non_neural_substractures = ['[n+]', '']
# query_list = [Chem.MolFromSmarts(s)  for s in non_neural_substractures]
# def is_neural_molecule(mol: Chem.Mol | str):
#     if type(mol) == str:
#         if '[n+]' in mol:
#             return False
#         elif '[I+]' in mol:
#             return False
#         elif '[NH3+]' in mol:
#             return False
#         elif '[NH2+]' in mol:
#             return False
#         elif '[NH+]' in mol:
#             return False
#         elif '[S+]' in mol:
#             return False
#         elif '[s+]' in mol:
#             return False
#         elif '[nH+]' in mol:
#             return False
#         elif '[P+]' in mol:
#             return False
#         elif '[Cl+3]' in mol:
#             return False
#         mol = Chem.MolFromSmiles(mol)

#     query = Chem.MolFromSmarts("[N+;D4](-[*])(-[*])(-[*])-[*]")
#     if mol.HasSubstructMatch(query):
#         return False

#     total_charge = Chem.rdmolops.GetFormalCharge(mol)
#     if total_charge != 0:
#         return False
#     return True

# def _format_uspto_mol_idx(mol_idx) -> str:
#     if isinstance(mol_idx, bytes):
#         mol_idx = mol_idx.decode("utf-8")
#     if isinstance(mol_idx, str):
#         return mol_idx if mol_idx.isdigit() else f"{int(mol_idx):07d}"
#     return f"{int(mol_idx):07d}"


# def _save_h5_item_to_npz_dict(
#     item: h5py.Dataset | h5py.Group,
#     save_key: str,
#     npz_data: dict,
#     ):
#     for attr_key, attr_value in item.attrs.items():
#         npz_data[f"{save_key}_attr_{attr_key}"] = np.asarray(attr_value)

#     if isinstance(item, h5py.Dataset):
#         npz_data[save_key] = item[()]
#         return

#     for subkey in item.keys():
#         _save_h5_item_to_npz_dict(item[subkey], f"{save_key}_{subkey}", npz_data)


# def _copy_uspto_molecule_without_spectra(
#     src_file: h5py.File,
#     dst_file: h5py.File,
#     mol_idx,
#     ) -> bool:
#     mol_idx = _format_uspto_mol_idx(mol_idx)
#     if mol_idx not in src_file:
#         print(f"Missing molecule {mol_idx}; skipped.")
#         return False

#     src_group = src_file[mol_idx]
#     dst_group = dst_file.create_group(mol_idx)
#     dst_group.attrs["mol_idx"] = mol_idx
#     if "smiles" in src_group.attrs:
#         dst_group.attrs["smiles"] = src_group.attrs["smiles"]

#     if "atom_features" in src_group:
#         src_group.copy("atom_features", dst_group)
#     else:
#         print(f"Molecule {mol_idx} has no atom_features; saved attrs only.")

    # return True


# def read_uspto_h5(
#     file_path: str,
#     save_dir: str | Path | None = None,
#     split_name: str = "split_indices_dedup",
#     ):
#     """
#     Split a large USPTO molecules.h5 by split_indices_dedup.

#     Molecular data are saved as one HDF5 file per split. Each molecule group
#     only contains mol_idx, smiles and atom_features; spectra are not copied.
#     The selected split group and valid_indices* are flattened and saved to
#     indices.npz.
#     """
#     save_dir = osp.dirname(file_path) if save_dir is None else str(save_dir)
#     os.makedirs(save_dir, exist_ok=True)

#     file_list = os.listdir(save_dir)
#     split_list = ["train", "val", "test"]
#     for file in file_list:
#         if file.endswith(".h5"):
#             if "train" in file: 
#                 split_list.remove("train")
#             elif "val" in file:
#                 split_list.remove("val")
#             elif "test" in file:
#                 split_list.remove("test")
#     if len(split_list) == 0:
#         log.info("All splits are already processed")
#         return 

#     indices = {}
#     split_counts = {}
#     with h5py.File(file_path, 'r', swmr=True) as f:
#         if split_name not in f:
#             raise KeyError(f"Cannot find split group '{split_name}' in {file_path}")

#         for key in [split_name, "valid_indices_h", "valid_indices_c"]:
#             if key in f:
#                 _save_h5_item_to_npz_dict(f[key], key, indices)

#         split_group = f[split_name]
#         for split in split_list:
#             if split not in split_group:
#                 raise KeyError(f"Cannot find '{split}' in split group '{split_name}'")

#             split_indices = split_group[split][()]
#             out_path = osp.join(save_dir, f"{split}_molecules.h5")
#             copied = 0
#             with h5py.File(out_path, "w") as out_f:
#                 out_f.attrs["source_file"] = file_path
#                 out_f.attrs["split_name"] = split_name
#                 out_f.attrs["split"] = split
#                 for mol_idx in tqdm(split_indices, desc=f"Saving {split} molecules", unit="mol"):
#                     copied += int(_copy_uspto_molecule_without_spectra(f, out_f, mol_idx))

#             split_counts[split] = copied

#     indices["exported_train_count"] = np.asarray(split_counts.get("train", 0), dtype=np.int64)
#     indices["exported_val_count"] = np.asarray(split_counts.get("val", 0), dtype=np.int64)
#     indices["exported_test_count"] = np.asarray(split_counts.get("test", 0), dtype=np.int64)
#     np.savez(osp.join(save_dir, "indices.npz"), **indices)


# def read_parquets(
#     parquet_dir='../data/uspto/download/data/multimodal_spectroscopic_dataset',
#     save_dir='../data/uspto/preprocessed',
#     ):
#     parquet_files = sorted(os.listdir(parquet_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))
#     data_list = []
#     for file in tqdm(parquet_files, total=len(parquet_files), desc="Reading parquet files", unit="file"):
#         df = pd.read_parquet(osp.join(parquet_dir, file))
#         data_list.extend(read_parquet_to_nmr_peaks(df))
#         del df
    
 
#     os.makedirs(save_dir, exist_ok=True)
#     torch.save(data_list, osp.join(save_dir, "nmr_peaks.pt"))
#     del data_list
#     return data_list

# def read_str(value):
#     if isinstance(value, bytes):
#         return value.decode("utf-8")
#     return value

# def preprocess_uspto():

#     """
#     item from lmdb:
#         atoms: list, atom symbols
#         coordinates: np.ndarray, (n_atoms, 3)
#         atoms_target: np.ndarray, (n_atoms, 1), nmr shifts
#         atoms_target_mask: np.ndarray, (n_atoms, 1), nmr shifts mask
#         smiles: str
#         db_id: str
#         mol: Chem.Mol
#         inchikey: str
#     """

#     # Download
#     download_dir = os.path.join(target_dir, "download")
#     os.makedirs(download_dir, exist_ok=True)
#     fname = osp.join(download_dir, "uspto.tar.gz")
#     if not osp.exists(fname):
#         log.info(f"The downloaded files will be placed in '{download_dir}'.")
#         log.info(f"Downloading '{URL_USPTO}'...")
#         urllib.request.urlretrieve(URL_USPTO, fname)
#         log.info(f"Done downloading '{URL_USPTO}'...")

#     data_dir = osp.join(download_dir, "data")
#     if not osp.exists(data_dir):
#         # with zipfile.ZipFile(fname, 'r') as zip_ref:
#             # zip_ref.extractall(download_dir)
#         with tarfile.open(fname, 'r') as tar_ref:
#             tar_ref.extractall(download_dir)
#         log.info(f"Done extracting '{fname}'...")
    
#     log.info(f"Preprocessing USPTO...")
#     h5_path = osp.join(data_dir, "uspto/molecules.h5")
#     preprocessed_dir = osp.join(download_dir, "preprocessed")
    
#     read_uspto_h5(h5_path, save_dir=preprocessed_dir, split_name="split_indices_dedup")
#     log.info(f"Done preprocessing USPTO. Saved files in '{preprocessed_dir}'.")


#     nmr_peaks_file = osp.join(preprocessed_dir, "nmr_peaks.pt")
#     if not osp.exists(nmr_peaks_file):
#         nmr_peaks_data = read_parquets(parquet_dir=osp.join(data_dir, "multimodal_spectroscopic_dataset"), save_dir=preprocessed_dir)
#         log.info(f"Done preprocessing NMR peaks ({len(nmr_peaks_data)} data). Saved files in {nmr_peaks_file}.")
#     else:
#         log.info(f"NMR peaks already processed. Loading from '{nmr_peaks_file}'...")
#         nmr_peaks_data = torch.load(nmr_peaks_file)
#         log.info(f"Done loading NMR peaks ({len(nmr_peaks_data)} data).")
        

#     smiles_to_idx = {}
#     for i, data in tqdm(enumerate(nmr_peaks_data), total=len(nmr_peaks_data), desc="SMILES to NMR data index", unit="data"):
#         if data.smiles not in smiles_to_idx:
#             smiles_to_idx[data.smiles] = i
#         else:
#             if type(smiles_to_idx[data.smiles]) == list:
#                 smiles_to_idx[data.smiles].append(i)
#             else:
#                 smiles_to_idx[data.smiles] = [smiles_to_idx[data.smiles], i]
    
#     save_dir = osp.join(target_dir, "preprocessed")
#     os.makedirs(save_dir, exist_ok=True)

#     for split in ['val', 'test', 'train']:
#     # for split in ['test', 'train']:
#         h5_file_path = osp.join(preprocessed_dir, f"{split}_molecules.h5")
#         log.info(f"Processing {split} molecules...")

#         invalid_mol = {
#             "None_mol": [],
#             "not_neural_molecule": [],
#             "I_3_bonds": [],
#             "I_5_bonds": [],
#             "num_total_mol_with_I": 0,
#             "max_pairwise_dist": 30,
#             "dist_too_large": [],
#             "dataset_max_pairwise_dist": 0,
#             "max_num_atoms": 170,
#             "num_atoms_too_large": [],
#             "not_found_in_nmr_data": [],
#             "geo_error": [],
#         }
#         heavy_atom_count = {}
#         total_heavy_atoms_count = {}
#         split_list = [] 
#         num_mol = 0
#         num_entries = 0

#         # split_metrics = USPTOPreprocessMetrics(summarize_hidden=True, hidden_prefix="")
     
#         with h5py.File(h5_file_path, 'r', swmr=True) as f:
#             for mol_idx in tqdm(f, total=len(f), desc=f"Processing {split} molecules", unit="mol"):
#                 mol = f[mol_idx]
#                 original_smiles = read_str(mol.attrs["smiles"])
#                 id = read_str(mol.attrs["mol_idx"])
             
#                 # === check smiles ===
#                 smiles = get_clean_smiles(original_smiles)
#                 if smiles is None:
#                     invalid_mol["None_mol"].append(id)
#                     continue
#                 if not is_neural_molecule(smiles):
#                     invalid_mol["not_neural_molecule"].append(id)
#                     continue
#                 has_I, is_I_3_bonds, is_I_5_bonds = check_I_valence(smiles)
#                 if has_I:
#                     if is_I_3_bonds:
#                         invalid_mol["I_3_bonds"].append(id)
#                         continue
#                     elif is_I_5_bonds:
#                         invalid_mol["I_5_bonds"].append(id)
#                         continue
                
#                 # === check nmr peaks ===
#                 if original_smiles not in smiles_to_idx:
#                     log.info(f"SMILES {original_smiles} not found in nmr data")
#                     invalid_mol["not_found_in_nmr_data"].append(id)
#                     continue

#                 # === process atoms ===
#                 if "atom_features" not in mol:
#                     raise KeyError(f"{mol_idx} has no atom_features group")

#                 atom_features = mol["atom_features"]
#                 atom_mask = atom_features["atom_mask"][()]
#                 num_atoms = int(atom_mask.sum())

#                 atom_charges = atom_features["atom_charges"][()] # np.ndarray, (num_atoms, 3)
#                 assert atom_charges.shape[0] == num_atoms, f"atom_charges.shape[0] != num_atoms: {atom_charges.shape[0]} != {num_atoms}"
                
#                 atom_coords = atom_features["atom_coords"][()] # np.ndarray, (3, num_atoms, 3),  3 ground-truth conformers
#                 assert atom_coords.shape[1] == num_atoms, f"atom_coords.shape[1] != num_atoms: {atom_coords.shape[1]} != {num_atoms}"

#                 # check if the molecule is too large (max pairwise distance: 30 Å)
#                 pos_list = []
#                 # assert atom_coords.shape[0] == 3, f"atom_coords.shape[0] != 3: {atom_coords.shape}"
#                 for i in range(atom_coords.shape[0]):
#                     pos = atom_coords[i]
#                     dist_mat = np.linalg.norm(
#                         pos[:, None, :] - pos[None, :, :],
#                         axis=-1
#                     )

#                     max_dist = dist_mat.max()
#                     if max_dist > invalid_mol["max_pairwise_dist"]:
#                         log.info(f"molecule {id}, conformer {i} | pairwise distance {max_dist} > {invalid_mol['max_pairwise_dist']} Å: {smiles}")
#                         continue
#                     pos_list.append(pos)
#                 if len(pos_list) == 0:
#                     invalid_mol["dist_too_large"].append(id)
#                     continue
                
#                 # === get data ===
#                 heavy_atoms_mask = (atom_charges != 1)
#                 num_heavy_atoms = int(heavy_atoms_mask.sum())
#                 if num_heavy_atoms not in total_heavy_atoms_count:
#                     total_heavy_atoms_count[num_heavy_atoms] = 0
#                 total_heavy_atoms_count[num_heavy_atoms] += 1

#                 heavy_atoms = atom_charges[heavy_atoms_mask]
#                 for heavy_atom in heavy_atoms:
#                     heavy_atom = int(heavy_atom)
#                     if heavy_atom not in heavy_atom_count:
#                         heavy_atom_count[heavy_atom] = 0
#                     heavy_atom_count[heavy_atom] += 1
                
#                 nmr_idx = smiles_to_idx[original_smiles]
#                 if type(nmr_idx) == list:
#                     nmr_data = []
#                     for idx in nmr_idx:
#                         nmr_data.append(nmr_peaks_data[idx])
#                 else:
#                     nmr_data = [nmr_peaks_data[nmr_idx]]
#                 data_list = process_atoms(
#                     atom_charges, pos_list,
#                     id, smiles, original_smiles,
#                     nmr_data,
#                     split,
#                     ) 

                
#                 geo_checked_data_list = []
#                 for data in data_list:
#                     try:
#                         split_metrics(data)
#                     except Exception as e:
#                         log.info(f"Geometry Error: Error processing molecule {id}")
#                         log.info(f"Data: {data}")
#                         xyz_txt = get_xyz_txt(
#                             [CHARGE_TO_SYMBOL[int(h)] for h in data.h], 
#                             data.pos.tolist(), 
#                             data.smiles
#                             )
#                         log.info(xyz_txt)
#                         invalid_mol["geo_error"].append(data.id)
#                         continue
#                     geo_checked_data_list.append(data)
                
#                 split_list.extend(geo_checked_data_list) 
#                 num_mol += 1
#                 num_entries += len(geo_checked_data_list)

                
                
        
#         log.info(f"Summarizing infos...")
#         infos_path = os.path.join(save_dir, f"{split}_infos.json")
#         agg_infos = split_metrics.summarize()
#         del split_metrics
#         agg_infos['num_mol'] = num_mol
#         agg_infos['num_entries'] = num_entries
#         # save_json(agg_infos, infos_path)

#         pt_path = osp.join(save_dir, f"{split}.pt")
#         torch.save(split_list, pt_path)
#         log.info(f"Done processing split: '{split}'. Saved in '{pt_path}'.")
    
#     heavy_atom_info = {
#             "heavy_atom_count": heavy_atom_count,
#             "total_heavy_atoms_count": total_heavy_atoms_count
#         }

BOND_TYPE_TO_idx = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
    }

SYMBOL_TO_CHARGE = {
    'H': 1,
    'C': 6, 'N': 7, 'O': 8, 'F': 9,
    'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
    'Br': 35, 'I': 53, 
}

CHARGE_TO_SYMBOL = {
        1:  'H',
        6:  'C',  7:  'N', 8:  'O', 9:  'F',
        14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 
        35: 'Br', 53: 'I',
    }
CHARGES = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

BOND_TYPE_CANDIDATES = [
    '1-1',
    '6-1', '6-2', '6-3', '6-4',
    '7-1', '7-2', '7-3', '7-4',
    '8-1', '8-2', '8-4',
    '9-1', 
    '15-1', '15-2', '15-4',
    '16-1', '16-2', '16-4',
    '17-1', 
    '35-1', 
    '53-1', 
]


def get_heavy_atom_local_label(atom: Chem.Atom, mol: Chem.Mol=None, num_max_count: int=4):
    """
    return: 
        [neighbor_type, count]
    """
    assert atom.GetAtomicNum() != 1, f"Atom {atom.GetSymbol()} is H"
        
    heavy_atom_neighbors = torch.zeros(len(BOND_TYPE_CANDIDATES), dtype=torch.int32) 
    
    neighbor_type_counts = {}
    for neighbor in atom.GetNeighbors():
        neighbor_charge = neighbor.GetAtomicNum()
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        bond_type_idx = BOND_TYPE_TO_idx[bond.GetBondType()]
        neighbor_type = f"{neighbor_charge}-{bond_type_idx}"
        assert neighbor_type in BOND_TYPE_CANDIDATES, f"Invalid neighbor type: {neighbor_type}"
        if neighbor_type not in neighbor_type_counts:
            neighbor_type_counts[neighbor_type] = 0
        neighbor_type_counts[neighbor_type] += 1
    
    for neighbor_type, count in neighbor_type_counts.items():
        heavy_atom_neighbors[BOND_TYPE_CANDIDATES.index(neighbor_type)] = count
    
    return heavy_atom_neighbors
    


def smiles_to_local_label(smiles: str):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol=Chem.AddHs(mol)
    
    canno_atoms = []
    
    hydrogen_neighbors = []
    
    is_aromatic = []
    heavy_atom_local_labels = []

    
    for atom in mol.GetAtoms():

        charge = atom.GetAtomicNum()
        canno_atoms.append(charge)

        if charge == 1:
            hydrogen_neighbors.append(atom.GetNeighbors()[0].GetAtomicNum())
            continue

        is_aromatic.append(int(atom.GetIsAromatic()))
        
        label = get_heavy_atom_local_label(atom, mol)
        heavy_atom_local_labels.append(label)

    return canno_atoms, hydrogen_neighbors, is_aromatic, heavy_atom_local_labels

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
        canno_atoms, hydrogen_neighbors, is_aromatic_heavy_atoms, heavy_atom_local_labels = smiles_to_local_label(smiles)
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
            canno_h = torch.tensor(canno_atoms),
            hydrogen_neighbors = torch.tensor(hydrogen_neighbors), # [N_H, ]
            is_aromatic_heavy_atoms = torch.tensor(is_aromatic_heavy_atoms), # [N_heavy_atoms, ]
            heavy_atom_local_labels = torch.stack(heavy_atom_local_labels), # [N_heavy_atoms, NUM_NEIGHBOR_TYPES]
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
            smiles = getattr(data, "isomeric_smiles", data.smiles)
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
    _atomic_torch_save(data_list, path)
    split_infos = read_json(info_path)
    split_infos["categorical_mapping_version"] = CATEGORICAL_MAPPING_VERSION
    split_infos["categorical_vocab_source"] = "dataset_infos_train.json"
    save_json(split_infos, info_path)
    print(f"Saved {len(data_list)} mapped samples to {path}")


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

    if data_files_exist and already_mapped:
        print("Found complete, already-mapped split artifacts; nothing to do")
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
            data_list = _load_torch(paths[split]["data"])
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
            data_list = _load_torch(paths[split]["data"])
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
