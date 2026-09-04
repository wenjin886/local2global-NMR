import urllib.request
import zipfile
from pathlib import Path

import lmdb
import os
import pickle
from functools import lru_cache

import random
from typing import List, Tuple

from tqdm import tqdm
from rxn.chemutils.tokenization import tokenize_smiles
import regex as re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import json
import matplotlib.pyplot as plt
import numpy as np

# from xyz2mol_final import XYZ2MOL, NotCompleteMoleculeError
import sys
sys.path.append("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_IR/peak_assign_nmr/code/utils")
from xyz2mol import XYZ2MOL as XYZ2MOL_WITH_FG
from xyz2mol import NotCompleteMoleculeError

# print(XYZ2MOL_WITH_FG)
# print(XYZ2MOL_WITH_FG.__module__)

ATOM_TYPE_TO_CHARGE = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Br': 35,
    'I': 53,
}

# def download_data():
# download from NMRNet
#     url = "https://zenodo.org/records/18232165/files/data.zip?download=1"
#     urllib.request.urlretrieve(url, "data.zip")

#     print("Data downloaded successfully")

#     with zipfile.ZipFile("data.zip", "r") as z:
#         z.extractall("./")


import pandas as pd
class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self.split = os.path.splitext(os.path.basename(self.db_path))[0]
            self.dbid_file = os.path.join(os.path.dirname(self.db_path), f'{self.split}_dbid.pkl')
            if os.path.isfile(self.dbid_file):
                with open(self.dbid_file, 'rb') as f:
                    self._keys = pickle.load(f)
            else:
                self._keys = list(txn.cursor().iternext(values=False))
                with open(self.dbid_file, 'wb') as f:
                    pickle.dump(self._keys, f)

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        return data


from src.data.dataset import canonicalize_smiles_without_stereo
def screen_nmr_data(db_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    files = os.listdir(db_dir)

    idx_chnmr_dict = {}
    clean_smiles_chnmr_dict = {}
    for f in files:
        if not f.endswith('.lmdb'):
            continue
        
        f_name = f.split('.')[0]
        idx_chnmr_dict[f_name] = []
        clean_smiles_chnmr_dict[f_name] = []

        f_path = os.path.join(db_dir, f)
        dataset = LMDBDataset(f_path)
        for idx in tqdm(range(len(dataset)), total=len(dataset), desc=f"Processing dataset: {f}"):
            data = dataset[idx]
            
            atoms = data['atoms']
            skip_mol = False
            for a in set(atoms):
                if a not in ATOM_TYPE_TO_CHARGE:
                    skip_mol = True
                    break
            if skip_mol:
                print(f"Skipping data {idx} due to invalid atom type {a}.")
                continue

            atoms_target_mask = data['atoms_target_mask'] # np.array
            charges = np.array([ATOM_TYPE_TO_CHARGE[atom] for atom in atoms])
            nmr_type = set(charges[atoms_target_mask.astype(bool)].tolist())
            if not (1 in nmr_type and 6 in nmr_type): # H and C
                
                print(f"Skipping data {idx} due to invalid NMR type {nmr_type}.")
                continue

            idx_chnmr_dict[f_name].append(idx)
            smiles = canonicalize_smiles_without_stereo(data['smiles'])
            if smiles not in clean_smiles_chnmr_dict[f_name]:
                clean_smiles_chnmr_dict[f_name].append(smiles)

        print(f"{f_name}: Number of molecules with H and C NMR: {len(idx_chnmr_dict[f_name])} from {len(dataset)}")
    print(f"Total number of molecules with H and C NMR: {sum([len(idx_chnmr_dict[f_name]) for f_name in idx_chnmr_dict])} from {len(dataset)}")

    with open(os.path.join(output_dir, "idx_chnmr.json"), "w") as f:
        json.dump(idx_chnmr_dict, f, indent=4)
    
    with open(os.path.join(output_dir, "clean_smiles_chnmr.json"), "w") as f:
        json.dump(clean_smiles_chnmr_dict, f, indent=4)
    print(f"Saved idx_chnmr.json and clean_smiles_chnmr.json to {output_dir}")

                
  
  

if __name__ == "__main__":
    db_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/nmrnet-model-data/data/nmrshiftdb2_2024/All"
    # db_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/local2global/data/nmrshifdb2"
    # output_dir = "../data/nmrexp/processed"
    output_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/nmrshiftdb_2024"
    # output_dir = db_dir
    screen_nmr_data(db_dir, output_dir)
