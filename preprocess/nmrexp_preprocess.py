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

def download_data():

    url = "https://zenodo.org/records/18232165/files/data.zip?download=1"
    urllib.request.urlretrieve(url, "data.zip")

    print("Data downloaded successfully")

    with zipfile.ZipFile("data.zip", "r") as z:
        z.extractall("./")

import pandas as pd
# df = pd.read_parquet("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_IR/DATASET/Labels2Literature/data/raw_data/NMRexp_10to24_1_0811.parquet")
# print(df.head())
# print(df.columns)
# print(df.shape)
#                       filename                                             smiles  page_in_file_smiles  ...  atom_number atom_number_diff_env atom_number_abstract
# 0  10.1021_acs.orglett.3c00718  [Si](CC)(CC)(CC)/C(F)=C(/Cc1ccc2ccccc2c1)C(=O)OCC                 15.0  ...           22                   18                 17.0
# 1  10.1021_acs.orglett.3c00718  C(=O)(OC(C)(C)C)/C(Cc1ccccc1)=C(\F)[Si](CC)(CC)CC                 15.0  ...           31                    7                 31.0
# 2  10.1021_acs.orglett.3c00718  C(=O)(OC(C)(C)C)/C(Cc1ccccc1)=C(\F)[Si](CC)(CC)CC                 15.0  ...            1                    1                  1.0
# 3  10.1021_acs.orglett.3c00718  C(=O)(OC(C)(C)C)/C(Cc1ccccc1)=C(\F)[Si](CC)(CC)CC                 15.0  ...           20                   12                 12.0
# 4  10.1021_acs.orglett.3c00718       [Si](CC)(CC)(CC)/C(F)=C(/CCc1ccccc1)C(=O)OCC                 15.0  ...           29                    9                 29.0

# [5 rows x 15 columns]
# Index(['filename', 'smiles', 'page_in_file_smiles', 'page_in_file_para',
#        'location_in_page_smiles', 'location_in_page_para', 'nmr_type',
#        'nmr_frequency', 'nmr_solvent', 'nmr_shift', 'nmr_note',
#        'nmr_processed', 'atom_number', 'atom_number_diff_env',
#        'atom_number_abstract'],
#       dtype='str')
# (3372987, 15)

#        10.1021_acs.orglett.3c00718
# smiles                     [Si](CC)(CC)(CC)/C(F)=C(/Cc1ccc2ccccc2c1)C(=O)OCC
# page_in_file_smiles                                                     15.0
# page_in_file_para                                                         15
# location_in_page_smiles        [0.05914522 0.28183596 0.20726104 0.32089846]
# location_in_page_para          [0.05459559 0.32343751 0.94227949 0.45234376]
# nmr_type                                                             13C NMR
# nmr_frequency                                                        101 MHz
# nmr_solvent                                                            CDCl3
# nmr_shift                  180.0 (d, J = 300.8 Hz), 167.6 (d, J = 22.7 Hz...
# nmr_note                                                                 NaN
# nmr_processed              [(180.0, 'd', 300.8), (167.6, 'd', 22.7), (137...
# atom_number                                                               22
# atom_number_diff_env                                                      18
# atom_number_abstract                                                    17.0
# Name: 0, dtype: object
# filename                                         10.1021_acs.orglett.3c00718
# smiles                     [Si](CC)(CC)(CC)/C(F)=C(/Cc1ccc2ccccc2c1)C(=O)OCC
# page_in_file_smiles                                                     15.0
# page_in_file_para                                                         15
# location_in_page_smiles        [0.05914522 0.28183596 0.20726104 0.32089846]
# location_in_page_para          [0.05459559 0.32343751 0.94227949 0.45234376]
# nmr_type                                                             13C NMR
# nmr_frequency                                                        101 MHz
# nmr_solvent                                                            CDCl3
# nmr_shift                  180.0 (d, J = 300.8 Hz), 167.6 (d, J = 22.7 Hz...
# nmr_note                                                                 NaN
# nmr_processed              [(180.0, 'd', 300.8), (167.6, 'd', 22.7), (137...
# atom_number                                                               22
# atom_number_diff_env                                                      18
# atom_number_abstract                                                    17.0

from src.data.dataset import canonicalize_smiles_without_stereo
def process_nmrexp_parquet(parquet_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(parquet_file)
    smi_nmr_check_dict = {}

    print(f"Number of nmr data in the parquet file: {len(df)}")
    df = df[df['nmr_solvent'] == 'CDCl3']
    print(f"Number of nmr data from CDCl3 in the parquet file: {len(df)}")
    df = df[df['nmr_type'].isin(['13C NMR', '1H NMR'])]
    print(f"Number of nmr data from 13C and 1H NMR in the parquet file: {len(df)}")

    grouped_df = df.groupby(['smiles', 'filename'])
    processed_data = {'smiles': [], 'cnmr': [], 'hnmr': [], 'filename': [], 'clean_smiles': []}

    print(f"Number of grouped data: {len(grouped_df)} with key (smiles, filename)")
    for (smiles, filename), group in tqdm(grouped_df, total=len(grouped_df), desc="Processing grouped data"):
        # print(smiles, filename)
        # print(group)

        if '.' in smiles: # skip molecules with multiple fragments
            continue
        if len(group) < 2: # skip molecules with less than 2 entries
            continue

        # smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Skipping molecule {smiles} due to invalid SMILES.")
            continue

        if smiles not in smi_nmr_check_dict:
            skip_mol = False
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                if atom_type not in ATOM_TYPE_TO_CHARGE:
                    skip_mol = True
                    break
            if skip_mol:
                continue
        
        # check if the molecule has both 13C and 1H NMR data
        num_h = (group['nmr_type'] == '1H NMR').sum()
        num_c = (group['nmr_type'] == '13C NMR').sum()
        if num_h != 1 or num_c != 1:
            continue

        clean_smiles = canonicalize_smiles_without_stereo(smiles)
        processed_data['smiles'].append(smiles)
        processed_data['cnmr'].append(
            group[group['nmr_type'] == '13C NMR']['nmr_processed'].iloc[0])
        processed_data['hnmr'].append(
            group[group['nmr_type'] == '1H NMR']['nmr_processed'].iloc[0])
        processed_data['filename'].append(filename)
        processed_data['clean_smiles'].append(clean_smiles)

        # break

      
    df_processed = pd.DataFrame(processed_data)
    print(df_processed)
    # raise Exception("Stop here")
    df_processed.to_parquet(os.path.join(output_dir, "nmrexp_processed.parquet"), index=False)

def check_nmrshiftdb2_data(nmrshiftdb2_smi_path: str, nmrexp_parquet_path: str, output_dir: str):
    smi_nmr_check_dict = json.load(open(nmrshiftdb2_smi_path, 'r'))
    smi_nmrshiftdb2_list = smi_nmr_check_dict['valid'] + smi_nmr_check_dict['train']

    df = pd.read_parquet(nmrexp_parquet_path)
    smiles_list = df['clean_smiles'].unique().tolist()
    exist_smiles_list = []
    for smiles in tqdm(smiles_list, total=len(smiles_list), desc="Checking molecules in nmrshiftdb2"):
        if smiles in smi_nmrshiftdb2_list:
            exist_smiles_list.append(smiles)
    num_not_exist_smiles = set(smi_nmrshiftdb2_list) - set(exist_smiles_list)
    print(f"Number of not exist smiles: {len(num_not_exist_smiles)}")

if __name__ == "__main__":
    nmrexp_file = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_IR/DATASET/Labels2Literature/data/raw_data/NMRexp_10to24_1_0811.parquet"
    # output_dir = "../data/nmrexp/processed"
    output_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/nmrexp"
    # process_nmrexp_parquet(nmrexp_file, output_dir)
    
    nmrshiftdb2_smi_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/nmrshiftdb_2024/clean_smiles_chnmr.json"
    nmrexp_parquet_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/nmrexp/nmrexp_processed.parquet"
    output_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/nmrexp"
    check_nmrshiftdb2_data(nmrshiftdb2_smi_path, nmrexp_parquet_path, output_dir)