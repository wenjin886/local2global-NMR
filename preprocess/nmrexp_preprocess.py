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

def process_nmrexp_parquet(parquet_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_parquet(parquet_file)
    smi_nmr_check_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        
        
        nmr_solvent = row['nmr_solvent']
        if nmr_solvent != 'CDCl3':
            print(f"Skipping molecule {row['smiles']} due to unsupported solvent: {nmr_solvent}")
            continue
        
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

        if smiles not in smi_nmr_check_dict:
            skip_mol = False
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                if atom_type not in ATOM_TYPE_TO_CHARGE:
                    print(f"Skipping molecule {smiles} due to unsupported atom type: {atom_type}")
                    skip_mol = True
                    break
            if skip_mol:
                continue

        if '13C' in row['nmr_type']:
            if smiles not in smi_nmr_check_dict:
                smi_nmr_check_dict[smiles] = {'cnmr': 0, 'hnmr': 0}
            smi_nmr_check_dict[smiles]['cnmr'] += 1
        elif '1H' in row['nmr_type']:
            if smiles not in smi_nmr_check_dict:
                smi_nmr_check_dict[smiles] = {'cnmr': 0, 'hnmr': 0}
            smi_nmr_check_dict[smiles]['hnmr'] += 1

        if len(smi_nmr_check_dict) == 300:
            break
    
    df_smi_check = pd.DataFrame(smi_nmr_check_dict).T.reset_index().rename(columns={'index': 'smiles'})
    
    df_smi_chnmr = df_smi_check[(df_smi_check['cnmr'] > 0) & (df_smi_check['hnmr'] > 0)]
    print(f"Number of molecules with both 13C and 1H NMR data in CDCl3: {len(df_smi_chnmr)} from {len(df_smi_check)} molecules.")

    
    processed_data = {'smiles': [], 'cnmr': [], 'hnmr': []}
    for _, row in tqdm(df_smi_chnmr.iterrows(), total=len(df_smi_chnmr), desc="Processing molecules with both 13C and 1H NMR data"):
        smiles = row['smiles']
        df_smiles = df[df['smiles'] == smiles]
        
        if len(df_smiles) > 2:
        
            file_name = df_smiles.iloc[0]['filename']
            df_smiles = df_smiles[df_smiles['filename'] == file_name]

            if len(df_smiles[df_smiles['nmr_type'] == '13C NMR']) != 1:
                print(f"Skipping molecule {smiles} due to multiple 13C NMR entries in the same file.")
                continue
            if len(df_smiles[df_smiles['nmr_type'] == '1H NMR']) != 1:
                print(f"Skipping molecule {smiles} due to multiple 1H NMR entries in the same file.")
                continue
        processed_data['smiles'].append(smiles)
        processed_data['cnmr'].append(
            df_smiles[df_smiles['nmr_type'] == '13C NMR']['nmr_processed'].iloc[0])
        processed_data['hnmr'].append(
            df_smiles[df_smiles['nmr_type'] == '1H NMR']['nmr_processed'].iloc[0])
            
      
      
    df_processed = pd.DataFrame(processed_data)
    print(df_processed)
    df_processed.to_parquet(os.path.join(output_dir, "nmrexp_processed.parquet"), index=False)





    # # Save the dictionary as a pickle file
    # output_file = os.path.join(file_dir, f"{smiles}.pkl")
    # with open(output_file, 'wb') as f:
    #     pickle.dump(data_dict, f)
  

if __name__ == "__main__":
    nmrexp_file = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_IR/DATASET/Labels2Literature/data/raw_data/NMRexp_10to24_1_0811.parquet"
    process_nmrexp_parquet(nmrexp_file, output_dir="../data/nmrexp/processed")
