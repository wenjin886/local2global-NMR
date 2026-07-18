from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch_geometric.data import Data
from .base import Metrics, discrete_histogram, ContextMetrics, read_json

class USPTOPreprocessMetrics(Metrics):
    def __init__(self,
                 json_path: Optional[str | Path] = None,
                 summarize_hidden: bool = False,
                 hidden_prefix: str = "_"):

        self.encoder = {s: idx for idx, s in enumerate(atom_types_str)}
        self.max_num_atoms = 0
        self.max_num_heavy_atoms = 0
        self.max_neighbor_type_cout = 0

        if json_path:
            dataset_infos = read_json(json_path=json_path)
            ref_smiles = set(dataset_infos.get("smiles"))
            ref_atom_hist = np.array(dataset_infos.get("atom_hist"))
        else:
            ref_smiles = set([])
            ref_atom_hist = None

        self.ref_smiles = ref_smiles
        self.ref_atom_hist = ref_atom_hist

        self.summarize_hidden = summarize_hidden
        self.hidden_prefix = hidden_prefix


        self.hnmr_shifts = ...
        self.cnmr_shifts = ...
        self.n_hnmr_shifts_per_mol = ...
        self.n_cnmr_shifts_per_mol = ...

        self.reset()

    def update(self, data: list[Data] | Data):
        if isinstance(data, Data):
            data = [data]

        for d in data:
            num_atoms = d.h.shape[0]
            if num_atoms > self.max_num_atoms:
                self.max_num_atoms = num_atoms
            
            heavy_mask =  (d.h > 1)
            num_heavy_atoms = heavy_mask.sum()
            if num_heavy_atoms > self.max_num_heavy_atoms:
                self.max_num_heavy_atoms = num_heavy_atoms
            
            fragment_labels = (
                d.heavy_fragment_labels
                if hasattr(d, "heavy_fragment_labels")
                else d.heavy_atom_local_labels
            )
            max_count = fragment_labels.max()
            if max_count > self.max_neighbor_type_cout:
                self.max_neighbor_type_cout = max_count
            

            h_nmr = d.h_nmr.tolist()
            self.hnmr_shifts.extend(h_nmr)
            self.n_hnmr_shifts_per_mol.append(len(h_nmr))

            c_nmr = d.c_nmr.tolist()
            self.cnmr_shifts.extend(c_nmr)
            self.n_cnmr_shifts_per_mol.append(len(c_nmr))

    def summarize(self) -> dict:
  
        summary = {}
        summary["max_num_atoms"] = self.max_num_atoms


        # print(self.hnmr_shifts)
        self.hnmr_shifts = np.array(self.hnmr_shifts)
        self.cnmr_shifts = np.array(self.cnmr_shifts)
        self.n_hnmr_shifts_per_mol = np.array(self.n_hnmr_shifts_per_mol)
        self.n_cnmr_shifts_per_mol = np.array(self.n_cnmr_shifts_per_mol)
        
        summary[f"{self.hidden_prefix}hnmr_shift_min"] = round(float(np.min(self.hnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}hnmr_shift_max"] = round(float(np.max(self.hnmr_shifts)), 4)    
        summary[f"{self.hidden_prefix}hnmr_shift_mean"] = round(float(np.mean(self.hnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}hnmr_shift_std"] = round(float(np.std(self.hnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}hnmr_shift_median"] = round(float(np.median(self.hnmr_shifts)), 4)

        summary[f"{self.hidden_prefix}cnmr_shift_min"] = round(float(np.min(self.cnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}cnmr_shift_max"] = round(float(np.max(self.cnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}cnmr_shift_mean"] = round(float(np.mean(self.cnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}cnmr_shift_std"] = round(float(np.std(self.cnmr_shifts)), 4)
        summary[f"{self.hidden_prefix}cnmr_shift_median"] = round(float(np.median(self.cnmr_shifts)), 4)

        summary[f"{self.hidden_prefix}n_hnmr_shifts_per_mol_min"] = int(np.min(self.n_hnmr_shifts_per_mol))
        summary[f"{self.hidden_prefix}n_hnmr_shifts_per_mol_max"] = int(np.max(self.n_hnmr_shifts_per_mol))

        summary[f"{self.hidden_prefix}n_cnmr_shifts_per_mol_min"] = int(np.min(self.n_cnmr_shifts_per_mol))
        summary[f"{self.hidden_prefix}n_cnmr_shifts_per_mol_max"] = int(np.max(self.n_cnmr_shifts_per_mol))

        # print(summary)
        

        return summary

    def reset(self):

        self.hnmr_shifts = []
        self.cnmr_shifts = []
        self.n_hnmr_shifts_per_mol = []
        self.n_cnmr_shifts_per_mol = []
