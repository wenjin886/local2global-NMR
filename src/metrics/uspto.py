from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch_geometric.data import Data
from .base import Metrics, discrete_histogram, ContextMetrics


_SYMBOLS_USPTO = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

def data_to_atoms(data: Data) -> ase.Atoms:
    symbols = data.h.tolist()
    positions = data.pos.tolist()
    atoms = ase.Atoms(symbols=symbols, positions=positions)
    return atoms

class USPTOPreprocessMetrics(Metrics):
    def __init__(self,
                 atom_types_str: str = _SYMBOLS_USPTO,
                 json_path: Optional[str | Path] = None,
                 summarize_hidden: bool = False,
                 hidden_prefix: str = "_"):

        self.encoder = {s: idx for idx, s in enumerate(atom_types_str)}
        self.max_num_atoms = 0

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

        self.smiles = ...
        self.n_atoms = ...
        self.atom_hist = ...

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
            if d.num_atoms > self.max_num_atoms:
                self.max_num_atoms = d.num_atoms
            
            heavy_mask =  (d.h > 1)
            num_heavy_atoms = heavy_mask.sum()
            
            
            self.smiles.append(data.smiles)
        
            h_nmr = d.h_nmr.tolist()
            self.hnmr_shifts.extend(h_nmr)
            self.n_hnmr_shifts_per_mol.append(len(h_nmr))

            c_nmr = d.c_nmr.tolist()
            self.cnmr_shifts.extend(c_nmr)
            self.n_cnmr_shifts_per_mol.append(len(c_nmr))

    def summarize(self) -> dict:
        assert len(self.valid) == len(self.valid_connected)
        assert len(self.valid) == len(self.molecule_stable)
        assert len(self.valid) == len(self.smiles)

        n_samples = len(self.valid)
        n_atoms = sum(self.n_atoms)

        summary = {}
        summary["max_num_atoms"] = self.max_num_atoms

        summary["atom_stable"] = sum(self.atom_stable) / n_atoms
        summary["molecule_stable"] = sum(self.molecule_stable) / n_samples

        summary["valid"] = sum(self.valid) / n_samples
        summary["valid_connected"] = sum(self.valid_connected) / n_samples

        valid_unique_smiles = set([smiles for (v, smiles) in zip(self.valid, self.smiles) if v])
        summary["valid_unique"] = len(valid_unique_smiles) / n_samples
        if self.ref_smiles is not None:
            vun_smiles = valid_unique_smiles.difference(self.ref_smiles)
            summary["valid_unique_novel"] = len(vun_smiles) / n_samples

        atom_hist = np.sum(np.stack(self.atom_hist, axis=0), axis=0)
        atom_hist = (atom_hist / atom_hist.sum())
        if self.ref_atom_hist is not None:
            summary["tv_atom"] = np.sum(np.abs(self.ref_atom_hist - atom_hist)).item()

        if self.summarize_hidden:
            summary[f"{self.hidden_prefix}atom_hist"] = atom_hist
            summary[f"{self.hidden_prefix}num_atoms_hist"] = discrete_histogram(self.n_atoms,
                                                                                encoder={idx: idx for idx in
                                                                                         range(self.max_num_atoms + 1)},
                                                                                norm=True)
            summary[f"{self.hidden_prefix}smiles"] = list(valid_unique_smiles)

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
        self.smiles = []
        self.valid = []
        self.valid_connected = []
        self.n_atoms = []
        self.molecule_stable = []
        self.atom_hist = []
        self.atom_stable = []

        self.hnmr_shifts = []
        self.cnmr_shifts = []
        self.n_hnmr_shifts_per_mol = []
        self.n_cnmr_shifts_per_mol = []


