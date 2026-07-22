import torch
import os
import os.path as osp
import rdkit
from rdkit import Chem
from tqdm import tqdm

import json

class AtomBondCounter:
    def __init__(self):
        self.atom_maxbondcounts = {}

    def update(self, mol):
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            atom_idx = atom.GetAtomicNum()
            if atom_idx not in self.atom_maxbondcounts:
                self.atom_maxbondcounts[atom_idx] = 0
            num_neighbors = len(atom.GetNeighbors())
            self.atom_maxbondcounts[atom_idx] = max(
                self.atom_maxbondcounts[atom_idx], num_neighbors
            )    

    def summarize(self):
        self.atom_maxbondcounts = dict(
            sorted(self.atom_maxbondcounts.items(), key=lambda x: x[0], reverse=False)
        )
        return self.atom_maxbondcounts

def count_dataset(data_dir):
    splits = ["val", "test", "train"]
    count_info = {}
    for split in splits:
        f_name = osp.join(data_dir, f"{split}.pt")
        print("Loading data from:", f_name)
        data_list = torch.load(f_name)
        print(f"Done loading.")
        counter = AtomBondCounter()
        for data in tqdm(data_list, desc=f"Counting {split}"):
            mol = Chem.MolFromSmiles(data.isomeric_smiles)
            counter.update(mol)
        count_info[split] = counter.summarize()
    with open(osp.join(data_dir, "atom_bond_count.json"), "w") as f:
        json.dump(count_info, f, indent=4)

if __name__ == "__main__":
    # smiles = "O=C(NC[C@H]1CN(c2ccc3c(c2)CCC(=O)N3)C(=O)CC1)c1cnn(CC(F)(F)F)c1"
    # mol = Chem.MolFromSmiles(smiles)
    # counter = AtomBondCounter()
    # counter.update(mol)
    # counts = counter.summarize()
    # print(counts)
    data_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/preprocessed"
    count_dataset(data_dir)

