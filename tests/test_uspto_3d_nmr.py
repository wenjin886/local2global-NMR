import numpy as np
import torch
from rdkit import Chem

from preprocess.uspto_3d_nmr import (
    _normalize_coordinates,
    collapse_carbon_lines,
    expand_hydrogen_shifts,
    hydrogen_peak_tensors,
    symmetry_classes,
)


def test_hydrogen_integrations_form_an_equal_cardinality_multiset():
    targets = expand_hydrogen_shifts(
        [{"delta": 1.2, "nH": 3}, {"delta": 7.1, "nH": 2}]
    )
    assert torch.equal(
        targets, torch.tensor([1.2, 1.2, 1.2, 7.1, 7.1])
    )
    shifts, counts = hydrogen_peak_tensors(
        [{"delta": 7.1, "nH": 2}, {"delta": 1.2, "nH": 3}]
    )
    assert torch.equal(shifts, torch.tensor([1.2, 7.1]))
    assert torch.equal(counts, torch.tensor([3, 2]))


def test_carbon_multiplet_lines_are_collapsed_to_known_class_count():
    peaks = [
        {"delta (ppm)": 20.0, "integral": 1.0},
        {"delta (ppm)": 99.0, "integral": 1.0},
        {"delta (ppm)": 100.0, "integral": 2.0},
        {"delta (ppm)": 101.0, "integral": 1.0},
    ]
    targets = collapse_carbon_lines(peaks, class_count=2)
    assert targets is not None
    assert torch.allclose(targets, torch.tensor([20.0, 100.0]))


def test_coordinate_normalization_keeps_all_conformers_and_applies_mask():
    coords = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    result = _normalize_coordinates(coords, np.array([1, 1, 0, 0]))
    assert result.shape == (2, 2, 3)
    assert np.array_equal(result, coords[:, :2])


def test_symmetry_classes_group_equivalent_explicit_hydrogens():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC"))
    classes = symmetry_classes(molecule)
    hydrogen_classes = classes[
        torch.tensor([atom.GetAtomicNum() == 1 for atom in molecule.GetAtoms()])
    ]
    assert torch.unique(hydrogen_classes).numel() == 1
