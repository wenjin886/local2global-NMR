import numpy as np
import pytest
import torch
from rdkit import Chem

from local2geo_demo import generate_geometry, simulate_soft_graph, write_xyz


def test_simulated_soft_graph_is_symmetric_and_normalized():
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    graph = simulate_soft_graph(molecule, logit_noise=0.2, seed=3)

    assert torch.allclose(
        graph.edge_probabilities, graph.edge_probabilities.transpose(0, 1)
    )
    assert torch.allclose(
        graph.edge_probabilities.sum(dim=-1),
        torch.ones(
            molecule.GetNumAtoms(),
            molecule.GetNumAtoms(),
            dtype=graph.edge_probabilities.dtype,
        ),
    )
    assert torch.equal(graph.projected_adjacency, graph.projected_adjacency.T)


@pytest.mark.parametrize("smiles", ["CCO", "C=C", "c1ccccc1", "C1CCCCC1"])
def test_generate_geometry_is_finite_and_locally_reasonable(smiles):
    result = generate_geometry(smiles, relaxation_steps=250)

    assert result.coordinates.shape == (len(result.symbols), 3)
    assert np.isfinite(result.coordinates).all()
    assert result.diagnostics["num_components"] == 1
    assert result.diagnostics["bond_mae_angstrom"] < 0.20
    assert result.diagnostics["max_bond_error_angstrom"] < 0.45
    assert result.diagnostics["final_prior_energy"] <= (
        result.diagnostics["initial_prior_energy"] + 1e-7
    )


def test_xyz_writer_and_invalid_smiles(tmp_path):
    result = generate_geometry("CO", relaxation_steps=100)
    path = tmp_path / "methanol.xyz"
    write_xyz(path, result)
    lines = path.read_text().splitlines()

    assert int(lines[0]) == len(result.symbols)
    assert len(lines) == len(result.symbols) + 2
    with pytest.raises(ValueError, match="Invalid SMILES"):
        generate_geometry("not-a-smiles")
