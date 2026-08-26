import torch

from end2end_module.metrics import graph_exact_match_vectors
from end2end_module.prior_lit_module import PriorOnlyNMRModule
from local2geo_module.topology_prior import SoftTopologyPrior
from src.data.constants import BOND_TYPE_CANDIDATES
from src.data.dataset import GraphBatch
from src.model.loss import NMRGraphLoss
from src.model.nmr_to_graph import NMRToGraph


def _batch() -> GraphBatch:
    atom_types = torch.tensor([[1, 1, 6, 6]])
    atom_mask = torch.ones_like(atom_types, dtype=torch.bool)
    bond_types = torch.zeros((1, 4, 4), dtype=torch.long)
    bond_types[0, 2, 3] = bond_types[0, 3, 2] = 1
    fragments = torch.full(
        (1, 4, len(BOND_TYPE_CANDIDATES)), -100, dtype=torch.long
    )
    return GraphBatch(
        atom_types=atom_types,
        atom_mask=atom_mask,
        h_nmr=torch.tensor([[1.0, 1.2]]),
        h_nmr_mask=torch.tensor([[True, True]]),
        h_nmr_integration=torch.zeros((1, 2)),
        h_nmr_integration_mask=torch.zeros((1, 2), dtype=torch.bool),
        h_nmr_multiplicity=torch.zeros((1, 2), dtype=torch.long),
        h_nmr_multiplicity_mask=torch.zeros((1, 2), dtype=torch.bool),
        h_nmr_j=torch.zeros((1, 2, 1)),
        h_nmr_j_mask=torch.zeros((1, 2, 1), dtype=torch.bool),
        c_nmr=torch.tensor([[20.0, 24.0]]),
        c_nmr_mask=torch.tensor([[True, True]]),
        bond_types=bond_types,
        h_attachment=torch.tensor([[2, 3, -100, -100]]),
        heavy_fragment_labels=fragments,
        h_parent_fragment_labels=fragments.clone(),
        h_parent_types=torch.full((1, 4), -100, dtype=torch.long),
        smiles_input_ids=torch.ones((1, 1), dtype=torch.long),
        smiles_input_mask=torch.ones((1, 1), dtype=torch.bool),
        smiles_target_ids=torch.full((1, 1), 2, dtype=torch.long),
    )


def _module() -> PriorOnlyNMRModule:
    hidden_dim = 32
    graph_model = NMRToGraph(
        hidden_dim=hidden_dim,
        num_heads=4,
        num_joint_layers=1,
        num_atom_interaction_layers=1,
        num_fourier_features=8,
        max_num_atoms=8,
        max_fragment_count=2,
        attachment_dim=16,
        use_h_integration=False,
        use_h_multiplicity=False,
        use_h_j=False,
        use_smiles_loss=False,
        predict_attachments=True,
        predict_edges=True,
        use_graph_joint_encoder=False,
    )
    criterion = NMRGraphLoss(
        heavy_fragment_weight=0.0,
        heavy_fragment_presence_weight=0.0,
        heavy_neighbor_count_weight=0.0,
        h_parent_fragment_weight=0.0,
        h_parent_presence_weight=0.0,
        h_parent_type_weight=0.0,
        h_attachment_weight=1.0,
        h_count_weight=0.25,
        edge_weight=1.0,
        edge_total_neighbor_count_weight=1.0,
        carbon_valence_weight=0.25,
        fragment_edge_consistency_weight=0.0,
        smiles_weight=0.0,
    )
    return PriorOnlyNMRModule(
        nmr_to_graph=graph_model,
        graph_criterion=criterion,
        topology_prior=SoftTopologyPrior(
            hidden_dim=16, num_layers=1, dropout=0.0
        ),
    )


def test_prior_only_truncates_3d_and_updates_only_prior():
    module = _module()
    assert not hasattr(module, "geometry_solver")
    assert not hasattr(module, "coordinate_refiner")
    assert not hasattr(module, "shift_model")
    assert not any(
        parameter.requires_grad for parameter in module.nmr_to_graph.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in module.topology_prior.parameters()
    )
    batch = _batch()
    output = module(batch, teacher_force_smiles=True)
    greedy_output = module(batch, teacher_force_smiles=False)
    assert greedy_output["raw"]["heavy_edge_logits"].shape == (
        1, 4, 4, 5
    )
    losses = module._losses(batch, output)
    assert torch.isfinite(losses["loss"])
    assert not losses["loss_raw_graph"].requires_grad
    losses["loss"].backward()
    assert all(
        parameter.grad is None for parameter in module.nmr_to_graph.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in module.topology_prior.parameters()
    )


def test_graph_exact_match_is_hydrogen_permutation_invariant():
    batch = _batch()
    edge_logits = torch.full((1, 4, 4, 5), -5.0)
    edge_logits[..., 0] = 5.0
    edge_logits[0, 2, 3, 0] = edge_logits[0, 3, 2, 0] = -5.0
    edge_logits[0, 2, 3, 1] = edge_logits[0, 3, 2, 1] = 5.0
    attachment_probabilities = torch.zeros((1, 4, 4))
    # Hydrogen identities are swapped, but each heavy atom still has one H.
    attachment_probabilities[0, 0, 3] = 1.0
    attachment_probabilities[0, 1, 2] = 1.0
    exact = graph_exact_match_vectors(
        batch.atom_types,
        batch.atom_mask,
        batch.bond_types,
        batch.h_attachment,
        edge_logits,
        attachment_probabilities,
    )
    assert exact["typed_exact"].item() == 1.0
    assert exact["connectivity_exact"].item() == 1.0

    edge_logits[0, 2, 3, 1] = edge_logits[0, 3, 2, 1] = -5.0
    edge_logits[0, 2, 3, 2] = edge_logits[0, 3, 2, 2] = 5.0
    bond_order_wrong = graph_exact_match_vectors(
        batch.atom_types,
        batch.atom_mask,
        batch.bond_types,
        batch.h_attachment,
        edge_logits,
        attachment_probabilities,
    )
    assert bond_order_wrong["typed_exact"].item() == 0.0
    assert bond_order_wrong["connectivity_exact"].item() == 1.0
