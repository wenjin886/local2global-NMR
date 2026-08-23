import torch

from end2end_module.lit_module import EndToEndNMRModule
from end2end_module.refiner import SpectrumConditionedEGNNRefiner
from local2geo_module.geometry_solver import DifferentiableGeometrySolver
from local2geo_module.topology_prior import SoftTopologyPrior
from shift3d_module.lit_module import Shift3DModule
from src.data.constants import BOND_TYPE_CANDIDATES
from src.data.dataset import GraphBatch
from src.model.loss import NMRGraphLoss
from src.model.nmr_to_graph import NMRToGraph


def _batch() -> GraphBatch:
    atom_types = torch.tensor([[1, 1, 6, 6]])
    atom_mask = torch.ones_like(atom_types, dtype=torch.bool)
    bond_types = torch.zeros((1, 4, 4), dtype=torch.long)
    bond_types[0, 2, 3] = bond_types[0, 3, 2] = 1
    h_attachment = torch.tensor([[2, 3, -100, -100]])
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
        h_attachment=h_attachment,
        heavy_fragment_labels=fragments,
        h_parent_fragment_labels=fragments.clone(),
        h_parent_types=torch.full((1, 4), -100, dtype=torch.long),
        smiles_input_ids=torch.ones((1, 1), dtype=torch.long),
        smiles_input_mask=torch.ones((1, 1), dtype=torch.bool),
        smiles_target_ids=torch.full((1, 1), 2, dtype=torch.long),
    )


def _module(use_smiles_decoder=False, **curriculum) -> EndToEndNMRModule:
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
        use_smiles_loss=use_smiles_decoder,
        use_smiles_joint_bixt=use_smiles_decoder,
        num_smiles_layers=1,
        max_smiles_length=4,
        smiles_vocab_size=8 if use_smiles_decoder else None,
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
        h_count_weight=0.0,
        edge_weight=1.0,
        edge_total_neighbor_count_weight=0.0,
        carbon_valence_weight=0.0,
        fragment_edge_consistency_weight=0.0,
        smiles_weight=0.0,
    )
    return EndToEndNMRModule(
        nmr_to_graph=graph_model,
        graph_criterion=criterion,
        topology_prior=SoftTopologyPrior(
            hidden_dim=16, num_layers=1, dropout=0.0
        ),
        geometry_solver=DifferentiableGeometrySolver(
            seed_mode="differentiable", num_steps=1
        ),
        coordinate_refiner=SpectrumConditionedEGNNRefiner(
            input_dim=hidden_dim,
            hidden_dim=16,
            num_layers=2,
            num_rbf=8,
            dropout=0.0,
        ),
        shift_model=Shift3DModule(
            hidden_dim=16,
            num_interactions=1,
            num_rbf=8,
            equivalence_loss_weight=0.0,
            log_prediction_plots=False,
        ),
        freeze_topology_prior=True,
        freeze_shift_model=True,
        input_shifts_are_normalized=False,
        **curriculum,
    )


def test_pipeline_generates_coordinates_without_xyz_and_keeps_shift_frozen():
    module = _module()
    batch = _batch()
    output = module(batch)
    assert output["refined"]["coordinates"].shape == (1, 4, 3)
    assert not any(parameter.requires_grad for parameter in module.shift_model.parameters())
    assert not any(parameter.requires_grad for parameter in module.topology_prior.parameters())

    loss = module._losses(batch, output)["loss"]
    loss.backward()
    assert all(parameter.grad is None for parameter in module.shift_model.parameters())
    assert module.coordinate_refiner.layers[0].coordinate_gate.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in module.nmr_to_graph.parameters()
        if parameter.requires_grad
    )


def test_refiner_is_translation_equivariant():
    torch.manual_seed(4)
    refiner = SpectrumConditionedEGNNRefiner(
        input_dim=8, hidden_dim=16, num_layers=2, num_rbf=8
    )
    with torch.no_grad():
        refiner.layers[0].coordinate_gate.weight.normal_(std=0.05)
        refiner.layers[1].coordinate_gate.weight.normal_(std=0.05)
    coordinates = torch.randn(1, 4, 3)
    atom_features = torch.randn(1, 4, 8)
    h_peaks = torch.randn(1, 2, 8)
    c_peaks = torch.randn(1, 3, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    edges = torch.softmax(torch.randn(1, 4, 4, 5), dim=-1)
    arguments = dict(
        graph_atom_features=atom_features,
        h_peak_features=h_peaks,
        h_peak_mask=torch.ones(1, 2, dtype=torch.bool),
        c_peak_features=c_peaks,
        c_peak_mask=torch.ones(1, 3, dtype=torch.bool),
        edge_probabilities=edges,
        atom_mask=mask,
    )
    first = refiner(coordinates=coordinates, **arguments)["coordinates"]
    translation = torch.tensor([[[2.0, -3.0, 0.5]]])
    second = refiner(
        coordinates=coordinates + translation, **arguments
    )["coordinates"]
    # The refiner centres every molecule, so translated inputs produce the
    # same centred output (translation invariance under the chosen gauge).
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)


def test_teacher_forced_smiles_loss_and_greedy_curriculum_schedule():
    module = _module(
        use_smiles_decoder=True,
        greedy_probability_start=0.0,
        greedy_probability_end=1.0,
        teacher_only_steps=2,
        greedy_transition_steps=4,
        greedy_schedule="linear",
    )
    assert module._greedy_probability_for_elapsed(0) == 0.0
    assert module._greedy_probability_for_elapsed(2) == 0.0
    assert module._greedy_probability_for_elapsed(4) == 0.5
    assert module._greedy_probability_for_elapsed(6) == 1.0

    batch = _batch()
    output = module(batch, teacher_force_smiles=True)
    losses = module._losses(batch, output, include_smiles_loss=True)
    assert torch.isfinite(losses["loss_smiles"])
    assert float(losses["loss_smiles"]) > 0.0
