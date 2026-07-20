import torch

from src.data.constants import BOND_TYPE_CANDIDATES
from src.data.dataset import GraphSample, collate_nmr_graph
from src.model.loss import NMRGraphLoss
from src.model.nmr_to_graph import NMRToGraph


def fragment(**counts):
    target = torch.zeros(len(BOND_TYPE_CANDIDATES), dtype=torch.long)
    for candidate, count in counts.items():
        target[BOND_TYPE_CANDIDATES.index(candidate.replace("_", "-"))] = count
    return target


def make_sample():
    # Element-sorted explicit-H methanol: H,H,H,H,C,O.
    atom_types = torch.tensor([1, 1, 1, 1, 6, 8])
    bond_types = torch.zeros((6, 6), dtype=torch.long)
    bond_types[4, 5] = bond_types[5, 4] = 1

    carbon_fragment = fragment(**{"1_1": 3, "8_1": 1})
    oxygen_fragment = fragment(**{"1_1": 1, "6_1": 1})
    heavy_fragments = torch.full(
        (6, len(BOND_TYPE_CANDIDATES)), -100, dtype=torch.long
    )
    heavy_fragments[4] = carbon_fragment
    heavy_fragments[5] = oxygen_fragment
    h_parent_fragments = torch.full_like(heavy_fragments, -100)
    h_parent_fragments[:3] = carbon_fragment
    h_parent_fragments[3] = oxygen_fragment

    return GraphSample(
        h=atom_types,
        h_nmr=torch.tensor([3.2, 4.7]),
        c_nmr=torch.tensor([50.0]),
        h_nmr_integration=torch.tensor([3.0, 1.0]),
        h_nmr_integration_mask=torch.tensor([True, True]),
        h_nmr_multiplicity=torch.tensor([3, 10]),
        h_nmr_multiplicity_mask=torch.tensor([True, True]),
        h_nmr_j=torch.tensor([[0.0, 0.0], [7.0, 2.0]]),
        h_nmr_j_mask=torch.tensor([[False, False], [True, True]]),
        bond_types=bond_types,
        h_attachment=torch.tensor([4, 4, 4, 5, -100, -100]),
        heavy_fragment_labels=heavy_fragments,
        h_parent_fragment_labels=h_parent_fragments,
        h_parent_types=torch.tensor([6, 6, 6, 8, -100, -100]),
        smiles="CO",
        canonical_smiles="CO",
        isomeric_smiles="CO",
        smiles_token_ids=torch.tensor([4, 5]),
    )


def make_model(**kwargs):
    return NMRToGraph(
        hidden_dim=32,
        num_heads=4,
        num_spectrum_layers=1,
        num_atom_spectrum_layers=1,
        num_atom_interaction_layers=1,
        num_fourier_features=16,
        max_num_atoms=16,
        attachment_dim=16,
        **kwargs,
    )


def test_forward_masks_fragments_and_probabilities():
    batch = collate_nmr_graph([make_sample(), make_sample()])
    model = make_model()
    outputs = model(**batch.model_inputs())

    assert outputs["fragment_logits"].shape == (
        2, 6, len(BOND_TYPE_CANDIDATES), 5
    )
    assert outputs["h_parent_fragment_logits"].shape == (
        2, 6, len(BOND_TYPE_CANDIDATES), 5
    )
    assert outputs["heavy_edge_logits"].shape == (2, 6, 6, 5)
    assert outputs["h_peak_features"].shape == (2, 2, 32)
    assert outputs["c_peak_features"].shape == (2, 1, 32)
    assert torch.allclose(
        outputs["heavy_edge_logits"],
        outputs["heavy_edge_logits"].transpose(1, 2),
        atol=1e-6,
    )
    assert outputs["heavy_edge_mask"].sum().item() == 4
    row_sums = outputs["h_attachment_probabilities"].sum(dim=-1)
    assert torch.allclose(row_sums[outputs["hydrogen_mask"]], torch.ones(8))
    assert torch.all(row_sums[outputs["heavy_mask"]] == 0)


def test_full_graph_loss_backpropagates_without_valence_term():
    batch = collate_nmr_graph([make_sample(), make_sample()])
    model = make_model()
    criterion = NMRGraphLoss()
    outputs = model(**batch.model_inputs())
    loss, losses = criterion(
        outputs=outputs,
        atom_types=batch.atom_types,
        bond_types=batch.bond_types,
        h_attachment=batch.h_attachment,
        heavy_fragment_labels=batch.heavy_fragment_labels,
        h_parent_fragment_labels=batch.h_parent_fragment_labels,
        h_parent_types=batch.h_parent_types,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert "valence" not in losses
    assert model.fragment_readout.readout[-1].weight.grad is not None
    assert model.attachment_readout.hydrogen_projection.weight.grad is not None
    assert model.edge_readout.mlp[-1].weight.grad is not None


def test_fragment_only_stage_has_no_edge_or_attachment_gradient():
    batch = collate_nmr_graph([make_sample()])
    model = make_model()
    criterion = NMRGraphLoss(
        h_attachment_weight=0.0,
        h_count_weight=0.0,
        edge_weight=0.0,
        fragment_edge_consistency_weight=0.0,
    )
    outputs = model(**batch.model_inputs())
    loss, _ = criterion(
        outputs=outputs,
        atom_types=batch.atom_types,
        bond_types=batch.bond_types,
        h_attachment=batch.h_attachment,
        heavy_fragment_labels=batch.heavy_fragment_labels,
        h_parent_fragment_labels=batch.h_parent_fragment_labels,
        h_parent_types=batch.h_parent_types,
    )
    loss.backward()

    assert model.fragment_readout.readout[-1].weight.grad is not None
    edge_gradient = model.edge_readout.mlp[-1].weight.grad
    assert edge_gradient is None or torch.count_nonzero(edge_gradient) == 0


def test_smiles_teacher_forcing_loss_and_greedy_conditioning():
    batch = collate_nmr_graph([make_sample(), make_sample()])
    model = make_model(
        use_smiles_loss=True,
        use_smiles_conditioning=True,
        num_smiles_layers=1,
        max_smiles_length=32,
        smiles_vocab_size=6,
    )
    model.train()
    outputs = model(**batch.model_inputs())
    assert outputs["smiles_teacher_forced"] is True
    assert outputs["smiles_logits"].shape[:2] == batch.smiles_target_ids.shape
    criterion = NMRGraphLoss(smiles_weight=1.0)
    loss, losses = criterion(
        outputs=outputs,
        atom_types=batch.atom_types,
        bond_types=batch.bond_types,
        h_attachment=batch.h_attachment,
        heavy_fragment_labels=batch.heavy_fragment_labels,
        h_parent_fragment_labels=batch.h_parent_fragment_labels,
        h_parent_types=batch.h_parent_types,
        smiles_target_ids=batch.smiles_target_ids,
    )
    loss.backward()
    assert torch.isfinite(losses["smiles"])
    assert model.smiles_decoder.output_projection.weight.grad is not None

    model.eval()
    with torch.no_grad():
        generated = model(**batch.model_inputs())
    assert generated["smiles_teacher_forced"] is False
    assert generated["smiles_token_ids"].shape == batch.smiles_target_ids.shape
