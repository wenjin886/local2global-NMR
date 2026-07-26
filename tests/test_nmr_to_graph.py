import json

import torch
import torch.nn.functional as F

from src.data.constants import BOND_TYPE_CANDIDATES, SMILES_PAD_INDEX
from src.data.dataset import GraphSample, TransformingCollator, collate_nmr_graph
from src.data.transforms import NormalizeNMR
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
        isomeric_smiles="CO",
        smiles_token_ids=torch.tensor([4, 5]),
    )


def make_model(**kwargs):
    return NMRToGraph(
        hidden_dim=32,
        num_heads=4,
        num_joint_layers=1,
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
    assert outputs["joint_features"].shape == (2, 9, 32)
    assert outputs["attention"]["joint"].shape[-2:] == (9, 9)
    assert outputs["attention"]["heavy_query_to_joint"].shape[-2:] == (6, 9)
    assert torch.allclose(
        outputs["heavy_edge_logits"],
        outputs["heavy_edge_logits"].transpose(1, 2),
        atol=1e-6,
    )
    assert outputs["heavy_edge_mask"].sum().item() == 4
    row_sums = outputs["h_attachment_probabilities"].sum(dim=-1)
    assert torch.allclose(row_sums[outputs["hydrogen_mask"]], torch.ones(8))
    assert torch.all(row_sums[outputs["heavy_mask"]] == 0)


def test_collator_expands_compact_dtypes_and_normalizes_batch(tmp_path):
    sample = make_sample()
    sample.h = sample.h.to(torch.uint8)
    sample.bond_types = sample.bond_types.to(torch.uint8)
    sample.h_attachment = sample.h_attachment.to(torch.int16)
    sample.heavy_fragment_labels = sample.heavy_fragment_labels.to(torch.int8)
    sample.h_parent_fragment_labels = sample.h_parent_fragment_labels.to(torch.int8)
    sample.h_parent_types = sample.h_parent_types.to(torch.int8)
    sample.h_nmr_multiplicity = sample.h_nmr_multiplicity.to(torch.int16)
    sample.smiles_token_ids = sample.smiles_token_ids.to(torch.int16)

    stats = {
        "hnmr_shift_mean": 4.0,
        "hnmr_shift_std": 2.0,
        "cnmr_shift_mean": 40.0,
        "cnmr_shift_std": 10.0,
        "hnmr_integration_mean": 2.0,
        "hnmr_integration_std": 1.0,
        "hnmr_j_mean": 5.0,
        "hnmr_j_std": 2.0,
    }
    path = tmp_path / "dataset_infos.json"
    path.write_text(json.dumps(stats))
    normalizer = NormalizeNMR(
        str(path), encode_multiplicity=False, encode_smiles=False
    )
    batch = TransformingCollator(normalizer)([sample])

    assert batch.atom_types.dtype == torch.long
    assert batch.bond_types.dtype == torch.long
    assert batch.heavy_fragment_labels.dtype == torch.long
    assert batch.smiles_target_ids.dtype == torch.long
    assert torch.allclose(batch.h_nmr, torch.tensor([[-0.4, 0.35]]))
    assert torch.allclose(batch.c_nmr, torch.tensor([[1.0]]))


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
    model = make_model(predict_attachments=False, predict_edges=False)
    criterion = NMRGraphLoss(
        h_attachment_weight=0.0,
        h_count_weight=0.0,
        edge_weight=0.0,
        fragment_edge_consistency_weight=0.0,
    )
    outputs = model(**batch.model_inputs())
    assert outputs["h_attachment_logits"] is None
    assert outputs["heavy_edge_logits"] is None
    assert outputs["graph_atom_features"] is outputs["atom_features"]
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


def test_heavy_neighbor_count_loss_penalizes_only_neighbor_overflow():
    criterion = NMRGraphLoss(
        heavy_neighbor_count_weight=1.0,
        max_heavy_neighbor_counts={6: 4, 7: 4},
    )
    atom_types = torch.tensor([[6, 7]])
    heavy_mask = torch.tensor([[True, True]])
    logits = torch.full((1, 2, len(BOND_TYPE_CANDIDATES), 5), -20.0)
    logits[..., 0] = 20.0
    assert criterion.heavy_neighbor_count_overflow_loss(
        logits, atom_types, heavy_mask
    ).item() < 1e-8

    # Carbon predicts count=4 for two independent neighbor/bond categories:
    # eight predicted neighbors exceed its dataset-observed cap of four.
    logits[0, 0, 0, 0] = -20.0
    logits[0, 0, 0, 4] = 20.0
    logits[0, 0, 1, 0] = -20.0
    logits[0, 0, 1, 4] = 20.0
    assert criterion.heavy_neighbor_count_overflow_loss(
        logits, atom_types, heavy_mask
    ).item() > 0


def test_edge_total_neighbor_count_includes_soft_edges_and_h_attachments():
    atom_types = torch.tensor([[1, 6, 8]])
    heavy_mask = torch.tensor([[False, True, True]])
    edge_mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    edge_mask[0, 1, 2] = edge_mask[0, 2, 1] = True
    edge_logits = torch.full((1, 3, 3, 5), -2.0)
    edge_logits[..., 0] = 2.0
    edge_logits[0, 1, 2, 0] = edge_logits[0, 2, 1, 0] = -2.0
    edge_logits[0, 1, 2, 1] = edge_logits[0, 2, 1, 1] = 2.0
    edge_logits.requires_grad_()
    attachment_probabilities = torch.zeros((1, 3, 3))
    attachment_probabilities[0, 0, 1] = 1.0
    outputs = {
        "fragment_logits": torch.zeros(
            1, 3, len(BOND_TYPE_CANDIDATES), 5
        ),
        "heavy_edge_logits": edge_logits,
        "heavy_edge_mask": edge_mask,
        "h_attachment_probabilities": attachment_probabilities,
        "heavy_mask": heavy_mask,
    }

    permissive = NMRGraphLoss()
    assert permissive.edge_total_neighbor_count_overflow_loss(
        outputs, atom_types
    ).item() == 0.0

    constrained = NMRGraphLoss(
        max_heavy_neighbor_counts={6: 1, 8: 1}
    )
    loss = constrained.edge_total_neighbor_count_overflow_loss(
        outputs, atom_types
    )
    assert loss.item() > 0.0
    loss.backward()
    assert torch.count_nonzero(edge_logits.grad) > 0


def test_edge_loss_uses_shared_none_and_bond_class_weights():
    logits = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]],
        requires_grad=True,
    )
    edge_mask = torch.ones((1, 3, 3), dtype=torch.bool)
    targets = torch.tensor([[
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]])
    outputs = {
        "heavy_edge_logits": logits,
        "heavy_edge_mask": edge_mask,
    }
    criterion = NMRGraphLoss(
        edge_none_class_weight=0.2,
        edge_bond_class_weight=1.0,
    )

    actual = criterion.edge_loss(outputs, targets)
    upper_triangle_logits = torch.stack((logits[0, 0, 1], logits[0, 0, 2],
                                         logits[0, 1, 2]))
    upper_triangle_targets = torch.tensor([0, 0, 1])
    expected = F.cross_entropy(
        upper_triangle_logits,
        upper_triangle_targets,
        weight=torch.tensor([0.2, 1.0, 1.0]),
    )

    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad) > 0


def test_legacy_edge_class_weight_list_remains_supported():
    criterion = NMRGraphLoss(edge_class_weights=[0.1, 1.0, 1.0])
    assert torch.allclose(
        criterion.edge_class_weights, torch.tensor([0.1, 1.0, 1.0])
    )
    assert "edge_class_weights" in criterion.state_dict()

    scalar_criterion = NMRGraphLoss(
        edge_none_class_weight=0.2,
        edge_bond_class_weight=1.0,
    )
    assert "edge_class_weights" not in scalar_criterion.state_dict()


def test_legacy_degree_aliases_are_checkpoint_safe():
    legacy_config_criterion = NMRGraphLoss(
        heavy_degree_weight=0.25,
        max_heavy_degrees={6: 4, 8: 2},
    )
    assert legacy_config_criterion.heavy_neighbor_count_weight == 0.25
    caps = legacy_config_criterion.heavy_neighbor_count_caps(
        torch.tensor([6, 8])
    )
    assert caps.tolist() == [4.0, 2.0]
    assert "max_neighbor_count_lookup" not in legacy_config_criterion.state_dict()

    # A state dict created before the lookup existed can still load strictly.
    resumed_criterion = NMRGraphLoss(
        heavy_neighbor_count_weight=0.0,
    )
    resumed_criterion.load_state_dict(
        legacy_config_criterion.state_dict(), strict=True
    )


def test_refined_smiles_memory_backpropagates_into_atomic_refinement():
    batch = collate_nmr_graph([make_sample(), make_sample()])
    model = make_model(
        use_smiles_loss=True,
        use_smiles_conditioning=False,
        smiles_memory="refined_atom_nmr",
        num_smiles_layers=1,
        max_smiles_length=32,
        smiles_vocab_size=6,
        predict_attachments=False,
        predict_edges=False,
    )
    outputs = model(**batch.model_inputs())
    smiles_loss = F.cross_entropy(
        outputs["smiles_logits"].reshape(-1, 6),
        batch.smiles_target_ids.reshape(-1),
        ignore_index=SMILES_PAD_INDEX,
    )
    smiles_loss.backward()

    assert outputs["smiles_memory"] == "refined_atom_nmr"
    assert model.heavy_query_decoder.q_projection.weight.grad is not None
    interaction_gradient = (
        model.atom_interaction_layers[0]
        .hydrogen_reads_heavy.q_projection.weight.grad
    )
    assert interaction_gradient is not None
    assert torch.count_nonzero(interaction_gradient) > 0


def test_smiles_teacher_forcing_loss_and_greedy_generation():
    batch = collate_nmr_graph([make_sample(), make_sample()])
    model = make_model(
        use_smiles_loss=True,
        use_smiles_conditioning=False,
        num_smiles_layers=1,
        max_smiles_length=32,
        smiles_vocab_size=6,
    )
    model.train()
    outputs = model(**batch.model_inputs())
    assert outputs["smiles_teacher_forced"] is True
    assert outputs["smiles_logits"].shape[:2] == batch.smiles_target_ids.shape
    assert outputs["attention"]["atom_to_smiles"] is None
    assert torch.count_nonzero(
        model.smiles_decoder.token_embedding.weight[SMILES_PAD_INDEX]
    ) == 0
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
