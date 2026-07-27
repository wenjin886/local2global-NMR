import unittest

import torch

from local2geo_module.data import collate_local2geo, graph_from_smiles
from local2geo_module.geometry_solver import DifferentiableGeometrySolver
from local2geo_module.seed_generator import SoftDistanceStressSeed
from local2geo_module.soft_graph_simulator import SoftGraphSimulator
from local2geo_module.topology_prior import SoftTopologyPrior


def oracle_priors(batch):
    return {
        "one_three_probability": batch["one_three_targets"],
        "one_four_probability": batch["one_four_targets"],
        "one_three_distance_ratio": batch[
            "one_three_log_ratio"
        ].exp(),
        "one_four_distance_ratio": batch[
            "one_four_log_ratio"
        ].exp(),
        "one_four_validity": (
            batch["heavy_mask"][:, :, None]
            & batch["heavy_mask"][:, None, :]
        ).float(),
    }


class SoftDistanceStressSeedTest(unittest.TestCase):
    def test_full_prior_seed_path_reaches_raw_graph_logits(self):
        batch = collate_local2geo([graph_from_smiles("CCCC")])
        raw = SoftGraphSimulator(corruption_boost=6.0)(
            batch, corrupted=True, seed=7
        )
        raw["heavy_edge_logits"].requires_grad_(True)
        raw["h_attachment_logits"].requires_grad_(True)
        prior = SoftTopologyPrior(hidden_dim=32, num_layers=1)
        learned = prior(
            atomic_numbers=batch["atomic_numbers"],
            formal_charges=batch["formal_charges"],
            atom_mask=batch["atom_mask"],
            heavy_mask=batch["heavy_mask"],
            hydrogen_mask=batch["hydrogen_mask"],
            pair_mask=batch["pair_mask"],
            heavy_pair_mask=batch["heavy_pair_mask"],
            attachment_mask=batch["attachment_mask"],
            raw_heavy_edge_logits=raw["heavy_edge_logits"],
            raw_h_attachment_logits=raw["h_attachment_logits"],
        )
        solver = DifferentiableGeometrySolver(
            num_steps=1,
            seed_mode="soft_stress",
            soft_stress_steps=3,
        )
        output = solver(
            atomic_numbers=batch["atomic_numbers"],
            atom_mask=batch["atom_mask"],
            heavy_mask=batch["heavy_mask"],
            hydrogen_mask=batch["hydrogen_mask"],
            heavy_edge_logits=learned[
                "corrected_heavy_edge_logits"
            ],
            h_attachment_logits=learned[
                "corrected_h_attachment_logits"
            ],
            differentiable=True,
            geometry_probabilities_override=torch.softmax(
                learned["geometry_logits"], dim=-1
            ),
            local_geometry_priors={
                key: learned[key]
                for key in (
                    "one_three_probability",
                    "one_four_probability",
                    "one_three_distance_ratio",
                    "one_four_distance_ratio",
                )
            },
            coordinate_seed=3,
        )
        gradients = torch.autograd.grad(
            output["coordinates"].square().sum(),
            (
                raw["heavy_edge_logits"],
                raw["h_attachment_logits"],
            ),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_parameter_free_and_gradient_reaches_both_logits(self):
        batch = collate_local2geo([graph_from_smiles("CCCC")])
        raw = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=3)
        raw["heavy_edge_logits"].requires_grad_(True)
        raw["h_attachment_logits"].requires_grad_(True)
        graph_builder = DifferentiableGeometrySolver(num_steps=0)
        graph = graph_builder.soft_graph(
            raw["heavy_edge_logits"],
            raw["h_attachment_logits"],
            batch["pair_mask"],
            batch["heavy_pair_mask"],
            batch["attachment_mask"],
        )
        module = SoftDistanceStressSeed(num_steps=3)
        coordinates = module(
            atom_mask=batch["atom_mask"],
            heavy_mask=batch["heavy_mask"],
            probabilities=graph["edge_probabilities"],
            covalent_radii=batch["covalent_radii"],
            vdw_radii=batch["vdw_radii"],
            bond_length_scales=graph_builder.bond_length_scales,
            local_geometry_priors=oracle_priors(batch),
            differentiable=True,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(sum(p.numel() for p in module.parameters()), 0)
        gradients = torch.autograd.grad(
            coordinates.square().sum(),
            (
                raw["heavy_edge_logits"],
                raw["h_attachment_logits"],
            ),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_long_chain_is_more_extended_than_legacy_soft_seed(self):
        batch = collate_local2geo([
            graph_from_smiles("CCCCCCCCCCCCC")
        ])
        raw = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=3)
        geometry = torch.nn.functional.one_hot(
            batch["geometry_classes"].clamp_min(0), 7
        ).float()

        def seed(mode):
            return DifferentiableGeometrySolver(
                num_steps=0,
                seed_mode=mode,
                soft_stress_steps=96,
            )(
                batch["atomic_numbers"],
                batch["atom_mask"],
                batch["heavy_mask"],
                batch["hydrogen_mask"],
                raw["heavy_edge_logits"],
                raw["h_attachment_logits"],
                differentiable=False,
                geometry_probabilities_override=geometry,
                local_geometry_priors=(
                    oracle_priors(batch)
                    if mode == "soft_stress" else None
                ),
                coordinate_seed=3,
            )["seed_coordinates"][0]

        legacy = seed("differentiable")
        stress = seed("soft_stress")
        heavy = batch["heavy_mask"][0]
        legacy_diameter = torch.cdist(
            legacy[heavy], legacy[heavy]
        ).max()
        stress_diameter = torch.cdist(
            stress[heavy], stress[heavy]
        ).max()
        self.assertGreater(
            float(stress_diameter), 1.5 * float(legacy_diameter)
        )

    def test_hydrogens_are_placed_outside_the_ethane_skeleton(self):
        batch = collate_local2geo([graph_from_smiles("CC")])
        raw = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=3)
        solver = DifferentiableGeometrySolver(
            num_steps=0,
            seed_mode="soft_stress",
            soft_stress_steps=96,
        )
        coordinates = solver(
            batch["atomic_numbers"],
            batch["atom_mask"],
            batch["heavy_mask"],
            batch["hydrogen_mask"],
            raw["heavy_edge_logits"],
            raw["h_attachment_logits"],
            differentiable=False,
            local_geometry_priors=oracle_priors(batch),
            coordinate_seed=3,
        )["seed_coordinates"][0]
        heavy = batch["heavy_mask"][0]
        bond = batch["bond_types"][0].ne(0)
        projections = []
        for hydrogen in torch.where(batch["hydrogen_mask"][0])[0]:
            parent = int(batch["h_attachment"][0, hydrogen])
            heavy_neighbours = torch.where(
                bond[parent] & heavy
            )[0]
            outward = (
                coordinates[parent]
                - coordinates[heavy_neighbours]
            ).sum(dim=0)
            h_vector = coordinates[hydrogen] - coordinates[parent]
            projections.append(torch.dot(h_vector, outward))
        projections = torch.stack(projections)
        self.assertTrue(projections.gt(0.0).all())

    def test_coordinate_seed_is_reproducible(self):
        batch = collate_local2geo([graph_from_smiles("CCCCC")])
        raw = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=3)
        solver = DifferentiableGeometrySolver(
            num_steps=0,
            seed_mode="soft_stress",
            soft_stress_steps=4,
        )
        arguments = dict(
            atomic_numbers=batch["atomic_numbers"],
            atom_mask=batch["atom_mask"],
            heavy_mask=batch["heavy_mask"],
            hydrogen_mask=batch["hydrogen_mask"],
            heavy_edge_logits=raw["heavy_edge_logits"],
            h_attachment_logits=raw["h_attachment_logits"],
            differentiable=False,
            local_geometry_priors=oracle_priors(batch),
            coordinate_seed=19,
        )
        first = solver(**arguments)["seed_coordinates"]
        second = solver(**arguments)["seed_coordinates"]
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
