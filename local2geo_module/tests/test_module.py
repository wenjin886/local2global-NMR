import tempfile
import unittest
from pathlib import Path

import torch

from local2geo_module.data import collate_local2geo, graph_from_smiles
from local2geo_module.eval import build_parser, evaluate_smiles
from local2geo_module.geometry_solver import DifferentiableGeometrySolver
from local2geo_module.soft_graph_simulator import SoftGraphSimulator


REAL_SMILES = [
    "COC(=O)Cc1c(C)oc2cc(N)ccc2c1=O",
    "C=C(CSCCCSc1ccc(C(=O)C(C)(C)N2CCOCC2)cc1)C(=O)OC",
    "O=C(CC(F)(F)F)NC[C@H]1CN(c2ccc3c(c2)CCCc2cn[nH]c2-3)C(=O)O1",
]


class ParameterFreeGeometrySolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = [graph_from_smiles(smiles) for smiles in REAL_SMILES]
        cls.batch = collate_local2geo(cls.samples)

    def _soft_graph(self, corrupted=False):
        return SoftGraphSimulator()(
            self.batch, corrupted=corrupted, seed=1729
        )

    def _solve(self, solver, soft_graph, differentiable):
        return solver(
            atomic_numbers=self.batch["atomic_numbers"],
            atom_mask=self.batch["atom_mask"],
            heavy_mask=self.batch["heavy_mask"],
            hydrogen_mask=self.batch["hydrogen_mask"],
            heavy_edge_logits=soft_graph["heavy_edge_logits"],
            h_attachment_logits=soft_graph["h_attachment_logits"],
            differentiable=differentiable,
        )

    def test_real_smiles_use_explicit_h_nmr_slot_order(self):
        self.assertEqual(
            self.batch["atom_mask"].sum(dim=-1).tolist(),
            [31, 60, 48],
        )
        for sample in self.samples:
            hydrogens = sample["num_hydrogens"]
            self.assertTrue(
                sample["atomic_numbers"][:hydrogens].eq(1).all()
            )
            self.assertTrue(
                sample["atomic_numbers"][hydrogens:].ne(1).all()
            )

    def test_simulator_is_separate_and_matches_nmr_graph_shapes(self):
        clean = self._soft_graph(corrupted=False)
        corrupted = SoftGraphSimulator(
            bond_type_confusion_probability=1.0,
            false_positive_probability=1.0,
            false_negative_probability=1.0,
            attachment_confusion_probability=1.0,
        )(self.batch, corrupted=True, seed=1729)
        expected_edge_shape = (*self.batch["bond_types"].shape, 5)
        self.assertTrue({
            "atom_types",
            "atom_mask",
            "heavy_mask",
            "hydrogen_mask",
            "heavy_edge_logits",
            "h_attachment_logits",
            "h_attachment_probabilities",
            "assigned_h_count",
        }.issubset(clean))
        self.assertEqual(
            clean["heavy_edge_logits"].shape, expected_edge_shape
        )
        self.assertEqual(
            clean["h_attachment_logits"].shape,
            self.batch["bond_types"].shape,
        )
        self.assertTrue(torch.isfinite(clean["heavy_edge_logits"]).all())
        self.assertTrue(torch.isfinite(clean["h_attachment_logits"]).all())
        self.assertTrue(torch.allclose(
            clean["heavy_edge_logits"],
            clean["heavy_edge_logits"].transpose(1, 2),
        ))
        self.assertFalse(torch.equal(
            clean["heavy_edge_logits"],
            corrupted["heavy_edge_logits"],
        ))

    def test_solver_is_parameter_free_fp32_and_differentiable(self):
        soft_graph = {
            key: self._soft_graph()[key].to(
                torch.bfloat16
            ).requires_grad_(True)
            for key in ("heavy_edge_logits", "h_attachment_logits")
        }
        solver = DifferentiableGeometrySolver(num_steps=2)
        self.assertEqual(sum(p.numel() for p in solver.parameters()), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = self._solve(
                solver, soft_graph, differentiable=True
            )
        self.assertEqual(outputs["coordinates"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(outputs["coordinates"]).all())
        loss = sum(outputs["geometry_terms"].values())
        gradients = torch.autograd.grad(
            loss,
            (
                soft_graph["heavy_edge_logits"],
                soft_graph["h_attachment_logits"],
            ),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_detached_mds_seed_keeps_relaxation_gradient_to_logits(self):
        soft_graph = {
            key: self._soft_graph()[key].clone().requires_grad_(True)
            for key in ("heavy_edge_logits", "h_attachment_logits")
        }
        outputs = self._solve(
            DifferentiableGeometrySolver(
                num_steps=2,
                seed_mode="mds",
            ),
            soft_graph,
            differentiable=True,
        )
        self.assertEqual(outputs["seed_mode"], "mds")
        self.assertFalse(outputs["seed_coordinates"].requires_grad)
        self.assertTrue(torch.isfinite(outputs["coordinates"]).all())
        gradients = torch.autograd.grad(
            outputs["coordinates"].square().sum(),
            (
                soft_graph["heavy_edge_logits"],
                soft_graph["h_attachment_logits"],
            ),
        )
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_mds_seed_is_more_extended_for_a_long_chain(self):
        batch = collate_local2geo([graph_from_smiles("CCCCCCCCCCCCC")])
        graph = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=1)

        def seed(mode):
            return DifferentiableGeometrySolver(
                num_steps=0,
                seed_mode=mode,
            )(
                batch["atomic_numbers"],
                batch["atom_mask"],
                batch["heavy_mask"],
                batch["hydrogen_mask"],
                graph["heavy_edge_logits"],
                graph["h_attachment_logits"],
                differentiable=False,
            )["seed_coordinates"]

        differentiable_seed = seed("differentiable")
        mds_seed = seed("mds")

        def diameter(coordinates):
            return torch.cdist(coordinates, coordinates).amax()

        self.assertGreater(
            float(diameter(mds_seed)),
            1.5 * float(diameter(differentiable_seed)),
        )

    def test_mds_seed_stress_restores_tetrahedral_chain_angles(self):
        batch = collate_local2geo([graph_from_smiles("CCCCCCCCCCCCC")])
        graph = SoftGraphSimulator(logit_noise_std=0.0)(batch, seed=1)
        outputs = DifferentiableGeometrySolver(
            num_steps=0,
            seed_mode="mds",
        )(
            batch["atomic_numbers"],
            batch["atom_mask"],
            batch["heavy_mask"],
            batch["hydrogen_mask"],
            graph["heavy_edge_logits"],
            graph["h_attachment_logits"],
            differentiable=False,
        )
        coordinates = outputs["seed_coordinates"][0]
        hard_types = outputs["seed_hard_bond_types"][0]
        atomic_numbers = batch["atomic_numbers"][0]
        angles = []
        for center in atomic_numbers.eq(6).nonzero().flatten():
            neighbors = (
                hard_types[center].gt(0) & atomic_numbers.eq(6)
            ).nonzero().flatten()
            if neighbors.numel() != 2:
                continue
            first = coordinates[neighbors[0]] - coordinates[center]
            second = coordinates[neighbors[1]] - coordinates[center]
            cosine = torch.dot(first, second) / (
                first.norm() * second.norm()
            )
            angles.append(torch.rad2deg(torch.acos(cosine.clamp(-1, 1))))
        mean_angle = torch.stack(angles).mean()
        self.assertLess(float(mean_angle), 115.0)
        self.assertGreater(float(mean_angle), 104.0)

    def test_eval_defaults_to_mds_seed(self):
        args = build_parser().parse_args(["--smiles", "CC"])
        self.assertEqual(args.seed_mode, "mds")
        self.assertEqual(args.mds_stress_steps, 384)

    def test_fixed_prior_solver_reduces_total_local_energy(self):
        solver = DifferentiableGeometrySolver(num_steps=64)
        outputs = self._solve(
            solver, self._soft_graph(), differentiable=False
        )
        seed_terms = solver.terms(
            outputs["seed_coordinates"],
            outputs["edge_probabilities"],
            outputs["geometry_probabilities"],
            self.batch["atom_mask"],
            outputs["pair_mask"],
            outputs["covalent_radii"],
            outputs["vdw_radii"],
        )
        self.assertTrue(torch.isfinite(outputs["coordinates"]).all())
        self.assertLess(
            float(solver.total(outputs["geometry_terms"])),
            float(solver.total(seed_terms)),
        )
        self.assertLess(
            float(outputs["geometry_terms"]["bond"]),
            float(seed_terms["bond"]),
        )
        self.assertLess(
            float(outputs["geometry_terms"]["angle"]),
            float(seed_terms["angle"]),
        )

    def test_one_three_pairs_are_excluded_from_generic_clash(self):
        solver = DifferentiableGeometrySolver(num_steps=0)
        # H(0)-C(1)-H(2): both C-H pairs are bonded, while H...H is a
        # one-three pair below the generic vdW cutoff. Its distance is governed
        # by the local VSEPR/angle term and must not be pushed apart by clash.
        positions = torch.tensor(
            [[[-0.80, 0.0, 0.0], [0.0, 0.0, 0.0], [0.80, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        probabilities = torch.zeros((1, 3, 3, 5), dtype=torch.float32)
        probabilities[..., 0] = 1.0
        probabilities[:, 0, 1, 0] = 0.0
        probabilities[:, 1, 0, 0] = 0.0
        probabilities[:, 1, 2, 0] = 0.0
        probabilities[:, 2, 1, 0] = 0.0
        probabilities[:, 0, 1, 1] = 1.0
        probabilities[:, 1, 0, 1] = 1.0
        probabilities[:, 1, 2, 1] = 1.0
        probabilities[:, 2, 1, 1] = 1.0
        atom_mask = torch.ones((1, 3), dtype=torch.bool)
        pair_mask = ~torch.eye(3, dtype=torch.bool)[None]
        geometry_probabilities = torch.zeros((1, 3, 7))
        geometry_probabilities[..., 0] = 1.0
        covalent_radii = torch.tensor([[0.31, 0.76, 0.31]])
        vdw_radii = torch.tensor([[1.20, 1.70, 1.20]])

        local_terms = solver.terms(
            positions,
            probabilities,
            geometry_probabilities,
            atom_mask,
            pair_mask,
            covalent_radii,
            vdw_radii,
        )
        self.assertLess(float(local_terms["clash"]), 1e-8)

        # With the two bonds removed, the same close coordinates are ordinary
        # nonlocal unbonded pairs and must still receive clash repulsion.
        unbonded_probabilities = torch.zeros_like(probabilities)
        unbonded_probabilities[..., 0] = 1.0
        nonlocal_terms = solver.terms(
            positions,
            unbonded_probabilities,
            geometry_probabilities,
            atom_mask,
            pair_mask,
            covalent_radii,
            vdw_radii,
        )
        self.assertGreater(float(nonlocal_terms["clash"]), 1e-3)

    def test_batch_members_do_not_change_each_others_coordinates(self):
        one = collate_local2geo([self.samples[0]])
        two = collate_local2geo([self.samples[0], self.samples[0]])
        simulator = SoftGraphSimulator(logit_noise_std=0.0)
        solver = DifferentiableGeometrySolver(num_steps=2)

        def solve(batch):
            graph = simulator(batch, seed=1)
            return solver(
                batch["atomic_numbers"],
                batch["atom_mask"],
                batch["heavy_mask"],
                batch["hydrogen_mask"],
                graph["heavy_edge_logits"],
                graph["h_attachment_logits"],
                differentiable=False,
            )["coordinates"]

        single_coordinates = solve(one)
        double_coordinates = solve(two)
        self.assertTrue(torch.allclose(
            single_coordinates[0], double_coordinates[0], atol=1e-4
        ))
        self.assertTrue(torch.allclose(
            double_coordinates[0], double_coordinates[1], atol=1e-4
        ))

    def test_eval_writes_explicit_h_xyz_and_sdf(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = evaluate_smiles(
                smiles=[REAL_SMILES[0]],
                solver=DifferentiableGeometrySolver(num_steps=2),
                simulator=SoftGraphSimulator(logit_noise_std=0.0),
                output=None,
                output_dir=Path(directory),
                corrupted=False,
                seed=1,
                device=torch.device("cpu"),
                write_sdf_files=True,
            )
            xyz = paths[0]
            self.assertTrue(xyz.is_file())
            self.assertEqual(int(xyz.read_text().splitlines()[0]), 31)
            self.assertTrue(xyz.with_suffix(".sdf").is_file())


if __name__ == "__main__":
    unittest.main()
