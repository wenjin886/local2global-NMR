import tempfile
import unittest
from pathlib import Path

import torch

from local2geo_module.corruption import SoftGraphCorruptor
from local2geo_module.data import Local2GeoDataset, collate_local2geo
from local2geo_module.eval import clean_attachment_logits, clean_edge_logits
from local2geo_module.geometry import DifferentiableLocalRelaxation
from local2geo_module.loss import ProjectionGeometryLoss
from local2geo_module.model import (
    LearnedCoordinateSeed,
    Local2GeoModel,
    SoftGraphProjector,
)
from local2geo_module.visualization import graph_image, write_sdf


class Local2GeoModuleTest(unittest.TestCase):
    def _batch(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "samples.pt"
        torch.save([
            {"isomeric_smiles": "CCO"},
            {"isomeric_smiles": "c1ccccc1"},
        ], path)
        dataset = Local2GeoDataset(str(path))
        return collate_local2geo([dataset[0], dataset[1]])

    def test_dataset_and_corruption_are_batched_and_symmetric(self):
        batch = self._batch()
        self.assertEqual(
            int(batch["atomic_numbers"][0, batch["atom_mask"][0]].numel()),
            9,
        )
        noisy = SoftGraphCorruptor()(
            batch["bond_types"],
            batch["heavy_pair_mask"],
            batch["h_attachment"],
            batch["attachment_mask"],
        )
        logits = noisy["heavy_edge_logits"]
        self.assertEqual(logits.shape[:3], batch["bond_types"].shape)
        self.assertTrue(torch.allclose(logits, logits.transpose(1, 2)))
        self.assertTrue(torch.equal(batch["pair_mask"], batch["pair_mask"].transpose(1, 2)))
        attachment = noisy["h_attachment_logits"]
        self.assertEqual(attachment.shape, batch["bond_types"].shape)
        self.assertTrue(torch.isfinite(attachment).all())
        self.assertTrue(batch["hydrogen_mask"].any())

    def test_local_losses_reach_input_logits_with_finite_gradient(self):
        batch = self._batch()
        noisy = SoftGraphCorruptor()(
            batch["bond_types"],
            batch["heavy_pair_mask"],
            batch["h_attachment"],
            batch["attachment_mask"],
        )
        noisy["heavy_edge_logits"].requires_grad_(True)
        noisy["h_attachment_logits"].requires_grad_(True)
        model = Local2GeoModel(
            projector=SoftGraphProjector(
                hidden_dim=16, pair_hidden_dim=16, num_layers=1
            ),
            coordinate_seed=LearnedCoordinateSeed(
                hidden_dim=16, max_num_atoms=64
            ),
            relaxation=DifferentiableLocalRelaxation(num_steps=2),
        )
        outputs = model(batch, noisy, differentiable_relaxation=True)
        clean_terms = model.clean_geometry_terms(batch, outputs["coordinates"])
        loss, _ = ProjectionGeometryLoss()(outputs, batch, clean_terms)
        loss.backward()

        self.assertEqual(outputs["coordinates"].shape, (*batch["atom_mask"].shape, 3))
        self.assertTrue(torch.isfinite(outputs["coordinates"]).all())
        self.assertIsNotNone(noisy["heavy_edge_logits"].grad)
        self.assertTrue(torch.isfinite(noisy["heavy_edge_logits"].grad).all())
        self.assertGreater(
            float(noisy["heavy_edge_logits"].grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(noisy["h_attachment_logits"].grad)
        self.assertTrue(
            torch.isfinite(noisy["h_attachment_logits"].grad).all()
        )
        self.assertGreater(
            float(noisy["h_attachment_logits"].grad.abs().sum()), 0.0
        )

    def test_clean_eval_logits_follow_graph_and_masks(self):
        batch = self._batch()
        logits = clean_edge_logits(
            batch["bond_types"], batch["heavy_pair_mask"], margin=4.0
        )
        predicted = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(
            predicted[batch["heavy_pair_mask"]],
            batch["bond_types"][batch["heavy_pair_mask"]],
        ))
        self.assertTrue(torch.equal(
            predicted[~batch["heavy_pair_mask"]],
            torch.zeros_like(predicted[~batch["heavy_pair_mask"]]),
        ))
        self.assertTrue(torch.allclose(logits, logits.transpose(1, 2)))
        attachment = clean_attachment_logits(
            batch["h_attachment"], batch["attachment_mask"], margin=4.0
        )
        h_rows = batch["hydrogen_mask"]
        self.assertTrue(torch.equal(
            attachment[h_rows].argmax(dim=-1),
            batch["h_attachment"][h_rows],
        ))

    def test_relaxation_is_invariant_to_repeated_batch_members(self):
        sample = self._batch()
        sample = {
            key: (
                value[:1]
                if isinstance(value, torch.Tensor)
                else value[:1]
            )
            for key, value in sample.items()
        }
        repeated = {
            key: (
                value.repeat(2, *([1] * (value.ndim - 1)))
                if isinstance(value, torch.Tensor)
                else value * 2
            )
            for key, value in sample.items()
        }
        relaxation = DifferentiableLocalRelaxation(num_steps=2)

        def run(batch):
            probabilities = torch.nn.functional.one_hot(
                batch["bond_types"].clamp_min(0), num_classes=5
            ).float()
            geometry = torch.nn.functional.one_hot(
                batch["geometry_classes"].clamp_min(0), num_classes=7
            ).float()
            coordinates, _ = relaxation(
                probabilities,
                geometry,
                batch["atom_mask"],
                batch["pair_mask"],
                batch["covalent_radii"],
                batch["vdw_radii"],
                differentiable=False,
            )
            return coordinates

        single = run(sample)
        double = run(repeated)
        self.assertTrue(torch.allclose(single[0], double[0], atol=1e-6))
        self.assertTrue(torch.allclose(double[0], double[1], atol=1e-6))

    def test_visualization_writes_explicit_h_sdf_and_2d_graph(self):
        batch = self._batch()
        size = int(batch["atom_mask"][0].sum())
        atomic_numbers = batch["atomic_numbers"][0, :size]
        charges = batch["formal_charges"][0, :size]
        bonds = batch["bond_types"][0, :size, :size]
        image = graph_image(atomic_numbers, charges, bonds)
        self.assertEqual(image.size, (700, 500))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ethanol.sdf"
            coordinates = torch.randn(size, 3)
            write_sdf(
                path,
                atomic_numbers,
                charges,
                bonds,
                coordinates,
                "ethanol",
            )
            self.assertTrue(path.is_file())
            self.assertIn(" H ", path.read_text())


if __name__ == "__main__":
    unittest.main()
