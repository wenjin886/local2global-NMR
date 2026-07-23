import tempfile
import unittest
from pathlib import Path

import torch

from local2geo_module.corruption import SoftGraphCorruptor
from local2geo_module.data import Local2GeoDataset, collate_local2geo
from local2geo_module.eval import clean_edge_logits
from local2geo_module.geometry import DifferentiableLocalRelaxation
from local2geo_module.loss import ProjectionGeometryLoss
from local2geo_module.model import (
    LearnedCoordinateSeed,
    Local2GeoModel,
    SoftGraphProjector,
)


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
        logits = SoftGraphCorruptor()(batch["bond_types"], batch["pair_mask"])
        self.assertEqual(logits.shape[:3], batch["bond_types"].shape)
        self.assertTrue(torch.allclose(logits, logits.transpose(1, 2)))
        self.assertTrue(torch.equal(batch["pair_mask"], batch["pair_mask"].transpose(1, 2)))

    def test_local_losses_reach_input_logits_with_finite_gradient(self):
        batch = self._batch()
        noisy = SoftGraphCorruptor()(
            batch["bond_types"], batch["pair_mask"]
        ).requires_grad_(True)
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
        self.assertIsNotNone(noisy.grad)
        self.assertTrue(torch.isfinite(noisy.grad).all())
        self.assertGreater(float(noisy.grad.abs().sum()), 0.0)

    def test_clean_eval_logits_follow_graph_and_masks(self):
        batch = self._batch()
        logits = clean_edge_logits(
            batch["bond_types"], batch["pair_mask"], margin=4.0
        )
        predicted = logits.argmax(dim=-1)
        self.assertTrue(torch.equal(
            predicted[batch["pair_mask"]],
            batch["bond_types"][batch["pair_mask"]],
        ))
        self.assertTrue(torch.equal(
            predicted[~batch["pair_mask"]],
            torch.zeros_like(predicted[~batch["pair_mask"]]),
        ))
        self.assertTrue(torch.allclose(logits, logits.transpose(1, 2)))


if __name__ == "__main__":
    unittest.main()
