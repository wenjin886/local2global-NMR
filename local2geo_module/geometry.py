from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .constants import (
    BOND_LENGTH_SCALES,
    GEOMETRY_COSINES,
    NONE,
    PLANAR_GEOMETRY_INDEX,
)


class DifferentiableLocalRelaxation(nn.Module):
    """Batched O(BN^2) local-prior relaxation with unrolled gradients."""

    def __init__(
        self,
        num_steps: int = 8,
        step_size: float = 0.04,
        bond_weight: float = 40.0,
        angle_weight: float = 8.0,
        planar_weight: float = 1.0,
        clash_weight: float = 0.02,
        bond_probability_power: float = 4.0,
        angle_probability_power: float = 3.0,
        clash_distance_scale: float = 0.62,
        clash_softness: float = 0.12,
        gradient_clip: float = 10.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.planar_weight = planar_weight
        self.clash_weight = clash_weight
        self.bond_probability_power = bond_probability_power
        self.angle_probability_power = angle_probability_power
        self.clash_distance_scale = clash_distance_scale
        self.clash_softness = clash_softness
        self.gradient_clip = gradient_clip
        self.register_buffer(
            "bond_length_scales", torch.tensor(BOND_LENGTH_SCALES)
        )
        self.register_buffer(
            "geometry_cosines", torch.tensor(GEOMETRY_COSINES)
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    @staticmethod
    def _seed(atom_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch, atoms = atom_mask.shape
        index = torch.arange(atoms, device=atom_mask.device, dtype=dtype) + 1.0
        golden = torch.pi * (3.0 - 5.0 ** 0.5)
        phase = index * golden
        z = 1.0 - 2.0 * (index - 0.5) / max(atoms, 1)
        radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        seed = torch.stack([
            radius * torch.cos(phase), radius * torch.sin(phase), z
        ], dim=-1)
        seed = 1.5 * seed.unsqueeze(0).expand(batch, -1, -1).clone()
        seed = seed * atom_mask.unsqueeze(-1)
        count = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
        seed = seed - seed.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        return seed * atom_mask.unsqueeze(-1)

    def target_lengths(
        self,
        probabilities: torch.Tensor,
        covalent_radii: torch.Tensor,
    ) -> torch.Tensor:
        bonded = probabilities[..., 1:]
        conditional = bonded / bonded.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (conditional * self.bond_length_scales[1:]).sum(dim=-1)
        return (covalent_radii[:, :, None] + covalent_radii[:, None, :]) * scale

    def terms(
        self,
        positions: torch.Tensor,
        probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dtype = positions.dtype
        pair_mask_f = pair_mask.to(dtype)
        vector = positions[:, None, :, :] - positions[:, :, None, :]
        distance = torch.sqrt(vector.square().sum(dim=-1) + 1e-8)
        unit = vector / distance.unsqueeze(-1)
        q = (1.0 - probabilities[..., NONE]) * pair_mask_f
        target = self.target_lengths(probabilities, covalent_radii).clamp_min(1e-4)

        upper = torch.triu(torch.ones_like(pair_mask_f), diagonal=1)
        bond_weights = q.pow(self.bond_probability_power) * upper
        bond_values = torch.log(distance.clamp_min(1e-4) / target).square()
        bond = self._masked_mean(bond_values, bond_weights)

        angle_weights = q.pow(self.angle_probability_power)
        # M_i = sum_j w_ij u_ij u_ij^T and s_i = sum_j w_ij u_ij.
        moment = torch.einsum("bij,bija,bijc->biac", angle_weights, unit, unit)
        direction_sum = torch.einsum("bij,bija->bia", angle_weights, unit)
        total_weight = angle_weights.sum(dim=-1)
        squared_weight = angle_weights.square().sum(dim=-1)
        target_cosine = (
            geometry_probabilities * self.geometry_cosines
        ).sum(dim=-1)
        moment_norm_sq = moment.square().sum(dim=(-1, -2))
        direction_norm_sq = direction_sum.square().sum(dim=-1)
        pair_weight = (total_weight.square() - squared_weight).clamp_min(0.0)
        angle_numerator = (
            moment_norm_sq
            - 2.0 * target_cosine * direction_norm_sq
            + target_cosine.square() * total_weight.square()
            - squared_weight * (1.0 - target_cosine).square()
        ).clamp_min(0.0)
        angle_per_atom = angle_numerator / pair_weight.clamp_min(1e-8)
        angle_valid = atom_mask & pair_weight.gt(1e-6)
        angle = self._masked_mean(
            angle_per_atom, angle_valid.to(dtype)
        )

        normalized_moment = moment / total_weight.clamp_min(1e-8)[..., None, None]
        # Expanded 3x3 determinant avoids linalg.det's inverse-based higher
        # derivatives, which are undefined for the intentionally rank-2
        # matrices of planar neighborhoods.
        determinant = (
            normalized_moment[..., 0, 0]
            * (
                normalized_moment[..., 1, 1] * normalized_moment[..., 2, 2]
                - normalized_moment[..., 1, 2] * normalized_moment[..., 2, 1]
            )
            - normalized_moment[..., 0, 1]
            * (
                normalized_moment[..., 1, 0] * normalized_moment[..., 2, 2]
                - normalized_moment[..., 1, 2] * normalized_moment[..., 2, 0]
            )
            + normalized_moment[..., 0, 2]
            * (
                normalized_moment[..., 1, 0] * normalized_moment[..., 2, 1]
                - normalized_moment[..., 1, 1] * normalized_moment[..., 2, 0]
            )
        )
        planar_probability = geometry_probabilities[..., PLANAR_GEOMETRY_INDEX]
        planar_weights = planar_probability * atom_mask.to(dtype) * pair_weight.gt(1e-6)
        planar = self._masked_mean(determinant.square(), planar_weights)

        nonbond_weights = (1.0 - q).pow(2) * pair_mask_f * upper
        minimum = self.clash_distance_scale * (
            vdw_radii[:, :, None] + vdw_radii[:, None, :]
        )
        clash_values = F.softplus(
            (minimum - distance) / self.clash_softness
        ).square()
        clash = self._masked_mean(clash_values, nonbond_weights)
        return {
            "bond": bond,
            "angle": angle,
            "planar": planar,
            "clash": clash,
        }

    def total(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.bond_weight * terms["bond"]
            + self.angle_weight * terms["angle"]
            + self.planar_weight * terms["planar"]
            + self.clash_weight * terms["clash"]
        )

    def forward(
        self,
        probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        initial_positions: torch.Tensor = None,
        differentiable: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.enable_grad():
            positions = (
                self._seed(atom_mask, probabilities.dtype)
                if initial_positions is None else initial_positions
            )
            if not positions.requires_grad:
                positions = positions.requires_grad_(True)
            for _ in range(self.num_steps):
                terms = self.terms(
                    positions,
                    probabilities,
                    geometry_probabilities,
                    atom_mask,
                    pair_mask,
                    covalent_radii,
                    vdw_radii,
                )
                gradient = torch.autograd.grad(
                    self.total(terms),
                    positions,
                    create_graph=differentiable,
                )[0]
                # The epsilon is required for finite second derivatives on
                # padded atoms whose force is exactly zero.
                norm = torch.sqrt(
                    gradient.square().sum(dim=-1, keepdim=True) + 1e-8
                )
                gradient = gradient / (
                    1.0 + norm / max(self.gradient_clip, 1e-8)
                )
                positions = positions - self.step_size * gradient
                count = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(positions.dtype)
                center = positions.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
                positions = (positions - center) * atom_mask.unsqueeze(-1)
            final_terms = self.terms(
                positions,
                probabilities,
                geometry_probabilities,
                atom_mask,
                pair_mask,
                covalent_radii,
                vdw_radii,
            )
        if not differentiable:
            positions = positions.detach()
            final_terms = {key: value.detach() for key, value in final_terms.items()}
        return positions, final_terms
