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
    """Batched all-atom relaxation with size-independent local forces."""

    def __init__(
        self,
        num_steps: int = 16,
        step_size: float = 0.04,
        bond_weight: float = 40.0,
        angle_weight: float = 8.0,
        planar_weight: float = 1.0,
        clash_weight: float = 1.0,
        bond_probability_power: float = 4.0,
        angle_probability_power: float = 3.0,
        one_three_distance_scale: float = 0.62,
        clash_distance_scale: float = 0.80,
        clash_softness: float = 0.12,
        clash_power: float = 4.0,
        initial_angle_scale: float = 0.5,
        initial_clash_scale: float = 0.25,
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
        self.one_three_distance_scale = one_three_distance_scale
        self.clash_distance_scale = clash_distance_scale
        self.clash_softness = clash_softness
        self.clash_power = clash_power
        self.initial_angle_scale = initial_angle_scale
        self.initial_clash_scale = initial_clash_scale
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
        return (
            covalent_radii[:, :, None] + covalent_radii[:, None, :]
        ) * scale

    def terms(
        self,
        positions: torch.Tensor,
        probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        if reduction not in {"mean", "force"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        dtype = positions.dtype
        pair_mask_f = pair_mask.to(dtype)
        vector = positions[:, None, :, :] - positions[:, :, None, :]
        distance = torch.sqrt(vector.square().sum(dim=-1) + 1e-8)
        unit = vector / distance.unsqueeze(-1)
        q = (1.0 - probabilities[..., NONE]) * pair_mask_f
        target = self.target_lengths(
            probabilities, covalent_radii
        ).clamp_min(1e-4)

        upper = torch.triu(torch.ones_like(pair_mask_f), diagonal=1)
        bond_weights = q.pow(self.bond_probability_power) * upper
        bond_values = torch.log(distance.clamp_min(1e-4) / target).square()
        if reduction == "force":
            # Interaction sums give each coordinate an O(1) local force,
            # independent of batch size or how many other bonds are present.
            bond = (bond_values * bond_weights).sum()
        else:
            bond = self._masked_mean(bond_values, bond_weights)

        angle_weights = q.pow(self.angle_probability_power)
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
        if reduction == "force":
            angle = (angle_per_atom * angle_valid.to(dtype)).sum()
        else:
            angle = self._masked_mean(
                angle_per_atom, angle_valid.to(dtype)
            )

        normalized_moment = (
            moment / total_weight.clamp_min(1e-8)[..., None, None]
        )
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
        planar_weights = (
            planar_probability
            * atom_mask.to(dtype)
            * pair_weight.gt(1e-6)
        )
        if reduction == "force":
            planar = (determinant.square() * planar_weights).sum()
        else:
            planar = self._masked_mean(
                determinant.square(), planar_weights
            )

        # Soft topology keeps the path differentiable. Direct neighbors are
        # excluded. Two-hop pairs get a permissive 1-3 threshold; all more
        # distant pairs receive stronger vdW repulsion.
        two_hop_path_mass = torch.bmm(q, q)
        two_hop = 1.0 - torch.exp(-two_hop_path_mass)
        not_direct = (1.0 - q).square() * pair_mask_f
        one_three_weight = not_direct * two_hop
        nonlocal_weight = not_direct * (1.0 - two_hop)
        radii_sum = vdw_radii[:, :, None] + vdw_radii[:, None, :]
        one_three_penetration = self.clash_softness * F.softplus(
            (
                self.one_three_distance_scale * radii_sum - distance
            ) / self.clash_softness
        )
        nonlocal_penetration = self.clash_softness * F.softplus(
            (
                self.clash_distance_scale * radii_sum - distance
            ) / self.clash_softness
        )
        penetration = (
            one_three_weight * one_three_penetration
            + nonlocal_weight * nonlocal_penetration
        )
        eye = torch.eye(
            penetration.size(1),
            dtype=torch.bool,
            device=penetration.device,
        )[None]
        penetration = penetration.masked_fill(eye, 0.0)
        # A per-atom p-norm behaves as a differentiable maximum, so a small
        # number of severe overlaps cannot disappear in an O(N^2) mean.
        per_atom_clash = (
            penetration.clamp_min(0.0).pow(self.clash_power).sum(dim=-1)
            + 1e-12
        ).pow(2.0 / self.clash_power)
        if reduction == "force":
            clash = 0.5 * (
                per_atom_clash * atom_mask.to(dtype)
            ).sum()
        else:
            clash = self._masked_mean(
                per_atom_clash, atom_mask.to(dtype)
            )
        return {
            "bond": bond,
            "angle": angle,
            "planar": planar,
            "clash": clash,
        }

    def total(
        self,
        terms: Dict[str, torch.Tensor],
        progress: float = 1.0,
    ) -> torch.Tensor:
        angle_scale = self.initial_angle_scale + (
            1.0 - self.initial_angle_scale
        ) * progress
        clash_scale = self.initial_clash_scale + (
            1.0 - self.initial_clash_scale
        ) * progress
        return (
            self.bond_weight * terms["bond"]
            + angle_scale * self.angle_weight * terms["angle"]
            + self.planar_weight * terms["planar"]
            + clash_scale * self.clash_weight * terms["clash"]
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
            for step in range(self.num_steps):
                force_terms = self.terms(
                    positions,
                    probabilities,
                    geometry_probabilities,
                    atom_mask,
                    pair_mask,
                    covalent_radii,
                    vdw_radii,
                    reduction="force",
                )
                progress = (step + 1) / max(self.num_steps, 1)
                gradient = torch.autograd.grad(
                    self.total(force_terms, progress=progress),
                    positions,
                    create_graph=differentiable,
                )[0]
                norm = torch.sqrt(
                    gradient.square().sum(dim=-1, keepdim=True) + 1e-8
                )
                gradient = gradient / (
                    1.0 + norm / max(self.gradient_clip, 1e-8)
                )
                positions = positions - self.step_size * gradient
                count = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(
                    positions.dtype
                )
                center = (
                    positions.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
                )
                positions = (
                    positions - center
                ) * atom_mask.unsqueeze(-1)
            final_terms = self.terms(
                positions,
                probabilities,
                geometry_probabilities,
                atom_mask,
                pair_mask,
                covalent_radii,
                vdw_radii,
                reduction="mean",
            )
        if not differentiable:
            positions = positions.detach()
            final_terms = {
                key: value.detach() for key, value in final_terms.items()
            }
        return positions, final_terms
