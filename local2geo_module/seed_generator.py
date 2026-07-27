from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class SoftDistanceStressSeed(nn.Module):
    """Fully soft, parameter-free distance-geometry coordinate seed.

    The module embeds corrected graph probabilities and learned local distance
    priors without neighbour selection or detached coordinate updates. D12,
    D13, and D14 are fitted by differentiable weighted SMACOF; non-local graph
    and vdW distances are lower bounds only and therefore cannot compact the
    molecule. Gradients from the final coordinates flow through every step to
    all incoming probabilities.
    """

    def __init__(
        self,
        num_steps: int = 96,
        step_size: float = 0.06,
        init_scale: float = 1.5,
        path_temperature: float = 0.08,
        uncertainty_penalty: float = 3.0,
        global_distance_scale: float = 0.90,
        global_distance_saturation: float = 20.0,
        vdw_scale: float = 0.72,
        lower_softness: float = 0.12,
        bond_weight: float = 12.0,
        one_three_weight: float = 12.0,
        one_four_weight: float = 4.0,
        global_weight: float = 0.8,
        confidence_power: float = 3.0,
        confidence_floor: float = 0.05,
        smacof_regularization: float = 1e-4,
        hydrogen_lower_weight: float = 1.0,
        heavy_stage_fraction: float = 0.65,
        hydrogen_stage_fraction: float = 0.20,
        outward_strength: float = 1.5,
        anchor_strength: float = 1.0,
        steric_outward_weight: float = 0.5,
        gradient_clip: float = 8.0,
        max_displacement: float = 0.30,
    ) -> None:
        super().__init__()
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        positive = {
            "step_size": step_size,
            "init_scale": init_scale,
            "path_temperature": path_temperature,
            "global_distance_saturation": global_distance_saturation,
            "lower_softness": lower_softness,
            "outward_strength": outward_strength,
            "anchor_strength": anchor_strength,
            "confidence_power": confidence_power,
            "smacof_regularization": smacof_regularization,
            "gradient_clip": gradient_clip,
            "max_displacement": max_displacement,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(
                "SoftDistanceStressSeed scales and step sizes must be positive"
            )
        if confidence_floor < 0:
            raise ValueError("confidence_floor must be non-negative")
        if (
            not 0.0 <= heavy_stage_fraction <= 1.0
            or not 0.0 <= hydrogen_stage_fraction <= 1.0
            or heavy_stage_fraction + hydrogen_stage_fraction > 1.0
        ):
            raise ValueError(
                "heavy/hydrogen stage fractions must be in [0, 1] and "
                "sum to at most 1"
            )
        self.num_steps = num_steps
        self.step_size = step_size
        self.init_scale = init_scale
        self.path_temperature = path_temperature
        self.uncertainty_penalty = uncertainty_penalty
        self.global_distance_scale = global_distance_scale
        self.global_distance_saturation = global_distance_saturation
        self.vdw_scale = vdw_scale
        self.lower_softness = lower_softness
        self.bond_weight = bond_weight
        self.one_three_weight = one_three_weight
        self.one_four_weight = one_four_weight
        self.global_weight = global_weight
        self.confidence_power = confidence_power
        self.confidence_floor = confidence_floor
        self.smacof_regularization = smacof_regularization
        self.hydrogen_lower_weight = hydrogen_lower_weight
        self.heavy_stage_fraction = heavy_stage_fraction
        self.hydrogen_stage_fraction = hydrogen_stage_fraction
        self.outward_strength = outward_strength
        self.anchor_strength = anchor_strength
        self.steric_outward_weight = steric_outward_weight
        self.gradient_clip = gradient_clip
        self.max_displacement = max_displacement

    @staticmethod
    def _pair_mask(atom_mask: torch.Tensor) -> torch.Tensor:
        atoms = atom_mask.size(1)
        diagonal = torch.eye(
            atoms, device=atom_mask.device, dtype=torch.bool
        )[None]
        return (
            atom_mask[:, :, None]
            & atom_mask[:, None, :]
            & ~diagonal
        )

    def _soft_min(
        self,
        direct: torch.Tensor,
        alternative: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth minimum without log-sum-exp path-count bias."""
        gate = torch.sigmoid(
            (alternative - direct) / self.path_temperature
        )
        return gate * direct + (1.0 - gate) * alternative

    def soft_path_distance(
        self,
        bond_probability: torch.Tensor,
        bond_target: torch.Tensor,
        atom_mask: torch.Tensor,
        differentiable: bool = True,
    ) -> torch.Tensor:
        """Continuous Floyd relaxation over uncertainty-aware edge costs."""
        pair_mask = self._pair_mask(atom_mask)
        probability = bond_probability * pair_mask
        direct = (
            bond_target
            + self.uncertainty_penalty
            * (-torch.log(probability + 1e-6))
        )
        large = direct.new_full(direct.shape, 1e3)
        distance = torch.where(pair_mask, direct, large)
        diagonal = torch.eye(
            distance.size(1),
            device=distance.device,
            dtype=torch.bool,
        )[None]
        distance = torch.where(
            diagonal,
            torch.zeros_like(distance),
            distance,
        )
        valid_pair = (
            atom_mask[:, :, None] & atom_mask[:, None, :]
        )
        for middle in range(distance.size(1)):
            def relax(
                current: torch.Tensor,
                middle_index: int = middle,
            ) -> torch.Tensor:
                through = (
                    current[:, :, middle_index, None]
                    + current[:, None, middle_index, :]
                )
                candidate = self._soft_min(current, through)
                middle_valid = atom_mask[
                    :, middle_index, None, None
                ]
                updated = torch.where(
                    middle_valid & valid_pair,
                    candidate,
                    current,
                )
                return torch.where(
                    diagonal,
                    torch.zeros_like(updated),
                    updated,
                )

            if differentiable:
                distance = checkpoint(
                    relax, distance, use_reentrant=False
                )
            else:
                distance = relax(distance).detach()
        return distance * pair_mask

    @staticmethod
    def _expanded_noise(
        atom_mask: torch.Tensor,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
        init_scale: float,
    ) -> torch.Tensor:
        batch, atoms = atom_mask.shape
        mask = atom_mask.unsqueeze(-1).to(dtype)
        coordinates = torch.randn(
            (batch, atoms, 3),
            device=atom_mask.device,
            dtype=dtype,
            generator=generator,
        ) * mask
        count = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        coordinates = coordinates - (
            coordinates.sum(dim=1, keepdim=True)
            / count.unsqueeze(-1)
        )
        rms = torch.sqrt(
            coordinates.square().sum(dim=(1, 2), keepdim=True)
            / (3.0 * count.unsqueeze(-1))
        ).clamp_min(1e-4)
        radius = init_scale * count.pow(1.0 / 3.0)
        return coordinates / rms * radius.unsqueeze(-1) * mask

    @staticmethod
    def _spherical_anchors(
        atoms: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Deterministic distinct directions used only to break H symmetry."""
        index = torch.arange(
            atoms, device=device, dtype=dtype
        ) + 0.5
        golden = torch.pi * (3.0 - 5.0 ** 0.5)
        z = 1.0 - 2.0 * index / max(atoms, 1)
        radial = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        return torch.stack(
            (
                radial * torch.cos(index * golden),
                radial * torch.sin(index * golden),
                z,
            ),
            dim=-1,
        )

    def _place_hydrogens(
        self,
        heavy_coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        bond_probability: torch.Tensor,
        bond_target: torch.Tensor,
    ) -> torch.Tensor:
        """Attach H to soft parents on the outside of the heavy skeleton.

        Parent positions and local outward directions are probability-weighted,
        so a downstream coordinate loss can still correct uncertain attachment
        and heavy-edge logits without a discrete parent selection.
        """
        dtype = heavy_coordinates.dtype
        hydrogen_mask = atom_mask & ~heavy_mask
        heavy_f = heavy_mask.to(dtype)
        hydrogen_f = hydrogen_mask.to(dtype)
        parent_weight = (
            bond_probability
            * hydrogen_f[:, :, None]
            * heavy_f[:, None, :]
        )
        parent_probability = parent_weight / parent_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        parent_position = torch.bmm(
            parent_probability, heavy_coordinates
        )
        parent_bond_length = (
            parent_probability * bond_target
        ).sum(dim=-1).clamp_min(0.6)

        vector = (
            heavy_coordinates[:, :, None, :]
            - heavy_coordinates[:, None, :, :]
        )
        distance = torch.sqrt(
            vector.square().sum(dim=-1) + 1e-8
        )
        unit = vector / distance.unsqueeze(-1)
        heavy_pair = (
            heavy_mask[:, :, None] & heavy_mask[:, None, :]
        ).to(dtype)
        eye = torch.eye(
            heavy_mask.size(1),
            device=heavy_mask.device,
            dtype=dtype,
        )[None]
        heavy_pair = heavy_pair * (1.0 - eye)
        heavy_bond = bond_probability * heavy_pair

        # The bonded-neighbour term points away from the local heavy-atom
        # cone. A shorter-range all-heavy term also resolves cyclic and
        # approximately symmetric neighbourhoods without a global-centroid
        # definition of "outside".
        bonded_outward = (
            heavy_bond.unsqueeze(-1) * unit
        ).sum(dim=2)
        steric_weight = (
            torch.exp(-distance / 2.0) * heavy_pair
        )
        steric_outward = (
            steric_weight.unsqueeze(-1) * unit
        ).sum(dim=2)
        parent_outward = (
            bonded_outward
            + self.steric_outward_weight * steric_outward
        )
        hydrogen_outward = torch.bmm(
            parent_probability, parent_outward
        )
        outward_norm = torch.sqrt(
            hydrogen_outward.square().sum(dim=-1, keepdim=True)
            + 1e-4
        )
        hydrogen_outward = hydrogen_outward / outward_norm

        anchors = self._spherical_anchors(
            atom_mask.size(1),
            atom_mask.device,
            dtype,
        )[None].expand(atom_mask.size(0), -1, -1)
        direction = (
            self.outward_strength * hydrogen_outward
            + self.anchor_strength * anchors
        )
        direction = direction / torch.sqrt(
            direction.square().sum(dim=-1, keepdim=True) + 1e-8
        )
        hydrogen_coordinates = (
            parent_position
            + parent_bond_length.unsqueeze(-1) * direction
        )
        return torch.where(
            hydrogen_mask.unsqueeze(-1),
            hydrogen_coordinates,
            heavy_coordinates,
        )

    @staticmethod
    def _target_derivative(
        distance: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """d [w log(d/t)^2] / d distance."""
        return (
            2.0
            * weight
            * torch.log(
                distance.clamp_min(1e-4) / target.clamp_min(1e-4)
            )
            / distance.clamp_min(1e-4)
        )

    def _lower_derivative(
        self,
        distance: torch.Tensor,
        lower: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """d [w softplus(lower-distance)^2] / d distance."""
        scaled = (lower - distance) / self.lower_softness
        violation = self.lower_softness * F.softplus(scaled)
        return (
            -2.0
            * weight
            * violation
            * torch.sigmoid(scaled)
        )

    def _confidence_weight(self, probability: torch.Tensor) -> torch.Tensor:
        """Sharp local confidence with a non-zero linear gradient path."""
        probability = probability.clamp(0.0, 1.0)
        return (
            probability.pow(self.confidence_power)
            + self.confidence_floor * probability
        )

    def _assemble_sparse_constraints(
        self,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        probabilities: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        bond_length_scales: torch.Tensor,
        local_geometry_priors: Dict[str, torch.Tensor],
        differentiable: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build one internally weighted D12/D13/D14 stress matrix."""
        pair_mask = self._pair_mask(atom_mask)
        pair_mask_f = pair_mask.to(probabilities.dtype)
        bonded = probabilities[..., 1:]
        bond_probability = bonded.sum(dim=-1) * pair_mask_f
        bond_share = bonded / bonded.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        radii_sum = (
            covalent_radii[:, :, None]
            + covalent_radii[:, None, :]
        ).clamp_min(0.5)
        target_by_type = (
            radii_sum.unsqueeze(-1) * bond_length_scales[1:]
        )
        target_12 = (
            bond_share * target_by_type
        ).sum(dim=-1).clamp_min(0.5)

        two_hop_mass = torch.bmm(
            bond_probability, bond_probability
        )
        normalizer = 1.0 - torch.exp(two_hop_mass.new_tensor(-1.0))
        analytic_p13 = (
            (1.0 - torch.exp(-two_hop_mass)) / normalizer
        ).clamp(0.0, 1.0)
        three_hop_mass = torch.bmm(
            two_hop_mass, bond_probability
        )
        analytic_p14 = (
            (1.0 - torch.exp(-three_hop_mass)) / normalizer
        ).clamp(0.0, 1.0)
        skeleton = (
            heavy_mask[:, :, None] & heavy_mask[:, None, :]
        ).to(probabilities.dtype)

        shell_12 = bond_probability
        shell_13 = (
            (1.0 - shell_12) * analytic_p13 * pair_mask_f
        )
        shell_14 = (
            (1.0 - shell_12)
            * (1.0 - shell_13)
            * analytic_p14
            * skeleton
            * pair_mask_f
        )
        target_13 = radii_sum * local_geometry_priors[
            "one_three_distance_ratio"
        ]
        target_14 = radii_sum * local_geometry_priors[
            "one_four_distance_ratio"
        ]
        weight_12 = (
            self.bond_weight * self._confidence_weight(shell_12)
        )
        weight_13 = (
            self.one_three_weight
            * self._confidence_weight(shell_13)
        )
        weight_14 = (
            self.one_four_weight
            * self._confidence_weight(shell_14)
        )
        sparse_weight = (
            weight_12 + weight_13 + weight_14
        ) * pair_mask_f
        sparse_target = (
            weight_12 * target_12
            + weight_13 * target_13
            + weight_14 * target_14
        ) / sparse_weight.clamp_min(1e-8)
        sparse_target = sparse_target.clamp_min(0.5)

        far_membership = (
            (1.0 - shell_12)
            * (1.0 - shell_13)
            * (1.0 - shell_14)
            * pair_mask_f
        )
        lower_scale = (
            self.hydrogen_lower_weight
            + (1.0 - self.hydrogen_lower_weight) * skeleton
        )
        lower_weight = (
            self.global_weight * far_membership * lower_scale
        )
        vdw_lower = self.vdw_scale * (
            vdw_radii[:, :, None] + vdw_radii[:, None, :]
        )
        path_distance = self.soft_path_distance(
            bond_probability,
            target_12,
            atom_mask,
            differentiable=differentiable,
        )
        graph_lower = (
            self.global_distance_scale
            * path_distance
            / (
                1.0
                + path_distance / self.global_distance_saturation
            )
        )
        lower_bound = self.lower_softness * torch.logsumexp(
            torch.stack([vdw_lower, graph_lower], dim=-1)
            / self.lower_softness,
            dim=-1,
        )
        return {
            "bond_probability": bond_probability,
            "bond_target": target_12,
            "sparse_target": sparse_target,
            "sparse_weight": sparse_weight,
            "lower_bound": lower_bound,
            "lower_weight": lower_weight,
            "shell_12": shell_12,
            "shell_13": shell_13,
            "shell_14": shell_14,
        }

    def _smacof_update(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        lower_bound: torch.Tensor,
        lower_weight: torch.Tensor,
        stage_pair: torch.Tensor,
        movable_mask: torch.Tensor,
        center_mask: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One differentiable anchored SMACOF step plus weak vdW repulsion."""
        dtype = current.dtype
        distance = torch.sqrt(
            (
                current[:, :, None, :]
                - current[:, None, :, :]
            ).square().sum(dim=-1)
            + 1e-8
        )
        local_weight = weight * stage_pair
        active_lower_weight = (
            lower_weight
            * stage_pair
            * torch.sigmoid(
                (lower_bound - distance) / self.lower_softness
            )
        )
        pair_weight = local_weight + active_lower_weight
        pair_target = (
            local_weight * target
            + active_lower_weight * lower_bound
        ) / pair_weight.clamp_min(1e-8)
        pair_weight = 0.5 * (
            pair_weight + pair_weight.transpose(1, 2)
        )
        pair_target = 0.5 * (
            pair_target + pair_target.transpose(1, 2)
        )
        ratio = (
            pair_weight * pair_target / distance.clamp_min(1e-4)
        )
        atoms = current.size(1)
        eye = torch.eye(
            atoms, device=current.device, dtype=dtype
        )[None]
        ratio = ratio * (1.0 - eye)
        b_matrix = -ratio
        b_matrix = b_matrix - torch.diag_embed(
            b_matrix.sum(dim=-1)
        )
        laplacian = (
            torch.diag_embed(pair_weight.sum(dim=-1))
            - pair_weight
        )
        rhs = torch.bmm(b_matrix, current)

        fixed = (~movable_mask.bool()) | (~atom_mask)
        system = laplacian + self.smacof_regularization * eye
        system = torch.where(
            fixed[:, :, None],
            eye.expand_as(system),
            system,
        )
        rhs = torch.where(
            fixed.unsqueeze(-1), current, rhs
        )
        updated = torch.linalg.solve(system, rhs)

        # Sparse stress has no attractive long-range target. A weak hinge
        # prevents non-local overlap without making the molecule compact.
        vector = (
            updated[:, :, None, :]
            - updated[:, None, :, :]
        )
        distance = torch.sqrt(
            vector.square().sum(dim=-1) + 1e-8
        )
        unit = vector / distance.unsqueeze(-1)
        derivative = self._lower_derivative(
            distance,
            lower_bound,
            lower_weight * stage_pair,
        )
        repulsion = (
            derivative.unsqueeze(-1) * unit
        ).sum(dim=2)
        norm = torch.linalg.vector_norm(
            repulsion, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        repulsion = repulsion / (
            1.0 + norm / self.gradient_clip
        )
        updated = updated - (
            0.25
            * self.step_size
            * repulsion
            * movable_mask.unsqueeze(-1)
        )
        center_count = center_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        center = (
            updated * center_mask.unsqueeze(-1)
        ).sum(dim=1, keepdim=True) / center_count.unsqueeze(-1)
        updated = updated - center * center_mask.unsqueeze(-1)
        return updated * atom_mask.unsqueeze(-1)

    def forward(
        self,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        probabilities: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        bond_length_scales: torch.Tensor,
        local_geometry_priors: Dict[str, torch.Tensor],
        differentiable: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        required = {
            "one_three_probability",
            "one_four_probability",
            "one_three_distance_ratio",
            "one_four_distance_ratio",
        }
        missing = required.difference(local_geometry_priors)
        if missing:
            raise ValueError(
                "SoftDistanceStressSeed requires learned local priors: "
                + ", ".join(sorted(missing))
            )
        constraints = self._assemble_sparse_constraints(
            atom_mask,
            heavy_mask,
            probabilities,
            covalent_radii,
            vdw_radii,
            bond_length_scales,
            local_geometry_priors,
            differentiable,
        )
        coordinates = self._expanded_noise(
            heavy_mask,
            probabilities.dtype,
            generator,
            self.init_scale,
        )

        def run_stage(
            current: torch.Tensor,
            steps: int,
            stage_pair: torch.Tensor,
            movable_mask: torch.Tensor,
            center_mask: torch.Tensor,
        ) -> torch.Tensor:
            stage_inputs = (
                constraints["sparse_target"],
                constraints["sparse_weight"],
                constraints["lower_bound"],
                constraints["lower_weight"],
                stage_pair,
                movable_mask.to(current.dtype),
                center_mask.to(current.dtype),
                atom_mask,
            )
            for _ in range(steps):
                if differentiable:
                    current = checkpoint(
                        self._smacof_update,
                        current,
                        *stage_inputs,
                        use_reentrant=False,
                    )
                else:
                    current = self._smacof_update(
                        current, *stage_inputs
                    ).detach()
            return current

        heavy_steps = int(round(
            self.num_steps * self.heavy_stage_fraction
        ))
        hydrogen_steps = min(
            self.num_steps - heavy_steps,
            int(round(
                self.num_steps * self.hydrogen_stage_fraction
            )),
        )
        joint_steps = self.num_steps - heavy_steps - hydrogen_steps
        heavy_pair = (
            heavy_mask[:, :, None] & heavy_mask[:, None, :]
        ).to(probabilities.dtype)
        all_pair = self._pair_mask(atom_mask).to(probabilities.dtype)
        coordinates = run_stage(
            coordinates,
            heavy_steps,
            heavy_pair,
            heavy_mask,
            heavy_mask,
        )

        # Stage 2 uses the soft H-parent distribution and the optimized local
        # heavy environment to build an outward all-atom proposal.
        coordinates = self._place_hydrogens(
            coordinates,
            atom_mask,
            heavy_mask,
            constraints["bond_probability"],
            constraints["bond_target"],
        )
        hydrogen_mask = atom_mask & ~heavy_mask
        coordinates = run_stage(
            coordinates,
            hydrogen_steps,
            all_pair,
            hydrogen_mask,
            torch.zeros_like(atom_mask),
        )

        # Stage 3 releases all atoms for a short joint reconciliation.
        coordinates = run_stage(
            coordinates,
            joint_steps,
            all_pair,
            atom_mask,
            atom_mask,
        )
        return coordinates if differentiable else coordinates.detach()

    def refine(
        self,
        coordinates: torch.Tensor,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        probabilities: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        bond_length_scales: torch.Tensor,
        local_geometry_priors: Dict[str, torch.Tensor],
        num_steps: int,
        differentiable: bool = True,
    ) -> torch.Tensor:
        """Refine an existing seed with the identical sparse objective."""
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        constraints = self._assemble_sparse_constraints(
            atom_mask,
            heavy_mask,
            probabilities,
            covalent_radii,
            vdw_radii,
            bond_length_scales,
            local_geometry_priors,
            differentiable,
        )
        all_pair = self._pair_mask(atom_mask).to(coordinates.dtype)
        update_inputs = (
            constraints["sparse_target"],
            constraints["sparse_weight"],
            constraints["lower_bound"],
            constraints["lower_weight"],
            all_pair,
            atom_mask.to(coordinates.dtype),
            atom_mask.to(coordinates.dtype),
            atom_mask,
        )
        current = coordinates
        for _ in range(num_steps):
            if differentiable:
                current = checkpoint(
                    self._smacof_update,
                    current,
                    *update_inputs,
                    use_reentrant=False,
                )
            else:
                current = self._smacof_update(
                    current, *update_inputs
                ).detach()
        return current if differentiable else current.detach()


def graph_smoothed_seed(
    atom_mask: torch.Tensor,
    bond_probability: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Original fully differentiable deterministic seed."""
    batch, atoms = atom_mask.shape
    dtype = bond_probability.dtype
    index = torch.arange(
        atoms, device=atom_mask.device, dtype=dtype
    ) + 1.0
    count = atom_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
    golden = torch.pi * (3.0 - 5.0 ** 0.5)
    phase = index[None, :] * golden
    z = 1.0 - 2.0 * (index[None, :] - 0.5) / count
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    anchors = torch.stack(
        (
            radius * torch.cos(phase),
            radius * torch.sin(phase),
            z,
        ),
        dim=-1,
    )
    scale = 1.5 * count.pow(1.0 / 3.0)
    anchors = anchors * scale.unsqueeze(-1) * atom_mask.unsqueeze(-1)

    degree = bond_probability.sum(dim=-1)
    laplacian = torch.diag_embed(degree) - bond_probability
    identity = torch.eye(
        atoms, device=bond_probability.device, dtype=dtype
    ).unsqueeze(0)
    system = identity + smoothing * laplacian
    seed = torch.linalg.solve(system, anchors)
    seed = seed * atom_mask.unsqueeze(-1)
    center = seed.sum(dim=1, keepdim=True) / count.unsqueeze(-1)
    return (seed - center) * atom_mask.unsqueeze(-1)


def _floyd_warshall(edge_lengths: torch.Tensor) -> torch.Tensor:
    distance = edge_lengths.clone()
    for middle in range(distance.size(0)):
        through_middle = (
            distance[:, middle, None] + distance[None, middle, :]
        )
        distance = torch.minimum(distance, through_middle)
    return distance


def _classical_mds(distance: torch.Tensor) -> torch.Tensor:
    atoms = distance.size(0)
    identity = torch.eye(
        atoms, device=distance.device, dtype=distance.dtype
    )
    centering = identity - torch.ones_like(identity) / max(atoms, 1)
    gram = -0.5 * centering @ distance.square() @ centering
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    take = min(3, atoms)
    indices = torch.arange(
        atoms - 1,
        atoms - take - 1,
        -1,
        device=distance.device,
    )
    values = eigenvalues[indices].clamp_min(0.0)
    coordinates = eigenvectors[:, indices] * torch.sqrt(values)[None, :]
    if take < 3:
        coordinates = torch.nn.functional.pad(
            coordinates, (0, 3 - take)
        )

    # Eigenvectors are sign-ambiguous. Canonicalizing each sign makes repeated
    # eval calls deterministic without changing any pairwise distance.
    for axis in range(3):
        pivot = coordinates[:, axis].abs().argmax()
        sign = torch.where(
            coordinates[pivot, axis] < 0,
            coordinates.new_tensor(-1.0),
            coordinates.new_tensor(1.0),
        )
        coordinates[:, axis] = coordinates[:, axis] * sign
    return coordinates


def _apply_local_one_three_bounds(
    distance: torch.Tensor,
    hard_types: torch.Tensor,
    bond_targets: torch.Tensor,
    target_cosines: torch.Tensor,
) -> torch.Tensor:
    """Replace path-sum 1--3 distances with VSEPR-consistent chords."""
    bounded = distance.clone()
    bonded = hard_types.ne(0)
    for center in range(distance.size(0)):
        neighbors = bonded[center].nonzero(as_tuple=False).flatten().tolist()
        cosine = target_cosines[center].clamp(-1.0, 1.0)
        for first, second in combinations(neighbors, 2):
            left = bond_targets[first, center]
            right = bond_targets[center, second]
            chord = torch.sqrt(
                (
                    left.square()
                    + right.square()
                    - 2.0 * left * right * cosine
                ).clamp_min(1e-6)
            )
            bounded[first, second] = chord
            bounded[second, first] = chord
    return bounded


def _deterministic_jitter(
    atoms: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    index = torch.arange(atoms, device=device, dtype=dtype) + 1.0
    jitter = torch.stack(
        (
            torch.sin(index * 1.61803398875),
            torch.sin(index * 2.41421356237 + 0.7),
            torch.sin(index * 3.14159265359 + 1.3),
        ),
        dim=-1,
    )
    return jitter - jitter.mean(dim=0, keepdim=True)


def _break_linear_local_geometry(
    coordinates: torch.Tensor,
    hard_types: torch.Tensor,
    target_cosines: torch.Tensor,
    displacement: float = 0.45,
) -> torch.Tensor:
    """Give non-linear centers a transverse kick before local stress."""
    position = coordinates.clone()
    bonded = hard_types.ne(0)
    jitter = _deterministic_jitter(
        position.size(0), position.device, position.dtype
    )
    for center in range(position.size(0)):
        if target_cosines[center] < -0.8:
            continue
        neighbors = bonded[center].nonzero(as_tuple=False).flatten()
        if neighbors.numel() < 2:
            continue
        vectors = position[neighbors] - position[center]
        unit = vectors / torch.linalg.vector_norm(
            vectors, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        cosines = unit @ unit.transpose(0, 1)
        cosines.fill_diagonal_(1.0)
        flat = cosines.argmin()
        first = flat // cosines.size(1)
        second = flat % cosines.size(1)
        if cosines[first, second] > -0.90:
            continue
        axis = unit[first] - unit[second]
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-6)
        transverse = jitter[center] - (jitter[center] * axis).sum() * axis
        transverse = transverse / torch.linalg.vector_norm(
            transverse
        ).clamp_min(1e-6)
        position[center] = position[center] + displacement * transverse
    return position


def _seed_stress_refinement(
    coordinates: torch.Tensor,
    hard_types: torch.Tensor,
    bond_targets: torch.Tensor,
    graph_distances: torch.Tensor,
    target_cosines: torch.Tensor,
    planar_probabilities: torch.Tensor,
    num_steps: int,
    step_size: float,
) -> torch.Tensor:
    """Refine a detached MDS proposal with local chemistry constraints."""
    if num_steps <= 0 or coordinates.size(0) <= 1:
        return coordinates

    device = coordinates.device
    bonded = hard_types.ne(0)
    edge_indices = torch.triu(bonded, diagonal=1).nonzero(
        as_tuple=False
    )
    edge_target = (
        bond_targets[edge_indices[:, 0], edge_indices[:, 1]]
        if edge_indices.numel()
        else coordinates.new_empty(0)
    )

    one_three_pairs = []
    one_three_targets = []
    planar_triplets = []
    planar_weights = []
    for center in range(coordinates.size(0)):
        neighbors = bonded[center].nonzero(as_tuple=False).flatten().tolist()
        for first, second in combinations(neighbors, 2):
            left = bond_targets[first, center]
            right = bond_targets[center, second]
            cosine = target_cosines[center].clamp(-1.0, 1.0)
            target_squared = (
                left.square()
                + right.square()
                - 2.0 * left * right * cosine
            ).clamp_min(1e-6)
            one_three_pairs.append((first, second))
            one_three_targets.append(torch.sqrt(target_squared))
        if len(neighbors) >= 3 and planar_probabilities[center] > 0.05:
            for triplet in combinations(neighbors, 3):
                planar_triplets.append((center, *triplet))
                planar_weights.append(planar_probabilities[center])

    if one_three_pairs:
        one_three_indices = torch.tensor(
            one_three_pairs, device=device, dtype=torch.long
        )
        one_three_target = torch.stack(one_three_targets)
    else:
        one_three_indices = torch.empty(
            (0, 2), device=device, dtype=torch.long
        )
        one_three_target = coordinates.new_empty(0)
    if planar_triplets:
        planar_indices = torch.tensor(
            planar_triplets, device=device, dtype=torch.long
        )
        planar_weight = torch.stack(planar_weights)
    else:
        planar_indices = torch.empty(
            (0, 4), device=device, dtype=torch.long
        )
        planar_weight = coordinates.new_empty(0)

    # The global term is a lower bound, not a target distance. It preserves
    # the expanded topology supplied by MDS without forcing graph paths to be
    # perfectly straight. Local 1--3 constraints determine bond angles.
    hops = _floyd_warshall(
        torch.where(
            bonded,
            torch.ones_like(graph_distances),
            torch.full_like(graph_distances, torch.inf),
        ).fill_diagonal_(0.0)
    )
    global_mask = (
        torch.isfinite(graph_distances)
        & hops.ge(3.0)
        & torch.triu(torch.ones_like(bonded), diagonal=1)
    )
    global_indices = global_mask.nonzero(as_tuple=False)
    global_lower = (
        (0.72 * graph_distances[
            global_indices[:, 0], global_indices[:, 1]
        ]).clamp_max(4.5)
        if global_indices.numel()
        else coordinates.new_empty(0)
    )

    # Alternating distance projections are an effective symmetry breaker for
    # nearly collinear MDS chains: ordinary distance gradients have almost no
    # transverse component at 180 degrees. These are proposal-only updates.
    position = coordinates.detach().clone()
    with torch.no_grad():
        for _ in range(min(48, num_steps)):
            for indices, targets, fraction in (
                (edge_indices, edge_target, 0.35),
                (one_three_indices, one_three_target, 0.25),
            ):
                if not indices.numel():
                    continue
                first, second = indices.unbind(dim=-1)
                vector = position[first] - position[second]
                distance_value = torch.linalg.vector_norm(
                    vector, dim=-1, keepdim=True
                ).clamp_min(1e-6)
                correction = (
                    fraction
                    * (
                        distance_value
                        - targets.unsqueeze(-1)
                    )
                    * vector
                    / distance_value
                )
                update = torch.zeros_like(position)
                update.index_add_(0, first, -0.5 * correction)
                update.index_add_(0, second, 0.5 * correction)
                position = position + update
            position = position - position.mean(dim=0, keepdim=True)

    first_moment = torch.zeros_like(position)
    second_moment = torch.zeros_like(position)
    beta1, beta2 = 0.9, 0.999
    with torch.enable_grad():
        for iteration in range(1, num_steps + 1):
            position = position.detach().requires_grad_(True)
            energy = position.sum() * 0.0
            if edge_indices.numel():
                edge_distance = torch.linalg.vector_norm(
                    position[edge_indices[:, 0]]
                    - position[edge_indices[:, 1]],
                    dim=-1,
                )
                energy = energy + 120.0 * torch.log(
                    edge_distance.clamp_min(1e-4)
                    / edge_target.clamp_min(1e-4)
                ).square().mean()
            if one_three_indices.numel():
                one_three_distance = torch.linalg.vector_norm(
                    position[one_three_indices[:, 0]]
                    - position[one_three_indices[:, 1]],
                    dim=-1,
                )
                energy = energy + 48.0 * torch.log(
                    one_three_distance.clamp_min(1e-4)
                    / one_three_target.clamp_min(1e-4)
                ).square().mean()
            if planar_indices.numel():
                center = position[planar_indices[:, 0]]
                vectors = [
                    position[planar_indices[:, index]] - center
                    for index in range(1, 4)
                ]
                unit = [
                    vector / torch.linalg.vector_norm(
                        vector, dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    for vector in vectors
                ]
                signed_volume = (
                    torch.linalg.cross(unit[0], unit[1], dim=-1)
                    * unit[2]
                ).sum(dim=-1)
                energy = energy + 2.0 * (
                    planar_weight * signed_volume.square()
                ).sum() / planar_weight.sum().clamp_min(1e-8)
            if global_indices.numel():
                global_distance = torch.linalg.vector_norm(
                    position[global_indices[:, 0]]
                    - position[global_indices[:, 1]],
                    dim=-1,
                )
                violation = F.softplus(
                    (global_lower - global_distance) / 0.10
                ) * 0.10
                energy = energy + 0.01 * violation.square().mean()

            gradient = torch.autograd.grad(energy, position)[0]
            gradient_norm = torch.linalg.vector_norm(
                gradient, dim=-1, keepdim=True
            )
            gradient = gradient / (1.0 + gradient_norm / 5.0)
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = (
                beta2 * second_moment
                + (1.0 - beta2) * gradient.square()
            )
            corrected_first = first_moment / (1.0 - beta1 ** iteration)
            corrected_second = second_moment / (1.0 - beta2 ** iteration)
            position = position - step_size * corrected_first / (
                corrected_second.sqrt() + 1e-8
            )
            position = position - position.mean(dim=0, keepdim=True)
    return position.detach()


def detached_graph_distance_mds_seed(
    atom_mask: torch.Tensor,
    probabilities: torch.Tensor,
    covalent_radii: torch.Tensor,
    bond_length_scales: torch.Tensor,
    geometry_probabilities: torch.Tensor | None = None,
    geometry_cosines: torch.Tensor | None = None,
    planar_geometry_index: int = 3,
    inflation: float = 1.15,
    jitter_scale: float = 0.08,
    disconnected_margin: float = 4.0,
    stress_steps: int = 384,
    stress_step_size: float = 0.03,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard shortest-path MDS seed used only as a detached proposal.

    Returns the padded coordinates and hard bond-type matrix. Relaxation still
    consumes the original soft probabilities, so graph-logit gradients do not
    depend on this discrete seed construction.
    """
    batch, padded_atoms = atom_mask.shape
    seeds = torch.zeros(
        (batch, padded_atoms, 3),
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    hard_types = torch.zeros(
        (batch, padded_atoms, padded_atoms),
        device=probabilities.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        for batch_index in range(batch):
            atoms = int(atom_mask[batch_index].sum())
            if atoms <= 1:
                continue
            sample_probabilities = probabilities[
                batch_index, :atoms, :atoms
            ]
            sample_types = sample_probabilities.argmax(dim=-1)
            sample_types = torch.triu(sample_types, diagonal=1)
            sample_types = sample_types + sample_types.transpose(0, 1)
            hard_types[
                batch_index, :atoms, :atoms
            ] = sample_types

            radii_sum = (
                covalent_radii[batch_index, :atoms, None]
                + covalent_radii[batch_index, None, :atoms]
            )
            target = radii_sum * bond_length_scales[sample_types]
            bonded = sample_types.ne(0)
            infinity = torch.full_like(target, torch.inf)
            edge_lengths = torch.where(bonded, target, infinity)
            edge_lengths.fill_diagonal_(0.0)
            distance = _floyd_warshall(edge_lengths)

            finite = torch.isfinite(distance)
            finite_max = distance[finite].max() if finite.any() else (
                distance.new_tensor(1.0)
            )
            distance = torch.where(
                finite,
                distance,
                finite_max + disconnected_margin,
            )
            sample_geometry = None
            target_cosines = None
            if (
                geometry_probabilities is not None
                and geometry_cosines is not None
            ):
                sample_geometry = geometry_probabilities[
                    batch_index, :atoms
                ].detach()
                target_cosines = (
                    sample_geometry * geometry_cosines.detach()
                ).sum(dim=-1)
                distance = _apply_local_one_three_bounds(
                    distance,
                    sample_types,
                    target,
                    target_cosines,
                )
            coordinates = _classical_mds(distance) * inflation
            coordinates = coordinates + jitter_scale * _deterministic_jitter(
                atoms, coordinates.device, coordinates.dtype
            )
            if (
                sample_geometry is not None
                and target_cosines is not None
                and stress_steps > 0
            ):
                coordinates = _break_linear_local_geometry(
                    coordinates,
                    sample_types,
                    target_cosines,
                )
                coordinates = _seed_stress_refinement(
                    coordinates,
                    sample_types,
                    target,
                    distance,
                    target_cosines,
                    sample_geometry[..., planar_geometry_index],
                    num_steps=stress_steps,
                    step_size=stress_step_size,
                )
            coordinates = coordinates - coordinates.mean(
                dim=0, keepdim=True
            )
            seeds[batch_index, :atoms] = coordinates
    return seeds.detach(), hard_types.detach()
