from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .constants import (
    AROMATIC,
    BOND_LENGTH_SCALES,
    DOUBLE,
    GEOMETRY_COSINES,
    NONE,
    NUM_BOND_TYPES,
    PLANAR_GEOMETRY_INDEX,
    SINGLE,
    TRIPLE,
)
from .seed_generator import (
    SoftDistanceStressSeed,
    detached_graph_distance_mds_seed,
    graph_smoothed_seed,
)


class DifferentiableGeometrySolver(nn.Module):
    """Parameter-free soft-graph to all-atom coordinate initializer."""

    def __init__(
        self,
        num_steps: int = 64,
        step_size: float = 0.02,
        bond_weight: float = 20.0,
        angle_weight: float = 4.0,
        planar_weight: float = 0.5,
        clash_weight: float = 2.0,
        one_three_distance_weight: float = 0.0,
        one_four_distance_weight: float = 0.0,
        bond_probability_power: float = 3.0,
        angle_probability_power: float = 2.0,
        clash_distance_scale: float = 0.80,
        clash_softness: float = 0.10,
        clash_smoothmax_temperature: float = 0.02,
        geometry_temperature: float = 0.15,
        graph_seed_smoothing: float = 0.5,
        seed_mode: str = "differentiable",
        mds_inflation: float = 1.15,
        mds_jitter_scale: float = 0.08,
        mds_stress_steps: int = 384,
        mds_stress_step_size: float = 0.03,
        soft_stress_steps: int = 96,
        soft_stress_step_size: float = 0.06,
        soft_stress_init_scale: float = 1.5,
        soft_stress_path_temperature: float = 0.08,
        soft_stress_uncertainty_penalty: float = 3.0,
        soft_stress_global_weight: float = 0.8,
        edge_temperature: float = 0.7,
        attachment_temperature: float = 0.7,
        gradient_clip: float = 5.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.planar_weight = planar_weight
        self.clash_weight = clash_weight
        self.one_three_distance_weight = one_three_distance_weight
        self.one_four_distance_weight = one_four_distance_weight
        self.bond_probability_power = bond_probability_power
        self.angle_probability_power = angle_probability_power
        self.clash_distance_scale = clash_distance_scale
        self.clash_softness = clash_softness
        self.clash_smoothmax_temperature = clash_smoothmax_temperature
        self.geometry_temperature = geometry_temperature
        self.graph_seed_smoothing = graph_seed_smoothing
        if seed_mode not in {
            "differentiable", "soft_stress", "mds"
        }:
            raise ValueError(
                "seed_mode must be differentiable, soft_stress, or mds"
            )
        self.seed_mode = seed_mode
        self.mds_inflation = mds_inflation
        self.mds_jitter_scale = mds_jitter_scale
        self.mds_stress_steps = mds_stress_steps
        self.mds_stress_step_size = mds_stress_step_size
        self.soft_stress_seed = SoftDistanceStressSeed(
            num_steps=soft_stress_steps,
            step_size=soft_stress_step_size,
            init_scale=soft_stress_init_scale,
            path_temperature=soft_stress_path_temperature,
            uncertainty_penalty=soft_stress_uncertainty_penalty,
            global_weight=soft_stress_global_weight,
        )
        self.edge_temperature = edge_temperature
        self.attachment_temperature = attachment_temperature
        self.gradient_clip = gradient_clip
        self.register_buffer(
            "bond_length_scales",
            torch.tensor(BOND_LENGTH_SCALES, dtype=torch.float32),
        )
        self.register_buffer(
            "geometry_cosines",
            torch.tensor(GEOMETRY_COSINES, dtype=torch.float32),
        )
        # Fixed Pyykko-like single-bond covalent radii and common Bondi vdW
        # radii in angstrom. The explicit table keeps the production solver
        # independent of RDKit; SMILES parsing remains in the simulator path.
        covalent_values = {
            1: 0.31,
            5: 0.85,
            6: 0.76,
            7: 0.71,
            8: 0.66,
            9: 0.57,
            14: 1.11,
            15: 1.07,
            16: 1.05,
            17: 1.02,
            35: 1.20,
            53: 1.39,
        }
        vdw_values = {
            1: 1.20,
            5: 1.92,
            6: 1.70,
            7: 1.55,
            8: 1.52,
            9: 1.47,
            14: 2.10,
            15: 1.80,
            16: 1.80,
            17: 1.75,
            35: 1.85,
            53: 1.98,
        }
        covalent = torch.zeros(119, dtype=torch.float32)
        vdw = torch.zeros(119, dtype=torch.float32)
        for atomic_number, radius in covalent_values.items():
            covalent[atomic_number] = radius
        for atomic_number, radius in vdw_values.items():
            vdw[atomic_number] = radius
        self.register_buffer("covalent_radius_table", covalent)
        self.register_buffer("vdw_radius_table", vdw)

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _masked_attachment_softmax(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(
            logits.masked_fill(~mask, -20.0)
            / self.attachment_temperature,
            dim=-1,
        )
        probabilities = probabilities * mask.to(probabilities.dtype)
        normalizer = probabilities.sum(dim=-1, keepdim=True)
        return torch.where(
            normalizer > 0,
            probabilities / normalizer.clamp_min(1e-8),
            torch.zeros_like(probabilities),
        )

    def soft_graph(
        self,
        heavy_edge_logits: torch.Tensor,
        h_attachment_logits: torch.Tensor,
        pair_mask: torch.Tensor,
        heavy_pair_mask: torch.Tensor,
        attachment_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        heavy_logits = heavy_edge_logits.masked_fill(
            ~heavy_pair_mask.unsqueeze(-1), -20.0
        ).clone()
        heavy_logits[..., NONE] = torch.where(
            heavy_pair_mask,
            heavy_logits[..., NONE],
            torch.full_like(heavy_logits[..., NONE], 20.0),
        )
        heavy_probabilities = torch.softmax(
            heavy_logits / self.edge_temperature, dim=-1
        )
        attachment_probabilities = self._masked_attachment_softmax(
            h_attachment_logits, attachment_mask
        )
        probabilities = torch.zeros(
            (*pair_mask.shape, NUM_BOND_TYPES),
            dtype=torch.float32,
            device=pair_mask.device,
        )
        probabilities[..., NONE] = 1.0
        probabilities = torch.where(
            heavy_pair_mask.unsqueeze(-1),
            heavy_probabilities,
            probabilities,
        )
        h_single = (
            attachment_probabilities * attachment_mask
        )
        h_single = h_single + h_single.transpose(1, 2)
        probabilities[..., SINGLE] = torch.where(
            h_single.gt(0),
            h_single,
            probabilities[..., SINGLE],
        )
        probabilities[..., NONE] = torch.where(
            h_single.gt(0),
            1.0 - h_single,
            probabilities[..., NONE],
        )
        probabilities = probabilities * pair_mask.unsqueeze(-1)
        probabilities[..., NONE] = torch.where(
            pair_mask,
            probabilities[..., NONE],
            torch.ones_like(probabilities[..., NONE]),
        )
        return {
            "heavy_edge_probabilities": heavy_probabilities,
            "h_attachment_probabilities": attachment_probabilities,
            "edge_probabilities": probabilities,
        }

    def soft_geometry_probabilities(
        self,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed differentiable VSEPR-like rules, without argmax."""
        q = (1.0 - probabilities[..., NONE]) * atom_mask[:, None, :]
        degree = q.sum(dim=-1)
        degree_centers = torch.arange(
            1, 6, device=degree.device, dtype=degree.dtype
        )
        degree_gates = torch.softmax(
            -(
                degree.unsqueeze(-1) - degree_centers
            ).square() / self.geometry_temperature,
            dim=-1,
        )
        g1, g2, g3, g4, g5 = degree_gates.unbind(dim=-1)
        double_mass = probabilities[..., DOUBLE].sum(dim=-1)
        triple_mass = probabilities[..., TRIPLE].sum(dim=-1)
        aromatic_mass = probabilities[..., AROMATIC].sum(dim=-1)
        double_presence = 1.0 - torch.exp(-double_mass)
        triple_presence = 1.0 - torch.exp(-triple_mass)
        aromatic_presence = 1.0 - torch.exp(-aromatic_mass)
        multiple_presence = 1.0 - (
            1.0 - double_presence
        ) * (1.0 - triple_presence) * (1.0 - aromatic_presence)

        def element(*values: int) -> torch.Tensor:
            mask = torch.zeros_like(atom_mask)
            for value in values:
                mask |= atomic_numbers.eq(value)
            return mask.to(degree.dtype)

        h = element(1)
        carbon_like = element(5, 6, 14)
        nitrogen_like = element(7, 15)
        oxygen_like = element(8, 16)
        linear = g2 * (
            triple_presence
            + torch.sigmoid((double_mass - 1.5) / 0.2)
            + 0.02
        ) * (carbon_like + nitrogen_like).clamp_max(1.0)
        bent = (
            2.0 * g2 * oxygen_like * (1.0 - triple_presence)
            + 0.01 * g2
        )
        planar_signal = (
            aromatic_presence + double_presence + 0.05
        ).clamp_max(1.0)
        planar = (
            g3 + g2 * multiple_presence
        ) * planar_signal * (
            carbon_like + nitrogen_like
        ).clamp_max(1.0)
        pyramidal = (
            g3
            * nitrogen_like
            * (1.0 - aromatic_presence)
            * (1.0 - double_presence)
            + 0.01 * g3
        )
        tetrahedral = g4 * (
            carbon_like + nitrogen_like + oxygen_like + 0.1
        ).clamp_max(1.0)
        terminal = g1 * (1.0 + 2.0 * h)
        other = g5 + 0.01
        scores = torch.stack([
            terminal,
            linear,
            bent,
            planar,
            pyramidal,
            tetrahedral,
            other,
        ], dim=-1)
        scores = scores * atom_mask.unsqueeze(-1)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _fixed_seed(
        self,
        atom_mask: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        return graph_smoothed_seed(
            atom_mask,
            q,
            smoothing=self.graph_seed_smoothing,
        )

    def _make_seed(
        self,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        local_geometry_priors: Optional[
            Dict[str, torch.Tensor]
        ],
        differentiable: bool,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.seed_mode == "mds":
            return detached_graph_distance_mds_seed(
                atom_mask,
                probabilities,
                covalent_radii,
                self.bond_length_scales,
                geometry_probabilities=geometry_probabilities,
                geometry_cosines=self.geometry_cosines,
                planar_geometry_index=PLANAR_GEOMETRY_INDEX,
                inflation=self.mds_inflation,
                jitter_scale=self.mds_jitter_scale,
                stress_steps=self.mds_stress_steps,
                stress_step_size=self.mds_stress_step_size,
            )
        if self.seed_mode == "soft_stress":
            if local_geometry_priors is None:
                raise ValueError(
                    "seed_mode='soft_stress' requires "
                    "local_geometry_priors from SoftTopologyPrior"
                )
            seed = self.soft_stress_seed(
                atom_mask=atom_mask,
                heavy_mask=heavy_mask,
                probabilities=probabilities,
                covalent_radii=covalent_radii,
                vdw_radii=vdw_radii,
                bond_length_scales=self.bond_length_scales,
                local_geometry_priors=local_geometry_priors,
                differentiable=differentiable,
                generator=generator,
            )
            # Hard types are metadata used by the legacy MDS diagnostics only.
            # Returning zeros avoids inserting argmax into the soft seed path.
            hard_types = torch.zeros(
                probabilities.shape[:-1],
                device=probabilities.device,
                dtype=torch.long,
            )
            return seed, hard_types
        q = (1.0 - probabilities[..., NONE]) * (
            atom_mask[:, :, None] & atom_mask[:, None, :]
        )
        seed = self._fixed_seed(atom_mask, q)
        hard_types = probabilities.argmax(dim=-1)
        return seed, hard_types

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
        local_geometry_priors: Optional[
            Dict[str, torch.Tensor]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        dtype = positions.dtype
        pair_mask_f = pair_mask.to(dtype)
        vector = positions[:, None, :, :] - positions[:, :, None, :]
        distance = torch.sqrt(vector.square().sum(dim=-1) + 1e-12)
        unit = vector / distance.unsqueeze(-1).clamp_min(1e-6)
        q = (1.0 - probabilities[..., NONE]) * pair_mask_f
        upper = torch.triu(torch.ones_like(pair_mask_f), diagonal=1)

        radii_sum = (
            covalent_radii[:, :, None] + covalent_radii[:, None, :]
        )
        target_by_type = (
            radii_sum.unsqueeze(-1) * self.bond_length_scales
        )
        bonded_probabilities = probabilities[..., 1:]
        bond_weights = (
            bonded_probabilities.pow(self.bond_probability_power)
            * pair_mask_f.unsqueeze(-1)
            * upper.unsqueeze(-1)
        )
        bond_values = torch.log(
            distance.unsqueeze(-1).clamp_min(1e-4)
            / target_by_type[..., 1:].clamp_min(1e-4)
        ).square()
        bond = (
            (bond_values * bond_weights).sum()
            if reduction == "force"
            else self._masked_mean(bond_values, bond_weights)
        )

        angle_weights = q.pow(self.angle_probability_power)
        moment = torch.einsum(
            "bij,bija,bijc->biac", angle_weights, unit, unit
        )
        direction_sum = torch.einsum(
            "bij,bija->bia", angle_weights, unit
        )
        total_weight = angle_weights.sum(dim=-1)
        squared_weight = angle_weights.square().sum(dim=-1)
        target_cosine = (
            geometry_probabilities * self.geometry_cosines
        ).sum(dim=-1)
        pair_weight = (
            total_weight.square() - squared_weight
        ).clamp_min(0.0)
        angle_numerator = (
            moment.square().sum(dim=(-1, -2))
            - 2.0
            * target_cosine
            * direction_sum.square().sum(dim=-1)
            + target_cosine.square() * total_weight.square()
            - squared_weight * (1.0 - target_cosine).square()
        ).clamp_min(0.0)
        angle_per_atom = (
            angle_numerator / pair_weight.clamp_min(1e-8)
        )
        angle_valid = atom_mask & pair_weight.gt(1e-6)
        angle = (
            (angle_per_atom * angle_valid).sum()
            if reduction == "force"
            else self._masked_mean(
                angle_per_atom, angle_valid.to(dtype)
            )
        )

        normalized_moment = (
            moment / total_weight.clamp_min(1e-8)[..., None, None]
        )
        determinant = (
            normalized_moment[..., 0, 0]
            * (
                normalized_moment[..., 1, 1]
                * normalized_moment[..., 2, 2]
                - normalized_moment[..., 1, 2]
                * normalized_moment[..., 2, 1]
            )
            - normalized_moment[..., 0, 1]
            * (
                normalized_moment[..., 1, 0]
                * normalized_moment[..., 2, 2]
                - normalized_moment[..., 1, 2]
                * normalized_moment[..., 2, 0]
            )
            + normalized_moment[..., 0, 2]
            * (
                normalized_moment[..., 1, 0]
                * normalized_moment[..., 2, 1]
                - normalized_moment[..., 1, 1]
                * normalized_moment[..., 2, 0]
            )
        )
        planar_weights = (
            geometry_probabilities[..., PLANAR_GEOMETRY_INDEX]
            * atom_mask
            * pair_weight.gt(1e-6)
        )
        planar = (
            (determinant.square() * planar_weights).sum()
            if reduction == "force"
            else self._masked_mean(
                determinant.square(), planar_weights
            )
        )

        # Generic vdW clash applies only beyond local 1--3 geometry. Bonded
        # 1--2 pairs are removed by (1 - q), while soft two-hop connectivity
        # removes 1--3 pairs whose distances are already controlled by the
        # angle/VSEPR term. Normalizing the saturating map makes one clean
        # two-edge path correspond to full exclusion without a hard threshold.
        two_hop_mass = torch.bmm(q, q)
        one_path_normalizer = 1.0 - torch.exp(
            two_hop_mass.new_tensor(-1.0)
        )
        analytic_one_three_probability = (
            (1.0 - torch.exp(-two_hop_mass)) / one_path_normalizer
        ).clamp(0.0, 1.0)
        one_three_probability = analytic_one_three_probability
        if local_geometry_priors is not None:
            learned_one_three = local_geometry_priors[
                "one_three_probability"
            ].clamp(0.0, 1.0)
            # Soft union retains the analytic path and lets the learned prior
            # restore a missed/low-confidence path without a hard threshold.
            one_three_probability = 1.0 - (
                1.0 - analytic_one_three_probability
            ) * (1.0 - learned_one_three)
        unbonded_weight = (
            (1.0 - q)
            * (1.0 - one_three_probability)
            * pair_mask_f
        )
        vdw_sum = vdw_radii[:, :, None] + vdw_radii[:, None, :]
        penetration = self.clash_softness * F.softplus(
            (
                self.clash_distance_scale * vdw_sum - distance
            ) / self.clash_softness
        )
        eye = torch.eye(
            penetration.size(1),
            device=penetration.device,
            dtype=torch.bool,
        )[None]
        valid_neighbor = pair_mask & ~eye
        # Apply the soft unbonded probability to the energy rather than to
        # penetration before squaring, so it retains its probability meaning.
        squared_penetration = unbonded_weight * penetration.square()
        temperature = self.clash_smoothmax_temperature
        logits = (
            squared_penetration / temperature
        ).masked_fill(~valid_neighbor, -torch.inf)
        valid_center = valid_neighbor.any(dim=-1)
        logits = torch.where(
            valid_center.unsqueeze(-1), logits, torch.zeros_like(logits)
        )
        valid_count = valid_neighbor.sum(dim=-1).clamp_min(1).to(dtype)
        per_atom_clash = temperature * (
            torch.logsumexp(logits, dim=-1) - valid_count.log()
        )
        per_atom_clash = torch.where(
            atom_mask & valid_center,
            per_atom_clash,
            torch.zeros_like(per_atom_clash),
        ).clamp_min(0.0)
        clash = (
            0.5 * (per_atom_clash * atom_mask).sum()
            if reduction == "force"
            else self._masked_mean(
                per_atom_clash, atom_mask.to(dtype)
            )
        )
        zero = positions.sum() * 0.0
        local_one_three = zero
        local_one_four = zero
        if local_geometry_priors is not None:
            radii_baseline = (
                covalent_radii[:, :, None]
                + covalent_radii[:, None, :]
            ).clamp_min(0.5)
            target_13 = (
                radii_baseline
                * local_geometry_priors["one_three_distance_ratio"]
            )
            target_14 = (
                radii_baseline
                * local_geometry_priors["one_four_distance_ratio"]
            )
            learned_13 = local_geometry_priors[
                "one_three_probability"
            ] * (1.0 - q)
            learned_14 = (
                local_geometry_priors["one_four_probability"]
                * local_geometry_priors.get(
                    "one_four_validity",
                    torch.ones_like(
                        local_geometry_priors[
                            "one_four_probability"
                        ]
                    ),
                )
                * (1.0 - q)
                * (1.0 - one_three_probability)
            )
            learned_13 = learned_13 * pair_mask_f * upper
            learned_14 = learned_14 * pair_mask_f * upper
            value_13 = torch.log(
                distance.clamp_min(1e-4) / target_13.clamp_min(1e-4)
            ).square()
            value_14 = torch.log(
                distance.clamp_min(1e-4) / target_14.clamp_min(1e-4)
            ).square()
            if reduction == "force":
                local_one_three = (value_13 * learned_13).sum()
                local_one_four = (value_14 * learned_14).sum()
            else:
                local_one_three = self._masked_mean(
                    value_13, learned_13
                )
                local_one_four = self._masked_mean(
                    value_14, learned_14
                )
        return {
            "bond": bond,
            "angle": angle,
            "planar": planar,
            "clash": clash,
            "one_three_distance": local_one_three,
            "one_four_distance": local_one_four,
        }

    def total(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.bond_weight * terms["bond"]
            + self.angle_weight * terms["angle"]
            + self.planar_weight * terms["planar"]
            + self.clash_weight * terms["clash"]
            + self.one_three_distance_weight
            * terms.get("one_three_distance", 0.0)
            + self.one_four_distance_weight
            * terms.get("one_four_distance", 0.0)
        )

    def _solve(
        self,
        probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        differentiable: bool,
        local_geometry_priors: Optional[
            Dict[str, torch.Tensor]
        ] = None,
        coordinate_seed: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        generator = None
        if coordinate_seed is not None:
            generator = torch.Generator(device=probabilities.device)
            generator.manual_seed(coordinate_seed)
        seed, seed_hard_types = self._make_seed(
            atom_mask,
            heavy_mask,
            probabilities,
            geometry_probabilities,
            covalent_radii,
            vdw_radii,
            local_geometry_priors,
            differentiable,
            generator,
        )
        with torch.enable_grad():
            positions = seed
            if not positions.requires_grad:
                positions = positions.detach().clone().requires_grad_(True)
            for _ in range(self.num_steps):
                force_terms = self.terms(
                    positions,
                    probabilities,
                    geometry_probabilities,
                    atom_mask,
                    pair_mask,
                    covalent_radii,
                    vdw_radii,
                    reduction="force",
                    local_geometry_priors=local_geometry_priors,
                )
                gradient = torch.autograd.grad(
                    self.total(force_terms),
                    positions,
                    create_graph=differentiable,
                )[0]
                norm = torch.sqrt(
                    gradient.square().sum(dim=-1, keepdim=True) + 1e-12
                )
                gradient = gradient / (
                    1.0 + norm / self.gradient_clip
                )
                positions = positions - self.step_size * gradient
                count = atom_mask.sum(
                    dim=1, keepdim=True
                ).clamp_min(1).to(positions.dtype)
                center = (
                    positions.sum(dim=1, keepdim=True)
                    / count.unsqueeze(-1)
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
                local_geometry_priors=local_geometry_priors,
            )
        if not differentiable:
            positions = positions.detach()
            seed = seed.detach()
            final_terms = {
                key: value.detach() for key, value in final_terms.items()
            }
        return seed, positions, final_terms, seed_hard_types

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor,
        heavy_mask: torch.Tensor,
        hydrogen_mask: torch.Tensor,
        heavy_edge_logits: torch.Tensor,
        h_attachment_logits: torch.Tensor,
        differentiable: bool = True,
        geometry_probabilities_override: Optional[
            torch.Tensor
        ] = None,
        local_geometry_priors: Optional[
            Dict[str, torch.Tensor]
        ] = None,
        coordinate_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        del heavy_mask, hydrogen_mask  # masks are encoded below by atom types.
        device_type = heavy_edge_logits.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            atomic_numbers = atomic_numbers.long()
            atom_mask = atom_mask.bool()
            supported = (
                self.covalent_radius_table[atomic_numbers].gt(0)
                & self.vdw_radius_table[atomic_numbers].gt(0)
            ) | ~atom_mask
            if not supported.all():
                unsupported = atomic_numbers[
                    atom_mask & ~supported
                ].unique().tolist()
                raise ValueError(
                    "No fixed radius prior for atomic numbers "
                    f"{unsupported}; extend the solver radius table."
                )
            heavy_mask = atom_mask & atomic_numbers.ne(1)
            hydrogen_mask = atom_mask & atomic_numbers.eq(1)
            atoms = atom_mask.size(1)
            diagonal = torch.eye(
                atoms, device=atom_mask.device, dtype=torch.bool
            )[None]
            pair_mask = (
                atom_mask[:, :, None] & atom_mask[:, None, :] & ~diagonal
            )
            heavy_pair_mask = (
                heavy_mask[:, :, None]
                & heavy_mask[:, None, :]
                & ~diagonal
            )
            attachment_mask = (
                hydrogen_mask[:, :, None] & heavy_mask[:, None, :]
            )
            graph = self.soft_graph(
                heavy_edge_logits.float(),
                h_attachment_logits.float(),
                pair_mask,
                heavy_pair_mask,
                attachment_mask,
            )
            analytic_geometry_probabilities = (
                self.soft_geometry_probabilities(
                    atomic_numbers,
                    atom_mask,
                    graph["edge_probabilities"],
                )
            )
            if geometry_probabilities_override is None:
                geometry_probabilities = analytic_geometry_probabilities
            else:
                learned_geometry = (
                    geometry_probabilities_override.float()
                    * atom_mask.unsqueeze(-1)
                )
                learned_geometry = learned_geometry / learned_geometry.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                # Retain an analytic gradient path from corrected edges while
                # allowing the pretrained head to resolve soft hybridisation.
                geometry_probabilities = (
                    0.2 * analytic_geometry_probabilities
                    + 0.8 * learned_geometry
                )
            covalent_radii = self.covalent_radius_table[atomic_numbers]
            vdw_radii = self.vdw_radius_table[atomic_numbers]
            seed, coordinates, terms, seed_hard_types = self._solve(
                graph["edge_probabilities"],
                geometry_probabilities,
                atom_mask,
                heavy_mask,
                pair_mask,
                covalent_radii,
                vdw_radii,
                differentiable=differentiable,
                local_geometry_priors=local_geometry_priors,
                coordinate_seed=coordinate_seed,
            )
        return {
            **graph,
            "geometry_probabilities": geometry_probabilities,
            "seed_coordinates": seed,
            "seed_hard_bond_types": seed_hard_types,
            "seed_mode": self.seed_mode,
            "coordinates": coordinates,
            "geometry_terms": terms,
            "covalent_radii": covalent_radii,
            "vdw_radii": vdw_radii,
            "pair_mask": pair_mask,
            "heavy_pair_mask": heavy_pair_mask,
            "attachment_mask": attachment_mask,
        }
