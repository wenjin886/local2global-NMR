"""Parameter-free differentiable soft-graph to 3D coordinate initializer.

Design goals
------------
* Fully differentiable w.r.t. the incoming soft edge probabilities. There is no
  argmax, no shortest-path search and no eigendecomposition anywhere in the
  forward pass, so gradients reach the 2D-graph predictor unimpeded.
* Every local geometric target is derived analytically from chemical priors:
    - 1--2 distance  = sum of covalent radii, scaled by bond order.
    - 1--3 distance  = law of cosines with the VSEPR hybridisation angle.
    - 1--4 / longer  = one-sided lower bounds from dense soft walk mass, soft
                       graph distance and van der Waals excluded volume.
* No top-k neighbour selection or hard graph thresholds occur in the forward
  path. Low-confidence and mis-ranked edges therefore retain bond, angle and
  long-range gradients instead of falling out of a discrete support.
* Zero learnable parameters. Optional pairwise torsion logits can continuously
  mix compact / gauche-like / extended 1--4 lower-bound anchors.

The output is a bounds-matrix embedding produced by unrolled weighted stress
minimisation. By default all iterations build an autograd graph. Setting
`unroll_steps` explicitly enables truncated backpropagation as an optional
memory/gradient-accuracy trade-off.

Chirality is deliberately undetermined: the objective depends only on pairwise
distances, so enantiomers are degenerate. This matches a downstream
E(3)-equivariant refiner and a parity-invariant shift decoder.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import (
    AROMATIC,
    BOND_LENGTH_SCALES,
    GEOMETRY_COSINES,
)

# Torsion anchors: cos(0deg) syn, cos(60deg) gauche, cos(180deg) anti.
TORSION_ANCHOR_COSINES = (1.0, 0.5, -1.0)
NUM_TORSION_ANCHORS = len(TORSION_ANCHOR_COSINES)

COVALENT_RADII = {
    1: 0.31, 5: 0.85, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}
VDW_RADII = {
    1: 1.20, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}


class PriorGeometryInitializer(nn.Module):
    """Soft bond probabilities -> 3D coordinates, no learnable parameters."""

    def __init__(
        self,
        max_degree: int = 5,
        support_threshold: float = 0.15,
        num_hops: int = 6,
        hop_scale: float = 1.20,
        hop_saturation: float = 6.0,
        vdw_scale: float = 0.80,
        anchor_prior_strength: float = 3.0,
        bond_weight: float = 1.0,
        angle_weight: float = 2.00,
        torsion_weight: float = 1.00,
        conjugation_weight: float = 2.00,
        clash_weight: float = 0.05,
        bound_softness: float = 0.20,
        attraction_radius: float = 0.5,
        reach_scale: float = 0.20,
        anneal_hold: float = 0.65,
        anneal_floor: float = 0.45,
        num_steps: int = 400,
        unroll_steps: Optional[int] = None,
        step_size: float = 0.10,
        momentum: float = 0.90,
        max_displacement: float = 0.35,
        init_scale: float = 2.0,
        num_restarts: int = 1,
    ) -> None:
        super().__init__()
        # Kept only for checkpoint/config compatibility. Dense soft support
        # deliberately ignores both legacy sparsification settings.
        self.max_degree = max_degree
        self.support_threshold = support_threshold
        self.num_hops = num_hops
        self.hop_scale = hop_scale
        self.hop_saturation = hop_saturation
        self.vdw_scale = vdw_scale
        self.anchor_prior_strength = anchor_prior_strength
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.torsion_weight = torsion_weight
        self.conjugation_weight = conjugation_weight
        self.clash_weight = clash_weight
        self.bound_softness = bound_softness
        self.attraction_radius = attraction_radius
        self.reach_scale = reach_scale
        self.anneal_hold = anneal_hold
        self.anneal_floor = anneal_floor
        self.num_steps = num_steps
        self.unroll_steps = (
            num_steps if unroll_steps is None else min(num_steps, unroll_steps)
        )
        if self.unroll_steps < 0:
            raise ValueError("unroll_steps must be non-negative or None")
        self.step_size = step_size
        self.momentum = momentum
        self.max_displacement = max_displacement
        self.init_scale = init_scale
        self.num_restarts = max(1, num_restarts)

        self.register_buffer(
            "bond_length_scales",
            torch.tensor(BOND_LENGTH_SCALES, dtype=torch.float32),
        )
        self.register_buffer(
            "geometry_cosines",
            torch.tensor(GEOMETRY_COSINES, dtype=torch.float32),
        )
        covalent = torch.zeros(119, dtype=torch.float32)
        vdw = torch.zeros(119, dtype=torch.float32)
        for number, radius in COVALENT_RADII.items():
            covalent[number] = radius
        for number, radius in VDW_RADII.items():
            vdw[number] = radius
        self.register_buffer(
            "torsion_cosines",
            torch.tensor(TORSION_ANCHOR_COSINES, dtype=torch.float32),
        )
        self.register_buffer("covalent_radius_table", covalent)
        self.register_buffer("vdw_radius_table", vdw)

    # ------------------------------------------------------------------
    # graph support
    # ------------------------------------------------------------------

    def _soft_hop_distance(
        self,
        q: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable expected graph distance via truncated walk counts."""
        walk = q
        reach = 1.0 - torch.exp(-walk)
        hop = reach
        previous = reach
        for order in range(2, self.num_hops + 1):
            walk = torch.bmm(walk, q).clamp_max(64.0)
            reach = 1.0 - torch.exp(-walk)
            hop = hop + order * (reach - previous).clamp_min(0.0)
            previous = torch.maximum(previous, reach)
        hop = hop + (self.num_hops + 1) * (1.0 - previous)
        return hop * pair_mask

    def _soft_reach(self, path_mass: torch.Tensor) -> torch.Tensor:
        """Continuous probability-like map from non-negative walk mass.

        Quadratic near the origin rather than linear. With a dense support the
        number of spurious low-mass pairs grows as N^2, so a map with
        reach(x) ~ x lets their aggregate weight rival the real shell. A
        quadratic onset suppresses mass ~1e-2 by two orders of magnitude while
        leaving a genuinely uncertain edge (mass ~0.3) essentially untouched.
        """
        scaled = path_mass.clamp_min(0.0) / self.reach_scale
        return -torch.expm1(-scaled.square())

    # ------------------------------------------------------------------
    # analytic distance targets
    # ------------------------------------------------------------------

    def _bond_targets(
        self,
        probabilities: torch.Tensor,
        covalent_radii: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expected 1--2 length and total bond probability."""
        bonded = probabilities[..., 1:]
        q = bonded.sum(dim=-1) * pair_mask
        share = bonded / bonded.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        radii_sum = (
            covalent_radii[:, :, None] + covalent_radii[:, None, :]
        )
        by_type = radii_sum.unsqueeze(-1) * self.bond_length_scales[1:]
        length = (share * by_type).sum(dim=-1)
        return length.clamp_min(0.5), q

    def _dense_one_three(
        self,
        q: torch.Tensor,
        bond_length: torch.Tensor,
        cos_theta: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense soft law-of-cosines average over every i--j--k path.

        The centre loop has static tensor indices and does not select graph
        support. Complexity is O(B*N^3) with O(B*N^2) peak memory.
        """
        batch, atoms, _ = q.shape
        numerator = q.new_zeros((batch, atoms, atoms))
        path_mass = torch.zeros_like(numerator)
        for center in range(atoms):
            left_weight = q[:, :, center]
            right_weight = q[:, center, :]
            weight = (
                left_weight.unsqueeze(-1)
                * right_weight.unsqueeze(-2)
                * pair_mask
            )
            left_length = bond_length[:, :, center].unsqueeze(-1)
            right_length = bond_length[:, center, :].unsqueeze(-2)
            cosine = cos_theta[:, center, None, None].clamp(-1.0, 1.0)
            chord = torch.sqrt(
                (
                    left_length.square()
                    + right_length.square()
                    - 2.0 * left_length * right_length * cosine
                ).clamp_min(1e-4)
            )
            numerator = numerator + weight * chord
            path_mass = path_mass + weight
        target = numerator / path_mass.clamp_min(1e-8)
        return target, path_mass


    @staticmethod
    def _gather_rows(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch, atoms, channels = source.shape
        degree = index.size(-1)
        flat = index.reshape(batch, atoms * degree, 1).expand(-1, -1, channels)
        return torch.gather(source, 1, flat).reshape(
            batch, atoms, degree, channels
        )

    def _one_four_anchors(
        self,
        q: torch.Tensor,
        bond_length: torch.Tensor,
        cos_theta: torch.Tensor,
        heavy: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact four-body 1--4 distances at three torsion anchors.

        Enumerated on a top-k support (three-hop paths cannot be formed densely
        below O(N^4)), but *scattered into a dense (B, N, N, 3) tensor* so that
        the torsion mixture and any learned logits stay dense and pairwise.
        Restricting only this term to a support costs little gradient quality:
        the signal that corrects a missing or mis-ranked bond flows through the
        1--2 and 1--3 terms, which remain fully dense.
        """
        batch, atoms, _ = q.shape
        device = q.device
        degree = min(self.max_degree, atoms)
        weights, index = q.topk(degree, dim=-1)
        neighbor_is_atom = torch.gather(
            atom_mask.unsqueeze(1).expand(-1, atoms, -1).to(torch.bool), 2, index
        )
        valid = weights.gt(self.support_threshold) & neighbor_is_atom
        valid = valid & atom_mask.unsqueeze(-1).to(torch.bool)
        weights = weights * valid

        flat_index = index.reshape(batch, atoms * degree)
        neighbor_length = torch.gather(bond_length, 2, index)
        neighbor_of_k = self._gather_rows(index, index)
        weight_of_k = self._gather_rows(weights, index)
        valid_of_k = self._gather_rows(valid.to(q.dtype), index).gt(0.5)
        length_of_k = self._gather_rows(neighbor_length, index)
        cos_of_k = torch.gather(cos_theta, 1, flat_index).reshape(
            batch, atoms, degree
        )

        d_ij = neighbor_length[:, :, :, None, None]
        d_jk = neighbor_length[:, :, None, :, None]
        d_kl = length_of_k[:, :, None, :, :]
        cos_j = cos_theta[:, :, None, None, None].clamp(-1.0, 1.0)
        cos_k = cos_of_k[:, :, None, :, None].clamp(-1.0, 1.0)
        sin_j = (1.0 - cos_j.square()).clamp_min(0.0).sqrt()
        sin_k = (1.0 - cos_k.square()).clamp_min(0.0).sqrt()

        base = (
            d_ij.square() + d_jk.square() + d_kl.square()
            - 2.0 * d_ij * d_jk * cos_j
            - 2.0 * d_jk * d_kl * cos_k
            + 2.0 * d_ij * d_kl * cos_j * cos_k
        )
        swing = 2.0 * d_ij * d_kl * sin_j * sin_k
        anchors = self.torsion_cosines.view(1, 1, 1, 1, 1, -1)
        anchor_distance = (
            base.unsqueeze(-1) - swing.unsqueeze(-1) * anchors
        ).clamp_min(1e-4).sqrt()

        index_i = index[:, :, :, None, None].expand(-1, -1, -1, degree, degree)
        index_l = neighbor_of_k[:, :, None, :, :].expand(-1, -1, degree, -1, -1)
        centre = torch.arange(atoms, device=device).view(1, -1, 1, 1, 1)
        slot_eye = torch.eye(degree, device=device, dtype=torch.bool)
        path_valid = (
            valid[:, :, :, None, None]
            & valid[:, :, None, :, None]
            & valid_of_k[:, :, None, :, :]
            & ~slot_eye[None, None, :, :, None]
            & index_l.ne(centre)
            & index_i.ne(index_l)
        )
        path_weight = (
            weights[:, :, :, None, None]
            * weights[:, :, None, :, None]
            * weight_of_k[:, :, None, :, :]
        ) * path_valid

        # A torsion preference may only be imposed on the heavy skeleton: four
        # H...H pairs share one central bond and cannot all be anti at once.
        # Paths touching a hydrogen collapse onto the syn anchor, which is the
        # unconditional lower bound over every torsion angle.
        skeleton = (
            torch.gather(heavy, 1, index_i.reshape(batch, -1))
            * torch.gather(heavy, 1, index_l.reshape(batch, -1))
        ).reshape_as(path_weight)
        anchor_distance = (
            skeleton.unsqueeze(-1) * anchor_distance
            + (1.0 - skeleton).unsqueeze(-1)
            * anchor_distance[..., :1]
        )

        flat = (index_i * atoms + index_l).reshape(batch, -1)
        mass = q.new_zeros((batch, atoms * atoms)).scatter_add(
            1, flat, path_weight.reshape(batch, -1)
        )
        numerator = q.new_zeros((batch, atoms * atoms, NUM_TORSION_ANCHORS))
        numerator = numerator.scatter_add(
            1,
            flat.unsqueeze(-1).expand(-1, -1, NUM_TORSION_ANCHORS),
            (anchor_distance * path_weight.unsqueeze(-1)).reshape(
                batch, -1, NUM_TORSION_ANCHORS
            ),
        )
        mass = mass.reshape(batch, atoms, atoms)
        numerator = numerator.reshape(batch, atoms, atoms, NUM_TORSION_ANCHORS)
        return numerator / mass.clamp_min(1e-8).unsqueeze(-1), mass

    # ------------------------------------------------------------------
    # bounds assembly
    # ------------------------------------------------------------------

    def bounds(
        self,
        probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        covalent_radii: torch.Tensor,
        vdw_radii: torch.Tensor,
        heavy: torch.Tensor,
        torsion_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pair_mask_f = pair_mask.to(probabilities.dtype)
        bond_length, q = self._bond_targets(
            probabilities, covalent_radii, pair_mask_f
        )
        cos_theta = (
            geometry_probabilities * self.geometry_cosines
        ).sum(dim=-1)

        chord, two_hop_mass = self._dense_one_three(
            q, bond_length, cos_theta, pair_mask_f
        )
        # Sequential soft memberships avoid hard q/weight thresholds while
        # assigning each pair predominantly to its shortest supported scale.
        bond_membership = q.clamp(0.0, 1.0)
        one_three_membership = (
            (1.0 - bond_membership)
            * self._soft_reach(two_hop_mass)
        )
        three_hop_mass = torch.bmm(torch.bmm(q, q), q) * pair_mask_f
        one_four_membership = (
            (1.0 - bond_membership)
            * (1.0 - one_three_membership)
            * self._soft_reach(three_hop_mass)
        )
        nonlocal_membership = (
            (1.0 - bond_membership)
            * (1.0 - one_three_membership)
            * (1.0 - one_four_membership)
        ) * pair_mask_f

        # Two-sided targets: exact 1--2 and 1--3 geometry. Note that three
        # 120-degree angles around one centre sum to 360 and therefore force
        # that centre to be planar, so conjugated systems need no separate
        # improper term as long as the 1--3 constraints are tight.
        bond_target_weight = self.bond_weight * bond_membership
        angle_target_weight = self.angle_weight * one_three_membership
        target_weight = (
            bond_target_weight + angle_target_weight
        ) * pair_mask_f
        target = (
            bond_target_weight * bond_length
            + angle_target_weight * chord
        ) / target_weight.clamp_min(1e-8)

        # One-sided lower bounds: torsion for 1--4, excluded volume beyond.
        hop = self._soft_hop_distance(q, pair_mask_f)
        saturating = self.hop_scale * hop / (1.0 + hop / self.hop_saturation)
        vdw_sum = vdw_radii[:, :, None] + vdw_radii[:, None, :]
        excluded = self.vdw_scale * vdw_sum
        long_range = torch.maximum(saturating, excluded) * pair_mask_f

        # Pairwise torsion logits softly mix compact, gauche-like and extended
        # lower bounds. The old neighbour-slot logits depended on top-k indices
        # and therefore cannot provide continuous graph correction.
        geometric_anchor, anchor_mass = self._one_four_anchors(
            q, bond_length, cos_theta, heavy, atom_mask
        )
        aromatic = probabilities[..., AROMATIC] * pair_mask_f
        conjugated_mass = torch.bmm(torch.bmm(q, aromatic), q)
        anchor_lower = torch.maximum(
            excluded.unsqueeze(-1), geometric_anchor
        )
        # A pair reachable by two or more independent three-hop paths closes a
        # ring, and a ring cannot adopt the anti torsion its acyclic prior
        # would ask for (cyclohexane 1--4 is 2.96 A, benzene para 2.78 A, both
        # far below the 3.85 A of an extended chain). Forcing anti there
        # strains the ring and the optimiser compensates by stretching bonds,
        # so the prior must be gated on this ring evidence.
        ring_gate = (anchor_mass - 1.0).clamp(0.0, 1.0)
        aromatic_share = self._soft_reach(conjugated_mass) * one_four_membership
        if torsion_logits is None:
            strength = self.anchor_prior_strength
            prior_logits = torch.stack(
                [
                    # syn: aromatic rings are flat and cis-locked
                    strength * aromatic_share,
                    # gauche: saturated rings must be able to close
                    strength * ring_gate * (1.0 - aromatic_share),
                    # anti: acyclic chains extend
                    strength * (1.0 - ring_gate),
                ],
                dim=-1,
            )
            torsion_mixture = torch.softmax(prior_logits, dim=-1)
        else:
            expected_shape = (*q.shape, NUM_TORSION_ANCHORS)
            if torsion_logits.shape != expected_shape:
                raise ValueError(
                    "torsion_logits must have dense pairwise shape "
                    f"{expected_shape}, got {tuple(torsion_logits.shape)}"
                )
            torsion_mixture = torch.softmax(torsion_logits.float(), dim=-1)
        one_four_lower = (
            torsion_mixture * anchor_lower
        ).sum(dim=-1)
        lower = (
            one_four_membership * one_four_lower
            + (1.0 - one_four_membership) * long_range
        ) * pair_mask_f

        # q @ aromatic @ q is the dense soft mass of i-j-k-l paths whose
        # central bond is aromatic. It continuously retains conjugated lower
        # bounds during annealing.
        conjugated_membership = (
            one_four_membership
            * self._soft_reach(conjugated_mass)
        )
        flexible_one_four = (
            one_four_membership - conjugated_membership
        ).clamp_min(0.0)
        rigid_lower_weight = (
            self.conjugation_weight
            * self.torsion_weight
            * conjugated_membership
        ) * pair_mask_f
        lower_weight = (
            self.torsion_weight * flexible_one_four
            + self.clash_weight * nonlocal_membership
        ) * pair_mask_f

        return {
            "target": target * pair_mask_f,
            "target_weight": target_weight,
            "lower": lower * pair_mask_f,
            "lower_weight": lower_weight,
            "rigid_lower_weight": rigid_lower_weight,
            "bond_probability": q,
            "hop": hop,
            "one_three_weight": one_three_membership,
            "one_four_weight": one_four_membership,
        }

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------

    def _stress(
        self,
        positions: torch.Tensor,
        bounds: Dict[str, torch.Tensor],
        pair_mask: torch.Tensor,
        bound_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        difference = positions[:, :, None, :] - positions[:, None, :, :]
        distance = difference.square().sum(dim=-1).clamp_min(1e-6).sqrt()
        unit = difference / distance.unsqueeze(-1)

        # Bounded attractive residual. With a dense soft support there are
        # O(N^2) near-zero-weight pairs; a linear w*(d-t) lets their aggregate
        # force dominate the O(N) real bonds, because (d-t) grows with
        # separation while w does not shrink fast enough. Saturating the
        # residual keeps a full gradient for genuinely uncertain edges while
        # making distant spurious pairs harmless.
        offset = distance - bounds["target"]
        radius = self.attraction_radius
        residual = bounds["target_weight"] * offset / (
            1.0 + offset.abs() / radius
        )
        softness = self.bound_softness
        violation = softness * F.softplus(
            (bounds["lower"] - distance) / softness
        )
        gate = torch.sigmoid((bounds["lower"] - distance) / softness)
        # Annealed bounds decay; conjugation-locked ones stay at full force.
        lower_weight = (
            bound_scale * bounds["lower_weight"]
            + bounds["rigid_lower_weight"]
        )
        residual = residual - 2.0 * lower_weight * violation * gate

        residual = residual * pair_mask
        force = (residual.unsqueeze(-1) * unit).sum(dim=2)

        saturated = radius * (
            offset.abs() - radius * torch.log1p(offset.abs() / radius)
        )
        energy = (
            bounds["target_weight"] * saturated
            + lower_weight * violation.square()
        )
        energy = (energy * pair_mask).sum(dim=(1, 2)) * 0.5
        return force, energy

    def _embed(
        self,
        bounds: Dict[str, torch.Tensor],
        atom_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        differentiable: bool,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, atoms = atom_mask.shape
        dtype = bounds["target"].dtype
        device = bounds["target"].device
        mask = atom_mask.unsqueeze(-1).to(dtype)
        pair_mask_f = pair_mask.to(dtype)

        # Random start is not an implementation detail: a deterministic
        # permutation-equivariant map cannot break graph automorphisms, so
        # symmetric molecules would collapse onto a point without it.
        positions = torch.randn(
            (batch, atoms, 3),
            device=device,
            dtype=dtype,
            generator=generator,
        ) * self.init_scale
        positions = (positions - positions.mean(dim=1, keepdim=True)) * mask
        velocity = torch.zeros_like(positions)

        graph_from = max(0, self.num_steps - self.unroll_steps)
        for step in range(self.num_steps):
            # Anneal the one-sided bounds: hold them at full strength while the
            # global shape is decided, then decay so that the exact 1--2 / 1--3
            # geometry can settle without being strained by torsion and
            # excluded-volume pressure. Without this the optimiser prefers to
            # open bond angles rather than rotate a torsion, and sp3 centres
            # end up near 120 degrees.
            progress = step / max(1, self.num_steps - 1)
            if progress <= self.anneal_hold:
                bound_scale = 1.0
            else:
                decay = (progress - self.anneal_hold) / max(
                    1e-6, 1.0 - self.anneal_hold
                )
                bound_scale = max(self.anneal_floor, 1.0 - decay)
            build_graph = differentiable and step >= graph_from

            def update(
                current_positions: torch.Tensor,
                current_velocity: torch.Tensor,
                current_bound_scale: float = bound_scale,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                force, _ = self._stress(
                    current_positions,
                    bounds,
                    pair_mask_f,
                    current_bound_scale,
                )
                next_velocity = self.momentum * current_velocity + force
                displacement = self.step_size * next_velocity
                norm = displacement.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                scale = (norm / self.max_displacement).clamp_min(1.0)
                next_positions = current_positions - displacement / scale
                next_positions = next_positions - (
                    (next_positions * mask).sum(dim=1, keepdim=True)
                    / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                )
                return next_positions * mask, next_velocity

            if build_graph:
                # Non-reentrant checkpointing recomputes each stress update
                # during backward, retaining the exact full-unroll gradient
                # without storing all O(num_steps*N^2) activations.
                positions, velocity = checkpoint(
                    update,
                    positions,
                    velocity,
                    use_reentrant=False,
                )
            else:
                with torch.no_grad():
                    positions, velocity = update(positions, velocity)
                positions = positions.detach()
                velocity = velocity.detach()

        _, energy = self._stress(positions, bounds, pair_mask_f)
        return positions, energy

    # ------------------------------------------------------------------

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atom_mask: torch.Tensor,
        edge_probabilities: torch.Tensor,
        geometry_probabilities: torch.Tensor,
        torsion_logits: Optional[torch.Tensor] = None,
        differentiable: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Soft graph -> coordinates.

        Args:
            atomic_numbers: (B, N) long.
            atom_mask: (B, N) bool.
            edge_probabilities: (B, N, N, NUM_BOND_TYPES), symmetric, softmax
                over bond types. Compatible with
                `DifferentiableGeometrySolver.soft_graph`.
            geometry_probabilities: (B, N, len(GEOMETRY_NAMES)).
            torsion_logits: optional dense pairwise (B, N, N, 3) logits that
                mix compact, gauche-like and extended 1--4 lower bounds.
        """
        atomic_numbers = atomic_numbers.long()
        atom_mask = atom_mask.bool()
        atoms = atom_mask.size(1)
        diagonal = torch.eye(atoms, device=atom_mask.device, dtype=torch.bool)
        pair_mask = (
            atom_mask[:, :, None] & atom_mask[:, None, :] & ~diagonal[None]
        )
        covalent_radii = self.covalent_radius_table[atomic_numbers]
        vdw_radii = self.vdw_radius_table[atomic_numbers]

        bounds = self.bounds(
            edge_probabilities.float(),
            geometry_probabilities.float(),
            atom_mask,
            pair_mask,
            covalent_radii,
            vdw_radii,
            (atom_mask & atomic_numbers.ne(1)).to(edge_probabilities.dtype),
            torsion_logits,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=atom_mask.device)
            generator.manual_seed(seed)

        best_positions: Optional[torch.Tensor] = None
        best_energy: Optional[torch.Tensor] = None
        if differentiable and self.num_restarts != 1:
            raise ValueError(
                "Fully differentiable mode requires num_restarts=1; hard "
                "best-restart selection is only available for evaluation."
            )
        for _ in range(self.num_restarts):
            positions, energy = self._embed(
                bounds, atom_mask, pair_mask, differentiable, generator
            )
            if best_positions is None:
                best_positions, best_energy = positions, energy
            else:
                take = energy.lt(best_energy)[:, None, None]
                best_positions = torch.where(take, positions, best_positions)
                best_energy = torch.minimum(energy, best_energy)

        return {
            "coordinates": best_positions,
            "stress": best_energy,
            "pair_mask": pair_mask,
            **bounds,
        }
