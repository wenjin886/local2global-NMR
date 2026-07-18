import math
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .layers import MLP


class FourierEmbedding(nn.Module):
    """
    Random Fourier features (sine and cosine expansion).
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std: float = 1.0,
            trainable: bool = False,
    ):
        super(FourierEmbedding, self).__init__()
        assert (out_features % 2) == 0
        weight = torch.normal(mean=torch.zeros(out_features // 2, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, x):
        x = F.linear(x, self.weight)
        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)
        x = torch.cat((cos_features, sin_features), dim=-1)

        return x

class PeakFourierEmbedding(nn.Module):
    def __init__(self, 
                 num_nmr_fourier_features: int,
                 out_dim: int,
                 h_dim: Union[int, list],
                 n: int, # number of layers
                 activation: Union[Callable, nn.Module] = None,
                 last_linear: bool = True):
        super().__init__()
        self.fourier = FourierEmbedding(1, num_nmr_fourier_features, std=1.0)
        
        self.mlp = MLP(
            in_dim=num_nmr_fourier_features,
            out_dim=out_dim,
            h_dim=h_dim,
            n=n,
            activation=activation,
            last_linear=last_linear
        )
       
    
    def forward(self, delta_normalized):
        # delta_normalized 已经是 z-score normalized，直接用
        fourier_feat = self.fourier(delta_normalized)
        return self.mlp(fourier_feat)
    
    def reset_parameters(self):
        self.mlp.reset_parameters()


class AtomSlotEmbedding(nn.Module):
    """Embed atomic numbers and add learned slot queries.

    Slot embeddings deliberately break the symmetry between atoms of the same
    element.  The loss is responsible for treating exchangeable atoms in a
    permutation-invariant manner.
    """

    def __init__(
            self,
            hidden_dim: int,
            max_atomic_number: int = 100,
            max_num_atoms: int = 192,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.element_embedding = nn.Embedding(
            max_atomic_number + 1,
            hidden_dim,
            padding_idx=0,
        )
        self.slot_embedding = nn.Embedding(max_num_atoms, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        if atom_types.ndim != 2:
            raise ValueError("atom_types must have shape [batch, num_atoms]")
        if atom_types.size(1) > self.max_num_atoms:
            raise ValueError(
                "Received %d atoms, but max_num_atoms=%d"
                % (atom_types.size(1), self.max_num_atoms)
            )

        slots = torch.arange(atom_types.size(1), device=atom_types.device)
        slots = self.slot_embedding(slots)[None, :, :]
        features = self.element_embedding(atom_types) + slots
        return self.dropout(self.norm(features))


class NMRPeakEmbedding(nn.Module):
    """Shared embedding for unassigned proton and carbon peak tokens.

    ``nucleus_types`` uses 0 for padding, 1 for 1H, and 2 for 13C.  Shifts are
    standardized with nucleus-specific statistics before Fourier expansion.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_fourier_features: int = 64,
            fourier_std: float = 1.0,
            shift_mean: Sequence[float] = (0.0, 5.0, 100.0),
            shift_std: Sequence[float] = (1.0, 5.0, 60.0),
            dropout: float = 0.0,
    ):
        super().__init__()
        if num_fourier_features % 2 != 0:
            raise ValueError("num_fourier_features must be even")
        if len(shift_mean) != 3 or len(shift_std) != 3:
            raise ValueError("shift statistics must contain padding, 1H, and 13C")

        self.fourier = FourierEmbedding(
            in_features=1,
            out_features=num_fourier_features,
            std=fourier_std,
        )
        self.nucleus_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.shift_projection = nn.Linear(num_fourier_features, hidden_dim)
        self.integration_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer("shift_mean", torch.tensor(shift_mean, dtype=torch.float))
        self.register_buffer("shift_std", torch.tensor(shift_std, dtype=torch.float))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            shifts: torch.Tensor,
            nucleus_types: torch.Tensor,
            integrations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if shifts.shape != nucleus_types.shape:
            raise ValueError("shifts and nucleus_types must have identical shapes")
        nucleus_types = nucleus_types.long()
        mean = self.shift_mean[nucleus_types]
        std = self.shift_std[nucleus_types].clamp_min(1e-6)
        normalized = ((shifts - mean) / std).unsqueeze(-1)

        features = self.shift_projection(self.fourier(normalized))
        features = features + self.nucleus_embedding(nucleus_types)
        if integrations is not None:
            if integrations.shape != shifts.shape:
                raise ValueError("integrations and shifts must have identical shapes")
            features = features + self.integration_projection(
                torch.log1p(integrations.clamp_min(0.0)).unsqueeze(-1)
            )
        return self.dropout(self.norm(features))

def cosine_cutoff(edge_distances: torch.Tensor, cutoff: float):
    return torch.where(
        edge_distances < cutoff,
        .5 * (torch.cos(torch.pi * edge_distances / cutoff) + 1.),
        torch.tensor(0.0, device=edge_distances.device, dtype=edge_distances.dtype),
    )


class EdgeEmbedding(nn.Module):
    def __init__(self,
                 num_rbf_features: int = 64,
                 max_distance: float = 25.0,
                 trainable: bool = False,
                 norm: bool = True,
                 cutoff: bool = True):
        super().__init__()

        self.norm = norm
        self.n_rbf_features = num_rbf_features
        self.max_distance = max_distance
        self.cutoff = cutoff

        self.register_buffer("delta", torch.tensor(max_distance / num_rbf_features))
        offsets = torch.linspace(start=0., end=max_distance, steps=num_rbf_features).unsqueeze(0)
        if trainable:
            self.offsets = nn.Parameter(offsets)
        else:
            self.register_buffer("offsets", offsets)

    def forward(self,
                positions: torch.Tensor,
                edge_index: torch.Tensor,
                norm: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = self.norm if norm is None else norm
        dest, source = edge_index

        vectors = (
                positions[dest] - positions[source]
        )  # (n_edges, 3) vector (i - > j)

        distances = torch.sqrt(
            torch.sum(vectors ** 2, dim=-1, keepdim=True) + 1e-6
        )  # (n_edges, 1)
        d = self.featurize_distances(distances)

        cos = F.cosine_similarity(positions[dest], positions[source], dim=-1).unsqueeze(1)

        if norm:
            vectors = vectors / (distances + 1.0)

        return d, cos, vectors  # (n_edges, 1), (n_edges, 3)

    def featurize_distances(self, distances: torch.Tensor):
        distances = torch.clamp(distances, 0., self.max_distance)
        features = torch.exp((-((distances - self.offsets) ** 2)) / self.delta)

        if self.cutoff:
            features = features * cosine_cutoff(distances, cutoff=self.max_distance)

        return features


class CompositionEmbedding(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 h_dim: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Parameter(torch.randn(in_dim, h_dim[0]))
        self.mlp = MLP(in_dim=h_dim[0] * in_dim, out_dim=out_dim,
                       h_dim=h_dim,
                       n=len(h_dim),
                       activation=nn.SiLU())

    def forward(self, composition: torch.Tensor):
        norm_composition = composition / torch.sum(composition, dim=-1, keepdim=True)
        embedded_composition = norm_composition[..., None] * self.embedding[None, ...]
        embedded_composition = embedded_composition.flatten(start_dim=1)

        return self.mlp(embedded_composition)

    def reset_parameters(self):
        self.mlp.reset_parameters()
