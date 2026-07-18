import math
from typing import Optional, Tuple

import torch
from torch import nn


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that returns zeros instead of NaNs for fully masked rows."""
    mask = mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked_logits, dim=dim)
    probabilities = probabilities * mask.to(dtype=probabilities.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    return torch.where(
        normalizer > 0,
        probabilities / normalizer.clamp_min(torch.finfo(probabilities.dtype).eps),
        torch.zeros_like(probabilities),
    )


class MaskedCrossAttentionBlock(nn.Module):
    """Multi-head cross-attention with explicit query and key masks."""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tensor.shape
        tensor = tensor.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(
            self,
            query: torch.Tensor,
            context: torch.Tensor,
            query_mask: torch.Tensor,
            context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError("query and context must have shape [batch, tokens, hidden]")

        q = self._split_heads(self.q_projection(self.query_norm(query)))
        k = self._split_heads(self.k_projection(self.context_norm(context)))
        v = self._split_heads(self.v_projection(self.context_norm(context)))

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        pair_mask = query_mask[:, None, :, None] & context_mask[:, None, None, :]
        attention = _masked_softmax(logits, pair_mask, dim=-1)
        attended = torch.matmul(self.attention_dropout(attention), v)
        attended = attended.transpose(1, 2).contiguous().view_as(query)

        query_mask_float = query_mask.unsqueeze(-1).to(dtype=query.dtype)
        output = query + self.residual_dropout(self.out_projection(attended)) * query_mask_float
        output = output + self.residual_dropout(self.ffn(self.ffn_norm(output))) * query_mask_float
        output = output * query_mask_float
        return output, attention


class MaskedSelfAttentionEncoder(nn.Module):
    """Stacked masked self-attention implemented through cross-attention blocks."""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            mlp_ratio: int = 4,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MaskedCrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
            self,
            features: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention = None
        for layer in self.layers:
            features, attention = layer(features, features, mask, mask)
        return features, attention
