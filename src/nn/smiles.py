from typing import Dict, Optional

import torch
from torch import nn

from src.data.constants import (
    SMILES_BOS_INDEX,
    SMILES_EOS_INDEX,
    SMILES_PAD_INDEX,
)


class NMRToSMILESDecoder(nn.Module):
    """Causal SMILES decoder conditioned on the encoded NMR peak set."""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            vocab_size: int,
            max_length: int = 256,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=SMILES_PAD_INDEX
        )
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def decode(
            self,
            input_ids: torch.Tensor,
            input_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
    ):
        length = input_ids.size(1)
        if length > self.max_length:
            raise ValueError(
                f"SMILES length {length} exceeds max_smiles_length={self.max_length}"
            )
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones((length, length), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~input_mask.bool(),
            memory_key_padding_mask=~memory_mask.bool(),
        )
        hidden = self.norm(hidden)
        return hidden, self.output_projection(hidden)

    def teacher_force(
            self,
            input_ids: torch.Tensor,
            input_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            joint_fusion: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden, logits = self.decode(input_ids, input_mask, memory, memory_mask)
        updated_memory = memory
        fusion_attention = None
        if joint_fusion is not None:
            hidden, updated_memory, fusion_attention = joint_fusion(
                left=hidden,
                right=memory,
                left_mask=input_mask.bool(),
                right_mask=memory_mask.bool(),
            )
            logits = self.output_projection(hidden)
        return {
            "hidden_states": hidden,
            "logits": logits,
            "mask": input_mask.bool(),
            "token_ids": logits.argmax(dim=-1),
            "teacher_forced": True,
            "updated_memory": updated_memory,
            "fusion_attention": fusion_attention,
        }

    def generate(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            max_steps: int,
            joint_fusion: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        if max_steps > self.max_length:
            raise ValueError(
                f"Requested {max_steps} steps, but max_smiles_length={self.max_length}"
            )
        batch_size = memory.size(0)
        input_ids = torch.full(
            (batch_size, 1), SMILES_BOS_INDEX,
            dtype=torch.long, device=memory.device,
        )
        input_mask = torch.ones_like(input_ids, dtype=torch.bool)
        active = torch.ones(batch_size, dtype=torch.bool, device=memory.device)
        hidden_steps = []
        logit_steps = []
        output_masks = []
        generated_ids = []
        updated_memory = memory
        fusion_attention = None
        for _ in range(max_steps):
            hidden, logits = self.decode(
                input_ids, input_mask, memory, memory_mask
            )
            if joint_fusion is not None:
                hidden, updated_memory, fusion_attention = joint_fusion(
                    left=hidden,
                    right=memory,
                    left_mask=input_mask,
                    right_mask=memory_mask.bool(),
                )
                logits = self.output_projection(hidden)
            step_hidden = hidden[:, -1]
            step_logits = logits[:, -1]
            next_ids = step_logits.argmax(dim=-1)
            next_ids = torch.where(
                active, next_ids, torch.full_like(next_ids, SMILES_PAD_INDEX)
            )
            hidden_steps.append(step_hidden)
            logit_steps.append(step_logits)
            output_masks.append(active)
            generated_ids.append(next_ids)

            next_input_valid = active
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)
            input_mask = torch.cat(
                [input_mask, next_input_valid[:, None]], dim=1
            )
            active = active & next_ids.ne(SMILES_EOS_INDEX)

        return {
            "hidden_states": torch.stack(hidden_steps, dim=1),
            "logits": torch.stack(logit_steps, dim=1),
            "mask": torch.stack(output_masks, dim=1),
            "token_ids": torch.stack(generated_ids, dim=1),
            "teacher_forced": False,
            "updated_memory": updated_memory,
            "fusion_attention": fusion_attention,
        }
