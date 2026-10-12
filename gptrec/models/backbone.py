"""The GPT-2 decoder stack shared by the recommender and the reward model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import GPT2Config, GPT2Model

EMBEDDING_INIT_STD = 0.02  # GPT-2's initializer_range


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_positions: int = 128  # sequence length the model is trained on; a history plus the items generated after it must fit


class CausalTransformer(nn.Module):
    """Hugging Face GPT-2 blocks applied to embeddings supplied by the caller.

    Attention is causal and positions count from 0, which is all a right-padded sequence needs.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.max_positions = config.max_positions
        self.gpt2 = GPT2Model(
            GPT2Config(
                vocab_size=1,  # GPT-2's own token table stays unused: callers pass `inputs_embeds`
                n_positions=config.max_positions,
                n_embd=config.hidden_size,
                n_layer=config.num_layers,
                n_head=config.num_heads,
                resid_pdrop=config.dropout,
                embd_pdrop=config.dropout,
                attn_pdrop=config.dropout,
                use_cache=False,
                bos_token_id=None,  # the sequences have no special tokens
                eos_token_id=None,
            )
        )

    def forward(self, token_embeddings: Tensor) -> Tensor:
        """[B, L, hidden] -> [B, L, hidden]; output position t depends on inputs 0..t only."""
        sequence_length = token_embeddings.shape[1]
        if sequence_length > self.max_positions:
            raise ValueError(f"sequence of {sequence_length} tokens exceeds max_positions={self.max_positions}")
        return self.gpt2(inputs_embeds=token_embeddings, use_cache=False).last_hidden_state


def new_item_embedding(num_items: int, hidden_size: int, padding_item: int) -> nn.Embedding:
    """Embedding table for item tokens 0..num_items with GPT-2's initialisation and an all-zero padding row."""
    embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=padding_item)
    nn.init.normal_(embedding.weight, std=EMBEDDING_INIT_STD)
    with torch.no_grad():
        embedding.weight[padding_item].zero_()
    return embedding
