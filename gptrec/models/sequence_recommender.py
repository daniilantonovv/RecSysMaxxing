"""What every next-item model that ranks by "sequence state . item embedding" has in common."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gptrec.data.log import PAD_ITEM

# Finite stand-in for -inf: keeps p * log(p) and log(p) - log(q) free of NaN for items that cannot be chosen.
MASKED_LOGIT = -1e9


class NextItemScorer(Protocol):
    """Anything that can rank the catalogue for a user: all that score-and-rank recommendation needs."""

    def next_item_logits(self, items: Tensor, lengths: Tensor) -> Tensor:
        """Scores of the item that follows each right-padded sequence. items: long [B, L], lengths: long [B] -> [B, V+1]"""
        ...


class SequenceRecommender(nn.Module):
    """A causal sequence model whose item scores are dot products of its state with the item embeddings.

    Subclasses own `item_embedding` (row PAD_ITEM is padding) and implement `hidden_states`. Ranking,
    item-by-item generation and full-softmax training are then the same for all of them.
    """

    item_embedding: nn.Embedding

    def hidden_states(self, items: Tensor) -> Tensor:
        """long [B, L] -> [B, L, hidden]; state t summarises items 0..t and nothing after them."""
        raise NotImplementedError

    def item_logits(self, hidden: Tensor) -> Tensor:
        """[..., hidden] -> float32 [..., V+1] scores of every item token; PAD_ITEM can never be chosen."""
        logits = F.linear(hidden, self.item_embedding.weight).float()
        logits[..., PAD_ITEM] = MASKED_LOGIT
        return logits

    def forward(self, items: Tensor) -> Tensor:
        """long [B, L] -> [B, L, V+1]; row t scores the item that follows items 0..t."""
        return self.item_logits(self.hidden_states(items))

    def next_item_logits(self, items: Tensor, lengths: Tensor) -> Tensor:
        """Scores of the item that follows each right-padded sequence. [B, V+1]"""
        last_position = lengths - 1
        return self.item_logits(self.hidden_states(items)[torch.arange(len(items), device=items.device), last_position])
