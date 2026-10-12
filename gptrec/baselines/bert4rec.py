"""BERT4Rec: a bidirectional Transformer trained to fill in hidden items (Sun et al., 2019)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gptrec.data.batches import EventBatch
from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import EMBEDDING_INIT_STD, TransformerConfig, new_item_embedding
from gptrec.models.sequence_recommender import MASKED_LOGIT


class BERT4Rec(nn.Module):
    """Predicts the next item by appending a mask token to the history and filling it in."""

    def __init__(self, num_items: int, config: TransformerConfig) -> None:
        super().__init__()
        self.num_items = num_items
        self.mask_token = num_items + 1
        self.max_positions = config.max_positions
        self.item_embedding = new_item_embedding(num_items + 1, config.hidden_size, PAD_ITEM)  # one extra row for the mask token
        self.position_embedding = nn.Embedding(config.max_positions, config.hidden_size)
        nn.init.normal_(self.position_embedding.weight, std=EMBEDDING_INIT_STD)
        self.embedding_dropout = nn.Dropout(config.dropout)
        block = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_heads,
            dim_feedforward=4 * config.hidden_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, config.num_layers, norm=nn.LayerNorm(config.hidden_size), enable_nested_tensor=False)
        self.item_bias = nn.Parameter(torch.zeros(num_items + 1))

    def hidden_states(self, tokens: Tensor) -> Tensor:
        """long [B, L] of item and mask tokens, right-padded -> [B, L, hidden]. Every position sees all real positions."""
        length = tokens.shape[1]
        if length > self.max_positions:
            raise ValueError(f"sequence of {length} tokens exceeds max_positions={self.max_positions}")
        positions = torch.arange(length, device=tokens.device)
        embedded = self.embedding_dropout(self.item_embedding(tokens) + self.position_embedding(positions))
        return self.blocks(embedded, src_key_padding_mask=tokens == PAD_ITEM)

    def item_logits(self, hidden: Tensor) -> Tensor:
        """[..., hidden] -> float32 [..., V+1] scores of the item behind a mask; PAD_ITEM can never be chosen."""
        logits = F.linear(hidden, self.item_embedding.weight[: self.num_items + 1], self.item_bias).float()
        logits[..., PAD_ITEM] = MASKED_LOGIT
        return logits

    def logits_at(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """Item scores at one position per sequence. tokens: [B, L], positions: long [B] -> [B, V+1]"""
        return self.item_logits(self.hidden_states(tokens)[torch.arange(len(tokens), device=tokens.device), positions])

    def next_item_logits(self, items: Tensor, lengths: Tensor) -> Tensor:
        """Scores of the item that follows each right-padded sequence. [B, V+1]"""
        tokens = F.pad(items, (0, 1), value=PAD_ITEM)
        tokens[torch.arange(len(items), device=items.device), lengths] = self.mask_token
        return self.logits_at(tokens, lengths)


def cloze_loss(model: BERT4Rec, batch: EventBatch, mask_probability: float = 0.2) -> Tensor:
    """Cross-entropy of hidden target events given the rest of their window.

    Each target is hidden with `mask_probability`; the last event of every window is always hidden,
    because predicting it is the task the model is used for.
    """
    last_position = _last_position(batch)
    is_hidden = batch.is_target & (torch.rand(batch.items.shape, device=batch.items.device) < mask_probability)
    is_hidden[torch.arange(len(batch.items), device=batch.items.device), last_position] = True
    tokens = batch.items.masked_fill(is_hidden, model.mask_token)
    return F.cross_entropy(model.item_logits(model.hidden_states(tokens)[is_hidden]), batch.items[is_hidden])


def last_item_loss(model: BERT4Rec, batch: EventBatch) -> Tensor:
    """Cross-entropy of the last event of every window given all events before it: a deterministic validation loss."""
    last_position = _last_position(batch)
    batch_index = torch.arange(len(batch.items), device=batch.items.device)
    tokens = batch.items.clone()
    tokens[batch_index, last_position] = model.mask_token
    return F.cross_entropy(model.logits_at(tokens, last_position), batch.items[batch_index, last_position])


def one_target_per_window(batch: EventBatch) -> Tensor:
    """What `last_item_loss` averages over: one prediction per window. bool [B]"""
    return torch.ones(len(batch.items), dtype=torch.bool, device=batch.items.device)


def _last_position(batch: EventBatch) -> Tensor:
    """Index of the last real event of every right-padded window. long [B]"""
    return (batch.items != PAD_ITEM).sum(dim=1) - 1
