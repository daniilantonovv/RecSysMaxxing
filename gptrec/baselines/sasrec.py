"""SASRec: self-attentive sequential recommendation (Kang & McAuley, 2018).

A causal Transformer like the GPT recommender, but with the original training objective: binary
cross-entropy of the observed next item against sampled negatives, instead of the full softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gptrec.data.batches import EventBatch
from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import EMBEDDING_INIT_STD, TransformerConfig, new_item_embedding
from gptrec.models.sequence_recommender import SequenceRecommender
from gptrec.training.supervised import predicts_target


class SASRec(SequenceRecommender):
    def __init__(self, num_items: int, config: TransformerConfig) -> None:
        super().__init__()
        self.num_items = num_items
        self.item_embedding = new_item_embedding(num_items, config.hidden_size, PAD_ITEM)
        self.position_embedding = nn.Embedding(config.max_positions, config.hidden_size)
        nn.init.normal_(self.position_embedding.weight, std=EMBEDDING_INIT_STD)
        self.embedding_dropout = nn.Dropout(config.dropout)
        block = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_heads,
            dim_feedforward=config.hidden_size,  # the paper's point-wise feed-forward layer keeps the model width
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, config.num_layers, norm=nn.LayerNorm(config.hidden_size), enable_nested_tensor=False)

    def hidden_states(self, items: Tensor) -> Tensor:
        length = items.shape[1]
        positions = torch.arange(length, device=items.device)
        embedded = self.embedding_dropout(self.item_embedding(items) + self.position_embedding(positions))
        sees_the_future = torch.ones(length, length, dtype=torch.bool, device=items.device).triu(diagonal=1)
        return self.blocks(embedded, mask=sees_the_future)


def sampled_binary_cross_entropy(model: SASRec, batch: EventBatch, num_negatives: int = 1) -> Tensor:
    """Every target event against `num_negatives` items drawn uniformly from the catalogue.

    The paper draws negatives the user never interacted with; with a large catalogue a uniform draw
    hits one of the user's items rarely enough to skip that check.
    """
    positions = predicts_target(batch)
    states = model.hidden_states(batch.items)[:, :-1][positions]  # [n, hidden]
    observed = batch.items[:, 1:][positions]  # [n]
    sampled = torch.randint(1, model.num_items + 1, (len(observed), num_negatives), device=observed.device)  # [n, k]
    item_embedding = model.item_embedding.weight
    observed_logits = (states * item_embedding[observed]).sum(dim=-1).float()
    sampled_logits = torch.einsum("nd,nkd->nk", states, item_embedding[sampled].to(states.dtype)).float()
    observed_loss = F.binary_cross_entropy_with_logits(observed_logits, torch.ones_like(observed_logits))
    sampled_loss = F.binary_cross_entropy_with_logits(sampled_logits, torch.zeros_like(sampled_logits))
    return observed_loss + sampled_loss
