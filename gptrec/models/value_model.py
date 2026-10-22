"""Value models: how much reward the recommender will still collect from a state (PPO's critic), and what the rest
of a list is worth after a given item (the look-ahead of guided decoding)."""

from __future__ import annotations

from dataclasses import replace

from torch import Tensor, nn

from gptrec.data.log import PAD_ITEM
from gptrec.models.backbone import CausalTransformer, new_item_embedding
from gptrec.models.policy import NextItemPolicy, PolicyConfig


class ValueModel(nn.Module):
    """A GPT-2 stack like the recommender's, with one number per position instead of a distribution over items.

    The output at position t is the expected return of everything generated after items 0..t.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        hidden_size = config.transformer.hidden_size
        self.item_embedding = new_item_embedding(config.num_items, hidden_size, PAD_ITEM)
        self.backbone = CausalTransformer(config.transformer)
        self.value_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_policy(cls, policy: NextItemPolicy) -> ValueModel:
        """A value model that starts from a copy of the recommender's embeddings and transformer, as in Petrov & Macdonald."""
        value_model = cls(policy.config)
        value_model.item_embedding.load_state_dict(policy.item_embedding.state_dict())
        value_model.backbone.load_state_dict(policy.backbone.state_dict())
        return value_model

    def forward(self, items: Tensor) -> Tensor:
        """long [B, L] -> float32 [B, L]"""
        return self.value_head(self.backbone(self.item_embedding(items))).squeeze(-1).float()


class ContinuationValueModel(nn.Module):
    """What the rest of a list is worth after each of its items: the discounted reward of the items still to come.

    A GPT-2 stack over item tokens like the recommender's, plus an embedding of each item's position in the list
    (0 for the items of the history): what follows an item is worth more the more items are still to come.
    """

    def __init__(self, config: PolicyConfig, horizon: int) -> None:
        super().__init__()
        self.config = config
        self.horizon = horizon
        hidden_size = config.transformer.hidden_size
        self.item_embedding = new_item_embedding(config.num_items, hidden_size, PAD_ITEM)
        self.list_position_embedding = nn.Embedding(horizon + 1, hidden_size)
        nn.init.zeros_(self.list_position_embedding.weight)  # at first it reads exactly what the recommender reads
        self.backbone = CausalTransformer(config.transformer)
        self.value_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_policy(cls, policy: NextItemPolicy, horizon: int) -> ContinuationValueModel:
        """A model that starts from a copy of the recommender's embeddings and transformer."""
        value_model = cls(replace(policy.config, item_offsets=False), horizon)
        value_model.item_embedding.load_state_dict(policy.item_embedding.state_dict())
        value_model.backbone.load_state_dict(policy.backbone.state_dict())
        return value_model

    def forward(self, items: Tensor, list_positions: Tensor) -> Tensor:
        """items, list_positions: long [B, L] -> float32 [B, L]; output t is the worth of what follows the item at t."""
        embeddings = self.item_embedding(items) + self.list_position_embedding(list_positions)
        return self.value_head(self.backbone(embeddings)).squeeze(-1).float()
