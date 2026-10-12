"""The value model (critic) of PPO: how much reward the recommender will still collect from a state."""

from __future__ import annotations

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
